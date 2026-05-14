"""Stage-2 streaming trainer with held-out val eval.

Differences vs `stage2_overfit.py`:
  - Streams sequences per iter (random sample with replacement from a train
    token pool) instead of caching N sequences on GPU.
  - Honours an external train/val token split (e.g. Part-01 62-scene train
    vs 23-scene val from `out/stage2_part1_splits.json`).
  - Periodic val-eval pass → mIoU / per-class IoU / occupancy IoU.
  - Saves three kinds of ckpts:
      * `ckpt_best_train.pt` — best smoothed train loss (cooldown-gated)
      * `ckpt_best_val.pt`   — best val mIoU
      * `ckpt_periodic.pt`   — rolling, overwritten every N iters (crash-recovery)
    No end-of-training ckpt is saved by default.
  - Final full-val eval emits `eval_final.json` with raw mIoU + restricted
    mIoU (min voxel-count filter) + per-class table.

Run:
    PYTHONUNBUFFERED=1 python -u -m s2go.tools.stage2_train \
        --splits-json out/stage2_part1_splits.json \
        --from-scratch \
        --iters 1500 \
        --eval-every 500 \
        --eval-num-val 50 \
        --final-eval-num-val 914 \
        --out-dir out/stage2_row_a_train_v1
"""
import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR, LambdaLR

from ..datasets.nusc_loader import NuScenesLoader
from ..models.backbone.r50_fpn import R50FPNBackbone
from ..stage2 import (
    GRID_SHAPE, NUM_CLASSES, EMPTY_CLASS_ID, PC_RANGE, VOXEL_SIZE, CLASS_NAMES,
)
from ..stage2.stage2_segmentor import S2GOStage2
from ..stage2.g2v import make_g2v_layer
from ..stage2.losses import compute_stage2_loss
from ..stage2.occ_dataset import Stage2OccLoader
from ..stage2.miou import MeanIoU


def to_device(d: dict, device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in d.items()}


def _build_ckpt(backbone, model, arch, sem_loss, n_iters_done, extra=None):
    cfg = {
        'K': int(arch['K']), 'J': int(arch['J']),
        'embed_dims': int(arch['embed_dims']),
        'num_layers': int(arch['num_layers']),
        'num_pts': int(arch['num_pts']),
        'feedforward_channels': int(arch['feedforward_channels']),
        'pc_range': list(PC_RANGE), 'voxel_size': list(VOXEL_SIZE),
        'grid_shape': list(GRID_SHAPE),
        'num_classes': NUM_CLASSES, 'empty_class_id': EMPTY_CLASS_ID,
        'sem_loss': sem_loss,
        'query_init': getattr(model, 'query_init', 'learned'),
        'iters_trained': n_iters_done,
        'stage': 2,
    }
    if extra:
        cfg.update(extra)
    return {
        'backbone': backbone.state_dict(),
        'segmentor': model.segmentor.state_dict(),
        'semantic_head': model.semantic_head.state_dict(),
        'config': cfg,
    }


def _filter_indices_by_tokens(occ_loader: Stage2OccLoader,
                              tokens: Sequence[str]) -> List[int]:
    """Return positions in occ_loader.indices whose loader.start_token is in `tokens`."""
    tok_set = set(tokens)
    out = []
    for pos, loader_idx in enumerate(occ_loader.indices):
        if occ_loader.loader.start_tokens[loader_idx] in tok_set:
            out.append(pos)
    return out


def _restricted_miou(per_class: Dict[str, Dict],
                      train_class_seen: Dict[str, int],
                      min_train_voxels: int,
                      min_val_voxels: int) -> Dict:
    """mIoU restricted to classes well-represented in both train and val.

    A class is kept if it has ≥min_train_voxels in train accumulation AND
    ≥min_val_voxels in val accumulation. Empty class is always excluded.
    """
    kept = []
    kept_names = []
    dropped = []
    for cname, st in per_class.items():
        if cname == CLASS_NAMES[EMPTY_CLASS_ID]:
            continue
        val_seen = int(st['seen'])
        train_seen = int(train_class_seen.get(cname, 0))
        if train_seen >= min_train_voxels and val_seen >= min_val_voxels:
            iou = st['iou']
            if iou == iou:                        # not NaN
                kept.append(iou)
                kept_names.append(cname)
        else:
            dropped.append((cname, train_seen, val_seen))
    rmiou = (sum(kept) / len(kept)) if kept else float('nan')
    return {'restricted_mIoU': rmiou,
            'kept_classes':    kept_names,
            'dropped_classes': dropped,
            'n_kept':          len(kept),
            'n_dropped':       len(dropped)}


@torch.no_grad()
def _eval_pass(backbone, model, g2v_eval, occ_loader: Stage2OccLoader,
               val_positions: List[int], device, occ_thresh: float = 0.5,
               max_seqs: Optional[int] = None,
               verbose_every: int = 0) -> Dict:
    """Run val eval over `val_positions[:max_seqs]`. Returns MeanIoU stats."""
    backbone.eval(); model.eval()
    metric = MeanIoU(ignore_classes=[EMPTY_CLASS_ID])
    n = min(max_seqs, len(val_positions)) if max_seqs else len(val_positions)
    t_eval = time.time()
    for k in range(n):
        pos = val_positions[k]
        seq_cpu = occ_loader[pos]
        f = to_device(seq_cpu[0], device)            # T=1 eval
        feat, ss, lsi, _ = backbone(f['imgs'])
        frame = {**f, 'feat_flatten': feat,
                 'spatial_shapes': ss, 'level_start_index': lsi,
                 'pad_h': 256, 'pad_w': 704}
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            out = model.forward_one_frame(frame)
        torch.cuda.empty_cache()
        with torch.cuda.amp.autocast(enabled=False):
            G = out.raw.gaussians
            occ_pred, sem_logits = g2v_eval(
                means=G.means.float(),
                rotations=G.rotations.float(),
                scales=G.scales.float(),
                opacities=G.opacities.float(),
                class_logits=out.sem_logits_per_g.float(),
            )
        pred_argmax = sem_logits.argmax(dim=-1)
        empty_mask = (occ_pred < occ_thresh)
        pred_argmax = torch.where(empty_mask,
                                  torch.full_like(pred_argmax, EMPTY_CLASS_ID),
                                  pred_argmax)
        sem_gt = f['sem_voxel_gt']
        metric.update(pred_argmax[0], sem_gt[0])
        if verbose_every and (k + 1) % verbose_every == 0:
            print(f"      eval {k+1}/{n} ({time.time()-t_eval:.0f}s)",
                  flush=True)
    backbone.train(); model.train()
    stats = metric.compute()
    stats['eval_sec'] = time.time() - t_eval
    stats['n_sequences'] = n
    return stats


def _accumulate_train_class_counts(train_class_seen: Dict[str, int],
                                    sem_voxel_gt: torch.Tensor) -> None:
    """Add per-class voxel counts from one train GT tensor."""
    gt = sem_voxel_gt[0].reshape(-1)
    for c in range(NUM_CLASSES):
        n_c = int((gt == c).sum().item())
        if n_c > 0:
            train_class_seen[CLASS_NAMES[c]] = \
                train_class_seen.get(CLASS_NAMES[c], 0) + n_c


def main(splits_json: str,
         out_dir: str,
         stage1_ckpt: Optional[str] = None,
         from_scratch: bool = True,
         query_init: str = 'learned',
         num_layers_override: Optional[int] = None,
         num_pts_override: Optional[int] = None,
         ffn_override: Optional[int] = None,
         n_iters: int = 1500,
         t_seq: int = 1,
         t_queue: int = 1,
         lr: float = 2e-4,
         lr_backbone_mult: float = 0.25,
         lr_schedule: str = 'cosine',
         lr_min: float = 0.0,
         warmup_iters: int = 500,
         grad_accum_steps: int = 1,
         grad_clip_max_norm: float = 35.0,
         scale_min_override: float = 0.05,
         nan_abort_window: int = 50,
         sem_loss: str = 'both',
         w_occ: float = 1.0, w_kl: float = 1.0,
         w_ce: float = 10.0, w_lovasz: float = 1.0,
         occ_pos_weight: float = 1.0,
         ignore_empty_in_sem: bool = True,
         g2v_backend: str = 'cuda',
         g2v_chunk_voxels_train: int = 2048,
         g2v_chunk_voxels_eval: int = 4096,
         voxel_sample_size: int = 4096,
         occ_thresh: float = 0.5,
         eval_every: int = 500,
         eval_num_val: int = 50,
         final_eval_num_val: int = 914,
         save_periodic_every: int = 200,
         save_best_window: int = 50,
         save_best_train_cooldown: int = 100,
         restricted_min_train_voxels: int = 1000,
         restricted_min_val_voxels: int = 100,
         log_every: int = 25,
         seed: int = 0):
    device = "cuda"
    torch.manual_seed(seed)
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    torch.cuda.empty_cache()

    # ── [1/6] Loader + train/val split ────────────────────────────────────
    print(f"Stage-2 streaming train: {n_iters} iters  "
          f"({'FROM SCRATCH' if from_scratch else 'init from Stage 1 ckpt'})")
    print(f"  out_dir: {out_dir}")
    print(f"  splits_json: {splits_json}")
    print("[1/6] Building loader + splits…")
    with open(splits_json, 'r') as fp:
        splits = json.load(fp)
    train_tokens = splits['train_start_tokens']
    val_tokens   = splits['val_start_tokens']
    print(f"    splits: train={len(train_tokens)} val={len(val_tokens)} "
          f"(scenes: train={len(splits['train_scenes'])}, "
          f"val={len(splits['val_scenes'])})")

    base = NuScenesLoader(T=t_seq, verbose=False)
    occ_loader = Stage2OccLoader(base, restrict_to_covered=True)
    train_pos = _filter_indices_by_tokens(occ_loader, train_tokens)
    val_pos   = _filter_indices_by_tokens(occ_loader, val_tokens)
    print(f"    GT-covered base: {len(occ_loader)}")
    print(f"    after token filter: train_pos={len(train_pos)} "
          f"val_pos={len(val_pos)}")

    # ── [2/6] Model arch ──────────────────────────────────────────────────
    if from_scratch:
        print("[2/6] From-scratch arch (no Stage-1 ckpt)")
        # Paper-spec T2 (= paper Table 3 architecture, same across rows a-e).
        # The earlier T0-tier default (num_layers=2, num_pts=4, ffn=2048) was
        # a 12 GB GPU workaround from before the CUDA G2V backend dropped peak
        # graph memory ~70 GB → ~360 MB. Override at the CLI with --num-layers
        # / --num-pts / --ffn for dev / smoke runs if T2 OOMs on smaller cards.
        arch = dict(K=900, J=10, embed_dims=768,
                    num_layers=6, num_pts=13, feedforward_channels=3072)
        ckpt = None
    else:
        assert stage1_ckpt and os.path.isfile(stage1_ckpt), \
            f"stage1 ckpt not found: {stage1_ckpt}"
        print(f"[2/6] Reading Stage-1 ckpt: {stage1_ckpt}")
        ckpt = torch.load(stage1_ckpt, map_location=device)
        s1_cfg = ckpt.get('config', {})
        arch = dict(
            K=int(s1_cfg.get('K', 900)),
            J=int(s1_cfg.get('J', 10)),
            embed_dims=int(s1_cfg.get('embed_dims', 768)),
            num_layers=int(s1_cfg.get('num_layers', 6)),
            num_pts=int(s1_cfg.get('num_pts', 13)),
            feedforward_channels=int(s1_cfg.get('feedforward_channels', 3072)),
        )

    if num_layers_override is not None:
        assert from_scratch
        arch['num_layers'] = int(num_layers_override)
    if num_pts_override is not None:
        assert from_scratch
        arch['num_pts'] = int(num_pts_override)
    if ffn_override is not None:
        assert from_scratch
        arch['feedforward_channels'] = int(ffn_override)
    print(f"    arch: {arch}")

    # ── [3/6] Build backbone + model ──────────────────────────────────────
    print("[3/6] Building backbone + Stage 2 model…")
    backbone = R50FPNBackbone(embed_dims=arch['embed_dims'],
                              num_outs=4, pretrained=True).to(device)
    model = S2GOStage2(
        segmentor_kwargs=dict(
            K=arch['K'], J=arch['J'], embed_dims=arch['embed_dims'],
            num_layers=arch['num_layers'],
            num_heads=12, num_groups=12,
            num_levels=4, num_cams=6,
            num_pts=arch['num_pts'],
            feedforward_channels=arch['feedforward_channels'],
            T_queue=t_queue,
        ),
        num_classes=NUM_CLASSES,
        query_init=query_init,
    ).to(device)
    if ckpt is not None:
        bb_missing, bb_unexpected = backbone.load_state_dict(ckpt['backbone'], strict=False)
        seg_missing, seg_unexpected, head_partial = \
            model.load_stage1_state(ckpt, strict=False)
        print("    backbone + segmentor: Stage-1 weights loaded")
        print(f"      backbone:  missing={len(bb_missing)}, unexpected={len(bb_unexpected)}")
        print(f"      segmentor: missing={len(seg_missing)}, unexpected={len(seg_unexpected)}, "
              f"head_partial_transferred={head_partial}")
    else:
        print("    backbone: torchvision ResNet50 pretrained init")
        print("    segmentor + semantic head: random init")

    # Numerical stability: raise scale_min from default 0.01 m → scale_min_override
    # (typically 0.05 m). At s=0.01, 1/s² = 1e4 in Σ⁻¹ and 1/√det blows up under
    # gradient. At s=0.05 the gradient floor is 400× smaller (1/s² = 400).
    # Without aux losses (Stage-1's depth/RGB/denoise), nothing prevents
    # scale-collapse to the boundary, so we tighten the floor.
    prev_scale_min = float(model.segmentor.child_head.scale_min)
    model.segmentor.child_head.scale_min = float(scale_min_override)
    print(f"    scale_min override: {prev_scale_min} → "
          f"{model.segmentor.child_head.scale_min}  (Gaussian floor in metres)")

    n_back = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    trainable: backbone={n_back/1e6:.1f} M, "
          f"model={n_model/1e6:.1f} M, total={(n_back+n_model)/1e6:.1f} M")

    # ── [4/6] G2V layers (train sparse, eval dense) ───────────────────────
    g2v_train = make_g2v_layer(backend=g2v_backend,
                               chunk_voxels=g2v_chunk_voxels_train).to(device)
    g2v_eval  = make_g2v_layer(backend=g2v_backend,
                               chunk_voxels=g2v_chunk_voxels_eval).to(device)
    print(f"[4/6] G2V backend: {g2v_backend}  "
          f"(train sparse {voxel_sample_size} voxels/iter, eval dense)")

    optim = AdamW([
        {'params': [p for p in backbone.parameters() if p.requires_grad],
         'lr': lr * lr_backbone_mult},
        {'params': model.parameters(), 'lr': lr},
    ], weight_decay=0.01)

    # Gradient accumulation: N micro-iters per optimizer step (effective batch ≈ N).
    # Scheduler ticks per OPTIMIZER STEP, not per micro-iter, so T_max is in
    # optim-steps. `n_iters` remains the total micro-iter budget (samples seen).
    if grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    n_optim_steps = (n_iters + grad_accum_steps - 1) // grad_accum_steps

    if lr_schedule == 'cosine':
        scheduler = CosineAnnealingLR(optim, T_max=n_optim_steps, eta_min=lr_min)
        print(f"    LR schedule: cosine "
              f"(T_max={n_optim_steps} optim-steps, eta_min={lr_min:.0e})")
    elif lr_schedule == 'constant':
        scheduler = ConstantLR(optim, factor=1.0, total_iters=n_optim_steps)
        print(f"    LR schedule: constant "
              f"(peak lr held for all {n_optim_steps} optim-steps)")
    elif lr_schedule == 'warmup_cosine':
        # Linear warmup 0 → peak over `warmup_iters` micro-iters, then cosine
        # decay to peak × 0.10 over the remaining (n_iters - warmup_iters)
        # micro-iters. Scheduler ticks once per optim step, so convert both
        # to optim-step units (float, since fractional).
        warmup_optim = warmup_iters / grad_accum_steps
        if warmup_optim >= n_optim_steps:
            raise ValueError(
                f"warmup_iters={warmup_iters} ≥ n_iters={n_iters} after "
                f"dividing by grad_accum_steps={grad_accum_steps}; warmup "
                f"spans the whole run.")
        def _warmup_cosine_lambda(s):
            if s < warmup_optim:
                return s / warmup_optim
            progress = min((s - warmup_optim) / (n_optim_steps - warmup_optim), 1.0)
            cosine_mul = 0.5 * (1.0 + math.cos(math.pi * progress))
            return 0.10 + 0.90 * cosine_mul
        scheduler = LambdaLR(optim, lr_lambda=_warmup_cosine_lambda)
        print(f"    LR schedule: warmup_cosine "
              f"(warmup {warmup_iters} micro-iters ≈ {warmup_optim:.1f} optim-steps, "
              f"cosine to peak×0.10 over remaining "
              f"{n_optim_steps - warmup_optim:.1f})")
    else:
        raise ValueError(
            f"unknown lr_schedule '{lr_schedule}' "
            f"(use 'cosine', 'constant', or 'warmup_cosine')")
    if grad_accum_steps > 1:
        print(f"    gradient accumulation: {grad_accum_steps} micro-iters per "
              f"optimizer step → {n_optim_steps} optim-steps over {n_iters} "
              f"micro-iters (effective batch ≈ {grad_accum_steps})")
    sem_ignore_idx = EMPTY_CLASS_ID if ignore_empty_in_sem else -1

    # ── [5/6] Train loop ──────────────────────────────────────────────────
    print(f"[5/6] Streaming training for {n_iters} iters "
          f"(log {log_every}, eval every {eval_every}, save_periodic_every "
          f"{save_periodic_every}, NaN-guarded, grad-clip={grad_clip_max_norm})\n")
    print(f"  iter |   total |  occ_bce |  sem_kl |  sem_ce | lovasz |  gnorm "
          f"| opa_mean | sec | mem MB")
    print(f"  -----+---------+----------+---------+---------+--------+--------"
          f"+----------+-----+-------")
    backbone.train(); model.train()

    history: List[Dict] = []
    eval_history: List[Dict] = []
    train_class_seen: Dict[str, int] = {}
    best_smoothed_train = float('inf')
    last_best_train_save_iter = -10**9
    n_best_train_saves = 0
    best_val_miou = -1.0
    n_best_val_saves = 0
    n_skipped = 0
    n_skipped_windows = 0
    window_skipped = False
    t_start = time.time()

    ckpt_best_train = os.path.join(out_dir, 'ckpt_best_train.pt')
    ckpt_best_val   = os.path.join(out_dir, 'ckpt_best_val.pt')
    ckpt_periodic   = os.path.join(out_dir, 'ckpt_periodic.pt')

    for i in range(n_iters):
        # Sample a train sequence (with replacement)
        pos = train_pos[random.randrange(len(train_pos))]
        seq_cpu = occ_loader[pos]
        seq = [to_device(f, device) for f in seq_cpu]
        for f in seq:
            _accumulate_train_class_counts(train_class_seen, f['sem_voxel_gt'])

        sequence_with_feat = []
        for f in seq:
            feat, ss, lsi, _ = backbone(f['imgs'])
            sequence_with_feat.append({**f,
                'feat_flatten': feat,
                'spatial_shapes': ss,
                'level_start_index': lsi,
                'pad_h': 256, 'pad_w': 704})

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            outs = model(sequence_with_feat)

        torch.cuda.empty_cache()
        L_total_seq = torch.zeros((), device=device, dtype=torch.float32)
        diag_acc = {'occ_bce': 0.0, 'sem_kl': 0.0, 'sem_ce': 0.0,
                    'sem_lovasz': 0.0, 'total': 0.0}
        for t, out in enumerate(outs):
            G = out.raw.gaussians
            sem_per_g = out.sem_logits_per_g
            sem_voxel_gt = sequence_with_feat[t]['sem_voxel_gt']
            occ_voxel_gt = sequence_with_feat[t]['occ_voxel_gt']
            with torch.cuda.amp.autocast(enabled=False):
                # Sparse: balanced occupied/empty sampling (matches overfit.py)
                occ_flat = occ_voxel_gt[0].reshape(-1)
                occ_idx = occ_flat.nonzero(as_tuple=False).squeeze(-1)
                emp_idx = (~occ_flat).nonzero(as_tuple=False).squeeze(-1)
                n_pos = min(voxel_sample_size // 2, occ_idx.numel())
                n_neg = voxel_sample_size - n_pos
                pos_pick = occ_idx[torch.randperm(occ_idx.numel(),
                                                  device=device)[:n_pos]]
                neg_pick = emp_idx[torch.randperm(emp_idx.numel(),
                                                  device=device)[:n_neg]]
                sample_idx = torch.cat([pos_pick, neg_pick], dim=0).unsqueeze(0)
                sem_gt_flat = sem_voxel_gt[0].reshape(-1).index_select(0, sample_idx[0])
                occ_gt_flat = occ_voxel_gt[0].reshape(-1).index_select(0, sample_idx[0])
                occ_pred, sem_logits = g2v_train.forward_sparse(
                    means=G.means.float(),
                    rotations=G.rotations.float(),
                    scales=G.scales.float(),
                    opacities=G.opacities.float(),
                    class_logits=sem_per_g.float(),
                    voxel_flat_idx=sample_idx,
                )
                occ_pred_re = occ_pred.unsqueeze(-1).unsqueeze(-1)
                sem_logits_re = sem_logits.unsqueeze(-2).unsqueeze(-2)
                occ_gt_re = occ_gt_flat.view(1, -1, 1, 1)
                sem_gt_re = sem_gt_flat.view(1, -1, 1, 1)
                L_t, diag_t = compute_stage2_loss(
                    occ_pred=occ_pred_re, sem_logits=sem_logits_re,
                    occ_gt=occ_gt_re, sem_gt=sem_gt_re,
                    sem_loss=sem_loss,
                    w_occ=w_occ, w_kl=w_kl, w_ce=w_ce, w_lovasz=w_lovasz,
                    occ_pos_weight=occ_pos_weight,
                    ignore_index=sem_ignore_idx,
                )
            L_total_seq = L_total_seq + L_t
            for k in diag_acc:
                diag_acc[k] += diag_t[k]
        L = L_total_seq / len(outs)
        for k in diag_acc:
            diag_acc[k] /= len(outs)

        # ── Grad-accum bookkeeping ───────────────────────────────────────
        # N micro-iters form one optimizer-step "window". zero_grad at the
        # window start, backward scales loss by 1/N so the accumulated
        # gradient is the per-sample MEAN (not sum) — keeps the effective LR
        # consistent with paper-spec batched training. clip + step +
        # scheduler.step + NaN-diag fire at window end only.
        is_accum_start = (i % grad_accum_steps == 0)
        is_accum_end   = ((i + 1) % grad_accum_steps == 0) or (i == n_iters - 1)
        if is_accum_start:
            optim.zero_grad(set_to_none=True)
            window_skipped = False

        L_micro = float(L.item())
        L_finite = math.isfinite(L_micro)
        if not window_skipped and L_finite:
            (L / grad_accum_steps).backward()
        elif not L_finite and not window_skipped:
            # First micro-iter NaN in this window → latch and abandon window.
            window_skipped = True
            n_skipped += 1
            print(f"  [skip] iter {i}: non-finite micro-loss "
                  f"(L={L_micro:.3g}, latching window)", flush=True)
            optim.zero_grad(set_to_none=True)        # nuke any poisoned grads
        else:
            # Window already poisoned — count this micro-iter as skipped too.
            n_skipped += 1

        opa_mean = float(outs[-1].raw.gaussians.opacities.detach()
                         .float().mean().item())

        # ── Window-end: clip, NaN-check accumulated grads, step, schedule ─
        if is_accum_end:
            if window_skipped:
                gnorm_val = float('nan')
                skipped   = True
            else:
                gnorm = nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(model.parameters()),
                    max_norm=grad_clip_max_norm)
                gnorm_val = float(gnorm.item())
                if not math.isfinite(gnorm_val):
                    # Accumulated grads went non-finite even though every
                    # micro-iter loss was finite — overflow happened in
                    # gsplat / G2V backward (most likely scale collapse on
                    # bf16 boundary). Skip step + dump rich diag on first
                    # window-level NaN.
                    n_skipped += 1
                    n_skipped_windows += 1
                    if n_skipped_windows == 1:
                        G_last = outs[-1].raw.gaussians
                        tok = seq[0].get('_sample_token', '?')
                        with torch.no_grad():
                            sc_min = float(G_last.scales.min().item())
                            sc_max = float(G_last.scales.max().item())
                            op_min = float(G_last.opacities.min().item())
                            op_max = float(G_last.opacities.max().item())
                        print(f"  [NaN-DIAG] first window-NaN at iter {i}: "
                              f"token={tok}", flush=True)
                        print(f"  [NaN-DIAG] L_micro={L_micro:.4g}  "
                              f"gnorm={gnorm_val:.4g}", flush=True)
                        print(f"  [NaN-DIAG] Gaussian scales:    "
                              f"min={sc_min:.4g}  max={sc_max:.4g}", flush=True)
                        print(f"  [NaN-DIAG] Gaussian opacities: "
                              f"min={op_min:.4g}  max={op_max:.4g}", flush=True)
                        grad_stats = []
                        for n, p in list(backbone.named_parameters()) + \
                                [(f"model.{nm}", pp)
                                 for nm, pp in model.named_parameters()]:
                            if p.grad is None:
                                continue
                            g = p.grad
                            has_nan = bool(torch.isnan(g).any().item())
                            has_inf = bool(torch.isinf(g).any().item())
                            if has_nan or has_inf:
                                grad_stats.append(
                                    (n, has_nan, has_inf,
                                     float(g.abs().nan_to_num(0).max().item())))
                        print(f"  [NaN-DIAG] {len(grad_stats)} params with "
                              f"NaN/Inf grad:", flush=True)
                        for n, hn, hi, maxabs in grad_stats[:15]:
                            print(f"    {n:60s}  NaN={hn} Inf={hi} "
                                  f"max|g|={maxabs:.4g}", flush=True)
                    print(f"  [skip-window] iter {i}: non-finite accumulated "
                          f"gnorm (gnorm={gnorm_val:.3g})", flush=True)
                    optim.zero_grad(set_to_none=True)
                    skipped = True
                else:
                    optim.step()
                    scheduler.step()    # advance LR only on real optimizer steps
                    skipped = False

            # Auto-abort if we've skipped N consecutive micro-iters at tail.
            # Under grad-accum=K, a fully-poisoned window contributes K skipped
            # entries (one per micro-iter), so to keep the wall-clock-equivalent
            # window unchanged, scale --nan-abort-window by K when using accum.
            if n_skipped >= nan_abort_window:
                consec = 0
                for h in reversed(history):
                    if h.get('skipped', False):
                        consec += 1
                    else:
                        break
                if consec >= nan_abort_window - 1:
                    print(f"  [ABORT] {consec+1} consecutive NaN-skipped "
                          f"iters >= {nan_abort_window} — stopping early.",
                          flush=True)
                    break
        else:
            # Mid-window: no optimizer state to report; carry NaN gnorm.
            gnorm_val = float('nan')
            skipped   = window_skipped

        history.append({**diag_acc, 'iter': i,
                         'gnorm': gnorm_val,
                         'lr': scheduler.get_last_lr()[0],
                         'skipped': skipped,
                         'window_end': is_accum_end,
                         'opa_mean': opa_mean})

        # Log line: print at every Kth optimizer step (window-end) so the
        # gnorm column is always meaningful (never NaN-from-mid-window). With
        # grad_accum_steps=1 this is every K micro-iters, matching old behavior.
        optim_step_idx = ((i + 1) // grad_accum_steps - 1) if is_accum_end else -1
        if (is_accum_end and optim_step_idx % log_every == 0) \
                or i == n_iters - 1:
            elapsed = time.time() - t_start
            mem = torch.cuda.max_memory_allocated() / 1024**2
            gnorm_disp = "  nan " if not math.isfinite(gnorm_val) \
                          else f"{gnorm_val:>6.2f}"
            print(f"  {i:>4} | {L_micro:>7.3f} | {diag_acc['occ_bce']:>8.4f} "
                  f"| {diag_acc['sem_kl']:>7.3f} | {diag_acc['sem_ce']:>7.3f} "
                  f"| {diag_acc['sem_lovasz']:>6.3f} | {gnorm_disp} "
                  f"| {opa_mean:>8.3f} | {elapsed:>3.0f} | {mem:>6.0f}",
                  flush=True)

        # ── Save-best by train smoothed loss ──────────────────────────────
        # Gate on window-end so we don't save mid-accumulation (where `skipped`
        # carries the latched window-skipped flag rather than a real verdict).
        if is_accum_end and not skipped and len(history) >= save_best_window:
            recent = [h['total'] for h in history[-save_best_window:]
                      if not h['skipped'] and h['total'] == h['total']]
            if len(recent) >= save_best_window // 2:
                smoothed = sum(recent) / len(recent)
                cooldown_ok = (i - last_best_train_save_iter) >= save_best_train_cooldown
                if smoothed < best_smoothed_train and cooldown_ok:
                    best_smoothed_train = smoothed
                    last_best_train_save_iter = i
                    n_best_train_saves += 1
                    extra = {'best_smoothed_train_loss': smoothed,
                             'best_window': save_best_window}
                    torch.save(_build_ckpt(backbone, model, arch, sem_loss,
                                            i + 1, extra),
                               ckpt_best_train)
                    print(f"  [best-train #{n_best_train_saves}] iter {i}: "
                          f"smoothed L({save_best_window}) = {smoothed:.4f} → "
                          f"{ckpt_best_train}", flush=True)

        # ── Periodic crash-recovery save ──────────────────────────────────
        # Gate on optimizer-step count (not micro-iter count) so the cadence
        # is independent of grad_accum_steps. With grad_accum_steps=16 and
        # save_periodic_every=200, the old `i % 200 == 0` gate was almost
        # never satisfied jointly with `is_accum_end` (modulo collision).
        if is_accum_end and save_periodic_every and optim_step_idx > 0 \
                and optim_step_idx % save_periodic_every == 0:
            torch.save(_build_ckpt(backbone, model, arch, sem_loss, i + 1,
                                    {'kind': 'periodic'}),
                       ckpt_periodic)

        # ── Periodic val eval ─────────────────────────────────────────────
        # Gate on optimizer-step count — see save-periodic block above.
        if is_accum_end and eval_every and optim_step_idx > 0 \
                and optim_step_idx % eval_every == 0:
            print(f"  [eval @ iter {i}] running over "
                  f"{min(eval_num_val, len(val_pos))} val sequences…",
                  flush=True)
            stats = _eval_pass(backbone, model, g2v_eval, occ_loader,
                                val_pos, device, occ_thresh,
                                max_seqs=eval_num_val)
            rstats = _restricted_miou(stats['per_class'], train_class_seen,
                                      restricted_min_train_voxels,
                                      restricted_min_val_voxels)
            eval_rec = {
                'iter': i,
                'mIoU':           stats['mIoU'],
                'occupancy_IoU':  stats['occupancy_IoU'],
                'restricted_mIoU': rstats['restricted_mIoU'],
                'n_kept_classes':  rstats['n_kept'],
                'eval_sec':        stats['eval_sec'],
                'n_sequences':     stats['n_sequences'],
            }
            eval_history.append(eval_rec)
            print(f"    mIoU={stats['mIoU']*100:.2f}%  "
                  f"restricted={rstats['restricted_mIoU']*100:.2f}% "
                  f"(over {rstats['n_kept']} classes)  "
                  f"occ_IoU={stats['occupancy_IoU']*100:.2f}%  "
                  f"({stats['eval_sec']:.0f}s)", flush=True)
            if stats['mIoU'] > best_val_miou:
                best_val_miou = stats['mIoU']
                n_best_val_saves += 1
                extra = {'best_val_mIoU': stats['mIoU'],
                         'best_val_restricted_mIoU': rstats['restricted_mIoU'],
                         'best_val_iter': i,
                         'kind': 'best_val'}
                torch.save(_build_ckpt(backbone, model, arch, sem_loss,
                                        i + 1, extra),
                           ckpt_best_val)
                print(f"    [best-val #{n_best_val_saves}] mIoU={stats['mIoU']*100:.2f}% "
                      f"→ {ckpt_best_val}", flush=True)

    print()
    print(f"Training done. best_smoothed_train={best_smoothed_train:.4f} "
          f"({n_best_train_saves} saves)   "
          f"best_val_mIoU={best_val_miou*100:.2f}% "
          f"({n_best_val_saves} saves)")

    # Save train + eval history
    with open(os.path.join(out_dir, 'training_history.json'), 'w') as fp:
        json.dump(history, fp, indent=2)
    with open(os.path.join(out_dir, 'eval_history.json'), 'w') as fp:
        json.dump(eval_history, fp, indent=2)
    print(f"  history → {out_dir}/training_history.json + eval_history.json")

    # ── [6/6] Final full-val eval ─────────────────────────────────────────
    print(f"[6/6] Final eval on {min(final_eval_num_val, len(val_pos))} "
          f"val sequences (out of {len(val_pos)} held-out)…")
    final = _eval_pass(backbone, model, g2v_eval, occ_loader,
                        val_pos, device, occ_thresh,
                        max_seqs=final_eval_num_val,
                        verbose_every=100)
    final_r = _restricted_miou(final['per_class'], train_class_seen,
                                restricted_min_train_voxels,
                                restricted_min_val_voxels)
    print(f"  raw mIoU (17 classes excl. empty): {final['mIoU']*100:.2f}%")
    print(f"  restricted mIoU (≥{restricted_min_train_voxels} train + "
          f"≥{restricted_min_val_voxels} val voxels, "
          f"{final_r['n_kept']} kept classes): "
          f"{final_r['restricted_mIoU']*100:.2f}%")
    print(f"  occupancy IoU: {final['occupancy_IoU']*100:.2f}%")
    print(f"  per-class IoU:")
    print(f"    {'class':>20s} | {'iou':>6s} | {'val_seen':>10s} | "
          f"{'train_seen':>12s}")
    print(f"    {'-'*20}-+-{'-'*6}-+-{'-'*10}-+-{'-'*12}")
    for cname, st in final['per_class'].items():
        if cname == CLASS_NAMES[EMPTY_CLASS_ID]:
            continue
        iou = st['iou']
        iou_s = f"{iou*100:.2f}%" if iou == iou else "  n/a"
        train_seen = train_class_seen.get(cname, 0)
        print(f"    {cname:>20s} | {iou_s:>6s} | {st['seen']:>10d} | "
              f"{train_seen:>12d}")

    final_out = {
        'mIoU':            final['mIoU'],
        'restricted_mIoU': final_r['restricted_mIoU'],
        'occupancy_IoU':   final['occupancy_IoU'],
        'n_voxels_seen':   final['n_voxels_seen'],
        'n_val_sequences': final['n_sequences'],
        'eval_sec':        final['eval_sec'],
        'per_class':       final['per_class'],
        'train_class_seen': train_class_seen,
        'restricted': {
            'min_train_voxels': restricted_min_train_voxels,
            'min_val_voxels':   restricted_min_val_voxels,
            'kept_classes':     final_r['kept_classes'],
            'dropped_classes':  final_r['dropped_classes'],
            'n_kept':           final_r['n_kept'],
            'n_dropped':        final_r['n_dropped'],
        },
        'best_val_mIoU':            best_val_miou,
        'best_smoothed_train_loss': best_smoothed_train,
        'occ_threshold': occ_thresh,
        'iters_total':   n_iters,
        'iters_skipped': n_skipped,
    }
    with open(os.path.join(out_dir, 'eval_final.json'), 'w') as fp:
        json.dump(final_out, fp, indent=2, default=float)
    print(f"  eval_final → {out_dir}/eval_final.json")
    print(f"  ckpts: best_train={'exists' if os.path.isfile(ckpt_best_train) else 'NONE'} "
          f"best_val={'exists' if os.path.isfile(ckpt_best_val) else 'NONE'} "
          f"periodic={'exists' if os.path.isfile(ckpt_periodic) else 'NONE'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits-json", type=str, required=True,
                    help="path to JSON with train_start_tokens + val_start_tokens")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--stage1-ckpt", type=str, default=None)
    p.add_argument("--from-scratch", action="store_true", default=False,
                    help="train Stage 2 from scratch (row (a) ablation). "
                         "Implicit when --stage1-ckpt is not provided.")
    p.add_argument("--query-init", choices=['learned', 'fps_lidar'],
                    default='learned')
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--num-pts", type=int, default=None)
    p.add_argument("--ffn", type=int, default=None)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--t-seq", type=int, default=1)
    p.add_argument("--t-queue", type=int, default=1)
    # NOTE: --lr default is set further below (2e-4, paper-aligned at batch 1).
    p.add_argument("--grad-clip", type=float, default=35.0,
                    help="Paper §3.4.4 value. Previously 10.0 as a defensive cap "
                         "during the v1 NaN cascade — that capped every iter; "
                         "with the cross-attn fixes in, 35.0 lets natural "
                         "gradient magnitudes flow.")
    p.add_argument("--sem-loss", choices=['kl', 'ce_lovasz', 'both'],
                    default='both')
    p.add_argument("--w-occ", type=float, default=1.0)
    p.add_argument("--w-kl", type=float, default=1.0)
    p.add_argument("--w-ce", type=float, default=10.0)
    p.add_argument("--w-lovasz", type=float, default=1.0)
    p.add_argument("--occ-pos-weight", type=float, default=1.0,
                    help="BCE pos_weight on the occupancy loss. Default 1.0 "
                         "matches the balanced sparse sampler (n_pos = "
                         "voxel_sample_size//2; positives are ~50pct of every "
                         "batch). Raise to ~16 only if you switch to dense "
                         "full-grid training where >95pct of voxels are empty. "
                         "Previously defaulted to 16.0, which double-corrected "
                         "the class imbalance under the sparse path and biased "
                         "the model toward predicting occupied everywhere.")
    p.add_argument("--no-ignore-empty", action="store_true")
    p.add_argument("--g2v-backend", choices=['torch', 'cuda'], default='cuda')
    p.add_argument("--voxel-sample-size", type=int, default=4096)
    p.add_argument("--occ-thresh", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-num-val", type=int, default=50)
    p.add_argument("--final-eval-num-val", type=int, default=914)
    p.add_argument("--save-periodic-every", type=int, default=200)
    p.add_argument("--save-best-window", type=int, default=50)
    p.add_argument("--save-best-train-cooldown", type=int, default=100)
    p.add_argument("--restricted-min-train-voxels", type=int, default=1000)
    p.add_argument("--restricted-min-val-voxels", type=int, default=100)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale-min", type=float, default=0.05,
                    help="Gaussian scale floor in metres. Default 0.05 is the "
                         "empirically-validated value (the only 1500-iter "
                         "row-(a) v2 run that finished with 0/1500 NaN-skips). "
                         "GF-2 / paper default is 0.01 but in our bf16 setup "
                         "that drives 1/s^4 backward magnitudes ~10^8 into the "
                         "bf16 mantissa cancellation cliff, producing a sharp "
                         "NaN cascade (smoke_tier_a iter 62; v4_10k iter 1071 "
                         "— always within 12-16 iters of the first scale-floor "
                         "touch). The v3 revert to 0.01 was based on the "
                         "working assumption that cross-attn fixes addressed "
                         "this; v4_10k disproved that.")
    p.add_argument("--nan-abort-window", type=int, default=50,
                    help="Stop training after N consecutive NaN-skipped iters.")
    p.add_argument("--lr", type=float, default=2e-4,
                    help="Paper §3.4.4 uses 4e-4 at batch 16. We're at batch 1; "
                         "2e-4 is the sqrt-scaling-rule equivalent. Previously "
                         "1e-4 as part of the v1 NaN workaround.")
    p.add_argument("--lr-schedule", choices=['cosine', 'constant', 'warmup_cosine'],
                    default='cosine',
                    help="'cosine' (default, v3 behavior): decays peak→eta_min over "
                         "n_optim_steps. 'constant': holds peak. 'warmup_cosine': "
                         "linear warmup 0→peak over --warmup-iters micro-iters, "
                         "then cosine to peak×0.10. The paper-spec choice for "
                         "long runs (use with --grad-accum-steps 16 for batch-16 "
                         "equivalent).")
    p.add_argument("--lr-min", type=float, default=0.0,
                    help="eta_min for the 'cosine' schedule. Ignored for the "
                         "others. Default 0.0 (matches v3).")
    p.add_argument("--warmup-iters", type=int, default=500,
                    help="warmup length in MICRO-ITERS for --lr-schedule "
                         "warmup_cosine. Ignored for other schedules. Default 500.")
    p.add_argument("--grad-accum-steps", type=int, default=1,
                    help="micro-iters per optimizer step. Effective batch ≈ this "
                         "value (per-sample loss scaled by 1/N, gradients "
                         "accumulated over N micro-iters). Default 1 (no accum, "
                         "bit-identical to v3). Use 16 to approximate paper-spec "
                         "effective batch on a single-sample loader. NOTE: "
                         "--nan-abort-window counts micro-iters, so a "
                         "fully-poisoned window contributes N skipped entries; "
                         "scale --nan-abort-window by N when using accum.")
    a = p.parse_args()
    # Coalesce: from-scratch is implicit when no --stage1-ckpt was given.
    # Without this, --from-scratch's argparse default was permanently True
    # and the Stage-1 bridge path could never run from CLI.
    from_scratch = a.from_scratch or (a.stage1_ckpt is None)
    main(splits_json=a.splits_json,
         out_dir=a.out_dir,
         stage1_ckpt=a.stage1_ckpt,
         from_scratch=from_scratch,
         query_init=a.query_init,
         num_layers_override=a.num_layers,
         num_pts_override=a.num_pts,
         ffn_override=a.ffn,
         n_iters=a.iters,
         t_seq=a.t_seq, t_queue=a.t_queue,
         lr=a.lr, lr_schedule=a.lr_schedule, lr_min=a.lr_min,
         warmup_iters=a.warmup_iters, grad_accum_steps=a.grad_accum_steps,
         grad_clip_max_norm=a.grad_clip,
         sem_loss=a.sem_loss,
         w_occ=a.w_occ, w_kl=a.w_kl, w_ce=a.w_ce, w_lovasz=a.w_lovasz,
         occ_pos_weight=a.occ_pos_weight,
         ignore_empty_in_sem=not a.no_ignore_empty,
         g2v_backend=a.g2v_backend,
         voxel_sample_size=a.voxel_sample_size,
         occ_thresh=a.occ_thresh,
         eval_every=a.eval_every,
         eval_num_val=a.eval_num_val,
         final_eval_num_val=a.final_eval_num_val,
         save_periodic_every=a.save_periodic_every,
         save_best_window=a.save_best_window,
         save_best_train_cooldown=a.save_best_train_cooldown,
         restricted_min_train_voxels=a.restricted_min_train_voxels,
         restricted_min_val_voxels=a.restricted_min_val_voxels,
         log_every=a.log_every,
         seed=a.seed,
         scale_min_override=a.scale_min,
         nan_abort_window=a.nan_abort_window)

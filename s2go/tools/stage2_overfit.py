"""Stage-2 smoke / overfit training entry point.

Pipeline per iter (matches Stage 1's overfit.py structure, no edits):
    [a] Load frame(s) via Stage2OccLoader → enriched dict with
        'sem_voxel_gt' and 'occ_voxel_gt'
    [b] Backbone R50+FPN (fp32) → feat_flatten + spatial_shapes
    [c] S2GOStage2 forward (bf16 autocast for segmentor, fp32 for SemHead)
        → per-Gaussian gaussians + sem_logits_per_g
    [d] G2VLayer (fp32, outside autocast) → voxel occ_prob + sem_logits
    [e] compute_stage2_loss vs GT → backward → NaN-guarded step
    [f] Per-iter log: total, occ_bce, sem_kl, sem_ce, sem_lovasz, gnorm,
        gauss state, peak_mem, skipped
    [g] Save training_history.json + ckpt at end.

The Stage 1 checkpoint is the init source: `backbone` state_dict +
`segmentor` state_dict (minus child_head.head rows, which have a
different shape in semantic mode). The new SemanticHead starts at its
init (empty-class-biased).

Run (smoke, 50 iters × 4 covered sequences):
    cd /media/skr/storage/self_driving/S2GO
    PYTHONUNBUFFERED=1 python -u -m s2go.tools.stage2_overfit \
        --stage1-ckpt out/barebones_full_part1_5000iter_v2/ckpt.pt \
        --iters 50 \
        --num-sequences 4 \
        --sem-loss both \
        --save-path out/stage2_smoke_50iter/ckpt.pt
"""
import argparse
import json
import os
import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR

from ..datasets.nusc_loader import NuScenesLoader
from ..models.backbone.r50_fpn import R50FPNBackbone
from ..stage2 import (
    GRID_SHAPE, NUM_CLASSES, EMPTY_CLASS_ID, PC_RANGE, VOXEL_SIZE,
)

# Resolve once so we can pass into both branches uniformly.
_SEM_IGNORE_IDX = -1  # set per-call below based on flag
from ..stage2.stage2_segmentor import S2GOStage2
from ..stage2.g2v import make_g2v_layer, G2VLayerCUDA
from ..stage2.losses import compute_stage2_loss
from ..stage2.occ_dataset import Stage2OccLoader


def to_device(d: dict, device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in d.items()}


def _build_ckpt(backbone, model, sem_loss, n_iters_done, arch, extra_config=None):
    """Build ckpt dict. `arch` is the actual model arch (auto-detected from
    Stage 1 ckpt or T0-tier defaults) — caller passes the dict used at
    `S2GOStage2` construction time, so the saved config matches reality."""
    cfg = {
        'K':                    int(arch['K']),
        'J':                    int(arch['J']),
        'embed_dims':           int(arch['embed_dims']),
        'num_layers':           int(arch['num_layers']),
        'num_pts':              int(arch['num_pts']),
        'feedforward_channels': int(arch['feedforward_channels']),
        'pc_range':             list(PC_RANGE),
        'voxel_size':           list(VOXEL_SIZE),
        'grid_shape':           list(GRID_SHAPE),
        'num_classes':          NUM_CLASSES,
        'empty_class_id':       EMPTY_CLASS_ID,
        'sem_loss':             sem_loss,
        'query_init':           getattr(model, 'query_init', 'learned'),
        'iters_trained':        n_iters_done,
        'stage':                2,
    }
    if extra_config:
        cfg.update(extra_config)
    return {
        'backbone':       backbone.state_dict(),
        'segmentor':      model.segmentor.state_dict(),
        'semantic_head':  model.semantic_head.state_dict(),
        'config':         cfg,
    }


def main(stage1_ckpt: str = None,
         from_scratch: bool = False,
         query_init: str = 'learned',
         num_layers_override: int = None,
         num_pts_override: int = None,
         ffn_override: int = None,
         n_iters: int = 50,
         num_sequences: int = 4,
         t_seq: int = 1,
         t_queue: int = 1,
         lr: float = 2e-4,
         lr_backbone_mult: float = 0.25,
         grad_clip_max_norm: float = 10.0,
         sem_loss: str = 'both',
         w_occ: float = 1.0,
         w_kl: float = 1.0,
         w_ce: float = 10.0,
         w_lovasz: float = 1.0,
         occ_pos_weight: float = 16.0,
         ignore_empty_in_sem: bool = True,
         g2v_backend: str = 'cuda',
         g2v_chunk_voxels: int = 2048,
         voxel_sample_size: int = 4096,
         dense_voxels: bool = False,
         log_every: int = 1,
         save_path: str = None,
         history_path: str = None,
         seed: int = 0):
    if not from_scratch:
        assert stage1_ckpt and os.path.isfile(stage1_ckpt), \
            f"stage1 ckpt not found: {stage1_ckpt} (or pass --from-scratch)"
    device = "cuda"
    torch.manual_seed(seed)
    torch.cuda.empty_cache()

    print(f"Stage-2 smoke run: {n_iters} iters × {num_sequences} sequences"
          f"  ({'FROM SCRATCH (T0-tier)' if from_scratch else 'init from Stage 1 ckpt'})")
    print(f"  sem_loss={sem_loss}  weights occ/kl/ce/lov = "
          f"{w_occ}/{w_kl}/{w_ce}/{w_lovasz}")
    print(f"  occ_pos_weight={occ_pos_weight}  "
          f"ignore_empty_in_sem={ignore_empty_in_sem}")
    sem_ignore_idx = EMPTY_CLASS_ID if ignore_empty_in_sem else -1

    # ── [1/5] Loader with GT coverage ─────────────────────────────────────
    print("[1/5] Building loader…")
    base = NuScenesLoader(T=t_seq, verbose=False)
    occ_loader = Stage2OccLoader(base, restrict_to_covered=True)
    print(f"    base sequences: {len(base)}, with-GT sequences: {len(occ_loader)}")
    n_overfit = min(num_sequences, len(occ_loader))
    print(f"    cache the first {n_overfit} GT-covered sequences "
          f"(loader idx = {occ_loader.indices[:n_overfit]})")
    sequences: List[List[Dict]] = []
    for i in range(n_overfit):
        seq_cpu = occ_loader[i]
        seq_gpu = [to_device(f, device) for f in seq_cpu]
        sequences.append(seq_gpu)

    # ── [2/5] Model arch ──────────────────────────────────────────────────
    if from_scratch:
        # T0-tier sizing by default (matches `s2go.tools.overfit`'s smoke runs).
        # Override via --num-layers / --num-pts / --ffn for paper-spec A/B.
        print("[2/5] From-scratch arch (no Stage-1 ckpt)")
        arch = dict(K=900, J=10, embed_dims=768,
                     num_layers=2, num_pts=4, feedforward_channels=2048)
        ckpt = None
    else:
        print(f"[2/5] Reading Stage-1 ckpt arch: {stage1_ckpt}")
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

    # CLI overrides (only valid with --from-scratch — would break ckpt load otherwise)
    overrides = [num_layers_override, num_pts_override, ffn_override]
    if any(v is not None for v in overrides):
        assert from_scratch, \
            "--num-layers / --num-pts / --ffn overrides require --from-scratch " \
            "(arch must match the loaded ckpt otherwise)"
        if num_layers_override is not None: arch['num_layers'] = int(num_layers_override)
        if num_pts_override    is not None: arch['num_pts']    = int(num_pts_override)
        if ffn_override        is not None: arch['feedforward_channels'] = int(ffn_override)
    print(f"    arch: {arch}")

    print("[3/5] Building backbone + Stage 2 model…")
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
    print(f"    query_init: {query_init}"
          f"{' (paper §3.4.1 — Stage2Lifter, ignores LiDAR)' if query_init == 'learned' else ' (legacy: FPS+ε on LiDAR)'}")
    if ckpt is not None:
        # Backbone weights load directly (R50+FPN shape didn't change)
        missing_b, unexpected_b = backbone.load_state_dict(ckpt['backbone'], strict=False)
        if missing_b: print(f"    backbone missing: {len(missing_b)}")
        if unexpected_b: print(f"    backbone unexpected: {len(unexpected_b)}")
        # Segmentor: partial child_head.head transfer (first 11 rows: offset/scale/rot/opa),
        # discard last 3 (RGB). See `S2GOStage2.load_stage1_state` for details (Gap 8 fix).
        missing_s, unexpected_s, head_partial = model.load_stage1_state(ckpt, strict=False)
        print(f"    segmentor: missing={len(missing_s)} unexpected={len(unexpected_s)}; "
              f"child_head.head partial-transferred (first 11 rows): {head_partial}")
    else:
        print("    backbone: torchvision ResNet50 pretrained init "
              "(no Stage-1 fine-tune)")
        print("    segmentor + semantic head: random init")
    n_back = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    trainable: backbone={n_back/1e6:.1f} M, model={n_model/1e6:.1f} M, "
          f"total={(n_back+n_model)/1e6:.1f} M")

    # ── [4/5] G2V + Optim ─────────────────────────────────────────────────
    print(f"[4/5] G2V backend: {g2v_backend} "
          f"({'DENSE — full 200×200×16 grid' if dense_voxels else f'SPARSE — {voxel_sample_size} voxels/iter'})")
    g2v = make_g2v_layer(backend=g2v_backend,
                          chunk_voxels=g2v_chunk_voxels).to(device)
    optim = AdamW([
        {'params': [p for p in backbone.parameters() if p.requires_grad],
         'lr': lr * lr_backbone_mult},
        {'params': model.parameters(), 'lr': lr},
    ], weight_decay=0.01)
    scheduler = ConstantLR(optim, factor=1.0, total_iters=n_iters)

    # ── [5/5] Train loop ──────────────────────────────────────────────────
    print(f"[5/5] Training for {n_iters} iters (log every {log_every}, "
          f"NaN-guarded, grad-clip={grad_clip_max_norm})…\n")
    print(f"  iter |   total |  occ_bce |  sem_kl |  sem_ce | lovasz |  gnorm | opa_mean | sec | mem MB")
    print(f"  -----+---------+----------+---------+---------+--------+--------+----------+-----+-------")
    backbone.train(); model.train()
    history = []
    t_start = time.time()
    n_skipped = 0

    for i in range(n_iters):
        seq = sequences[i % n_overfit]

        # [b] backbone — fp32 (FPN upsample_nearest2d has no bf16 kernel)
        sequence_with_feat = []
        for f in seq:
            feat, ss, lsi, _ = backbone(f['imgs'])
            sequence_with_feat.append({
                **f,
                'feat_flatten':       feat,
                'spatial_shapes':     ss,
                'level_start_index':  lsi,
                'pad_h': 256, 'pad_w': 704,
            })

        # [c] segmentor + semantic head — bf16 autocast
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            outs = model(sequence_with_feat)

        # [d] + [e] G2V + loss — fp32 (covariance inv/det are noisy in bf16).
        # Dense path (CUDA kernel): full 640k voxels, ~360 MB peak.
        # Sparse path (torch or CUDA): subset of voxels per iter, autograd-safe.
        torch.cuda.empty_cache()
        L_total_seq = torch.zeros((), device=device, dtype=torch.float32)
        diag_acc = {'occ_bce': 0.0, 'sem_kl': 0.0, 'sem_ce': 0.0,
                    'sem_lovasz': 0.0, 'total': 0.0}
        for t, out in enumerate(outs):
            G = out.raw.gaussians
            sem_per_g = out.sem_logits_per_g
            sem_voxel_gt = sequence_with_feat[t]['sem_voxel_gt']        # (1, Vx, Vy, Vz)
            occ_voxel_gt = sequence_with_feat[t]['occ_voxel_gt']        # (1, Vx, Vy, Vz)

            with torch.cuda.amp.autocast(enabled=False):
                if dense_voxels:
                    occ_pred, sem_logits = g2v(
                        means=G.means.float(),
                        rotations=G.rotations.float(),
                        scales=G.scales.float(),
                        opacities=G.opacities.float(),
                        class_logits=sem_per_g.float(),
                    )
                    L_t, diag_t = compute_stage2_loss(
                        occ_pred=occ_pred,                     # (1, Vx, Vy, Vz)
                        sem_logits=sem_logits,                  # (1, Vx, Vy, Vz, C)
                        occ_gt=occ_voxel_gt,                    # (1, Vx, Vy, Vz)
                        sem_gt=sem_voxel_gt,                    # (1, Vx, Vy, Vz)
                        sem_loss=sem_loss,
                        w_occ=w_occ, w_kl=w_kl, w_ce=w_ce, w_lovasz=w_lovasz,
                        occ_pos_weight=occ_pos_weight,
                        ignore_index=sem_ignore_idx,
                    )
                else:
                    # Sparse: balanced occupied/empty sampling
                    occ_flat = occ_voxel_gt[0].reshape(-1)
                    occ_idx = occ_flat.nonzero(as_tuple=False).squeeze(-1)
                    emp_idx = (~occ_flat).nonzero(as_tuple=False).squeeze(-1)
                    n_pos = min(voxel_sample_size // 2, occ_idx.numel())
                    n_neg = voxel_sample_size - n_pos
                    pos_pick = occ_idx[torch.randperm(occ_idx.numel(), device=device)[:n_pos]]
                    neg_pick = emp_idx[torch.randperm(emp_idx.numel(), device=device)[:n_neg]]
                    sample_idx = torch.cat([pos_pick, neg_pick], dim=0).unsqueeze(0)
                    sem_gt_flat = sem_voxel_gt[0].reshape(-1).index_select(0, sample_idx[0])
                    occ_gt_flat = occ_voxel_gt[0].reshape(-1).index_select(0, sample_idx[0])
                    occ_pred, sem_logits = g2v.forward_sparse(
                        means=G.means.float(),
                        rotations=G.rotations.float(),
                        scales=G.scales.float(),
                        opacities=G.opacities.float(),
                        class_logits=sem_per_g.float(),
                        voxel_flat_idx=sample_idx,
                    )
                    # Reshape sparse outputs/GT to (1, P, 1, 1[, C]) so the
                    # loss helper's flatten-and-reduce treats them uniformly.
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

        # Backward + NaN guard + step
        optim.zero_grad(set_to_none=True)
        L.backward()
        gnorm = nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(model.parameters()),
            max_norm=grad_clip_max_norm)
        finite = bool(torch.isfinite(L).item() and torch.isfinite(gnorm).item())
        skipped = not finite
        if skipped:
            n_skipped += 1
            tok = seq[0].get('_sample_token', f'iter{i}')
            print(f"  [skip] iter {i}: non-finite "
                  f"(L={L.item():.3g}, gnorm={gnorm.item():.3g}, token={tok})",
                  flush=True)
        else:
            optim.step()
        scheduler.step()

        # Gaussian-state quick stats (last frame's last Gaussians)
        opa_mean = float(outs[-1].raw.gaussians.opacities.detach()
                         .float().mean().item())

        history.append({
            **diag_acc,
            'iter':    i,
            'gnorm':   float(gnorm.item()),
            'lr':      scheduler.get_last_lr()[0],
            'skipped': skipped,
            'opa_mean': opa_mean,
        })

        if i % log_every == 0 or i == n_iters - 1:
            elapsed = time.time() - t_start
            mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  {i:>4} | {L.item():>7.3f} | {diag_acc['occ_bce']:>8.4f} | "
                  f"{diag_acc['sem_kl']:>7.3f} | {diag_acc['sem_ce']:>7.3f} | "
                  f"{diag_acc['sem_lovasz']:>6.3f} | {gnorm.item():>6.2f} | "
                  f"{opa_mean:>8.3f} | {elapsed:>3.0f} | {mem:>6.0f}")

    print()
    finite_hist = [h for h in history if not h.get('skipped', False)
                                          and h['total'] == h['total']]
    first_5 = sum(h['total'] for h in finite_hist[:5]) / max(len(finite_hist[:5]), 1)
    last_5  = sum(h['total'] for h in finite_hist[-5:]) / max(len(finite_hist[-5:]), 1)
    drop = first_5 - last_5
    print(f"Stage-2 smoke summary:")
    print(f"  iters total              : {n_iters}")
    print(f"  iters skipped (NaN guard): {n_skipped}  "
          f"({100.0 * n_skipped / max(n_iters, 1):.1f}%)")
    print(f"  loss avg first 5 finite  : {first_5:.4f}")
    print(f"  loss avg last  5 finite  : {last_5:.4f}")
    print(f"  Δ = {drop:+.4f}  "
          f"({'GOOD' if drop > 0 else 'BAD: loss did not decrease'})")

    # History + ckpt
    out_dir = os.path.dirname(save_path) if save_path else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if history_path:
        os.makedirs(os.path.dirname(history_path) or '.', exist_ok=True)
        with open(history_path, 'w') as fp:
            json.dump(history, fp, indent=2)
        print(f"  history → {history_path}")
    elif out_dir:
        hp = os.path.join(out_dir, 'training_history.json')
        with open(hp, 'w') as fp:
            json.dump(history, fp, indent=2)
        print(f"  history → {hp}")

    if save_path:
        ckpt_out = _build_ckpt(backbone, model, sem_loss, n_iters_done=n_iters,
                                arch=arch,
                                extra_config={'covered_indices_used':
                                              occ_loader.indices[:n_overfit]})
        torch.save(ckpt_out, save_path)
        sz_mb = sum(t.numel() * t.element_size()
                     for d in [ckpt_out['backbone'], ckpt_out['segmentor'],
                                ckpt_out['semantic_head']]
                     for t in d.values()) / 1024**2
        print(f"  ckpt    → {save_path}  ({sz_mb:.1f} MB)")

    return history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-ckpt", type=str,
                    default="out/barebones_full_part1_5000iter_v2/ckpt.pt",
                    help="path to Stage 1 .pt with {backbone, segmentor, config}")
    p.add_argument("--from-scratch", action="store_true",
                    help="skip Stage-1 ckpt load; build T0-tier model (num_layers=2, "
                         "num_pts=4, ffn=2048) and random-init everything. Fits in 12 GB "
                         "with full G2V graph; use for pipeline-validation smoke runs.")
    p.add_argument("--query-init", choices=['learned', 'fps_lidar'],
                    default='learned',
                    help="Query initialisation strategy (Gap-1 fix). 'learned' uses "
                         "Stage2Lifter (learnable query xyz + feat, paper §3.4.1 — "
                         "Stage 2 does NOT consume LiDAR in forward). 'fps_lidar' falls "
                         "back to the inherited Stage 1 lifter (FPS+ε on LiDAR) for "
                         "ablation/debugging. Default 'learned'.")
    p.add_argument("--num-layers", type=int, default=None,
                    help="Override decoder num_layers (only valid with --from-scratch). "
                         "Use 6 for paper-spec, 2 for T0-tier (default).")
    p.add_argument("--num-pts", type=int, default=None,
                    help="Override deformable num_pts (only valid with --from-scratch). "
                         "Use 13 for paper-spec, 4 for T0-tier (default).")
    p.add_argument("--ffn", type=int, default=None,
                    help="Override decoder feedforward_channels (only valid with "
                         "--from-scratch). Use 3072 for paper-spec, 2048 for T0-tier.")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--num-sequences", type=int, default=4)
    p.add_argument("--t-seq", type=int, default=1)
    p.add_argument("--t-queue", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--sem-loss", choices=['kl', 'ce_lovasz', 'both'],
                    default='both')
    p.add_argument("--w-occ", type=float, default=1.0)
    p.add_argument("--w-kl", type=float, default=1.0)
    p.add_argument("--w-ce", type=float, default=10.0)
    p.add_argument("--w-lovasz", type=float, default=1.0)
    p.add_argument("--occ-pos-weight", type=float, default=16.0,
                    help="positive-class weight for occupancy BCE. ~16 matches "
                         "typical Occ3D empty:occupied ratio; combats the "
                         "'predict empty everywhere' collapse seen on the dense "
                         "supervision path.")
    p.add_argument("--no-ignore-empty", action="store_true",
                    help="DON'T mask empty voxels from the semantic loss "
                         "(default: ignore_index=EMPTY_CLASS_ID, so the "
                         "semantic head learns 'what is this OCCUPIED voxel' "
                         "and leaves occupancy-vs-empty to the BCE head).")
    p.add_argument("--g2v-backend", choices=['torch', 'cuda'], default='cuda',
                    help="G2V backend. 'cuda' uses the compiled localagg_s2go "
                         "kernel — supports full dense grid in ~360 MB. "
                         "'torch' is the pure-python autograd reference, "
                         "constrained to sparse voxel sampling on 12 GB GPUs.")
    p.add_argument("--dense-voxels", action="store_true",
                    help="run G2V loss on every voxel (640k for 200×200×16 grid). "
                         "Recommended with --g2v-backend cuda. Off by default "
                         "to stay safe with the torch backend.")
    p.add_argument("--g2v-chunk-voxels", type=int, default=2048,
                    help="(torch backend only) chunk voxels for the dense torch path.")
    p.add_argument("--voxel-sample-size", type=int, default=4096,
                    help="(sparse only) P voxels sampled per iter, balanced "
                         "occupied/empty. Ignored when --dense-voxels is set.")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-path", type=str, default=None)
    p.add_argument("--history-path", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    main(stage1_ckpt=a.stage1_ckpt,
         from_scratch=a.from_scratch,
         query_init=a.query_init,
         num_layers_override=a.num_layers,
         num_pts_override=a.num_pts,
         ffn_override=a.ffn,
         n_iters=a.iters, num_sequences=a.num_sequences,
         t_seq=a.t_seq, t_queue=a.t_queue,
         lr=a.lr, grad_clip_max_norm=a.grad_clip,
         sem_loss=a.sem_loss,
         w_occ=a.w_occ, w_kl=a.w_kl, w_ce=a.w_ce, w_lovasz=a.w_lovasz,
         occ_pos_weight=a.occ_pos_weight,
         ignore_empty_in_sem=(not a.no_ignore_empty),
         g2v_backend=a.g2v_backend,
         g2v_chunk_voxels=a.g2v_chunk_voxels,
         voxel_sample_size=a.voxel_sample_size,
         dense_voxels=a.dense_voxels,
         log_every=a.log_every,
         save_path=a.save_path, history_path=a.history_path,
         seed=a.seed)

"""Standalone Stage-2 mIoU evaluator for any saved checkpoint.

Reconstructs the same backbone + segmentor + G2V eval path used inside
`stage2_train.py`'s `[6/6] Final eval` block, but loads model weights from
disk instead of using the in-memory end-of-training state.

Usage:
    python -m s2go.tools.stage2_eval_ckpt \\
        --ckpt        out/stage2_t2_b16_35k_parts01-05/ckpt_best_val.pt \\
        --splits-json out/stage2_parts01-05_splits.json \\
        --out-json    out/stage2_t2_b16_35k_parts01-05/eval_best_val_914.json \\
        --max-seqs    914

Writes a JSON identical in schema to `eval_final.json`'s top-level fields
(mIoU, occupancy_IoU, per_class, n_voxels_seen, n_val_sequences, eval_sec).
"""
from __future__ import annotations
import argparse
import json
import os
import time

import torch

from ..datasets.nusc_loader import NuScenesLoader
from ..models.backbone.r50_fpn import R50FPNBackbone
from ..stage2 import NUM_CLASSES, EMPTY_CLASS_ID
from ..stage2.stage2_segmentor import S2GOStage2
from ..stage2.g2v import make_g2v_layer
from ..stage2.occ_dataset import Stage2OccLoader

# Re-use the helpers from stage2_train.py — same semantics, same gates.
from .stage2_train import _filter_indices_by_tokens, _eval_pass


def main(ckpt_path: str, splits_json: str, out_json: str,
         max_seqs: int = 914, occ_thresh: float = 0.5,
         scale_min: float = 0.05, query_init: str = 'learned',
         t_seq: int = 1, t_queue: int = 1,
         g2v_backend: str = 'cuda', g2v_chunk_voxels: int = 4096) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Stage-2 ckpt eval:\n  ckpt={ckpt_path}")
    print(f"  splits_json={splits_json}")
    print(f"  out_json={out_json}")
    print(f"  max_seqs={max_seqs}  occ_thresh={occ_thresh}  scale_min={scale_min}\n")

    # ── [1/4] Loader + val split ──────────────────────────────────────────
    print("[1/4] Building loader + filtering val split…")
    with open(splits_json, 'r') as fp:
        splits = json.load(fp)
    val_tokens = splits['val_start_tokens']
    base = NuScenesLoader(T=t_seq, verbose=False)
    occ_loader = Stage2OccLoader(base, restrict_to_covered=True)
    val_pos = _filter_indices_by_tokens(occ_loader, val_tokens)
    print(f"    base={len(occ_loader)}  val_pos={len(val_pos)}  "
          f"(eval slice: {min(max_seqs, len(val_pos))})\n")

    # ── [2/4] Load ckpt, derive arch from its saved config ────────────────
    print(f"[2/4] Loading ckpt…")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get('config', {})
    arch = dict(
        K=int(cfg.get('K', 900)),
        J=int(cfg.get('J', 10)),
        embed_dims=int(cfg.get('embed_dims', 768)),
        num_layers=int(cfg.get('num_layers', 6)),
        num_pts=int(cfg.get('num_pts', 13)),
        feedforward_channels=int(cfg.get('feedforward_channels', 3072)),
    )
    iters_trained = cfg.get('iters_trained', '?')
    print(f"    iter_trained={iters_trained}  arch={arch}\n")

    # ── [3/4] Build model, load weights ───────────────────────────────────
    print("[3/4] Building backbone + Stage-2 model and loading weights…")
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

    backbone.load_state_dict(ckpt['backbone'])
    model.segmentor.load_state_dict(ckpt['segmentor'])
    model.semantic_head.load_state_dict(ckpt['semantic_head'])
    # Mirror the training-time scale_min override on the loaded child_head.
    prev_scale_min = float(model.segmentor.child_head.scale_min)
    model.segmentor.child_head.scale_min = float(scale_min)
    print(f"    weights loaded; scale_min: {prev_scale_min} → "
          f"{model.segmentor.child_head.scale_min}\n")

    g2v_eval = make_g2v_layer(backend=g2v_backend,
                               chunk_voxels=g2v_chunk_voxels).to(device)

    # ── [4/4] Eval ────────────────────────────────────────────────────────
    print(f"[4/4] Running eval pass over {min(max_seqs, len(val_pos))} val sequences…")
    t0 = time.time()
    stats = _eval_pass(backbone, model, g2v_eval, occ_loader,
                       val_pos, device, occ_thresh,
                       max_seqs=max_seqs,
                       verbose_every=100)
    stats['eval_sec'] = time.time() - t0
    stats['n_val_sequences'] = min(max_seqs, len(val_pos))
    stats['ckpt_path'] = ckpt_path
    stats['ckpt_iter'] = iters_trained
    stats['occ_threshold'] = occ_thresh

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n  raw mIoU (17 classes excl. empty): {stats['mIoU']*100:.2f}%")
    print(f"  occupancy IoU:                     {stats['occupancy_IoU']*100:.2f}%")
    print(f"  n_voxels_seen:                     {stats['n_voxels_seen']}")
    print(f"  eval_sec:                          {stats['eval_sec']:.1f}\n")
    print("  per-class IoU:")
    print(f"    {'class':>22s} | {'iou':>6s} | {'val_seen':>10s} | {'val_pred':>10s}")
    print(f"    {'-'*22}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}")
    for cname, st in stats['per_class'].items():
        iou = st['iou']
        iou_str = f"{iou*100:5.2f}%" if iou == iou else "   nan"
        print(f"    {cname:>22s} | {iou_str:>6s} | {st['seen']:>10d} | "
              f"{st['pred']:>10d}")

    # 16-class paper-comparable mIoU (drop the extra "other" class)
    paper16 = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
               'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
               'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
               'manmade', 'vegetation']
    p16 = [stats['per_class'][c]['iou'] for c in paper16
           if stats['per_class'][c]['iou'] == stats['per_class'][c]['iou']]
    if p16:
        stats['mIoU_16class_paper'] = sum(p16) / len(p16)
        print(f"\n  mIoU (16-class, paper convention): "
              f"{stats['mIoU_16class_paper']*100:.2f}%")

    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    with open(out_json, 'w') as fp:
        json.dump(stats, fp, indent=2, default=lambda x: str(x))
    print(f"\n  → wrote {out_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--splits-json", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--max-seqs", type=int, default=914,
                   help="cap on val sequences (default 914, matches "
                        "stage2_train --final-eval-num-val). Use 100000 for full val.")
    p.add_argument("--occ-thresh", type=float, default=0.5)
    p.add_argument("--scale-min", type=float, default=0.05)
    p.add_argument("--g2v-backend", choices=['torch', 'cuda'], default='cuda')
    a = p.parse_args()
    main(ckpt_path=a.ckpt, splits_json=a.splits_json, out_json=a.out_json,
         max_seqs=a.max_seqs, occ_thresh=a.occ_thresh,
         scale_min=a.scale_min, g2v_backend=a.g2v_backend)

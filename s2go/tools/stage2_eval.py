"""Stage-2 evaluation entry point.

Loads a Stage-2 checkpoint, forwards one frame of one held sequence, runs
the dense G2V (no_grad → full 200×200×16 grid fits in 12 GB without the
autograd intermediates), computes argmax labels, and:

  - Updates MeanIoU vs GT.
  - Saves PNG figures:
      eval_A_bev.png         — top-down argmax projection coloured by class
      eval_B_voxel_slices.png — 4 z-slices showing pred & GT side-by-side
      eval_C_confusion.png   — confusion matrix (18 × 18)
  - Saves eval_stats.json with per-class IoU + mean + occupancy IoU.

Run (eval on training sequence 0 of the smoke run):
    PYTHONUNBUFFERED=1 python -u -m s2go.tools.stage2_eval \
        --ckpt out/stage2_smoke_50iter/ckpt.pt \
        --seq-idx 0 \
        --out-dir out/stage2_smoke_50iter
"""
import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ..datasets.nusc_loader import NuScenesLoader
from ..models.backbone.r50_fpn import R50FPNBackbone
from ..stage2 import (
    GRID_SHAPE, NUM_CLASSES, EMPTY_CLASS_ID, PC_RANGE, VOXEL_SIZE, CLASS_NAMES,
)
from ..stage2.stage2_segmentor import S2GOStage2
from ..stage2.g2v import make_g2v_layer
from ..stage2.miou import MeanIoU
from ..stage2.occ_dataset import Stage2OccLoader


# A simple 18-class palette (taken from common Occ3D conventions; arbitrary
# but visually distinguishable). Last entry is empty (transparent in plots).
PALETTE = np.array([
    [255, 120,  50],   #  0 barrier         orange
    [255, 192, 203],   #  1 bicycle         pink
    [255, 255,   0],   #  2 bus             yellow
    [  0, 150, 245],   #  3 car             blue
    [  0, 255, 255],   #  4 construction    cyan
    [255, 127,   0],   #  5 motorcycle      dark orange
    [255,   0,   0],   #  6 pedestrian      red
    [255, 240, 150],   #  7 traffic cone    light yellow
    [135,  60,   0],   #  8 trailer         brown
    [160,  32, 240],   #  9 truck           purple
    [255,   0, 255],   # 10 drivable        magenta
    [139, 137, 137],   # 11 other_flat      grey
    [ 75,   0,  75],   # 12 sidewalk        dark purple
    [150, 240,  80],   # 13 terrain         light green
    [230, 230, 250],   # 14 manmade         lavender
    [  0, 175,   0],   # 15 vegetation      green
    [180, 180, 180],   # 16 other           light grey
    [255, 255, 255],   # 17 empty           white (background)
], dtype=np.float32) / 255.0


def to_device(d, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in d.items()}


def main(ckpt_path: str,
         seq_idx: int = 0,
         out_dir: str = "out/stage2_smoke_50iter",
         g2v_backend: str = 'cuda',
         g2v_chunk_voxels: int = 4096,
         occ_thresh: float = 0.5):
    device = "cuda"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Stage-2 eval — ckpt={ckpt_path}, seq_idx={seq_idx}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt['config']
    print(f"  ckpt arch: K={cfg['K']}, J={cfg['J']}, d={cfg['embed_dims']}, "
          f"L={cfg['num_layers']}, num_pts={cfg['num_pts']}, "
          f"ffn={cfg['feedforward_channels']}, iters={cfg['iters_trained']}")

    # Loader (T=1)
    base = NuScenesLoader(T=1, verbose=False)
    occ_loader = Stage2OccLoader(base, restrict_to_covered=True)

    # Pull the same index used by training (smoke recorded covered indices).
    used = cfg.get('covered_indices_used', [])
    if seq_idx < len(used):
        # Map smoke's cached index → loader index; eval uses occ_loader idx.
        # used[i] is a *loader* index; we need the position of that in
        # occ_loader.indices.
        target = used[seq_idx]
        try:
            occ_idx = occ_loader.indices.index(target)
        except ValueError:
            occ_idx = seq_idx
    else:
        occ_idx = seq_idx
    print(f"  using occ_loader index {occ_idx} "
          f"(loader index {occ_loader.indices[occ_idx]})")
    seq_cpu = occ_loader[occ_idx]
    seq = [to_device(f, device) for f in seq_cpu]

    # Build model from ckpt arch
    print("  building model…")
    backbone = R50FPNBackbone(embed_dims=cfg['embed_dims'],
                                num_outs=4, pretrained=True).to(device)
    # Use the same query_init the ckpt was trained with (stored in ckpt['config']).
    ckpt_query_init = cfg.get('query_init', 'learned')
    model = S2GOStage2(
        segmentor_kwargs=dict(
            K=cfg['K'], J=cfg['J'], embed_dims=cfg['embed_dims'],
            num_layers=cfg['num_layers'],
            num_heads=12, num_groups=12,
            num_levels=4, num_cams=6,
            num_pts=cfg['num_pts'],
            feedforward_channels=cfg['feedforward_channels'],
            T_queue=1,
        ),
        num_classes=NUM_CLASSES,
        query_init=ckpt_query_init,
    ).to(device)
    print(f"  query_init (from ckpt): {ckpt_query_init}")
    backbone.load_state_dict(ckpt['backbone'])
    model.segmentor.load_state_dict(ckpt['segmentor'])
    model.semantic_head.load_state_dict(ckpt['semantic_head'])
    backbone.eval(); model.eval()

    g2v = make_g2v_layer(backend=g2v_backend,
                          chunk_voxels=g2v_chunk_voxels).to(device)
    print(f"  G2V backend: {g2v_backend}")

    # Forward (no grad)
    print("  forwarding + dense G2V (no_grad)…")
    with torch.no_grad():
        f = seq[0]
        feat, ss, lsi, _ = backbone(f['imgs'])
        frame = {**f, 'feat_flatten': feat,
                  'spatial_shapes': ss, 'level_start_index': lsi,
                  'pad_h': 256, 'pad_w': 704}
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            out = model.forward_one_frame(frame)
        torch.cuda.empty_cache()

        with torch.cuda.amp.autocast(enabled=False):
            G = out.raw.gaussians
            occ_pred, sem_logits = g2v(
                means=G.means.float(),
                rotations=G.rotations.float(),
                scales=G.scales.float(),
                opacities=G.opacities.float(),
                class_logits=out.sem_logits_per_g.float(),
            )
        # Argmax → predicted class label. Force EMPTY where occ_prob < thresh.
        pred_argmax = sem_logits.argmax(dim=-1)                # (1, Vx, Vy, Vz)
        empty_mask  = (occ_pred < occ_thresh)
        pred_argmax = torch.where(empty_mask,
                                    torch.full_like(pred_argmax, EMPTY_CLASS_ID),
                                    pred_argmax)
        sem_gt = f['sem_voxel_gt']                             # (1, Vx, Vy, Vz)

    # mIoU
    print("  computing mIoU…")
    metric = MeanIoU(ignore_classes=[EMPTY_CLASS_ID])
    metric.update(pred_argmax[0], sem_gt[0])
    stats = metric.compute()
    print(f"  mIoU (excl. empty)   : {stats['mIoU']*100:.2f} %")
    print(f"  occupancy IoU        : {stats['occupancy_IoU']*100:.2f} %")
    print(f"  n_voxels evaluated   : {stats['n_voxels_seen']}")
    top_classes = sorted(stats['per_class'].items(),
                          key=lambda kv: -kv[1]['seen'])[:5]
    for name, st in top_classes:
        print(f"    {name:>20s}: IoU={st['iou']*100:5.1f}%  seen={st['seen']:>7d}")

    # ── Figures ───────────────────────────────────────────────────────────
    pred_np = pred_argmax[0].cpu().numpy()        # (Vx, Vy, Vz)
    gt_np   = sem_gt[0].cpu().numpy()             # (Vx, Vy, Vz)
    occ_np  = occ_pred[0].cpu().numpy()           # (Vx, Vy, Vz)
    Vx, Vy, Vz = pred_np.shape

    # (A) BEV — topmost non-empty z for each (x, y)
    def _bev_topdown(vol, empty_id):
        Vx, Vy, Vz = vol.shape
        nonempty = vol != empty_id
        zhit = np.where(nonempty.any(axis=-1),
                         (nonempty * np.arange(Vz)).argmax(axis=-1),
                         0)
        out = vol[np.arange(Vx)[:, None], np.arange(Vy)[None, :], zhit]
        out = np.where(nonempty.any(axis=-1), out, empty_id)
        return out
    bev_pred = _bev_topdown(pred_np, EMPTY_CLASS_ID)
    bev_gt   = _bev_topdown(gt_np,   EMPTY_CLASS_ID)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    cmap = ListedColormap(PALETTE)
    axes[0].imshow(bev_gt.T,   cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1,
                     origin='lower', interpolation='nearest')
    axes[0].set_title("GT BEV (top-down argmax)")
    axes[1].imshow(bev_pred.T, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1,
                     origin='lower', interpolation='nearest')
    axes[1].set_title(f"Pred BEV  (mIoU={stats['mIoU']*100:.1f}%, "
                       f"occ_IoU={stats['occupancy_IoU']*100:.1f}%)")
    for ax in axes:
        ax.set_xlabel("y voxel"); ax.set_ylabel("x voxel")
    fig.tight_layout()
    A = os.path.join(out_dir, "eval_A_bev.png")
    fig.savefig(A, dpi=110); plt.close(fig)
    print(f"  wrote {A}")

    # (B) z-slices — 4 horizontal slices
    z_slices = [Vz // 5, 2 * Vz // 5, 3 * Vz // 5, 4 * Vz // 5]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for i, z in enumerate(z_slices):
        axes[0, i].imshow(gt_np[:, :, z].T, cmap=cmap,
                            vmin=0, vmax=NUM_CLASSES - 1,
                            origin='lower', interpolation='nearest')
        axes[0, i].set_title(f"GT z={z}")
        axes[1, i].imshow(pred_np[:, :, z].T, cmap=cmap,
                            vmin=0, vmax=NUM_CLASSES - 1,
                            origin='lower', interpolation='nearest')
        axes[1, i].set_title(f"Pred z={z}")
        for ax in axes[:, i]:
            ax.set_xlabel("y"); ax.set_ylabel("x")
    fig.tight_layout()
    B = os.path.join(out_dir, "eval_B_voxel_slices.png")
    fig.savefig(B, dpi=110); plt.close(fig)
    print(f"  wrote {B}")

    # (C) Confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    g_flat = gt_np.flatten()
    p_flat = pred_np.flatten()
    for c in range(NUM_CLASSES):
        m = g_flat == c
        if m.any():
            for k in range(NUM_CLASSES):
                cm[c, k] = int((p_flat[m] == k).sum())
    # Normalize per GT row for readability
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(1, None)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(cm_norm, cmap='magma', vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=7)
    ax.set_yticklabels(CLASS_NAMES, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
    ax.set_title("Confusion matrix (row-normalised)")
    fig.tight_layout()
    C = os.path.join(out_dir, "eval_C_confusion.png")
    fig.savefig(C, dpi=110); plt.close(fig)
    print(f"  wrote {C}")

    # Stats JSON
    stats_out = {
        'mIoU':           stats['mIoU'],
        'occupancy_IoU':  stats['occupancy_IoU'],
        'n_voxels_seen':  stats['n_voxels_seen'],
        'per_class':      stats['per_class'],
        'occ_threshold':  occ_thresh,
        'occ_pred_range': [float(occ_pred.min().item()),
                            float(occ_pred.max().item())],
        'occ_pred_mean':  float(occ_pred.mean().item()),
        'pred_class_counts': {CLASS_NAMES[c]: int((p_flat == c).sum())
                                 for c in range(NUM_CLASSES)},
        'gt_class_counts':   {CLASS_NAMES[c]: int((g_flat == c).sum())
                                 for c in range(NUM_CLASSES)},
        'seq_idx':       seq_idx,
        'occ_loader_idx': occ_idx,
        'loader_idx':    occ_loader.indices[occ_idx],
    }
    J = os.path.join(out_dir, "eval_stats.json")
    with open(J, 'w') as fp:
        json.dump(stats_out, fp, indent=2, default=float)
    print(f"  wrote {J}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--seq-idx", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="out/stage2_smoke_50iter")
    p.add_argument("--g2v-backend", choices=['torch', 'cuda'], default='cuda')
    p.add_argument("--g2v-chunk-voxels", type=int, default=4096)
    p.add_argument("--occ-thresh", type=float, default=0.5)
    a = p.parse_args()
    main(ckpt_path=a.ckpt, seq_idx=a.seq_idx, out_dir=a.out_dir,
         g2v_backend=a.g2v_backend,
         g2v_chunk_voxels=a.g2v_chunk_voxels, occ_thresh=a.occ_thresh)

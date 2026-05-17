"""Reproduce the paper qualitative figures from a trained Stage-1 checkpoint.

Generates, from a depth-only Stage-1 ckpt (overfit.py output):

  Fig1a  — "Stage 1: Denoising & Rendering Pretraining"
             [left]  Query Denoising : noised init queries -> refined (denoised)
                                       vs noise-free FPS anchors (target), BEV
             [right] Depth Map Rendering : 6-cam gsplat depth (magma), 2x3 surround

  fig2   — paper Figure 2, *Stage-1-derivable columns only* (Pretrained row):
             [LiDAR] input cloud (BEV) | [Query Offsets] refined coloured by
             |refined-init| | [Gauss. Centers] gaussian means coloured by opacity
           (Occ.Pred + the No-Pretraining baseline are Stage-2 / second-model
            artifacts and are intentionally omitted — no Stage-2 ckpt available.)

Reuses the verified load/forward from s2go.tools.stage1_eval so model
construction stays single-sourced.

Run:
    cd /media/skr/storage/self_driving/S2GO
    python -m scripts.repro_paper_figs_stage1 \
        --checkpoint out/stage1_curated_270_depthonly-20260516-153143/ckpt_eval_best.pt \
        --scene-tokens-json dataset_stats/curated_val_50/tokens.json \
        --seq-idx 0 --out-dir .
"""
import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s2go.datasets.nusc_loader import NuScenesLoader
from s2go.tools.stage1_eval import load_model, run_forward
from s2go.render.gsplat_wrapper import render

# nuScenes camera order as produced by the loader (see stage1_eval.CAM_NAMES)
CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
# 2x3 surround layout (row-major): FL F FR / BL B BR
SURROUND = [('CAM_FRONT_LEFT', 'front-left'), ('CAM_FRONT', 'front'),
            ('CAM_FRONT_RIGHT', 'front-right'),
            ('CAM_BACK_LEFT', 'back-left'), ('CAM_BACK', 'back'),
            ('CAM_BACK_RIGHT', 'back-right')]
RANGE_M = 50.0


def _bev(ax, xy, c, cmap, s, title, cbar_label=None, dark=True):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=cmap, s=s,
                    alpha=0.85, linewidths=0, rasterized=True)
    ax.set_xlim(-RANGE_M, RANGE_M); ax.set_ylim(-RANGE_M, RANGE_M)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    fg = 'white' if dark else 'black'
    ax.set_title(title, color=fg, fontsize=11, pad=6)
    if dark:
        ax.set_facecolor('#0a0a12')
    for sp in ax.spines.values():
        sp.set_color('#444')
    if cbar_label is not None:
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(cbar_label, color=fg if dark else 'black', fontsize=8)
        cb.ax.tick_params(colors=fg if dark else 'black', labelsize=7)
    return sc


@torch.no_grad()
def main(checkpoint, scene_tokens_json, seq_idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda"
    backbone, seg, cfg = load_model(checkpoint, device)

    print(f"Loading curated-val sequence #{seq_idx} (T={cfg['T_queue']})…")
    loader = NuScenesLoader(T=cfg['T_queue'], verbose=True,
                            scene_tokens_json=scene_tokens_json)
    seq_idx = seq_idx % len(loader)
    seq_cpu = loader[seq_idx]
    sequence = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in f.items()} for f in seq_cpu]

    print("Forward pass…")
    outputs, _ = run_forward(backbone, seg, sequence, device)
    o = outputs[-1]
    frame = sequence[-1]

    init    = o.init_xyz[0].cpu().float().numpy()       # (K,3) noised queries
    refined = o.refined_xyz[0].cpu().float().numpy()     # (K,3) denoised pred
    anchors = o.anchors_xyz[0].cpu().float().numpy()     # (K,3) FPS target
    means   = o.gaussians.means[0].cpu().float().numpy() # (K*J,3)
    opac    = o.gaussians.opacities[0].cpu().float().numpy().reshape(-1)
    lidar   = frame['lidar_pts'][0].cpu().float().numpy()  # (M,3)
    off_mag = np.linalg.norm(refined - init, axis=1)       # (K,)

    # ── Depth render (depth-only ckpt → render_mode='D') ──────────────────
    G = o.gaussians._replace(
        means=o.gaussians.means.float(), scales=o.gaussians.scales.float(),
        rotations=o.gaussians.rotations.float(),
        opacities=o.gaussians.opacities.float(),
        velocity=o.gaussians.velocity.float())
    _, depth, _ = render(G, viewmats=frame['viewmats'][0].float(),
                         Ks=frame['cam_K'][0].float(),
                         height=256, width=704, render_mode='D')
    depth = depth.cpu().float().numpy()  # (C,H,W)
    dmax = float(np.percentile(depth[depth > 0], 95)) if (depth > 0).any() else 50.0

    # ════════════════════════════ Fig1a ═════════════════════════════════
    fig = plt.figure(figsize=(14, 5.6), facecolor='#0a0a12')
    fig.suptitle("Stage 1: Denoising & Rendering Pretraining",
                 color='white', fontsize=15, fontweight='bold', y=0.985)
    # col 0 = Query Denoising (spans both rows); cols 1-3 = 2x3 depth surround
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1],
                          width_ratios=[1.15, 1, 1, 1],
                          hspace=0.16, wspace=0.05,
                          left=0.015, right=0.995, top=0.86, bottom=0.02)
    # left: Query Denoising (full height)
    axq = fig.add_subplot(gs[:, 0])
    axq.scatter(lidar[:, 0], lidar[:, 1], s=0.4, c='#3a3a55', alpha=0.5,
                linewidths=0, rasterized=True, label='LiDAR')
    axq.scatter(init[:, 0], init[:, 1], s=7, c='#ff5d5d', alpha=0.45,
                linewidths=0, label='init (noised)')
    axq.scatter(refined[:, 0], refined[:, 1], s=9, c='#36d399', alpha=0.85,
                linewidths=0, label='refined (denoised)')
    axq.set_xlim(-RANGE_M, RANGE_M); axq.set_ylim(-RANGE_M, RANGE_M)
    axq.set_aspect('equal'); axq.set_xticks([]); axq.set_yticks([])
    axq.set_facecolor('#0a0a12')
    for sp in axq.spines.values():
        sp.set_color('#444')
    axq.set_title("Query Denoising", color='white', fontsize=12, pad=6)
    axq.legend(loc='lower right', fontsize=8, framealpha=0.25,
               facecolor='#0a0a12', labelcolor='white')
    # right: 6-cam depth (2x3 surround), both rows full height
    name2idx = {n: i for i, n in enumerate(CAM_NAMES)}
    for k, (cam, lbl) in enumerate(SURROUND):
        r, cc = divmod(k, 3)
        ax = fig.add_subplot(gs[r, 1 + cc])
        di = depth[name2idx[cam]]
        ax.imshow(np.ma.masked_where(di <= 0, di), cmap='magma',
                  vmin=0, vmax=dmax, aspect='auto')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(lbl, color='#bbb', fontsize=8, pad=2)
        for sp in ax.spines.values():
            sp.set_color('#333')
    fig.text(0.74, 0.90, "Depth Map Rendering", color='white',
             fontsize=12, ha='center')
    p1 = os.path.join(out_dir, "Fig1a_stage1_repro.png")
    fig.savefig(p1, dpi=140, facecolor='#0a0a12'); plt.close(fig)

    # ═══════════════════ fig2 (Stage-1 columns only) ════════════════════
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4), facecolor='white')
    li = lidar[(np.abs(lidar[:, 0]) < RANGE_M) & (np.abs(lidar[:, 1]) < RANGE_M)]
    _bev(ax[0], li[:, :2], li[:, 2], 'viridis', 0.6,
         "LiDAR (only for visualization)", "height z (m)", dark=False)
    _bev(ax[1], refined[:, :2], off_mag, 'plasma', 12,
         "Query Offsets (Pretrained)", "|refined - init| (m)", dark=False)
    mm = means[(np.abs(means[:, 0]) < RANGE_M) & (np.abs(means[:, 1]) < RANGE_M)]
    mo = opac[(np.abs(means[:, 0]) < RANGE_M) & (np.abs(means[:, 1]) < RANGE_M)]
    _bev(ax[2], mm[:, :2], mo, 'turbo', 2.0,
         "Gauss. Centers (Pretrained)", "opacity", dark=False)
    fig.suptitle("Figure 2 (Stage-1 reproduction) — Pretrained columns only; "
                 "Occ.Pred & No-Pretraining baseline omitted (need Stage-2)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = os.path.join(out_dir, "fig2_stage1_repro.png")
    fig.savefig(p2, dpi=140); plt.close(fig)

    # ═══════════ depth-vs-GT comparison (GT RGB | pred depth | GT LiDAR) ═══
    has_gtd = 'lidar_depth' in frame
    fig, ax = plt.subplots(6, 3, figsize=(13, 15), facecolor='white')
    for k, (cam, lbl) in enumerate(SURROUND):
        ci = name2idx[cam]
        rgb = frame['imgs'][0, ci].cpu().float().permute(1, 2, 0).numpy()
        ax[k, 0].imshow(rgb.clip(0, 1))
        ax[k, 0].set_ylabel(lbl, fontsize=10)
        ax[k, 0].set_title("GT RGB (input)" if k == 0 else "")
        di = depth[ci]
        ax[k, 1].imshow(np.ma.masked_where(di <= 0, di), cmap='magma',
                        vmin=0, vmax=dmax)
        ax[k, 1].set_title("predicted depth" if k == 0 else "")
        if has_gtd:
            gd = frame['lidar_depth'][0, ci].cpu().float().numpy()
            ax[k, 2].imshow(rgb.clip(0, 1))
            ax[k, 2].scatter(*np.where(gd > 0)[::-1], c=gd[gd > 0], cmap='magma',
                             vmin=0, vmax=dmax, s=2, linewidths=0)
            ax[k, 2].set_xlim(0, gd.shape[1]); ax[k, 2].set_ylim(gd.shape[0], 0)
            ax[k, 2].set_title("GT LiDAR depth (sparse)" if k == 0 else "")
        else:
            ax[k, 2].text(0.5, 0.5, "no GT depth in frame",
                          ha='center', va='center')
        for a in ax[k]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Stage-1 predicted depth vs ground truth — curated-val seq "
                 f"#{seq_idx} (ckpt_eval_best, BEV nn_dist 1.58 m)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p3 = os.path.join(out_dir, "Fig1a_depth_vs_gt.png")
    fig.savefig(p3, dpi=130); plt.close(fig)

    print(f"\nSaved:\n  {p1}\n  {p2}\n  {p3}")
    print(f"  seq #{seq_idx} | K={len(refined)} queries, {len(means)} gaussians, "
          f"{len(lidar)} lidar pts")
    print(f"  query offset |refined-init|  mean={off_mag.mean():.3f} m  "
          f"max={off_mag.max():.3f} m")
    print(f"  refined→anchor dist          mean="
          f"{np.linalg.norm(refined - anchors, axis=1).mean():.3f} m")
    print(f"  opacity  mean={opac.mean():.3f}  depth p95={dmax:.1f} m")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scene-tokens-json",
                   default="dataset_stats/curated_val_50/tokens.json")
    p.add_argument("--seq-idx", type=int, default=0)
    p.add_argument("--out-dir", default=".")
    a = p.parse_args()
    main(a.checkpoint, a.scene_tokens_json, a.seq_idx, a.out_dir)

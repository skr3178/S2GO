"""Standalone Query-Denoising visualization from a Stage-1 checkpoint.

Replicates the paper Fig1a "Query Denoising" left panel (BEV: dense LiDAR
in blue, per-query red->green arrows = noised init -> refined/denoised),
and adds two magnitude-aware views:

  [1] paper-style : LiDAR (blue) + red(init)->green(refined) arrows on black
  [2] vectors     : same arrows, coloured by |refined - init| (displacement)
  [3] heatmap     : |refined - init| hexbin heat overlaid on the BEV

Run:
    cd /media/skr/storage/self_driving/S2GO
    PYTHONPATH=. python scripts/viz_query_denoising.py \
        --checkpoint out/stage1_curated_270_depthonly-20260516-153143/ckpt_eval_best.pt \
        --scene-tokens-json dataset_stats/curated_val_50/tokens.json \
        --seq-idx 0 --range-m 40 --out query_denoising.png
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s2go.datasets.nusc_loader import NuScenesLoader
from s2go.tools.stage1_eval import load_model, run_forward


def _bev_axes(ax, R, dark=True):
    ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    if dark:
        ax.set_facecolor('black')
    for sp in ax.spines.values():
        sp.set_color('#444')
    # ego marker
    ax.plot(0, 0, marker='^', ms=9, color='#ffd166', zorder=6)


@torch.no_grad()
def main(checkpoint, scene_tokens_json, seq_idx, R, out):
    device = "cuda"
    backbone, seg, cfg = load_model(checkpoint, device)
    loader = NuScenesLoader(T=cfg['T_queue'], verbose=True,
                            scene_tokens_json=scene_tokens_json)
    seq_idx = seq_idx % len(loader)
    seq_cpu = loader[seq_idx]
    sequence = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in f.items()} for f in seq_cpu]
    outputs, _ = run_forward(backbone, seg, sequence, device)
    o, frame = outputs[-1], sequence[-1]

    init    = o.init_xyz[0].cpu().float().numpy()        # (K,3) noised
    refined = o.refined_xyz[0].cpu().float().numpy()     # (K,3) denoised
    anchors = o.anchors_xyz[0].cpu().float().numpy()     # (K,3) FPS target
    lidar   = frame['lidar_pts'][0].cpu().float().numpy()  # (M,3)
    d = refined - init                                    # (K,3)
    mag = np.linalg.norm(d[:, :2], axis=1)                # BEV displacement

    # ── Denoising statistics (over ALL K queries, 3-D) ───────────────────
    PCT = [10, 25, 50, 75, 90, 95]
    err_before = np.linalg.norm(init - anchors, axis=1)      # ‖init-FPS‖
    err_after = np.linalg.norm(refined - anchors, axis=1)    # ‖refined-FPS‖ = L_denoise
    disp = np.linalg.norm(d, axis=1)                          # ‖refined-init‖
    lt = torch.from_numpy(lidar).float()
    nn_init = torch.cdist(torch.from_numpy(init).float(), lt).min(1).values.numpy()
    nn_ref = torch.cdist(torch.from_numpy(refined).float(), lt).min(1).values.numpy()
    nn_anc = torch.cdist(torch.from_numpy(anchors).float(), lt).min(1).values.numpy()

    def _row(name, a):
        ps = np.percentile(a, PCT)
        return (f"  {name:<26s} mean={a.mean():6.3f}  std={a.std():6.3f}  "
                f"min={a.min():5.3f}  " +
                "  ".join(f"p{p}={v:5.3f}" for p, v in zip(PCT, ps)) +
                f"  max={a.max():6.3f}")

    print("\n================ Stage-1 Query-Denoising statistics "
          f"(K={len(init)} queries, seq #{seq_idx}) ================")
    print("‖query − FPS anchor‖  (the L_denoise term; 'before'=init, 'after'=refined):")
    print(_row("before (init/noised)", err_before))
    print(_row("after  (refined)",     err_after))
    red = 100.0 * (1 - err_after.mean() / err_before.mean())
    print(f"  → mean denoising error {err_before.mean():.3f} → "
          f"{err_after.mean():.3f} m  ({red:+.1f}%)")
    print("\n‖refined − init‖  (displacement applied):")
    print(_row("displacement", disp))
    print("\nnearest-LiDAR distance (alignment to real structure):")
    print(_row("init (before)",  nn_init))
    print(_row("refined (after)", nn_ref))
    print(_row("FPS anchors (ideal)", nn_anc))
    print("=" * 96)

    # keep queries whose endpoints are in view
    inv = (np.abs(init[:, 0]) < R) & (np.abs(init[:, 1]) < R) & \
          (np.abs(refined[:, 0]) < R) & (np.abs(refined[:, 1]) < R)
    li = lidar[(np.abs(lidar[:, 0]) < R) & (np.abs(lidar[:, 1]) < R)]
    ix, iy = init[inv, 0], init[inv, 1]
    ux, uy = d[inv, 0], d[inv, 1]
    rx, ry = refined[inv, 0], refined[inv, 1]
    m = mag[inv]

    fig, ax = plt.subplots(2, 2, figsize=(15, 14), facecolor='white')

    # ── [0,0] paper-style: dense blue LiDAR + red->green arrows ──────────
    ax[0, 0].scatter(li[:, 0], li[:, 1], s=2.6, c='#5b8fd6', alpha=0.65,
                     linewidths=0, rasterized=True)
    ax[0, 0].quiver(ix, iy, ux, uy, angles='xy', scale_units='xy', scale=1.0,
                    color='#ff5d5d', width=0.0030, headwidth=4.5, headlength=5,
                    alpha=0.9, zorder=4)
    ax[0, 0].scatter(ix, iy, s=10, c='#ff5d5d', alpha=0.85, linewidths=0, zorder=5)
    ax[0, 0].scatter(rx, ry, s=13, c='#36d399', alpha=0.95, linewidths=0, zorder=5)
    _bev_axes(ax[0, 0], R)
    ax[0, 0].set_title("Query Denoising  (red = init/noised → green = refined)",
                       fontsize=12)

    # ── [0,1] arrows coloured by displacement magnitude ──────────────────
    ax[0, 1].scatter(li[:, 0], li[:, 1], s=0.8, c='#2b2b40', alpha=0.5,
                     linewidths=0, rasterized=True)
    q = ax[0, 1].quiver(ix, iy, ux, uy, m, angles='xy', scale_units='xy',
                        scale=1.0, cmap='plasma', width=0.0034,
                        headwidth=4, headlength=5, alpha=0.95, zorder=4)
    cb = plt.colorbar(q, ax=ax[0, 1], fraction=0.046, pad=0.02)
    cb.set_label("|refined − init|  (m, BEV)", fontsize=9)
    _bev_axes(ax[0, 1], R)
    ax[0, 1].set_title("Denoising vectors — coloured by displacement", fontsize=12)

    # ── [1,0] |refined − init| heat overlaid on BEV ──────────────────────
    ax[1, 0].scatter(li[:, 0], li[:, 1], s=0.8, c='#dfe6ee', alpha=0.7,
                     linewidths=0, rasterized=True)
    hb = ax[1, 0].hexbin(rx, ry, C=m, gridsize=34, cmap='inferno',
                         reduce_C_function=np.mean, mincnt=1,
                         extent=(-R, R, -R, R), alpha=0.85)
    cb2 = plt.colorbar(hb, ax=ax[1, 0], fraction=0.046, pad=0.02)
    cb2.set_label("mean |refined − init| (m) per cell", fontsize=9)
    _bev_axes(ax[1, 0], R, dark=False)
    ax[1, 0].set_facecolor('white')
    ax[1, 0].set_title("Displacement heatmap (at refined positions)", fontsize=12)

    # ── [1,1] before/after distribution (the denoising metric) ───────────
    axd = ax[1, 1]
    bins = np.linspace(0, max(err_before.max(), 1.0), 60)
    axd.hist(err_before, bins=bins, color='#ff5d5d', alpha=0.55,
             label=f"before  ‖init−FPS‖  μ={err_before.mean():.2f} m", density=True)
    axd.hist(err_after, bins=bins, color='#36d399', alpha=0.6,
             label=f"after  ‖refined−FPS‖  μ={err_after.mean():.2f} m", density=True)
    axd.axvline(err_before.mean(), color='#c0392b', ls='--', lw=1.4)
    axd.axvline(err_after.mean(), color='#1e8449', ls='--', lw=1.4)
    axd.set_xlabel("‖query − FPS anchor‖  (m)  — the L_denoise target")
    axd.set_ylabel("density")
    axd.set_title(f"Denoising error before vs after  "
                  f"({100*(1-err_after.mean()/err_before.mean()):+.1f}% mean)",
                  fontsize=12)
    axd.legend(fontsize=9); axd.grid(alpha=0.25)
    txt = (f"percentiles (m)   p25 / p50 / p75 / p90 / p95\n"
           f"before : "
           + " / ".join(f"{v:.2f}" for v in np.percentile(err_before,[25,50,75,90,95]))
           + f"\nafter  : "
           + " / ".join(f"{v:.2f}" for v in np.percentile(err_after,[25,50,75,90,95])))
    axd.text(0.97, 0.55, txt, transform=axd.transAxes, ha='right', va='top',
             fontsize=8.5, family='monospace',
             bbox=dict(boxstyle='round', fc='#f4f4f4', ec='#ccc'))

    fig.suptitle(f"Stage-1 Query Denoising — curated-val seq #{seq_idx} "
                 f"(ckpt_eval_best · {inv.sum()}/{len(init)} queries in ±{R:g} m view)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nSaved: {out}")
    print(f"  queries shown: {int(inv.sum())}/{len(init)}  | "
          f"|Δ| BEV  mean={m.mean():.3f}  median={np.median(m):.3f}  "
          f"max={m.max():.3f} m")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scene-tokens-json",
                   default="dataset_stats/curated_val_50/tokens.json")
    p.add_argument("--seq-idx", type=int, default=0)
    p.add_argument("--range-m", type=float, default=40.0)
    p.add_argument("--out", default="query_denoising.png")
    a = p.parse_args()
    main(a.checkpoint, a.scene_tokens_json, a.seq_idx, a.range_m, a.out)

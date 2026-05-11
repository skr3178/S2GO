"""Stage-1 eval / sanity-check script — loads a saved checkpoint and runs the
four-check qualitative review described in Stage1_design.md §11.

  A. BEV scatter: refined query positions vs LiDAR points
  B. Rendered depth + RGB side-by-side with ground truth (CAM_FRONT)
  C. Gaussian-state histograms: opacity, scale L2, velocity magnitude
  D. Memory-queue inspection: count / uniqueness / topk-opacity spread (text)

This is qualitative. Quantitative pass/fail thresholds (e.g. PSNR > 23 dB,
masked-depth L1 < 1 m) require full-scale training; on a 4-overfit-50-iter
checkpoint we expect partial signal at best — the purpose of these figures is
"is anything obviously broken", not "is this paper-quality".

Run:
    cd /media/skr/storage/self_driving/S2GO
    python -m s2go.tools.stage1_eval \\
        --checkpoint /tmp/stage1_t2_50iter.pt \\
        --out-dir /tmp/stage1_eval/
"""
import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..datasets.nusc_loader import NuScenesLoader
from ..models.backbone.r50_fpn import R50FPNBackbone
from ..models.segmentor import S2GOSegmentor
from ..render.gsplat_wrapper import render


CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
              'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


# ────────────────────────────────────────────────────────────────────────────
# Load checkpoint + rebuild model
# ────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt['config']
    print(f"Loaded checkpoint trained for {cfg.get('iters_trained', '?')} iters")
    print(f"  arch: K={cfg['K']} J={cfg['J']} embed={cfg['embed_dims']} "
          f"layers={cfg['num_layers']} pts={cfg['num_pts']} ffn={cfg['feedforward_channels']}")

    backbone = R50FPNBackbone(embed_dims=cfg['embed_dims'], num_outs=4,
                               pretrained=True).to(device)
    seg = S2GOSegmentor(
        K=cfg['K'], J=cfg['J'], embed_dims=cfg['embed_dims'],
        num_layers=cfg['num_layers'], num_pts=cfg['num_pts'],
        feedforward_channels=cfg['feedforward_channels'],
        T_queue=cfg['T_queue'],
    ).to(device)
    backbone.load_state_dict(ckpt['backbone'])
    seg.load_state_dict(ckpt['segmentor'])
    backbone.eval(); seg.eval()
    return backbone, seg, cfg


# ────────────────────────────────────────────────────────────────────────────
# Run forward + cache outputs for all checks
# ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_forward(backbone, seg, sequence, device):
    seq_feat = []
    for f in sequence:
        feat, ss, lsi, _ = backbone(f['imgs'])
        seq_feat.append({**f, 'feat_flatten': feat, 'spatial_shapes': ss,
                          'level_start_index': lsi, 'pad_h': 256, 'pad_w': 704})
    outputs = seg(seq_feat)
    return outputs, seq_feat


# ────────────────────────────────────────────────────────────────────────────
# Check A — BEV scatter
# ────────────────────────────────────────────────────────────────────────────
def check_a_bev(outputs, sequence, out_path: str, range_m: float = 50.0):
    """Top-down (X-Y) view of refined queries vs LiDAR pts. Last frame only."""
    last_out = outputs[-1]
    last_frame = sequence[-1]
    refined = last_out.refined_xyz[0].cpu().float().numpy()           # (K, 3)
    anchors = last_out.anchors_xyz[0].cpu().float().numpy()           # (K, 3)
    lidar = last_frame['lidar_pts'][0].cpu().float().numpy()          # (M, 3)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(lidar[:, 0], lidar[:, 1], s=0.5, c='lightgrey', alpha=0.5, label='LiDAR')
    ax.scatter(anchors[:, 0], anchors[:, 1], s=8, c='C0', alpha=0.4, label='FPS anchors (target)')
    ax.scatter(refined[:, 0], refined[:, 1], s=10, c='C3', alpha=0.7, label='refined queries (pred)')
    ax.set_xlim(-range_m, range_m); ax.set_ylim(-range_m, range_m)
    ax.set_aspect('equal'); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(f'A. BEV — refined queries vs LiDAR (frame {len(outputs)-1})')
    ax.legend(loc='upper right'); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)

    # Numeric companion: per-query distance to nearest LiDAR point
    refined_t = torch.from_numpy(refined)
    lidar_t = torch.from_numpy(lidar)
    d = torch.cdist(refined_t, lidar_t).min(dim=1).values    # (K,)
    return {
        'nn_dist_mean': float(d.mean()), 'nn_dist_median': float(d.median()),
        'nn_dist_min':  float(d.min()),  'nn_dist_max':    float(d.max()),
    }


# ────────────────────────────────────────────────────────────────────────────
# Check B — Rendered depth + RGB vs GT (3 cams)
# ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def check_b_renders(outputs, sequence, out_path: str, cam_indices=(0, 1, 3)):
    """Side-by-side: [GT image | rendered RGB] and [GT depth | rendered depth].
    Done at t=0 only — neighbors would just look similar with mild parallax.
    """
    last_out = outputs[-1]; last_frame = sequence[-1]
    G = last_out.gaussians._replace(
        means=last_out.gaussians.means.float(),
        scales=last_out.gaussians.scales.float(),
        rotations=last_out.gaussians.rotations.float(),
        opacities=last_out.gaussians.opacities.float(),
        colors=last_out.gaussians.colors.float() if last_out.gaussians.colors is not None else None,
        velocity=last_out.gaussians.velocity.float(),
    )
    rgb, depth, _ = render(G, viewmats=last_frame['viewmats'][0].float(),
                            Ks=last_frame['cam_K'][0].float(),
                            height=256, width=704, render_mode='RGB+D')
    # rgb: (N_cam, H, W, 3); depth: (N_cam, H, W)

    n = len(cam_indices)
    fig, axes = plt.subplots(n, 4, figsize=(20, 4 * n))
    if n == 1:
        axes = axes[None, :]
    for row, ci in enumerate(cam_indices):
        gt_img   = last_frame['imgs'][0, ci].cpu().float().permute(1, 2, 0).numpy()
        gt_d     = last_frame['lidar_depth'][0, ci].cpu().float().numpy()
        rend_rgb = rgb[ci].cpu().float().numpy().clip(0, 1)
        rend_d   = depth[ci].cpu().float().numpy()
        # Mask depth GT (0 = no LiDAR return) — show only valid pixels
        gt_d_m = np.ma.masked_where(gt_d == 0, gt_d)

        axes[row, 0].imshow(gt_img.clip(0, 1));        axes[row, 0].set_title(f'{CAM_NAMES[ci]} GT image')
        axes[row, 1].imshow(rend_rgb);                  axes[row, 1].set_title('rendered RGB')
        axes[row, 2].imshow(gt_d_m, cmap='viridis', vmin=0, vmax=50)
        axes[row, 2].set_title('GT depth (LiDAR)')
        axes[row, 3].imshow(rend_d,  cmap='viridis', vmin=0, vmax=50)
        axes[row, 3].set_title('rendered depth')
        for ax in axes[row]: ax.axis('off')
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    return {
        'rgb_minmax':   (float(rgb.min()),   float(rgb.max())),
        'depth_minmax': (float(depth.min()), float(depth.max())),
    }


# ────────────────────────────────────────────────────────────────────────────
# Check C — Gaussian-state histograms
# ────────────────────────────────────────────────────────────────────────────
def check_c_histograms(outputs, out_path: str):
    G = outputs[-1].gaussians
    op = G.opacities.detach().float().flatten().cpu().numpy()
    sc = G.scales.detach().float().norm(dim=-1).flatten().cpu().numpy()
    v  = G.velocity.detach().float().norm(dim=-1).flatten().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(op, bins=50, color='C0', alpha=0.8); axes[0].set_title(f'opacity  (μ={op.mean():.3f}, σ={op.std():.3f})')
    axes[0].axvline(op.mean(), color='r', ls='--', lw=1); axes[0].set_xlabel('opacity')
    axes[1].hist(sc, bins=50, color='C1', alpha=0.8); axes[1].set_title(f'scale L2 (μ={sc.mean():.3f}m, σ={sc.std():.3f}m)')
    axes[1].axvline(sc.mean(), color='r', ls='--', lw=1); axes[1].set_xlabel('||scale|| (m)')
    axes[2].hist(v,  bins=50, color='C2', alpha=0.8); axes[2].set_title(f'|velocity|  (μ={v.mean():.3f}m/s, σ={v.std():.3f}m/s)')
    axes[2].axvline(v.mean(), color='r', ls='--', lw=1); axes[2].set_xlabel('||v|| (m/s)')
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
    return {
        'opacity': dict(mean=op.mean(), std=op.std(), min=op.min(), max=op.max()),
        'scale':   dict(mean=sc.mean(), std=sc.std(), min=sc.min(), max=sc.max()),
        'velocity':dict(mean=v.mean(),  std=v.std(),  min=v.min(),  max=v.max()),
    }


# ────────────────────────────────────────────────────────────────────────────
# Check D — Memory queue + propagator inspection (text)
# ────────────────────────────────────────────────────────────────────────────
def check_d_queue(seg, outputs) -> dict:
    q_rp = seg.queue.memory_reference_point
    if q_rp is None:
        return {'note': 'queue is empty (segmentor reset or never ran)'}
    rp = q_rp[0].detach().float()
    nonzero = (rp.abs().sum(dim=-1) > 0)
    n_pop = int(nonzero.sum().item())
    populated = rp[nonzero]
    n_uniq = int(torch.unique(populated.round(decimals=4), dim=0).shape[0]) if n_pop > 0 else 0
    last_prop = outputs[-1].prop
    topk = last_prop.opa.detach().float().flatten()
    return {
        'memory_capacity':    int(q_rp.shape[1]),
        'memory_populated':   n_pop,
        'memory_unique_xyz':  n_uniq,
        'topk_opacity_min':   float(topk.min()),
        'topk_opacity_max':   float(topk.max()),
        'topk_opacity_mean':  float(topk.mean()),
        'topk_opacity_std':   float(topk.std()),
    }


# ────────────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────────────
def main(checkpoint: str, out_dir: str, seq_idx: int):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda"
    backbone, seg, cfg = load_model(checkpoint, device)

    print(f"Loading sequence #{seq_idx} (T={cfg['T_queue']}) from nuScenes Part 1…")
    loader = NuScenesLoader(T=cfg['T_queue'], verbose=False)
    sequence_cpu = loader[seq_idx]
    sequence = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in f.items()} for f in sequence_cpu]

    print("Running forward pass…")
    outputs, _ = run_forward(backbone, seg, sequence, device)

    print("Check A — BEV scatter…")
    stats_a = check_a_bev(outputs, sequence, os.path.join(out_dir, 'A_bev.png'))
    print(f"  nearest-LiDAR distance per query (m): mean={stats_a['nn_dist_mean']:.3f} "
          f"median={stats_a['nn_dist_median']:.3f}  min={stats_a['nn_dist_min']:.3f}  "
          f"max={stats_a['nn_dist_max']:.3f}")

    print("Check B — depth + RGB renders…")
    stats_b = check_b_renders(outputs, sequence, os.path.join(out_dir, 'B_renders.png'))
    print(f"  rendered RGB range: [{stats_b['rgb_minmax'][0]:.3f}, {stats_b['rgb_minmax'][1]:.3f}]")
    print(f"  rendered depth range: [{stats_b['depth_minmax'][0]:.2f}, {stats_b['depth_minmax'][1]:.2f}] m")

    print("Check C — Gaussian-state histograms…")
    stats_c = check_c_histograms(outputs, os.path.join(out_dir, 'C_histograms.png'))
    for k, v in stats_c.items():
        print(f"  {k:>10s}: μ={v['mean']:.3f}  σ={v['std']:.3f}  "
              f"[{v['min']:.3f}, {v['max']:.3f}]")

    print("Check D — memory queue + propagator…")
    stats_d = check_d_queue(seg, outputs)
    for k, v in stats_d.items():
        print(f"  {k:<22s} {v}")

    # Dump all stats as JSON for offline analysis
    import json
    with open(os.path.join(out_dir, 'eval_stats.json'), 'w') as fp:
        json.dump({
            'checkpoint': checkpoint,
            'seq_idx':    seq_idx,
            'A_bev':      stats_a,
            'B_render':   {k: list(v) if isinstance(v, tuple) else v
                            for k, v in stats_b.items()},
            'C_gauss':    stats_c,
            'D_queue':    stats_d,
        }, fp, indent=2, default=float)
    print(f"\nFigures + stats saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to .pt saved by overfit.py")
    p.add_argument("--out-dir",    default="/tmp/stage1_eval/", help="output directory for figures")
    p.add_argument("--seq-idx",    type=int, default=0, help="which loader sequence to use (0..N-1)")
    a = p.parse_args()
    main(a.checkpoint, a.out_dir, a.seq_idx)

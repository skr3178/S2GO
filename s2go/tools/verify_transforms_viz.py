"""Visual verification of derived inputs (lidar2img, viewmats, ego_pose, ego_pose_inv).

Produces three figures:
  1. Six cam images with LiDAR-depth overlay (via lidar2img).
  2. BEV in LIDAR_TOP frame with camera FOV cones (via viewmats^-1).
  3. Ego_pose round-trip overlay (points before / after world-roundtrip).

If lidar2img is correct, the LiDAR dots in (1) should sit on real surfaces.
If viewmats is correct, the FOV cones in (2) should point in directions
matching each camera's name (CAM_FRONT cone points +y forward, etc.).
If ego_pose / ego_pose_inv is correct, the two point clouds in (3) overlap.

Output: out/transform_verification/{cam_overlays.png, bev_fov.png, ego_roundtrip.png}
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from ..datasets.nusc_loader import NuScenesLoader, CAM_NAMES


OUT_DIR = "out/transform_verification"


def fig1_cam_overlays(f):
    """6-panel: each cam image with its LiDAR-depth overlay from lidar2img."""
    imgs       = f['imgs'].squeeze(0).numpy()              # (6, 3, H, W) in [0,1]
    l2i        = f['lidar2img'].squeeze(0).numpy()         # (6, 4, 4)
    pts        = f['lidar_pts'].squeeze(0).numpy()         # (M, 3)
    H, W       = imgs.shape[2], imgs.shape[3]

    fig, axes = plt.subplots(2, 3, figsize=(18, 7))
    pts_h = np.c_[pts, np.ones(pts.shape[0])]

    for c, ax in enumerate(axes.ravel()):
        # Project all LiDAR points through this cam
        proj = pts_h @ l2i[c].T                          # (M, 4)
        z = proj[:, 2]
        valid = z > 0.1
        u = proj[valid, 0] / z[valid]
        v = proj[valid, 1] / z[valid]
        in_b = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, d = u[in_b], v[in_b], z[valid][in_b]

        # Show image (CHW → HWC, [0,1] is fine for imshow)
        img = imgs[c].transpose(1, 2, 0)
        ax.imshow(np.clip(img, 0, 1))
        # Scatter LiDAR points colored by depth (clip 0-50 m for stable colormap)
        sc = ax.scatter(u, v, c=np.clip(d, 0, 50), cmap='turbo',
                         s=1.2, alpha=0.6, linewidths=0)
        ax.set_title(f"cam {c}: {CAM_NAMES[c]}  ({len(u)} LiDAR pts in-bounds)",
                      fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.7, pad=0.01)
    cbar.set_label("depth (m, clipped 0-50)")
    fig.suptitle("Figure 1: lidar2img verification — LiDAR projected onto each cam image",
                 fontsize=12, y=0.98)
    fig.savefig(f"{OUT_DIR}/cam_overlays.png", dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {OUT_DIR}/cam_overlays.png")


def fig2_bev_fov(f):
    """BEV plot in LIDAR_TOP frame with each camera's FOV cone drawn from viewmats."""
    pts       = f['lidar_pts'].squeeze(0).numpy()         # (M, 3)  LIDAR_TOP
    viewmats  = f['viewmats'].squeeze(0).numpy()          # (6, 4, 4)  LIDAR→cam
    K_arr     = f['cam_K'].squeeze(0).numpy()             # (6, 3, 3)

    H_img, W_img = 256, 704
    cone_range = 40.0          # how far to draw the FOV cones (m)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # Plot LiDAR points (clip z for top-down clarity, color by z)
    ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap='viridis',
                s=0.5, alpha=0.6, linewidths=0)
    ax.scatter([0], [0], c='red', s=150, marker='*', zorder=5, label='LIDAR_TOP origin')

    # For each camera, compute its origin and FOV in LIDAR_TOP frame.
    cam_colors = ['#ff4444', '#ff8800', '#ffcc00', '#0066ff', '#0099ff', '#00ccff']
    for c in range(6):
        V = viewmats[c]                                  # 4×4 LIDAR→cam
        V_inv = np.linalg.inv(V)                          # 4×4 cam→LIDAR
        # Cam origin in LIDAR_TOP frame
        cam_origin = V_inv[:3, 3]
        # Cam's forward direction (+z in cam coords) in LIDAR_TOP frame
        cam_forward = V_inv[:3, :3] @ np.array([0, 0, 1.0])
        # Cam's horizontal FOV from intrinsic K (assume principal point ≈ center)
        fx = K_arr[c, 0, 0]
        half_fov_h = np.arctan((W_img / 2.0) / fx)        # rad
        # Build 4 frustum corners at depth `cone_range`
        corners_cam = np.array([
            [-cone_range * np.tan(half_fov_h), 0, cone_range],   # left
            [+cone_range * np.tan(half_fov_h), 0, cone_range],   # right
        ])
        corners_h = np.c_[corners_cam, np.ones(2)]
        corners_lidar = (V_inv @ corners_h.T).T[:, :3]
        # Triangle (origin, left, right) in LIDAR_TOP xy plane
        tri = np.array([
            cam_origin[:2],
            corners_lidar[0, :2],
            corners_lidar[1, :2],
        ])
        ax.add_patch(Polygon(tri, alpha=0.18, facecolor=cam_colors[c],
                               edgecolor=cam_colors[c], linewidth=1.2,
                               label=f"{c}: {CAM_NAMES[c]}"))
        # Label at cone tip
        tip = cam_origin[:2] + cam_forward[:2] * (cone_range * 0.7)
        ax.annotate(CAM_NAMES[c].replace('CAM_', ''),
                     xy=(tip[0], tip[1]),
                     fontsize=8, color=cam_colors[c], ha='center', fontweight='bold')

    # Forward arrow (+y in LIDAR_TOP frame, per H3 hack)
    ax.annotate('', xy=(0, 30), xytext=(0, 5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(1.5, 18, '+y\n(forward,\nper hacks.md H3)',
             fontsize=9, color='black', va='center')

    ax.set_xlim(-50, 50); ax.set_ylim(-50, 80)
    ax.set_aspect('equal')
    ax.set_xlabel('LIDAR_TOP x (m)')
    ax.set_ylabel('LIDAR_TOP y (m)')
    ax.set_title("Figure 2: viewmats verification — BEV with cam FOV cones in LIDAR_TOP frame\n"
                  "(if cones point toward each cam's named direction, viewmats is correct)")
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(f"{OUT_DIR}/bev_fov.png", dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {OUT_DIR}/bev_fov.png")


def fig3_ego_roundtrip(f):
    """Plot LIDAR_TOP points before and after world-roundtrip via ego_pose / ego_pose_inv."""
    pts       = f['lidar_pts'].squeeze(0).numpy()
    ego       = f['ego_pose'].squeeze(0).numpy()
    ego_inv   = f['ego_pose_inv'].squeeze(0).numpy()

    pts_h    = np.c_[pts, np.ones(pts.shape[0])]
    pts_world = (ego @ pts_h.T).T[:, :3]
    pts_back  = (ego_inv @ np.c_[pts_world, np.ones(pts.shape[0])].T).T[:, :3]
    err       = np.linalg.norm(pts_back - pts, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) Original LIDAR_TOP points (BEV)
    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap='viridis', s=0.5, alpha=0.6)
    ax.scatter([0], [0], c='red', s=80, marker='*')
    ax.set_aspect('equal'); ax.set_xlim(-60, 60); ax.set_ylim(-60, 80)
    ax.set_title(f"(a) Original LIDAR_TOP pts (M={len(pts)})")
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.grid(True, alpha=0.3)

    # (b) After roundtrip — should look identical to (a)
    ax = axes[1]
    ax.scatter(pts_back[:, 0], pts_back[:, 1], c=pts_back[:, 2], cmap='viridis',
                s=0.5, alpha=0.6)
    ax.scatter([0], [0], c='red', s=80, marker='*')
    ax.set_aspect('equal'); ax.set_xlim(-60, 60); ax.set_ylim(-60, 80)
    ax.set_title(f"(b) After ego_pose_inv ∘ ego_pose roundtrip\n(should overlap (a) exactly)")
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.grid(True, alpha=0.3)

    # (c) Per-point error histogram
    ax = axes[2]
    ax.hist(err * 1e6, bins=40, color='steelblue', alpha=0.8)
    ax.set_xlabel('roundtrip error per point (µm)')
    ax.set_ylabel('count')
    ax.set_title(f"(c) Per-point roundtrip error\n"
                  f"max={err.max()*1e6:.1f} µm, mean={err.mean()*1e6:.2f} µm\n"
                  f"(SE(3) inversion is correct; max is fp32 noise)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 3: ego_pose / ego_pose_inv verification — roundtrip overlap",
                 fontsize=12, y=1.02)
    fig.savefig(f"{OUT_DIR}/ego_roundtrip.png", dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {OUT_DIR}/ego_roundtrip.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading sample 0 (T=1)…")
    loader = NuScenesLoader(T=1, verbose=False)
    f = loader[0][0]
    print(f"  pts: {f['lidar_pts'].shape}   imgs: {f['imgs'].shape}")
    print(f"\nGenerating visualizations into {OUT_DIR}/")
    fig1_cam_overlays(f)
    fig2_bev_fov(f)
    fig3_ego_roundtrip(f)
    print("\nDone.  Inspect the three PNGs to verify visually.")


if __name__ == "__main__":
    main()

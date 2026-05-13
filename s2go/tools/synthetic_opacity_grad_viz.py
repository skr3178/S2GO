"""Visual companion to synthetic_opacity_grad_test.

Re-runs the 3-Gaussian opacity-gradient test and saves a 4-panel figure:

  (a) Side-view schematic in (z, x) plane — camera at origin, 3 Gaussians
      at their world positions, target depth marker, rendered depth marker.
      Bubble size ∝ opacity.
  (b) Rendered depth image (mode='D', 128×128) with center pixel marked.
  (c) Rendered alpha image (accumulated transmittance).
  (d) Bar chart of ∂L/∂opacity per Gaussian with expected-sign indicators.

Output: out/synthetic_opacity_grad_viz/result.png

Run:
    cd /media/skr/storage/self_driving/S2GO
    python -m s2go.tools.synthetic_opacity_grad_viz
"""
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow

from .synthetic_opacity_grad_test import (
    build_three_gaussians, build_synthetic_camera,
)
from ..render.gsplat_wrapper import render


OUT_DIR = "out/synthetic_opacity_grad_viz"


def main():
    device = "cuda"
    torch.manual_seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    G = build_three_gaussians(device)
    viewmat, K, H, W = build_synthetic_camera(device)

    rgb, depth, alpha = render(G, viewmats=viewmat, Ks=K,
                                height=H, width=W, render_mode="D")
    cy, cx = H // 2, W // 2
    d_center = depth[0, cy, cx]
    L = (d_center - 10.0).abs()
    L.backward()

    grads = G.opacities.grad.squeeze(0).squeeze(-1).cpu().numpy()
    means = G.means.detach().squeeze(0).cpu().numpy()       # (3, 3)
    opacities_val = G.opacities.detach().squeeze(0).squeeze(-1).cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ── (a) Side-view schematic ────────────────────────────────────────────
    ax = axes[0, 0]
    # Camera marker at origin
    ax.scatter([0], [0], c='black', s=200, marker='^', zorder=5, label='camera')
    ax.annotate('camera', (0, 0), xytext=(-1.0, -0.5), fontsize=9)
    # Camera-ray (looking +z, gsplat OpenCV convention)
    ax.annotate('', xy=(22, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4, lw=1))
    # Target depth marker
    ax.axvline(10.0, color='green', ls='--', alpha=0.8, lw=2, label='target depth = 10 m')
    # Rendered depth marker
    ax.axvline(d_center.item(), color='red', ls=':', alpha=0.8, lw=2,
                label=f'rendered D = {d_center.item():.2f} m')
    # Gaussian bubbles — radius ∝ opacity (visualization only)
    colors = ['#2ca02c', '#d62728', '#ff7f0e']
    labels = ['G1 on-surface (α=0.9)', 'G2 front empty (α=0.1)', 'G3 front empty (α=0.1)']
    for i, (z, a, c, lbl) in enumerate(zip([10, 4, 7], [0.9, 0.1, 0.1], colors, labels)):
        ax.add_patch(Circle((z, 0), radius=0.3 + 1.2*a, color=c, alpha=0.6,
                              label=lbl, zorder=4))
        ax.annotate(f'G{i+1}', (z, 0), color='white', ha='center', va='center',
                     fontweight='bold', fontsize=9, zorder=6)

    ax.set_xlim(-2, 22); ax.set_ylim(-3, 3)
    ax.set_xlabel('z (m) — camera ray (OpenCV +Z)')
    ax.set_ylabel('lateral (m)')
    ax.set_title('(a) Side-view schematic — bubble size ∝ opacity')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # ── (b) Rendered depth image ───────────────────────────────────────────
    ax = axes[0, 1]
    d_img = depth[0].detach().cpu().numpy()
    im = ax.imshow(d_img, cmap='turbo', vmin=0, vmax=12)
    ax.scatter([cx], [cy], facecolor='none', edgecolor='white', s=300, lw=2)
    ax.set_title(f'(b) Rendered depth (mode="D", unnormalized) — '
                  f'center px = {d_center.item():.2f} m')
    ax.set_xlabel('px'); ax.set_ylabel('px')
    plt.colorbar(im, ax=ax, label='depth (m)', shrink=0.85)

    # ── (c) Rendered alpha image ───────────────────────────────────────────
    ax = axes[1, 0]
    a_img = alpha[0, ..., 0].detach().cpu().numpy()
    im = ax.imshow(a_img, cmap='magma', vmin=0, vmax=1)
    ax.scatter([cx], [cy], facecolor='none', edgecolor='white', s=300, lw=2)
    a_center_val = a_img[cy, cx]
    ax.set_title(f'(c) Accumulated alpha — center px = {a_center_val:.3f}')
    ax.set_xlabel('px'); ax.set_ylabel('px')
    plt.colorbar(im, ax=ax, label='alpha', shrink=0.85)

    # ── (d) Opacity-gradient bar chart ─────────────────────────────────────
    ax = axes[1, 1]
    bar_colors = ['green' if (grads[0] < 0) else 'red',
                  'green' if (grads[1] > 0) else 'red',
                  'green' if (grads[2] > 0) else 'red']
    bars = ax.bar(['G1\n(on-surface)', 'G2\n(z=4 front)', 'G3\n(z=7 front)'],
                    grads, color=bar_colors, alpha=0.7, edgecolor='black')
    # Expected-sign overlay
    expected_signs = ['< 0', '> 0', '> 0']
    expected_colors = ['blue', 'blue', 'blue']
    for i, (bar, txt) in enumerate(zip(bars, expected_signs)):
        y = bar.get_height()
        offset = 0.5 if y > 0 else -0.8
        ax.annotate(f"expected: {txt}",
                     xy=(bar.get_x() + bar.get_width()/2, y + offset),
                     ha='center', fontsize=9, color='blue', fontweight='bold')
        ax.annotate(f"{grads[i]:+.3f}",
                     xy=(bar.get_x() + bar.get_width()/2, y/2),
                     ha='center', va='center', fontsize=11, fontweight='bold',
                     color='white' if abs(y) > 1.5 else 'black')
    ax.axhline(0, color='black', lw=1)
    ax.set_ylabel('∂L / ∂opacity')
    ax.set_title('(d) Opacity gradients\n(green bar = matches expected sign)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(min(grads) * 1.3 - 1, max(grads) * 1.3 + 1)

    verdict = "PASS" if (grads[0] < 0 and grads[1] > 0 and grads[2] > 0) else "FAIL"
    fig.suptitle(f"Synthetic 3-Gaussian opacity-gradient test  ·  "
                 f"loss = {L.item():.4f} m  ·  verdict: {verdict}",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    out_path = f"{OUT_DIR}/result.png"
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

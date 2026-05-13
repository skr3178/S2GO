"""Synthetic-Gaussian opacity-gradient unit test.

Verifies the single most-used differentiable operator in S2GO training:
the gsplat alpha-compositing backward pathway. Pure unit test — no
nuScenes data, no segmentor, no decoder.

Setup
-----
  3 Gaussians, all on the +Z ray from camera origin so they project to
  the same pixel:

    G1 ── on-LiDAR-surface ─── z=10 m, opacity=0.9
    G2 ── empty space (front) ─ z= 4 m, opacity=0.1
    G3 ── empty space (front) ─ z= 7 m, opacity=0.1

  All off-surface Gaussians are placed IN FRONT of the target so all three
  opacity gradients have unambiguous signs under unnormalized depth render
  (mode='D'). (With one front + one back, G3 behind the target can flip its
  sign — a property of unnormalized D, not a bug. Front-only avoids that.)

Synthetic target
----------------
  Depth map zeros everywhere except 10.0 m at the central pixel where the
  Gaussians project. L1 loss masked to that pixel only.

Expected gradient signs
-----------------------
  ∂L/∂α_1  <  0     pushing G1's opacity UP reduces loss
                       (dominate the alpha composite at z=10)
  ∂L/∂α_2  >  0     pushing G2's opacity DOWN reduces loss
                       (fade so it doesn't pull D toward 4)
  ∂L/∂α_3  >  0     pushing G3's opacity DOWN reduces loss
                       (fade so it doesn't pull D toward 7)

If all three signs match: the gsplat backward → opacity → loss pathway is
correct. The topk_opacity collapse seen in v2 streaming must be upstream
(decoder, queries, optimizer dynamics) rather than a render-wiring bug.

Run
---
    cd /media/skr/storage/self_driving/S2GO
    python -m s2go.tools.synthetic_opacity_grad_test
"""
import torch

from ..render.gsplat_wrapper import render
from ..models.encoder.assembly import Gaussians


def build_three_gaussians(device: str = "cuda"):
    """3 Gaussians on the +Z ray, hand-crafted opacities."""
    # (B=1, N=3, ...) — wrapper strips leading B and feeds gsplat
    means = torch.tensor([
        [0.0, 0.0, 10.0],     # G1: on-surface
        [0.0, 0.0,  4.0],     # G2: front, empty
        [0.0, 0.0,  7.0],     # G3: closer front, empty
    ], device=device).unsqueeze(0)                                 # (1, 3, 3)
    # Isotropic small Gaussians so they cover a few pixels but not too many
    scales = torch.full((1, 3, 3), 0.5, device=device)              # (1, 3, 3)
    # Identity rotation (wxyz convention used by gsplat)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]],
                              device=device).repeat(1, 3, 1)         # (1, 3, 4)
    # Opacities — the variable under test
    opacities = torch.tensor([0.9, 0.1, 0.1],
                              device=device).view(1, 3, 1)            # (1, 3, 1)
    opacities.requires_grad_(True)
    # No colors (depth-only render)
    colors = None
    velocity = torch.zeros((1, 3, 3), device=device)
    return Gaussians(means=means, scales=scales, rotations=rotations,
                     opacities=opacities, colors=colors, velocity=velocity)


def build_synthetic_camera(device: str = "cuda"):
    """Single camera at origin, looking +Z (OpenCV), 128×128, fx=fy=128.
    All 3 Gaussians (xy=0) project to image center."""
    H, W = 128, 128
    fx, fy = 128.0, 128.0
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], device=device).unsqueeze(0)   # (1, 3, 3)
    viewmat = torch.eye(4, device=device).unsqueeze(0)                # (1, 4, 4)
    return viewmat, K, H, W


def main():
    device = "cuda"
    torch.manual_seed(0)

    print("== Synthetic 3-Gaussian opacity gradient test ==\n")
    print("Setup: 3 Gaussians on +Z ray; target depth = 10.0 m at image center.")
    print("  G1: z=10.0, opacity=0.9  (on-surface; should be pushed UP → grad < 0)")
    print("  G2: z= 4.0, opacity=0.1  (front-empty; should be pushed DOWN → grad > 0)")
    print("  G3: z= 7.0, opacity=0.1  (front-empty; should be pushed DOWN → grad > 0)\n")

    G = build_three_gaussians(device)
    viewmat, K, H, W = build_synthetic_camera(device)

    # Forward through the same wrapper used in training.
    rgb, depth, alpha = render(G, viewmats=viewmat, Ks=K,
                                height=H, width=W, render_mode="D")
    # depth: (C=1, H, W)
    cy, cx = H // 2, W // 2
    d_center = depth[0, cy, cx]
    a_center = alpha[0, cy, cx, 0]
    print(f"Forward render at center pixel ({cx}, {cy}):")
    print(f"  rendered depth (mode='D', unnormalized) = {d_center.item():.4f} m")
    print(f"  alpha (accumulated)                     = {a_center.item():.4f}")
    print(f"  expected analytical D (front-to-back):")
    print(f"    D = 1·0.1·4 + (1-0.1)·0.9·10 + (1-0.1)(1-0.9)·0.1·20 …")
    print(f"    (computed value depends on gsplat's pixel-footprint weighting)\n")

    # Loss: L1 vs target=10 at the center pixel only.
    target = 10.0
    L = (d_center - target).abs()
    print(f"Loss at center pixel: |{d_center.item():.4f} - {target}| = {L.item():.4f}")

    # Backward.
    L.backward()
    grads = G.opacities.grad.squeeze(0).squeeze(-1)   # (3,)
    print(f"\nGradients ∂L/∂opacity:")
    for i, (name, expect_sign) in enumerate([
        ("G1 (z=10, on-surface)", "< 0"),
        ("G2 (z= 4, front)",     "> 0"),
        ("G3 (z= 7, front)",     "> 0"),
    ]):
        g = grads[i].item()
        sign_str = "+" if g > 0 else ("-" if g < 0 else "0")
        print(f"  α_{i+1}  ({name}): grad = {g:+.6f}   "
              f"sign={sign_str}   expected: {expect_sign}")

    # Pass/fail verdict.
    expectations = [grads[0].item() < 0,
                    grads[1].item() > 0,
                    grads[2].item() > 0]
    nonzero = [abs(grads[i].item()) > 1e-6 for i in range(3)]

    print(f"\nVerdict:")
    print(f"  sign of grad[0]  (expect < 0):  "
          f"{'PASS' if expectations[0] else 'FAIL'}")
    print(f"  sign of grad[1]  (expect > 0):  "
          f"{'PASS' if expectations[1] else 'FAIL'}")
    print(f"  sign of grad[2]  (expect > 0):  "
          f"{'PASS' if expectations[2] else 'FAIL'}")
    print(f"  magnitudes > 1e-6           :  "
          f"{'PASS' if all(nonzero) else 'FAIL'}")

    if all(expectations) and all(nonzero):
        print("\n  ALL CHECKS PASS — gsplat backward → opacity pathway is correct.")
        print("  Implication: the topk_opacity drop on streaming data is NOT a")
        print("  render-wiring bug. Look upstream (decoder, optimizer dynamics).")
    else:
        print("\n  AT LEAST ONE CHECK FAILED — structural bug in render/opacity path.")


if __name__ == "__main__":
    main()

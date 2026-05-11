"""S1.0 — gsplat smoke test in isolation.

Per Stage1_design.md §7 build order:
  "render 100 random Gaussians on one (synthetic) camera, eyeball depth + RGB
   output. No network, no loss."

What this verifies:
  1. gsplat.rasterization API works with our env (gsplat 1.5.3 + torch 2.0+cu118).
  2. Output shapes match Stage1_design.md §6 expectations:
       renders = (C=1, H, W, 4)  with last channel = depth
       alphas  = (C=1, H, W, 1)
  3. Gradients flow back to all 5 Gaussian parameters
       (means, quats, scales, opacities, colors).
  4. Render is visually plausible (saved to out/s1.0_smoke/{rgb,depth}.png).

Not in scope here: nuScenes intrinsics, multi-camera, real LiDAR depth.
Those come in S1.3 once the lifter + child head produce real Gaussians.

Run from project root:
    conda activate /media/skr/storage/conda_envs/selfocc
    python s2go/render/gsplat_smoke.py
"""
import os
import torch
from gsplat import rasterization
from PIL import Image


def make_synthetic_gaussians(N=100, device='cuda', seed=0):
    """N random Gaussians in a 4×4×4 m volume placed in front of camera.

    Camera convention (gsplat / OpenCV): world frame, camera at origin looking
    down +Z, viewmat=identity. So we place Gaussians at z ∈ [3, 7] m.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    means = torch.empty((N, 3), device=device).uniform_(-2.0, 2.0, generator=g)
    means[:, 2] = means[:, 2] + 5.0                              # z ∈ [3, 7]
    # Anisotropic scales so that rotation gradient is non-degenerate.
    scales = torch.empty((N, 3), device=device).uniform_(0.05, 0.20, generator=g)
    # Random unit quaternions so all 4 quat components contribute to rotation.
    quats = torch.empty((N, 4), device=device).uniform_(-1.0, 1.0, generator=g)
    quats = quats / quats.norm(dim=-1, keepdim=True)              # normalize
    opacities = torch.full((N,), 0.8, device=device)              # opaque-ish
    colors = torch.empty((N, 3), device=device).uniform_(0.0, 1.0, generator=g)
    return means, quats, scales, opacities, colors


def make_camera(width, height, fovy_deg=60.0, device='cuda'):
    """One pinhole camera, viewmat = identity (world frame == cam frame)."""
    fy = 0.5 * height / torch.tan(torch.tensor(fovy_deg) * torch.pi / 360)
    fx = fy
    K = torch.tensor([[fx,  0., width  / 2],
                      [0.,  fy, height / 2],
                      [0.,  0., 1.0]], device=device).unsqueeze(0)   # (1, 3, 3)
    viewmat = torch.eye(4, device=device).unsqueeze(0)               # (1, 4, 4)
    return K, viewmat


def main():
    device = 'cuda'
    H, W = 480, 640
    N = 100

    print(f"S1.0 smoke test — {N} synthetic Gaussians, {W}×{H} camera, gsplat rasterization")

    means, quats, scales, opacities, colors = make_synthetic_gaussians(N, device)
    K, viewmat = make_camera(W, H, device=device)

    # require_grad to test gradient flow
    means_g     = means.detach().requires_grad_(True)
    quats_g     = quats.detach().requires_grad_(True)
    scales_g    = scales.detach().requires_grad_(True)
    opacities_g = opacities.detach().requires_grad_(True)
    colors_g    = colors.detach().requires_grad_(True)

    # ── Forward ──────────────────────────────────────────────────────────
    renders, alphas, meta = rasterization(
        means=means_g, quats=quats_g, scales=scales_g,
        opacities=opacities_g, colors=colors_g,
        viewmats=viewmat, Ks=K,
        width=W, height=H,
        render_mode='RGB+D',
        near_plane=0.1, far_plane=20.0,
    )
    rgb   = renders[..., :3]    # (1, H, W, 3)
    depth = renders[..., 3]     # (1, H, W)

    # ── Shape assertions ─────────────────────────────────────────────────
    assert renders.shape == (1, H, W, 4), f"renders shape {renders.shape} != (1,{H},{W},4)"
    assert alphas.shape  == (1, H, W, 1), f"alphas shape {alphas.shape}  != (1,{H},{W},1)"

    print(f"  renders: {tuple(renders.shape)}  dtype={renders.dtype}")
    print(f"  rgb   range: [{rgb.min().item():.3f}, {rgb.max().item():.3f}]")
    print(f"  depth range: [{depth.min().item():.3f}, {depth.max().item():.3f}] m")
    print(f"  alpha range: [{alphas.min().item():.3f}, {alphas.max().item():.3f}]")

    # value-range sanity
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0 + 1e-3, "RGB outside [0,1]"
    assert depth.max() < 20.0, "depth exceeds far_plane"
    # gsplat 'D' mode = accumulated alpha-weighted depth (Σ w_i z_i), not normalized.
    # So low-alpha pixels naturally have small depth. Mask by alpha to test the
    # values in well-covered regions only — those should fall within [near, far].
    alpha = alphas[..., 0]
    well_covered = alpha > 0.5
    assert well_covered.any(), "no pixels with alpha > 0.5 — Gaussians too sparse / invisible?"
    # gsplat 'D' returns Σ w_i z_i; for well-covered pixels (Σ w_i ≈ alpha ≈ 1)
    # this is roughly the true depth. Allow generous slack: [near_plane, far_plane].
    well_covered_depth = depth[well_covered]
    assert well_covered_depth.min() > 0.05, \
        f"covered-pixel min depth {well_covered_depth.min().item():.3f} below sane lower bound"
    assert well_covered_depth.max() < 20.0, "covered-pixel depth exceeds far_plane"

    # ── Backward — gradient flow check ───────────────────────────────────
    # Use a non-trivial scalar so all 5 parameter tensors get non-zero grad.
    target_rgb   = torch.zeros_like(rgb)
    target_depth = torch.full_like(depth, 5.0)
    loss = (rgb - target_rgb).abs().mean() + (depth - target_depth).abs().mean()
    loss.backward()

    grads_ok = True
    for name, t in [('means', means_g), ('quats', quats_g), ('scales', scales_g),
                    ('opacities', opacities_g), ('colors', colors_g)]:
        if t.grad is None:
            print(f"  FAIL: {name}.grad is None")
            grads_ok = False
            continue
        norm = t.grad.norm().item()
        print(f"  ∂L/∂{name:<10} norm = {norm:.4e}  shape = {tuple(t.grad.shape)}")
        if norm == 0.0:
            print(f"  WARN: {name}.grad is all-zero")
    assert grads_ok, "some parameters did not receive gradient"

    # ── Save debug images ────────────────────────────────────────────────
    out_dir = 'out/s1.0_smoke'
    os.makedirs(out_dir, exist_ok=True)

    rgb_img = (rgb[0].detach().cpu().clamp(0, 1) * 255).byte().numpy()
    Image.fromarray(rgb_img).save(f'{out_dir}/rgb.png')

    d = depth[0].detach().cpu()
    d_vis = ((d - d.min()) / (d.max() - d.min() + 1e-9) * 255).byte().numpy()
    Image.fromarray(d_vis).save(f'{out_dir}/depth.png')

    a = alphas[0, ..., 0].detach().cpu()
    a_vis = (a.clamp(0, 1) * 255).byte().numpy()
    Image.fromarray(a_vis).save(f'{out_dir}/alpha.png')

    print(f"\n  Saved debug images to {out_dir}/  (rgb.png, depth.png, alpha.png)")
    print("\nS1.0 smoke test PASSED.")


if __name__ == '__main__':
    main()

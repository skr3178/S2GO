"""Forward-sanity test for a 3DGS diff-gaussian-rasterization build.

Usage:
    python test_dgr_forward_sanity.py <path-to-rasterizer-dir>

The path should point at a directory containing the `diff_gaussian_rasterization`
package (already built in-place, i.e. _C*.so present in the package dir).

Tests:
    1. import succeeds
    2. forward runs on N random Gaussians, output is finite, shape matches
    3. mean.backward() runs without error and produces finite grads
    4. simple timing report (forward + backward)
"""
import math
import os
import sys
import time

import torch


def make_camera(image_size: int = 256, fov_deg: float = 60.0, cam_dist: float = 4.0,
                device: str = "cuda"):
    """3DGS-convention camera: identity rotation, camera at world (0,0,-cam_dist),
    so the world origin is at view-space +cam_dist (3DGS expects +Z = forward).

    Returns (viewmatrix, projmatrix, campos, tanfovx, tanfovy) — viewmatrix and
    projmatrix are passed *transposed* (column-major), matching the 3DGS API.
    """
    fov = math.radians(fov_deg)
    tanfov = math.tan(fov / 2.0)
    znear, zfar = 0.01, 100.0

    # 3DGS world-to-view (Rt): R=I, camera position T in world.
    # world_view_transform[:3,:3] = R.T = I; [:3,3] = -R.T @ T
    R = torch.eye(3, device=device)
    T = torch.tensor([0.0, 0.0, -cam_dist], device=device)  # camera at -Z so origin is in front
    Rt = torch.eye(4, device=device)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = -(R.T @ T)
    Rt[3, 3] = 1.0

    # OpenGL-ish projection (matches 3DGS getProjectionMatrix with z_sign=+1)
    top = tanfov * znear
    right = tanfov * znear
    P = torch.zeros(4, 4, device=device)
    P[0, 0] = znear / right
    P[1, 1] = znear / top
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0

    viewmatrix = Rt.T.contiguous()
    full_proj = P @ Rt
    projmatrix = full_proj.T.contiguous()
    campos = T.clone()
    return viewmatrix, projmatrix, campos, tanfov, tanfov


def make_gaussians(num: int = 1024, device: str = "cuda", seed: int = 0):
    g = torch.Generator(device=device).manual_seed(seed)
    means = (torch.rand(num, 3, generator=g, device=device) * 2.0 - 1.0) * 0.5
    # opacities pre-sigmoid; pass sigmoid-applied to rasterizer
    opacities = torch.sigmoid(torch.randn(num, 1, generator=g, device=device) * 0.5 - 1.0)
    scales = torch.exp(torch.randn(num, 3, generator=g, device=device) * 0.3 - 3.0)  # ~ exp(-3) = 0.05
    rots = torch.randn(num, 4, generator=g, device=device)
    rots = rots / rots.norm(dim=-1, keepdim=True)
    colors = torch.rand(num, 3, generator=g, device=device)
    return means, opacities, scales, rots, colors


def main():
    if len(sys.argv) < 2:
        print("usage: python test_dgr_forward_sanity.py <rasterizer-dir>")
        sys.exit(2)
    raster_dir = os.path.abspath(sys.argv[1])
    print(f"[test] inserting {raster_dir} into sys.path")
    sys.path.insert(0, raster_dir)

    import diff_gaussian_rasterization as dgr
    print(f"[test] imported diff_gaussian_rasterization from {dgr.__file__}")

    device = "cuda"
    image_size = 256
    num_gaussians = 4096

    viewmatrix, projmatrix, campos, tanfovx, tanfovy = make_camera(
        image_size=image_size, device=device
    )
    means, opacities, scales, rots, colors = make_gaussians(num=num_gaussians, device=device)

    # Both APIs need means2D as a leaf grad-track tensor (used as gradient sink for 2D positions)
    means2D = torch.zeros_like(means, requires_grad=True)
    means.requires_grad_(True)
    opacities.requires_grad_(True)
    scales.requires_grad_(True)
    rots.requires_grad_(True)
    colors.requires_grad_(True)

    # Detect API variant by inspecting GaussianRasterizationSettings fields
    fields = dgr.GaussianRasterizationSettings._fields
    has_pixel_weights = "pixel_weights" in fields  # taming
    print(f"[test] settings fields: {fields}")
    print(f"[test] variant: {'taming-3dgs' if has_pixel_weights else 'vanilla 3dgs'}")

    bg = torch.zeros(3, device=device)
    common = dict(
        image_height=image_size,
        image_width=image_size,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
    )
    if has_pixel_weights:
        common["pixel_weights"] = torch.empty(0, device=device)

    settings = dgr.GaussianRasterizationSettings(**common)
    rasterizer = dgr.GaussianRasterizer(raster_settings=settings)

    # ---- Forward (use colors_precomp to bypass SH/DC API asymmetry) ----
    fwd_kwargs = dict(
        means3D=means,
        means2D=means2D,
        opacities=opacities,
        colors_precomp=colors,
        scales=scales,
        rotations=rots,
    )
    torch.cuda.synchronize()
    t0 = time.time()
    out = rasterizer(**fwd_kwargs)
    torch.cuda.synchronize()
    fwd_ms = (time.time() - t0) * 1e3

    # Output is (rendered_image, radii) for vanilla; taming may add extras.
    if isinstance(out, tuple):
        rendered = out[0]
        radii = out[1] if len(out) > 1 else None
    else:
        rendered = out
        radii = None

    print(f"[test] forward OK in {fwd_ms:.1f} ms, image shape={tuple(rendered.shape)}")

    # ---- Sanity asserts ----
    assert rendered.shape == (3, image_size, image_size), \
        f"unexpected shape: {rendered.shape}"
    assert torch.isfinite(rendered).all(), "rendered image has NaN/Inf"
    rmin, rmax = rendered.min().item(), rendered.max().item()
    print(f"[test] image range: [{rmin:.4f}, {rmax:.4f}]")
    if radii is not None:
        nvis = int((radii > 0).sum().item())
        print(f"[test] radii: {nvis}/{radii.numel()} Gaussians have positive radius")
        assert nvis > 0, "no Gaussians produced positive radius — camera/setup bug"

    # ---- Backward ----
    torch.cuda.synchronize()
    t0 = time.time()
    loss = rendered.mean()
    loss.backward()
    torch.cuda.synchronize()
    bwd_ms = (time.time() - t0) * 1e3
    print(f"[test] backward OK in {bwd_ms:.1f} ms, loss={loss.item():.6f}")

    for name, t in [("means", means), ("opacities", opacities),
                    ("scales", scales), ("rots", rots), ("colors", colors)]:
        assert t.grad is not None, f"{name}.grad is None"
        assert torch.isfinite(t.grad).all(), f"{name}.grad has NaN/Inf"
        print(f"[test]   d/d{name:<10} : norm={t.grad.norm().item():.4e}")

    # ---- Timing (warmed) ----
    for t in (means, opacities, scales, rots, colors):
        t.grad = None

    n_iters = 20
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        out = rasterizer(**fwd_kwargs)
        rendered = out[0] if isinstance(out, tuple) else out
        rendered.mean().backward()
    torch.cuda.synchronize()
    avg_ms = (time.time() - t0) * 1e3 / n_iters
    print(f"[test] mean fwd+bwd over {n_iters} iters: {avg_ms:.2f} ms "
          f"({num_gaussians} gaussians @ {image_size}x{image_size})")

    print("[test] PASS")


if __name__ == "__main__":
    main()

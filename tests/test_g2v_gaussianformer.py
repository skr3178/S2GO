"""Compare GaussianFormer's localagg_prob_fast G2V kernel against PyTorch reference.

This is the *naive baseline* G2V kernel that S2GO claims to beat 20×. Running it
through our test scaffold:
  1. Validates that our PyTorch reference math (`g2v_forward_reference_gf`)
     actually matches an independent CUDA implementation, not just itself.
  2. Establishes a perf baseline (ms/iter on the same synthetic scene) that
     the future S2GO kernel must beat.

Layout note: GaussianFormer's `local_aggregate_prob_fast` package is on the
import path because we already verified `python -c "import
local_aggregate_prob_fast"` works in the selfocc env.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent))
from g2v_reference import (
    g2v_forward_reference_gf,
    make_voxel_grid,
    quat_to_rotmat,
)
from local_aggregate_prob_fast import LocalAggregator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def synth_scene(num_gaussians: int, num_classes: int, grid_dim: int,
                voxel_size: float, *, seed: int = 0, device: str = "cuda"):
    """Random Gaussians scattered inside a cubic voxel grid."""
    g = torch.Generator(device=device).manual_seed(seed)
    grid_extent = grid_dim * voxel_size
    grid_origin = torch.full((3,), -grid_extent / 2, device=device)

    # Means inside the grid (with margin so means3D_int stays in [0, grid_dim))
    margin = voxel_size * 0.51
    means = grid_origin + margin + torch.rand(num_gaussians, 3, generator=g, device=device) \
            * (grid_extent - 2 * margin)
    # Scales such that 3σ ≈ ~3 voxels — large enough to overlap many voxels but
    # not blow up the grid
    scales = torch.exp(torch.randn(num_gaussians, 3, generator=g, device=device) * 0.2) \
             * voxel_size * 1.0                                          # σ ≈ voxel_size
    rots = torch.randn(num_gaussians, 4, generator=g, device=device)
    rots = rots / rots.norm(dim=-1, keepdim=True)
    opacities = torch.sigmoid(torch.randn(num_gaussians, generator=g, device=device) * 0.5)
    semantics = torch.softmax(
        torch.randn(num_gaussians, num_classes, generator=g, device=device), dim=-1
    )
    return grid_origin, means, rots, scales, opacities, semantics


def cov3D_inv_from_rot_scale(rotations: Tensor, scales: Tensor) -> Tensor:
    """(N, 4) quat + (N, 3) scales → (N, 6) packed Σ⁻¹ in [xx, yy, zz, xy, yz, xz] order."""
    R = quat_to_rotmat(rotations)                              # (N, 3, 3)
    inv_s_sq = 1.0 / (scales ** 2)                             # (N, 3)
    Sinv = (R * inv_s_sq.unsqueeze(-2)) @ R.transpose(-1, -2)  # (N, 3, 3) = R diag(1/s²) Rᵀ
    return torch.stack([
        Sinv[:, 0, 0], Sinv[:, 1, 1], Sinv[:, 2, 2],
        Sinv[:, 0, 1], Sinv[:, 1, 2], Sinv[:, 0, 2],
    ], dim=-1)


def compare(name: str, a: Tensor, b: Tensor, *, rtol=1e-4, atol=1e-5) -> bool:
    diff = (a - b).abs()
    rel = diff / b.abs().clamp_min(1e-12)
    ok = torch.allclose(a, b, rtol=rtol, atol=atol)
    print(f"  {name:<28} max_abs={diff.max().item():.3e}  "
          f"max_rel={rel.max().item():.3e}  {'OK' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda"
    num_gaussians = 256
    num_classes = 18  # hardcoded in localagg_prob_fast/src/config.h (NUM_CHANNELS)
    grid_dim = 8                  # 8x8x8 = 512 voxels
    voxel_size = 0.25
    seed = 0
    rtol, atol = 5e-4, 5e-5       # tight enough to catch mismatches but float32-tolerant

    print(f"[gf] N={num_gaussians} C={num_classes} grid={grid_dim}^3 voxel={voxel_size}")

    grid_origin, means, rots, scales, opas, sem = synth_scene(
        num_gaussians, num_classes, grid_dim, voxel_size, seed=seed, device=device
    )
    grid_extent = grid_dim * voxel_size

    # --- voxel grid (the "points" the kernel evaluates at) ---
    centers = make_voxel_grid(grid_origin, (grid_dim,) * 3, voxel_size).reshape(-1, 3)
    V = centers.shape[0]
    pts_int = torch.floor((centers - grid_origin) / voxel_size).to(torch.int32)
    assert pts_int.min() >= 0 and pts_int.max() < grid_dim
    means_int = torch.floor((means - grid_origin) / voxel_size).to(torch.int32)
    assert means_int.min() >= 0 and means_int.max() < grid_dim, \
        f"means_int out of grid: range [{means_int.min().item()}, {means_int.max().item()}]"

    # --- precision matrix (Σ⁻¹) for both reference and kernel ---
    cov3D_inv_flat = cov3D_inv_from_rot_scale(rots, scales).detach()    # (N, 6)

    # ============================================================
    # FORWARD-only correctness check (no_grad)
    # ============================================================
    print("\n[gf] forward correctness")

    # The LocalAggregator wrapper takes a (1, ...) batch dim and 3x3 cov tensors.
    # It will internally re-pack to 6-flat. To bypass its repack and pass our
    # already-flat Σ⁻¹, we feed a 3x3 tensor whose flatten[:,[0,4,8,1,5,2]]
    # matches our cov3D_inv_flat. Easier path: build a fake 3x3 with the right
    # entries on positions (0, 4, 8, 1, 5, 2). The wrapper indexes those 6.
    #
    # But the kernel only ever reads the 6-flat result, so we can construct a
    # symmetric 3x3 such that its flatten[:, 0,4,8,1,5,2] = our (xx,yy,zz,xy,yz,xz).
    # That just means: build the symmetric Σ⁻¹ as a 3x3 from rotations/scales.
    R = quat_to_rotmat(rots)
    Sinv33 = (R * (1.0 / scales ** 2).unsqueeze(-2)) @ R.transpose(-1, -2)  # (N, 3, 3)

    aggregator = LocalAggregator(
        scale_multiplier=3.0,         # 3σ radius
        H=grid_dim, W=grid_dim, D=grid_dim,
        pc_min=grid_origin.cpu().tolist(),
        grid_size=voxel_size,
        radii_min=grid_dim,           # force full coverage so it considers ALL pairs
                                      # → exact match with reference (no tile-binning culling)
    ).to(device)

    # Add batch dim for the wrapper
    pts_b = centers.unsqueeze(0)
    means_b = means.detach().unsqueeze(0)
    # opas as (1, N) — after the wrapper's squeeze(0) → (N,) which is what the
    # kernel's backward returns. (1, N, 1) would force backward to return (N, 1).
    opas_b = opas.detach().unsqueeze(0)                      # (1, N)
    sem_b = sem.detach().unsqueeze(0)                        # (1, N, C)
    scales_b = scales.detach().unsqueeze(0)
    cov3D_b = Sinv33.detach().unsqueeze(0)                   # (1, N, 3, 3)

    with torch.no_grad():
        logits_k, bin_logits_k, density_k = aggregator(
            pts_b, means_b, opas_b, sem_b, scales_b, cov3D_b
        )

    # Reference (same math)
    with torch.no_grad():
        logits_r, bin_logits_r, density_r = g2v_forward_reference_gf(
            means.detach(), cov3D_inv_flat.detach(),
            opas.detach(), sem.detach(), centers,
        )

    print(f"  kernel  bin_logits range: [{bin_logits_k.min().item():.4f}, "
          f"{bin_logits_k.max().item():.4f}], mean {bin_logits_k.mean().item():.4f}")
    print(f"  ref     bin_logits range: [{bin_logits_r.min().item():.4f}, "
          f"{bin_logits_r.max().item():.4f}], mean {bin_logits_r.mean().item():.4f}")

    fwd_ok = True
    fwd_ok &= compare("logits",     logits_k,     logits_r,     rtol=rtol, atol=atol)
    fwd_ok &= compare("bin_logits", bin_logits_k, bin_logits_r, rtol=rtol, atol=atol)
    fwd_ok &= compare("density",    density_k,    density_r,    rtol=rtol, atol=atol)

    # ============================================================
    # BACKWARD correctness — compare grads w.r.t. (means, opas, sem, cov3D)
    # ============================================================
    print("\n[gf] backward correctness (grads w.r.t. means, opas, sem)")

    # Random upstream grads
    g = torch.Generator(device=device).manual_seed(seed + 7)
    grad_logits = torch.randn(V, num_classes, generator=g, device=device)
    grad_bin = torch.randn(V, generator=g, device=device)
    grad_density = torch.randn(V, generator=g, device=device)

    # ---- kernel backward (via autograd.Function) ----
    means_k = means.detach().clone().unsqueeze(0).requires_grad_(True)
    opas_k = opas.detach().clone().unsqueeze(0).requires_grad_(True)   # (1, N)
    sem_k = sem.detach().clone().unsqueeze(0).requires_grad_(True)
    cov3D_k = Sinv33.detach().clone().unsqueeze(0).requires_grad_(True)
    logits_k, bin_logits_k, density_k = aggregator(
        pts_b, means_k, opas_k, sem_k, scales_b, cov3D_k
    )
    loss_k = (logits_k * grad_logits).sum() + (bin_logits_k * grad_bin).sum() \
             + (density_k * grad_density).sum()
    loss_k.backward()

    # ---- reference backward ----
    means_r = means.detach().clone().requires_grad_(True)
    opas_r = opas.detach().clone().requires_grad_(True)
    sem_r = sem.detach().clone().requires_grad_(True)
    cov3D_inv_r = cov3D_inv_flat.detach().clone().requires_grad_(True)
    logits_r, bin_logits_r, density_r = g2v_forward_reference_gf(
        means_r, cov3D_inv_r, opas_r, sem_r, centers,
    )
    loss_r = (logits_r * grad_logits).sum() + (bin_logits_r * grad_bin).sum() \
             + (density_r * grad_density).sum()
    loss_r.backward()

    bwd_ok = True
    bwd_ok &= compare("d/dmeans",   means_k.grad.squeeze(0), means_r.grad, rtol=rtol, atol=atol)
    bwd_ok &= compare("d/dopas",    opas_k.grad.squeeze(0),
                                     opas_r.grad, rtol=rtol, atol=atol)
    bwd_ok &= compare("d/dsem",     sem_k.grad.squeeze(0),  sem_r.grad,   rtol=rtol, atol=atol)
    # Convert kernel's cov3D grad (1, N, 3, 3) → 6-flat to match reference
    cov3D_k_grad_flat = torch.stack([
        cov3D_k.grad[0, :, 0, 0], cov3D_k.grad[0, :, 1, 1], cov3D_k.grad[0, :, 2, 2],
        cov3D_k.grad[0, :, 0, 1], cov3D_k.grad[0, :, 1, 2], cov3D_k.grad[0, :, 0, 2],
    ], dim=-1)
    # Off-diagonal grads from a symmetric matrix are split between (i,j) and (j,i)
    # in the 3x3 form, so add the symmetric pair:
    cov3D_k_grad_flat_sym = torch.stack([
        cov3D_k.grad[0, :, 0, 0],
        cov3D_k.grad[0, :, 1, 1],
        cov3D_k.grad[0, :, 2, 2],
        cov3D_k.grad[0, :, 0, 1] + cov3D_k.grad[0, :, 1, 0],
        cov3D_k.grad[0, :, 1, 2] + cov3D_k.grad[0, :, 2, 1],
        cov3D_k.grad[0, :, 0, 2] + cov3D_k.grad[0, :, 2, 0],
    ], dim=-1)
    bwd_ok &= compare("d/dcov3D (sum-sym)", cov3D_k_grad_flat_sym, cov3D_inv_r.grad,
                      rtol=rtol, atol=atol)

    # ============================================================
    # Timing
    # ============================================================
    print("\n[gf] timing (warmed, 50 iters)")
    n_iters = 50
    # warmup
    for _ in range(5):
        with torch.no_grad():
            _ = aggregator(pts_b, means_b, opas_b, sem_b, scales_b, cov3D_b)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = aggregator(pts_b, means_b, opas_b, sem_b, scales_b, cov3D_b)
    torch.cuda.synchronize()
    fwd_kernel_ms = (time.time() - t0) * 1e3 / n_iters

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = g2v_forward_reference_gf(means, cov3D_inv_flat, opas, sem, centers)
    torch.cuda.synchronize()
    fwd_ref_ms = (time.time() - t0) * 1e3 / n_iters

    print(f"  kernel  forward: {fwd_kernel_ms:.3f} ms")
    print(f"  ref     forward: {fwd_ref_ms:.3f} ms")
    print(f"  speedup: {fwd_ref_ms / max(fwd_kernel_ms, 1e-6):.1f}x")

    if fwd_ok and bwd_ok:
        print("\n[gf] PASS")
        sys.exit(0)
    else:
        print("\n[gf] FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()

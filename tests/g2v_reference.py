"""Pure-PyTorch reference implementation of the S2GO G2V (Gaussian-to-Voxel) kernel.

Implements the same math as Kernel_pseudocode.md Algorithms 1-3, but as a single
dense, broadcast-style PyTorch function — slow, *O(V*N)* memory, but autograd
gives forward + backward for free. This is the ground truth that any CUDA
kernel implementation must match (within fp tolerance).

Math (S2GO §3.4.3):
    Σ_g  = R(r_g) · diag(s_g²) · R(r_g)ᵀ                       (Eq. 3)
    q_gx = -½ (x - μ_g)ᵀ Σ_g⁻¹ (x - μ_g)
    α_gx = a_g · exp(q_gx)              [skipped if q_gx > 0]
    Occ(x)   = 1 - ∏_g (1 - min(α_gx, 0.9999))                  (Eq. 1)
    w_gx = a_g · exp(q_gx) / √det(Σ_g)                          (Eq. 4)
    Class(x) = Σ_g w_gx · c_g  /  Σ_g w_gx
"""
from __future__ import annotations
import math
from typing import Tuple

import torch
from torch import Tensor


def quat_to_rotmat(q: Tensor) -> Tensor:
    """(N, 4) quaternion (w, x, y, z) → (N, 3, 3) rotation matrix.

    Matches the 3DGS convention (`computeCov3D` in cuda_rasterizer/forward.cu).
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(dim=-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def make_voxel_grid(grid_origin: Tensor, grid_dims: Tuple[int, int, int],
                    voxel_size: float) -> Tensor:
    """Voxel-center coordinates for an axis-aligned grid.

    Returns: (Dx, Dy, Dz, 3) tensor of voxel centers in world coords.
    """
    Dx, Dy, Dz = grid_dims
    device = grid_origin.device
    ix = torch.arange(Dx, device=device, dtype=torch.float32)
    iy = torch.arange(Dy, device=device, dtype=torch.float32)
    iz = torch.arange(Dz, device=device, dtype=torch.float32)
    gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing="ij")
    centers = torch.stack([gx, gy, gz], dim=-1) + 0.5  # cell-center offset
    centers = grid_origin + centers * voxel_size
    return centers  # (Dx, Dy, Dz, 3)


def g2v_forward_reference(
    means: Tensor,         # (N, 3)
    rotations: Tensor,     # (N, 4) quaternion (w, x, y, z)
    scales: Tensor,        # (N, 3) — interpreted as direct std-dev scales (no log)
    opacities: Tensor,     # (N,) or (N, 1) — interpreted as final alpha (post-sigmoid)
    colors: Tensor,        # (N, C)
    voxel_centers: Tensor, # (V, 3) flat list of voxel centers
    *,
    alpha_clip: float = 0.9999,
    eps_w: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """Reference forward — returns (occupancy [V], class [V, C]).

    Differentiable through autograd. *O(V*N)* memory; intended for small tests
    (e.g. V=512, N=128).
    """
    if opacities.dim() == 2:
        opacities = opacities.squeeze(-1)
    N = means.shape[0]
    V = voxel_centers.shape[0]
    C = colors.shape[-1]
    assert opacities.shape == (N,), f"opacities {opacities.shape}"
    assert colors.shape == (N, C)
    assert voxel_centers.shape == (V, 3)

    R = quat_to_rotmat(rotations)                       # (N, 3, 3)
    s_sq = scales ** 2                                  # (N, 3)
    # Σ = R · diag(s²) · Rᵀ   →   (R * s²[None, None, :]) @ R.T (broadcast on cols)
    Sigma = (R * s_sq.unsqueeze(-2)) @ R.transpose(-1, -2)        # (N, 3, 3)
    Sigma_inv = torch.linalg.inv(Sigma)                            # (N, 3, 3)
    det_Sigma = torch.linalg.det(Sigma).clamp_min(1e-20)           # (N,)

    # d = x - μ for every (voxel, gaussian) pair
    d = voxel_centers.unsqueeze(1) - means.unsqueeze(0)            # (V, N, 3)
    # q = -½ dᵀ Σ⁻¹ d
    q = -0.5 * torch.einsum('vni,nij,vnj->vn', d, Sigma_inv, d)   # (V, N)

    # ----- occupancy via Eq. 1 -----
    # α = a · exp(q), clipped to [0, alpha_clip]; q > 0 means out of support → α = 0
    alpha_raw = opacities.unsqueeze(0) * torch.exp(q.clamp(max=0.0))   # (V, N), q≤0 here
    in_support = (q <= 0)
    alpha = torch.where(in_support, alpha_raw, torch.zeros_like(alpha_raw))
    alpha = alpha.clamp(max=alpha_clip)
    log_one_minus_alpha = torch.log1p(-alpha)                       # numerically stable
    log_T = log_one_minus_alpha.sum(dim=1)                          # (V,)
    occ = 1.0 - torch.exp(log_T)                                    # (V,)

    # ----- class mixture via Eq. 4 -----
    # w_g(x) = a_g · exp(q_gx) / √det(Σ_g)
    inv_sqrt_det = 1.0 / torch.sqrt(det_Sigma)                      # (N,)
    w_full = (opacities * inv_sqrt_det).unsqueeze(0) * torch.exp(q.clamp(max=0.0))
    w = torch.where(in_support, w_full, torch.zeros_like(w_full))   # (V, N)
    W = w.sum(dim=1)                                                # (V,)
    E = (w.unsqueeze(-1) * colors.unsqueeze(0)).sum(dim=1)          # (V, C)
    cls = E / W.unsqueeze(-1).clamp_min(eps_w)                      # (V, C)

    return occ, cls


def g2v_forward_reference_gf(
    means: Tensor,         # (N, 3)
    cov3D_inv: Tensor,     # (N, 6) — packed Σ⁻¹: [xx, yy, zz, xy, yz, xz]
    opacities: Tensor,     # (N,) or (N, 1)
    colors: Tensor,        # (N, C)
    voxel_centers: Tensor, # (V, 3)
) -> Tuple[Tensor, Tensor, Tensor]:
    """Reference matching GaussianFormer/localagg_prob_fast/forward.cu exactly.

    Differences from the paper-spec reference:
      - cov3D_inv stores Σ⁻¹ directly in 6-flat form (xx, yy, zz, xy, yz, xz)
      - bin_logit = 1 - ∏(1 - exp(q))   (NO opacity in the product — Eq.1 with α=exp(q))
      - logits = Σ w·c / Σ w  with  w = (2π)^(-3/2) · √det(Σ⁻¹) · exp(q) · a
      - density = Σ exp(q)   (sum, not product; not used in our paper math)
      - no q > 0 culling (kernel uses tile binning instead, but the support drops
        to ~0 at >3σ anyway)

    Returns (logits [V, C], bin_logits [V], density [V]).
    """
    if opacities.dim() == 2:
        opacities = opacities.squeeze(-1)
    N = means.shape[0]
    V = voxel_centers.shape[0]
    C = colors.shape[-1]
    assert cov3D_inv.shape == (N, 6)

    # Unpack Σ⁻¹: indexing [xx, yy, zz, xy, yz, xz]
    xx, yy, zz = cov3D_inv[:, 0], cov3D_inv[:, 1], cov3D_inv[:, 2]
    xy, yz, xz = cov3D_inv[:, 3], cov3D_inv[:, 4], cov3D_inv[:, 5]

    # d = μ - x (note kernel uses (μ - x); quadratic form is symmetric in sign)
    d = means.unsqueeze(0) - voxel_centers.unsqueeze(1)              # (V, N, 3)
    dx, dy, dz = d.unbind(-1)                                         # (V, N) each

    # power = exp(-½ dᵀ Σ⁻¹ d), expanded:
    #   diag part: dx²·xx + dy²·yy + dz²·zz
    #   off-diag (×2 from symmetry): 2·(dx·dy·xy + dy·dz·yz + dx·dz·xz)
    quad = (dx * dx * xx + dy * dy * yy + dz * dz * zz
            + 2.0 * (dx * dy * xy + dy * dz * yz + dx * dz * xz))
    power = torch.exp(-0.5 * quad)                                    # (V, N)

    # det(Σ⁻¹) from the 6-flat form
    deter = (xx * yy * zz + 2.0 * xy * yz * xz
             - xx * yz * yz - yy * xz * xz - zz * xy * xy)            # (N,)

    const_2pi = (2.0 * math.pi) ** -1.5
    prob = const_2pi * torch.sqrt(deter.clamp_min(1e-30)).unsqueeze(0) * power * opacities.unsqueeze(0)  # (V, N)

    # Mixture (kernel: out_logits[ch] = C[ch] / prob_sum, fall-back uniform)
    C_ch = (prob.unsqueeze(-1) * colors.unsqueeze(0)).sum(dim=1)      # (V, C)
    prob_sum = prob.sum(dim=1)                                        # (V,)
    safe = prob_sum > 1e-9
    uniform = torch.full_like(C_ch, 1.0 / max(C - 1, 1))
    # NB the kernel only writes channels [0..C-2] in the fallback (off-by-one bug?
    # see line 96-97 of forward.cu). Replicating exactly here would be unstable
    # for grads; we use a uniform-1/(C-1) fallback in *all* channels, which is
    # what the kernel does in practice when prob_sum>0 ~ always for our tests.
    logits = torch.where(safe.unsqueeze(-1),
                         C_ch / prob_sum.unsqueeze(-1).clamp_min(1e-30),
                         uniform)

    # bin_logit cumulative product, then 1 - bin_logit
    log_one_minus_power = torch.log1p(-power.clamp(max=1.0 - 1e-7))   # (V, N)
    bin_logit_prod = torch.exp(log_one_minus_power.sum(dim=1))        # (V,)
    bin_logits = 1.0 - bin_logit_prod                                  # (V,) — = occupancy

    density = power.sum(dim=1)                                        # (V,)

    return logits, bin_logits, density


def g2v_forward_reference_grid(
    means, rotations, scales, opacities, colors,
    grid_origin: Tensor, grid_dims: Tuple[int, int, int], voxel_size: float,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """Convenience wrapper: takes grid params, returns (occ, cls) shaped to grid."""
    Dx, Dy, Dz = grid_dims
    centers = make_voxel_grid(grid_origin, grid_dims, voxel_size).reshape(-1, 3)
    occ, cls = g2v_forward_reference(
        means, rotations, scales, opacities, colors, centers, **kwargs
    )
    return occ.reshape(Dx, Dy, Dz), cls.reshape(Dx, Dy, Dz, -1)

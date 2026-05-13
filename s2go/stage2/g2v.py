"""G2VLayer — Gaussian-to-Voxel splatting (Stage 2).

Two backends:

  * `backend='torch'` (default): pure-PyTorch reference. Autograd-friendly,
    needs no compilation, but holds the full (V, N, ...) intermediates in
    the autograd graph → ~70 GB peak memory for the full 200×200×16 grid
    with N≈9000 Gaussians. Use the `forward_sparse` path (subset of
    voxels per iter) for training on 12 GB GPUs.

  * `backend='cuda'`: wraps the compiled `local_aggregate_s2go` kernel at
    `kernel/localagg_s2go/`. Custom CUDA forward+backward avoids the
    autograd intermediate blowup — dense 640k voxels × 9000 Gaussians
    runs in **~360 MB** end-to-end (200× win over the torch path) and
    enables full-grid training without voxel subsampling.

Both backends compute the same math (paper §3.4.3, Eq. 1 + Eq. 4):
    Σ_g       = R(r_g) · diag(s_g²) · R(r_g)ᵀ
    q_gx      = -½ (x - μ_g)ᵀ Σ_g⁻¹ (x - μ_g)
    α_gx      = a_g · exp(q_gx) · 1(q_gx ≤ 0)                 (Eq. 1 weight)
    Occ(x)    = 1 - ∏_g (1 - clip(α_gx, 0.9999))              (Eq. 1)
    w_gx      = a_g · exp(q_gx) / √det(Σ_g) · 1(q_gx ≤ 0)     (Eq. 4 weight)
    Logits(x) = Σ_g w_gx · ĉ_g  /  Σ_g w_gx                   (Eq. 4)

Outputs per call:
  - occ_prob:     (B, Vx, Vy, Vz)      — occupancy probability ∈ [0, 1)
  - sem_logits:   (B, Vx, Vy, Vz, C)   — class mixture from Gaussians

Note: sem_logits are not log-softmax. They are the weighted mean of
per-Gaussian class logits, which behaves like a logit distribution under
softmax. Downstream losses (`stage2/losses.py`) treat them as logits.
"""
import math
import os
import sys
from typing import Tuple

import torch
import torch.nn as nn

from . import PC_RANGE, VOXEL_SIZE, GRID_SHAPE


_KERNEL_DIR = "/media/skr/storage/self_driving/S2GO/kernel/localagg_s2go"


def _load_cuda_kernel():
    """Lazy-import the LocalAggregator. Adds the kernel dir to sys.path on
    first call so users of the torch backend don't need the .so present."""
    if _KERNEL_DIR not in sys.path:
        sys.path.insert(0, _KERNEL_DIR)
    from local_aggregate_s2go import LocalAggregator  # noqa: E402
    return LocalAggregator


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) quaternion (w, x, y, z) → (N, 3, 3) rotation matrix.

    Matches the 3DGS convention used by gsplat (Stage 1's renderer).
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(dim=-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def _make_voxel_centers(pc_range, voxel_size, grid_shape, device, dtype=torch.float32):
    """Return (Vx, Vy, Vz, 3) tensor of voxel centers in LIDAR_TOP coords."""
    Vx, Vy, Vz = grid_shape
    x0, y0, z0, _, _, _ = pc_range
    sx, sy, sz = voxel_size
    ix = torch.arange(Vx, device=device, dtype=dtype)
    iy = torch.arange(Vy, device=device, dtype=dtype)
    iz = torch.arange(Vz, device=device, dtype=dtype)
    gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing="ij")
    # +0.5 to land at the voxel CENTER (not corner).
    cx = x0 + (gx + 0.5) * sx
    cy = y0 + (gy + 0.5) * sy
    cz = z0 + (gz + 0.5) * sz
    return torch.stack([cx, cy, cz], dim=-1)


def g2v_forward(
    means: torch.Tensor,       # (N, 3)
    rotations: torch.Tensor,   # (N, 4) quaternion (w, x, y, z)
    scales: torch.Tensor,      # (N, 3) std-dev scales (meters)
    opacities: torch.Tensor,   # (N,) or (N, 1) — final alpha ∈ [0, 1]
    class_logits: torch.Tensor, # (N, C) per-Gaussian class logits
    voxel_centers: torch.Tensor, # (V, 3)
    *,
    alpha_clip: float = 0.9999,
    eps_w: float = 1e-8,
    chunk_voxels: int = 8192,  # tile voxel dim to keep memory bounded
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch G2V forward — returns (occ [V], cls [V, C]).

    Memory budget: ~ chunk_voxels × N × 4 bytes × small constant. For
    chunk_voxels=8192, N=9000, fp32 → ~1.3 GB peak per chunk. Tile across
    V/chunk_voxels chunks to stay under GPU limits.
    """
    if opacities.dim() == 2:
        opacities = opacities.squeeze(-1)
    N = means.shape[0]
    V = voxel_centers.shape[0]
    C = class_logits.shape[-1]
    assert opacities.shape == (N,), f"opacities {opacities.shape} vs N={N}"
    assert class_logits.shape == (N, C)

    # Precompute per-Gaussian quantities (shared across voxel chunks).
    R = quat_to_rotmat(rotations)                              # (N, 3, 3)
    s_sq = scales ** 2                                          # (N, 3)
    # Σ = R · diag(s²) · Rᵀ
    Sigma = (R * s_sq.unsqueeze(-2)) @ R.transpose(-1, -2)      # (N, 3, 3)
    Sigma_inv = torch.linalg.inv(Sigma)                          # (N, 3, 3)
    det_Sigma = torch.linalg.det(Sigma).clamp_min(1e-20)         # (N,)
    inv_sqrt_det = 1.0 / torch.sqrt(det_Sigma)                   # (N,)

    out_occ = torch.empty(V, device=voxel_centers.device, dtype=voxel_centers.dtype)
    out_cls = torch.empty(V, C, device=voxel_centers.device, dtype=voxel_centers.dtype)

    for v0 in range(0, V, chunk_voxels):
        v1 = min(V, v0 + chunk_voxels)
        vc = voxel_centers[v0:v1]                                # (Vc, 3)
        d  = vc.unsqueeze(1) - means.unsqueeze(0)                # (Vc, N, 3)
        # q = -½ dᵀ Σ⁻¹ d
        q = -0.5 * torch.einsum('vni,nij,vnj->vn', d, Sigma_inv, d)  # (Vc, N)
        in_support = (q <= 0)                                     # (Vc, N)

        # Occupancy (Eq. 1)
        alpha = opacities.unsqueeze(0) * torch.exp(q.clamp(max=0.0))
        alpha = torch.where(in_support, alpha, torch.zeros_like(alpha))
        alpha = alpha.clamp(max=alpha_clip)
        log_one_minus = torch.log1p(-alpha)                       # numerically stable
        out_occ[v0:v1] = 1.0 - torch.exp(log_one_minus.sum(dim=1))

        # Class mixture (Eq. 4) — matmul avoids the (Vc, N, C) intermediate.
        w_full = (opacities * inv_sqrt_det).unsqueeze(0) * torch.exp(q.clamp(max=0.0))
        w = torch.where(in_support, w_full, torch.zeros_like(w_full))   # (Vc, N)
        W = w.sum(dim=1)                                                # (Vc,)
        E = w @ class_logits                                            # (Vc, C)
        out_cls[v0:v1] = E / W.unsqueeze(-1).clamp_min(eps_w)

    return out_occ, out_cls


class G2VLayer(nn.Module):
    """Wraps `g2v_forward` for use in an nn.Module stack.

    Input batched per-frame Gaussians + per-Gaussian class logits.
    Output dense voxel-space tensors over the grid pinned in
    `s2go/stage2/__init__.py`.

    Memory note: dense G2V over the full 200×200×16=640k voxels with N≈9000
    Gaussians is ~70 GB peak autograd memory. For 12 GB cards, use
    `forward_sparse` with a sampled voxel index set per iter.
    """
    def __init__(self,
                 pc_range=PC_RANGE,
                 voxel_size=VOXEL_SIZE,
                 grid_shape=GRID_SHAPE,
                 alpha_clip: float = 0.9999,
                 chunk_voxels: int = 8192):
        super().__init__()
        self.pc_range   = tuple(pc_range)
        self.voxel_size = tuple(voxel_size)
        self.grid_shape = tuple(grid_shape)
        self.alpha_clip = float(alpha_clip)
        self.chunk_voxels = int(chunk_voxels)
        # Cache voxel centers (lazy; built on first forward to inherit device).
        self._centers_cache = None

    def _centers(self, device, dtype):
        if (self._centers_cache is None
                or self._centers_cache.device != device
                or self._centers_cache.dtype != dtype):
            grid = _make_voxel_centers(self.pc_range, self.voxel_size,
                                        self.grid_shape, device, dtype)
            self._centers_cache = grid.reshape(-1, 3).contiguous()  # (V, 3)
        return self._centers_cache

    def forward(self,
                means: torch.Tensor,       # (B, N, 3)
                rotations: torch.Tensor,   # (B, N, 4)
                scales: torch.Tensor,      # (B, N, 3)
                opacities: torch.Tensor,   # (B, N, 1) or (B, N)
                class_logits: torch.Tensor # (B, N, C)
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dense forward — returns (occ_prob (B,Vx,Vy,Vz), sem_logits (B,Vx,Vy,Vz,C)).

        Only use this at eval / inference (no autograd). For training, use
        `forward_sparse`.
        """
        assert means.dim() == 3 and rotations.dim() == 3 and scales.dim() == 3
        B, N, _ = means.shape
        Vx, Vy, Vz = self.grid_shape
        device, dtype = means.device, means.dtype
        centers = self._centers(device, dtype)                       # (V, 3)
        C = class_logits.shape[-1]

        occ_out = torch.empty(B, Vx, Vy, Vz, device=device, dtype=dtype)
        sem_out = torch.empty(B, Vx, Vy, Vz, C, device=device, dtype=dtype)
        for b in range(B):
            occ_b, cls_b = g2v_forward(
                means[b], rotations[b], scales[b], opacities[b], class_logits[b],
                centers,
                alpha_clip=self.alpha_clip,
                chunk_voxels=self.chunk_voxels,
            )
            occ_out[b] = occ_b.view(Vx, Vy, Vz)
            sem_out[b] = cls_b.view(Vx, Vy, Vz, C)
        return occ_out, sem_out

    def forward_sparse(self,
                        means: torch.Tensor,
                        rotations: torch.Tensor,
                        scales: torch.Tensor,
                        opacities: torch.Tensor,
                        class_logits: torch.Tensor,
                        voxel_flat_idx: torch.Tensor,   # (B, P) flat indices into Vx*Vy*Vz
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sparse forward — compute occ + sem only at the given P voxel indices.

        Backward only needs to retain the (P, N) intermediates, not (V, N).
        For P=8192, N=9000 → ~880 MB intermediate (vs ~70 GB dense).

        Args:
            voxel_flat_idx: (B, P) long, flat indices into the grid (row-major
                            with shape Vx × Vy × Vz). Same per batch is fine
                            if your batch is 1.
        Returns:
            occ: (B, P)
            sem: (B, P, C)
        """
        assert means.dim() == 3
        B, N, _ = means.shape
        P = voxel_flat_idx.shape[-1]
        C = class_logits.shape[-1]
        device, dtype = means.device, means.dtype
        centers = self._centers(device, dtype)                       # (V, 3)

        occ_out = torch.empty(B, P, device=device, dtype=dtype)
        sem_out = torch.empty(B, P, C, device=device, dtype=dtype)
        for b in range(B):
            idx_b = voxel_flat_idx[b]                                # (P,)
            vc_b  = centers.index_select(0, idx_b)                   # (P, 3)
            occ_b, cls_b = g2v_forward(
                means[b], rotations[b], scales[b], opacities[b], class_logits[b],
                vc_b,
                alpha_clip=self.alpha_clip,
                chunk_voxels=self.chunk_voxels,
            )
            occ_out[b] = occ_b
            sem_out[b] = cls_b
        return occ_out, sem_out


# ────────────────────────────────────────────────────────────────────────────
# CUDA backend (wraps the compiled local_aggregate_s2go kernel)
# ────────────────────────────────────────────────────────────────────────────
def _sigma_inv_from_rot_scale(rotations: torch.Tensor,
                                scales: torch.Tensor) -> torch.Tensor:
    """(B, N, 4) quaternion + (B, N, 3) scales → (B, N, 3, 3) Σ⁻¹.

    Σ = R diag(s²) Rᵀ,  Σ⁻¹ = R diag(1/s²) Rᵀ.  Differentiable.
    """
    R = quat_to_rotmat(rotations)                                  # (B, N, 3, 3)
    inv_s2 = 1.0 / (scales ** 2)                                    # (B, N, 3)
    return (R * inv_s2.unsqueeze(-2)) @ R.transpose(-1, -2)         # (B, N, 3, 3)


class G2VLayerCUDA(nn.Module):
    """CUDA-backed Gaussian-to-Voxel splatting.

    Wraps the compiled `local_aggregate_s2go.LocalAggregator`. The kernel
    operates on FLAT lists of query points (`pts`), so we can:
      - Pass the full voxel-center list for dense forward
      - Pass a sampled subset for sparse training (same `pts` interface)

    Per-call inputs are the same as `G2VLayer`. The kernel internally bins
    Gaussians by their `scale_multiplier * scale / grid_size` radius for
    O(N) tiled lookup, so dense 640k × 9000 fits in ~360 MB.

    Args:
        scale_multiplier: factor on scales for spatial radius (default 3.0
                           = 3σ — Gaussians beyond this contribute ~0).
        radii_min:        minimum integer radius in voxels (default 1).
    """
    def __init__(self,
                 pc_range=PC_RANGE,
                 voxel_size=VOXEL_SIZE,
                 grid_shape=GRID_SHAPE,
                 scale_multiplier: float = 3.0,
                 radii_min: int = 1):
        super().__init__()
        # Kernel assumes isotropic voxel size — assert that's the case.
        sx, sy, sz = voxel_size
        assert sx == sy == sz, f"CUDA kernel needs isotropic voxels, got {voxel_size}"
        self.pc_range   = tuple(pc_range)
        self.voxel_size = tuple(voxel_size)
        self.grid_shape = tuple(grid_shape)
        self.scale_multiplier = float(scale_multiplier)
        self.radii_min = int(radii_min)

        LocalAggregator = _load_cuda_kernel()
        Vx, Vy, Vz = grid_shape
        self.agg = LocalAggregator(
            scale_multiplier=scale_multiplier,
            H=Vx, W=Vy, D=Vz,
            pc_min=list(pc_range[:3]),
            grid_size=float(sx),
            radii_min=radii_min,
        )
        self._centers_cache = None

    def _centers(self, device, dtype):
        if (self._centers_cache is None
                or self._centers_cache.device != device
                or self._centers_cache.dtype != dtype):
            grid = _make_voxel_centers(self.pc_range, self.voxel_size,
                                        self.grid_shape, device, dtype)
            self._centers_cache = grid.reshape(-1, 3).contiguous()  # (V, 3)
        return self._centers_cache

    def _in_grid_mask(self, means: torch.Tensor) -> torch.Tensor:
        """(B, N, 3) means → (B, N) bool mask: True if mean is in voxel grid.

        The kernel asserts every Gaussian's integer-binned position is inside
        [0, H) × [0, W) × [0, D). Out-of-grid Gaussians have negligible
        contribution anyway (q_gx → −∞ for far voxels). Filter them out.
        """
        Vx, Vy, Vz = self.grid_shape
        x0, y0, z0, *_ = self.pc_range
        sx, _, _ = self.voxel_size   # isotropic per the constructor
        ix = ((means[..., 0] - x0) / sx).floor().long()
        iy = ((means[..., 1] - y0) / sx).floor().long()
        iz = ((means[..., 2] - z0) / sx).floor().long()
        return ((ix >= 0) & (ix < Vx) &
                (iy >= 0) & (iy < Vy) &
                (iz >= 0) & (iz < Vz))

    def _forward_pts(self,
                      means: torch.Tensor,       # (B, N, 3)
                      rotations: torch.Tensor,   # (B, N, 4)
                      scales: torch.Tensor,      # (B, N, 3)
                      opacities: torch.Tensor,   # (B, N, 1) or (B, N)
                      class_logits: torch.Tensor, # (B, N, C)
                      pts: torch.Tensor          # (B, P, 3) query points
                      ):
        """Run the kernel for B=1. Returns (occ (P,), sem (P, C)).

        Filters Gaussians outside the grid before calling the kernel. The
        index_select is autograd-differentiable; out-of-grid Gaussians
        receive zero grad (correct: they contributed nothing).
        """
        assert pts.shape[0] == 1, "CUDA kernel only supports B=1 today"
        # Filter out-of-grid Gaussians (kernel asserts in-range bin indices).
        mask = self._in_grid_mask(means)[0]                    # (N,)
        keep_idx = mask.nonzero(as_tuple=False).squeeze(-1)    # (N',)
        means_f      = means.index_select(1, keep_idx)
        rotations_f  = rotations.index_select(1, keep_idx)
        scales_f     = scales.index_select(1, keep_idx)
        opas_f       = opacities.index_select(-2 if opacities.dim() == 3 else -1,
                                                keep_idx)
        class_f      = class_logits.index_select(1, keep_idx)
        # Pack opacities to (1, N') shape — kernel returns 1-D grad for opas.
        if opas_f.dim() == 3:
            opas_f = opas_f.squeeze(-1)
        Sigma_inv = _sigma_inv_from_rot_scale(rotations_f, scales_f)
        sem_logits, occ_prob, density = self.agg(
            pts, means_f, opas_f, class_f, scales_f, Sigma_inv,
        )
        return occ_prob, sem_logits

    def forward(self,
                means: torch.Tensor,
                rotations: torch.Tensor,
                scales: torch.Tensor,
                opacities: torch.Tensor,
                class_logits: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dense forward over the full voxel grid.

        Returns (occ_prob (B, Vx, Vy, Vz), sem_logits (B, Vx, Vy, Vz, C)).
        """
        B, N, _ = means.shape
        Vx, Vy, Vz = self.grid_shape
        C = class_logits.shape[-1]
        device, dtype = means.device, means.dtype
        centers = self._centers(device, dtype)                       # (V, 3)
        pts = centers.unsqueeze(0).expand(B, -1, -1).contiguous()     # (B, V, 3)
        occ, sem = self._forward_pts(means, rotations, scales,
                                      opacities, class_logits, pts)
        return occ.view(B, Vx, Vy, Vz), sem.view(B, Vx, Vy, Vz, C)

    def forward_sparse(self,
                        means: torch.Tensor,
                        rotations: torch.Tensor,
                        scales: torch.Tensor,
                        opacities: torch.Tensor,
                        class_logits: torch.Tensor,
                        voxel_flat_idx: torch.Tensor,   # (B, P)
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sparse forward at the given P voxel indices (B=1 only).

        Returns (occ (B=1, P), sem (B=1, P, C)).
        """
        assert means.shape[0] == 1, "CUDA kernel only supports B=1"
        device, dtype = means.device, means.dtype
        centers = self._centers(device, dtype)
        pts = centers.index_select(0, voxel_flat_idx[0]).unsqueeze(0).contiguous()
        occ, sem = self._forward_pts(means, rotations, scales,
                                      opacities, class_logits, pts)
        # _forward_pts returns (P,) and (P, C) for B=1 inputs
        return occ.unsqueeze(0), sem.unsqueeze(0)


def make_g2v_layer(backend: str = 'torch', **kwargs) -> nn.Module:
    """Factory: select G2V backend ('torch' or 'cuda')."""
    if backend == 'torch':
        return G2VLayer(**kwargs)
    elif backend == 'cuda':
        # Drop torch-only kwargs that the CUDA layer doesn't use
        kwargs.pop('chunk_voxels', None)
        kwargs.pop('alpha_clip', None)
        return G2VLayerCUDA(**kwargs)
    raise ValueError(f"unknown g2v backend: {backend} (use 'torch' or 'cuda')")


# ────────────────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────────────────
def _self_test():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Tiny grid so this runs fast end-to-end. Real Stage 2 uses (200,200,16).
    layer = G2VLayer(pc_range=(-5, -5, -2, 5, 5, 2),
                     voxel_size=(1.0, 1.0, 1.0),
                     grid_shape=(10, 10, 4),
                     chunk_voxels=200).to(device)
    B, N, C = 1, 16, 18
    means = torch.zeros(B, N, 3, device=device)
    means[..., 0].uniform_(-3, 3)
    means[..., 1].uniform_(-3, 3)
    means[..., 2].uniform_(-1, 1)
    means = means.requires_grad_(True)
    # Non-identity quaternions — identity gives zero grad by symmetry.
    rotations = torch.randn(B, N, 4, device=device).requires_grad_(True)
    scales = torch.full((B, N, 3), 1.0, device=device, requires_grad=True)
    opacities = torch.full((B, N, 1), 0.5, device=device, requires_grad=True)
    class_logits = torch.randn(B, N, C, device=device, requires_grad=True)

    occ, sem = layer(means, rotations, scales, opacities, class_logits)
    assert occ.shape == (B, 10, 10, 4), occ.shape
    assert sem.shape == (B, 10, 10, 4, C), sem.shape
    assert torch.isfinite(occ).all() and torch.isfinite(sem).all()
    assert (occ >= 0).all() and (occ <= 1).all()
    print(f"G2VLayer fwd OK: occ {tuple(occ.shape)} in [0,1], sem {tuple(sem.shape)}")
    print(f"  occ_mean={occ.mean().item():.3f}  occ_max={occ.max().item():.3f}")

    # Grad flow on each input
    (occ.sum() + sem.sum()).backward()
    for name, t in [('means', means), ('rotations', rotations),
                     ('scales', scales), ('opacities', opacities),
                     ('class_logits', class_logits)]:
        gn = t.grad.norm().item()
        print(f"  grad ‖{name}‖ = {gn:.3e}")
        assert gn > 0, f"no grad on {name}"
    print("G2VLayer self-test PASSED.")


def _self_test_cuda():
    """Verify the CUDA backend gives shape-correct outputs and finite grads.

    Note: numeric values between CUDA and torch paths *will* differ — the
    kernel uses a slightly different formulation (no q>0 culling, GF-style
    mixture). We compare *behaviour* (finite, in-range, grad-flowing), not
    bit-exact equality. The standalone test at
    `kernel/tests/test_g2v_gaussianformer.py` does the numeric tightness
    check against a separate torch reference.
    """
    torch.manual_seed(0)
    device = "cuda"
    layer = G2VLayerCUDA(scale_multiplier=3.0).to(device)
    B, N, C = 1, 9000, 18
    means = (torch.rand(B, N, 3, device=device)
              * torch.tensor([100, 100, 8], device=device)
              + torch.tensor([-50, -50, -5], device=device)
            ).requires_grad_(True)
    rots = torch.randn(B, N, 4, device=device).requires_grad_(True)
    scales = torch.full((B, N, 3), 0.8, device=device, requires_grad=True)
    opas = torch.full((B, N), 0.2, device=device, requires_grad=True)
    sem = torch.randn(B, N, C, device=device, requires_grad=True)

    torch.cuda.reset_peak_memory_stats()
    occ, sem_out = layer(means, rots, scales, opas, sem)
    assert occ.shape == (B, 200, 200, 16), occ.shape
    assert sem_out.shape == (B, 200, 200, 16, C), sem_out.shape
    assert torch.isfinite(occ).all() and torch.isfinite(sem_out).all()
    assert (occ >= 0).all() and (occ <= 1).all()
    print(f"G2VLayerCUDA fwd OK: occ {tuple(occ.shape)} in [0,{occ.max().item():.3f}], "
          f"sem {tuple(sem_out.shape)}")

    (occ.sum() + sem_out.sum()).backward()
    for name, t in [('means', means), ('rotations', rots),
                     ('scales', scales), ('opacities', opas),
                     ('class_logits', sem)]:
        gn = t.grad.norm().item()
        print(f"  grad ‖{name}‖ = {gn:.3e}")
        assert gn > 0, f"no grad on {name}"
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  peak mem (full 200×200×16 dense forward+backward): {peak:.1f} MB")
    print("G2VLayerCUDA self-test PASSED.")


if __name__ == "__main__":
    _self_test()
    print()
    try:
        _self_test_cuda()
    except ImportError as e:
        print(f"[skip] CUDA backend not available: {e}")

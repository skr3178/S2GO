"""S2GO G2V kernel test scaffold.

Methodology mirrors the 3DGS sanity test:
    1. synthesize a small scene of random 3D Gaussians
    2. set up a small voxel grid that the Gaussians overlap
    3. run forward, check shapes / no NaN / value ranges
    4. run backward (autograd), check finite grads
    5. compare a candidate kernel to the PyTorch reference (allclose on outputs
       and on grads w.r.t. all inputs)

The CUDA kernel doesn't exist yet, so by default this script tests the
reference against itself (a no-op consistency check that validates the
scaffold). Wire a real kernel by passing it as `--kernel <import.path>` once
implemented; it must expose

    g2v_forward(means, rotations, scales, opacities, colors, voxel_centers)
        -> (occupancy: [V], class: [V, C])

and be differentiable via autograd (e.g. wrapped as a torch.autograd.Function).

Usage:
    python test_g2v_scaffold.py                       # ref vs ref
    python test_g2v_scaffold.py --kernel mypkg.g2v    # ref vs CUDA kernel
"""
from __future__ import annotations
import argparse
import importlib
import math
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent))
from g2v_reference import (
    g2v_forward_reference,
    make_voxel_grid,
    quat_to_rotmat,
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def synth_gaussians(num: int, grid_origin: Tensor, grid_extent: float,
                    *, seed: int = 0, device: str = "cuda"):
    """Random Gaussians scattered inside the voxel-grid AABB.

    Scales are sized so that ~3σ fits inside ~1/8 of the grid extent — gives a
    healthy mix of voxels each Gaussian touches without any single Gaussian
    covering the whole grid (which would make the test trivial).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    means = grid_origin + torch.rand(num, 3, generator=g, device=device) * grid_extent
    scales = torch.exp(
        torch.randn(num, 3, generator=g, device=device) * 0.3
    ) * (grid_extent / 16.0)                                         # ~ extent/16
    rots = torch.randn(num, 4, generator=g, device=device)
    rots = rots / rots.norm(dim=-1, keepdim=True)
    opacities = torch.sigmoid(
        torch.randn(num, generator=g, device=device) * 0.5 + 0.0
    )                                                                 # ~ U around 0.5
    return means, rots, scales, opacities


def synth_colors(num: int, num_classes: int, *, seed: int = 1, device: str = "cuda"):
    """One-hot-ish softmax over class logits, so colors are valid class probabilities."""
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(num, num_classes, generator=g, device=device)
    return torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------

def load_kernel(import_path: str):
    """Resolve `pkg.module:fn_name` (or `pkg.module.fn_name`) to a callable."""
    if ":" in import_path:
        modname, attr = import_path.split(":", 1)
    else:
        modname, attr = import_path.rsplit(".", 1)
    mod = importlib.import_module(modname)
    return getattr(mod, attr)


def run_forward_backward(fn, means, rotations, scales, opacities, colors, voxel_centers,
                         grad_out_occ, grad_out_cls):
    """Run forward + a fake-loss backward; return outputs and grads."""
    inputs = {
        "means": means, "rotations": rotations, "scales": scales,
        "opacities": opacities, "colors": colors,
    }
    for v in inputs.values():
        v.requires_grad_(True)
        if v.grad is not None:
            v.grad = None

    occ, cls = fn(means, rotations, scales, opacities, colors, voxel_centers)
    loss = (occ * grad_out_occ).sum() + (cls * grad_out_cls).sum()
    loss.backward()
    grads = {k: v.grad.detach().clone() for k, v in inputs.items()}
    return occ.detach().clone(), cls.detach().clone(), grads, loss.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=str, default=None,
                        help="Import path of candidate kernel callable (e.g. "
                             "'s2go.g2v:forward'). If omitted, uses the reference "
                             "as a self-consistency test.")
    parser.add_argument("--num-gaussians", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--grid-dim", type=int, default=8)  # 8x8x8 = 512 voxels
    parser.add_argument("--voxel-size", type=float, default=0.25)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--double", action="store_true",
                        help="Run in float64 (tighter tolerances, useful for ref-vs-ref).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64 if args.double else torch.float32

    print(f"[g2v] device={device} dtype={dtype} "
          f"N={args.num_gaussians} C={args.num_classes} "
          f"grid={args.grid_dim}^3 voxel={args.voxel_size}")

    # --- voxel grid centered on origin ---
    grid_dims = (args.grid_dim, args.grid_dim, args.grid_dim)
    grid_extent = args.grid_dim * args.voxel_size
    grid_origin = torch.tensor([-grid_extent / 2] * 3, device=device, dtype=dtype)
    voxel_centers = make_voxel_grid(grid_origin, grid_dims, args.voxel_size)
    voxel_centers = voxel_centers.reshape(-1, 3).to(dtype)
    V = voxel_centers.shape[0]

    # --- random Gaussians inside the grid ---
    means, rots, scales, opacities = synth_gaussians(
        args.num_gaussians, grid_origin, grid_extent, seed=args.seed, device=device
    )
    means = means.to(dtype)
    rots = rots.to(dtype)
    scales = scales.to(dtype)
    opacities = opacities.to(dtype)
    colors = synth_colors(
        args.num_gaussians, args.num_classes, seed=args.seed + 1, device=device
    ).to(dtype)

    # --- random upstream gradients to broadcast to a scalar loss ---
    g = torch.Generator(device=device).manual_seed(args.seed + 2)
    grad_occ = torch.randn(V, generator=g, device=device, dtype=dtype)
    grad_cls = torch.randn(V, args.num_classes, generator=g, device=device, dtype=dtype)

    # --- run reference ---
    print("[g2v] running reference forward+backward...")
    t0 = time.time()
    occ_ref, cls_ref, grads_ref, loss_ref = run_forward_backward(
        g2v_forward_reference, means.clone(), rots.clone(), scales.clone(),
        opacities.clone(), colors.clone(), voxel_centers, grad_occ, grad_cls,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    ref_ms = (time.time() - t0) * 1e3
    print(f"[g2v] reference: {ref_ms:.1f} ms, loss={loss_ref:.6f}")

    # ---- sanity on the reference output itself ----
    assert torch.isfinite(occ_ref).all(), "ref occ has NaN/Inf"
    assert torch.isfinite(cls_ref).all(), "ref cls has NaN/Inf"
    assert (occ_ref >= 0).all() and (occ_ref <= 1.0001).all(), \
        f"ref occ out of [0,1]: range=[{occ_ref.min().item():.4f}, {occ_ref.max().item():.4f}]"
    cls_sum = cls_ref.sum(dim=-1)
    # cls is a w-weighted convex combination of softmax-normalized colors → also sums to ~1
    assert (cls_sum > 0.99).all() and (cls_sum < 1.01).all(), \
        f"ref cls per-voxel sum out of range: {cls_sum.min().item():.4f}..{cls_sum.max().item():.4f}"
    print(f"[g2v]   occ range:        [{occ_ref.min():.4f}, {occ_ref.max():.4f}]")
    print(f"[g2v]   occ mean:         {occ_ref.mean():.4f} "
          f"(fraction-occupied)")
    print(f"[g2v]   cls per-voxel sum:[{cls_sum.min():.4f}, {cls_sum.max():.4f}]")
    for k, v in grads_ref.items():
        assert torch.isfinite(v).all(), f"ref grad {k} has NaN/Inf"
        print(f"[g2v]   d/d{k:<10} norm={v.norm().item():.4e}")

    # ---- compare candidate kernel to reference ----
    if args.kernel is None:
        print("[g2v] no --kernel provided; running reference-vs-reference "
              "self-consistency check (validates scaffold)")
        kernel_fn = g2v_forward_reference
        label = "reference"
    else:
        print(f"[g2v] loading candidate kernel: {args.kernel}")
        kernel_fn = load_kernel(args.kernel)
        label = args.kernel

    print(f"[g2v] running candidate ({label}) forward+backward...")
    t0 = time.time()
    occ_can, cls_can, grads_can, loss_can = run_forward_backward(
        kernel_fn, means.clone(), rots.clone(), scales.clone(),
        opacities.clone(), colors.clone(), voxel_centers, grad_occ, grad_cls,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    can_ms = (time.time() - t0) * 1e3
    print(f"[g2v] candidate: {can_ms:.1f} ms, loss={loss_can:.6f}")

    # --- compare ---
    def cmp(name, a, b):
        diff = (a - b).abs()
        rel = diff / b.abs().clamp_min(1e-12)
        ok = torch.allclose(a, b, rtol=args.rtol, atol=args.atol)
        print(f"[g2v]   {name:<20} max_abs={diff.max().item():.3e} "
              f"max_rel={rel.max().item():.3e}  {'OK' if ok else 'FAIL'}")
        return ok

    print("[g2v] comparing forward outputs:")
    ok = True
    ok &= cmp("occ", occ_can, occ_ref)
    ok &= cmp("cls", cls_can, cls_ref)
    print("[g2v] comparing input gradients:")
    for k in grads_ref:
        ok &= cmp(f"d/d{k}", grads_can[k], grads_ref[k])

    print(f"[g2v] reference / candidate ratio: {ref_ms / max(can_ms, 1e-3):.2f}x "
          f"(>1 means candidate is faster)")

    if ok:
        print("[g2v] PASS")
        sys.exit(0)
    else:
        print("[g2v] FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()

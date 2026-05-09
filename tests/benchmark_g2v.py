"""Phase 0 baseline benchmark — local_aggregate_prob_fast (and our cloned _s2go).

Measures forward time, backward time, peak GPU memory across several
(N_gauss, voxels) configs. The new S2GO kernel will be measured the same way.

Targets per S2GO §3.4.3 (paper, A100):
    forward:   1.5x  speedup over GF baseline
    backward: 20.4x  speedup over GF baseline
    memory:    3.3x  reduction

We don't expect to match A100 absolutes on the 3060 — the *ratios* are what
matter for verification.

Usage:
    /media/skr/storage/conda_envs/selfocc/bin/python benchmark_g2v.py
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager

import torch

# Both ops should be installed in the env (pip install -e .)
import local_aggregate_prob_fast as gf_op
try:
    import local_aggregate_s2go as s2go_op
    HAS_S2GO = True
except ImportError:
    HAS_S2GO = False


@contextmanager
def cuda_timer():
    """Context manager returning elapsed-ms after exit."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    out = [0.0]
    start.record()
    try:
        yield out
    finally:
        end.record()
        torch.cuda.synchronize()
        out[0] = start.elapsed_time(end)


def make_inputs(n_pts: int, n_gauss: int, H: int, W: int, D: int,
                grid_size: float, pc_min, n_chan: int, device, seed: int = 0):
    g = torch.Generator(device=device).manual_seed(seed)
    pmin = torch.tensor(pc_min, device=device)
    pmax = pmin + torch.tensor([H, W, D], device=device).float() * grid_size

    def in_bounds(n):
        return (torch.rand(1, n, 3, device=device, generator=g) * (pmax - pmin) * 0.95) + pmin

    pts = in_bounds(n_pts)
    means = in_bounds(n_gauss)
    # NOTE: opas is (1, N) not (1, N, 1). The CUDA op's backward returns a 1-D grad,
    # so PyTorch expects opas to be shape (..., N) — see test_g2v_gaussianformer.py
    opas = torch.rand(1, n_gauss, device=device, generator=g)
    semantics = torch.rand(1, n_gauss, n_chan, device=device, generator=g)
    scales = torch.full((1, n_gauss, 3), 0.5, device=device)
    cov_diag = (torch.eye(3, device=device) * 0.5).unsqueeze(0).expand(n_gauss, 3, 3).contiguous()
    cov3D = cov_diag.unsqueeze(0).clone()

    # tensors that need grad
    means.requires_grad_(True)
    opas.requires_grad_(True)
    semantics.requires_grad_(True)
    cov3D.requires_grad_(True)

    return pts, means, opas, semantics, scales, cov3D


def measure(op_module, pts, means, opas, semantics, scales, cov3D,
            H, W, D, pc_min, grid_size, n_iters: int = 20, n_warmup: int = 5):
    aggregator = op_module.LocalAggregator(
        scale_multiplier=4, H=H, W=W, D=D, pc_min=pc_min, grid_size=grid_size
    ).to(means.device)

    def one_iter(do_backward: bool):
        # zero grads
        for t in (means, opas, semantics, cov3D):
            if t.grad is not None:
                t.grad = None
        logits, bin_logits, density = aggregator(pts, means, opas, semantics, scales, cov3D)
        if do_backward:
            loss = logits.float().sum() + bin_logits.float().sum() + density.float().sum()
            loss.backward()
        return logits

    # warmup
    for _ in range(n_warmup):
        one_iter(do_backward=True)
    torch.cuda.synchronize()

    # forward-only timing
    fwd_ms = []
    for _ in range(n_iters):
        with cuda_timer() as t:
            with torch.no_grad():
                aggregator(pts, means, opas, semantics, scales, cov3D)
        fwd_ms.append(t[0])
    fwd_ms.sort()
    fwd_med = fwd_ms[len(fwd_ms) // 2]

    # forward + backward timing
    fb_ms = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(n_iters):
        with cuda_timer() as t:
            one_iter(do_backward=True)
        fb_ms.append(t[0])
    fb_ms.sort()
    fb_med = fb_ms[len(fb_ms) // 2]
    bwd_med = fb_med - fwd_med
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "fwd_ms_median": round(fwd_med, 4),
        "fwd_ms_min":    round(fwd_ms[0], 4),
        "fwd_ms_max":    round(fwd_ms[-1], 4),
        "fwd_bwd_ms_median": round(fb_med, 4),
        "bwd_ms_median": round(bwd_med, 4),
        "peak_mb": round(peak_mb, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kernel/tests/baseline_3060.json")
    args = ap.parse_args()

    device = "cuda"
    torch.cuda.empty_cache()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GF op:    {gf_op.__file__}")
    if HAS_S2GO:
        print(f"S2GO op:  {s2go_op.__file__}")
    else:
        print(f"S2GO op:  not installed (skipping)")
    print()

    # paper config: 200x200x16 voxels, 0.5m grid, [-50,-50,-5] origin
    H, W, D = 200, 200, 16
    grid_size = 0.5
    pc_min = [-50.0, -50.0, -5.0]
    n_chan = 18  # nuScenes-SurroundOcc class count
    n_pts = H * W * D  # all voxels evaluated (paper: 640k)

    # Sweep over Gaussian counts roughly matching paper's 9k / 12.8k / 25.6k
    sweep = [
        ("Prob-64-eq",  6_400),
        ("Prob-128-eq", 12_800),
        ("Prob-256-eq", 25_600),
    ]

    results = {"gpu": torch.cuda.get_device_name(0), "configs": []}
    for label, n_gauss in sweep:
        print(f"=== {label}  N_gauss={n_gauss}  voxels={n_pts:,}  C={n_chan} ===")
        inputs = make_inputs(n_pts, n_gauss, H, W, D, grid_size, pc_min, n_chan, device)

        gf_metrics = measure(gf_op, *inputs, H=H, W=W, D=D, pc_min=pc_min, grid_size=grid_size)
        print(f"  GF baseline   : fwd {gf_metrics['fwd_ms_median']} ms | "
              f"bwd {gf_metrics['bwd_ms_median']} ms | mem {gf_metrics['peak_mb']} MB")

        cfg = {"label": label, "n_gauss": n_gauss, "voxels": n_pts, "n_chan": n_chan,
               "gf": gf_metrics}

        if HAS_S2GO:
            s2_metrics = measure(s2go_op, *inputs, H=H, W=W, D=D, pc_min=pc_min, grid_size=grid_size)
            print(f"  S2GO clone    : fwd {s2_metrics['fwd_ms_median']} ms | "
                  f"bwd {s2_metrics['bwd_ms_median']} ms | mem {s2_metrics['peak_mb']} MB")
            cfg["s2go"] = s2_metrics

            ratio_fwd = gf_metrics["fwd_ms_median"] / s2_metrics["fwd_ms_median"] if s2_metrics["fwd_ms_median"] > 0 else float("nan")
            ratio_bwd = gf_metrics["bwd_ms_median"] / s2_metrics["bwd_ms_median"] if s2_metrics["bwd_ms_median"] > 0 else float("nan")
            ratio_mem = gf_metrics["peak_mb"] / s2_metrics["peak_mb"] if s2_metrics["peak_mb"] > 0 else float("nan")
            print(f"  ratio (S2GO/GF): fwd {ratio_fwd:.2f}x | bwd {ratio_bwd:.2f}x | mem {ratio_mem:.2f}x")
            cfg["ratio"] = {"fwd": round(ratio_fwd, 3), "bwd": round(ratio_bwd, 3), "mem": round(ratio_mem, 3)}

        results["configs"].append(cfg)
        print()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"results saved to {args.out}")


if __name__ == "__main__":
    main()

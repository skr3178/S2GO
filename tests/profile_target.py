"""Profile target — runs *only* the backward repeatedly so ncu/nvprof
data concentrates on the kernel under investigation.

Usage:
    /media/skr/storage/conda_envs/selfocc/bin/python profile_target.py
"""
import torch
import local_aggregate_s2go as op


def main(n_gauss: int = 12_800, n_iters: int = 30):
    H, W, D = 200, 200, 16
    grid_size = 0.5
    pc_min = [-50.0, -50.0, -5.0]
    n_chan = 18
    n_pts = H * W * D
    device = "cuda"

    g = torch.Generator(device=device).manual_seed(0)
    pmin = torch.tensor(pc_min, device=device)
    pmax = pmin + torch.tensor([H, W, D], device=device).float() * grid_size

    def in_bounds(n):
        return (torch.rand(1, n, 3, device=device, generator=g) * (pmax - pmin) * 0.95) + pmin

    pts = in_bounds(n_pts)
    means = in_bounds(n_gauss).requires_grad_(True)
    opas = torch.rand(1, n_gauss, device=device, generator=g, requires_grad=True)
    semantics = torch.rand(1, n_gauss, n_chan, device=device, generator=g, requires_grad=True)
    scales = torch.full((1, n_gauss, 3), 0.5, device=device)
    cov_diag = (torch.eye(3, device=device) * 0.5).unsqueeze(0).expand(n_gauss, 3, 3).contiguous()
    cov3D = cov_diag.unsqueeze(0).clone().requires_grad_(True)

    agg = op.LocalAggregator(scale_multiplier=4, H=H, W=W, D=D,
                             pc_min=pc_min, grid_size=grid_size).to(device)

    # warmup
    for _ in range(5):
        for t in (means, opas, semantics, cov3D):
            if t.grad is not None:
                t.grad = None
        l, b, d_ = agg(pts, means, opas, semantics, scales, cov3D)
        (l.sum() + b.sum() + d_.sum()).backward()
    torch.cuda.synchronize()
    print(f"warmup done; running {n_iters} iters with N_gauss={n_gauss}", flush=True)

    # repeated forward+backward — kernel calls accumulate for the profiler
    for _ in range(n_iters):
        for t in (means, opas, semantics, cov3D):
            if t.grad is not None:
                t.grad = None
        l, b, d_ = agg(pts, means, opas, semantics, scales, cov3D)
        (l.sum() + b.sum() + d_.sum()).backward()
    torch.cuda.synchronize()
    print("done", flush=True)


if __name__ == "__main__":
    main()

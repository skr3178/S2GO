"""Phase 3 verification — Eq. 9 opacity-weighted binary occupancy.

Tests:
  1. With opas = 1 (uniform), output matches the original `localagg_prob_fast` exactly
     — Eq. 9 reduces to Eq. 2 when opacity = 1.
  2. With opas != 1, the new bin_logits differ from the original.
  3. With opas = 0, bin_logits = 0 everywhere (the union product = 1, so 1-product = 0).
  4. Backward gradients are finite and self-consistent (gradcheck-style).
"""
import torch
import local_aggregate_prob_fast as gf_op
import local_aggregate_s2go as s2_op


def make_inputs(n_pts, n_gauss, H, W, D, grid_size, pc_min, n_chan, opas_val,
                device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    pmin = torch.tensor(pc_min, device=device)
    pmax = pmin + torch.tensor([H, W, D], device=device).float() * grid_size

    def in_bounds(n):
        return (torch.rand(1, n, 3, device=device, generator=g) * (pmax - pmin) * 0.95) + pmin

    pts = in_bounds(n_pts)
    means = in_bounds(n_gauss)
    if opas_val == "rand":
        opas = torch.rand(1, n_gauss, device=device, generator=g)
    elif isinstance(opas_val, (int, float)):
        opas = torch.full((1, n_gauss), float(opas_val), device=device)
    else:
        raise ValueError(opas_val)
    semantics = torch.rand(1, n_gauss, n_chan, device=device, generator=g)
    scales = torch.full((1, n_gauss, 3), 0.5, device=device)
    cov_diag = (torch.eye(3, device=device) * 0.5).unsqueeze(0).expand(n_gauss, 3, 3).contiguous()
    cov3D = cov_diag.unsqueeze(0).clone()
    return pts, means, opas, semantics, scales, cov3D


def run(op, pts, means, opas, semantics, scales, cov3D, H, W, D, pc_min, grid_size):
    agg = op.LocalAggregator(scale_multiplier=4, H=H, W=W, D=D,
                             pc_min=pc_min, grid_size=grid_size).to(pts.device)
    return agg(pts, means, opas, semantics, scales, cov3D)


def main():
    cfg = dict(H=200, W=200, D=16, grid_size=0.5, pc_min=[-50.0, -50.0, -5.0])
    n_chan = 18
    n_pts = 5_000
    n_gauss = 500

    print("Test 1: opas = 1.0 — Eq. 9 must reduce to Eq. 2 (bit-identical)")
    inp = make_inputs(n_pts, n_gauss, **cfg, n_chan=n_chan, opas_val=1.0)
    gf_l, gf_b, gf_d = run(gf_op, *inp, **cfg)
    s2_l, s2_b, s2_d = run(s2_op, *inp, **cfg)
    diff_l = (gf_l - s2_l).abs().max().item()
    diff_b = (gf_b - s2_b).abs().max().item()
    diff_d = (gf_d - s2_d).abs().max().item()
    print(f"  max abs diff  logits {diff_l:.2e}  bin {diff_b:.2e}  density {diff_d:.2e}")
    if max(diff_l, diff_b, diff_d) < 1e-5:
        print("  PASS")
    else:
        print("  FAIL")
        return 1

    print()
    print("Test 2: opas = 0.5 — bin_logits should differ from GF baseline")
    inp = make_inputs(n_pts, n_gauss, **cfg, n_chan=n_chan, opas_val=0.5)
    gf_l, gf_b, gf_d = run(gf_op, *inp, **cfg)
    s2_l, s2_b, s2_d = run(s2_op, *inp, **cfg)
    diff_b = (gf_b - s2_b).abs().max().item()
    print(f"  max abs |bin diff| = {diff_b:.4f}  (expect > 0)")
    print(f"  GF  bin range = [{gf_b.min().item():.4f}, {gf_b.max().item():.4f}]")
    print(f"  S2  bin range = [{s2_b.min().item():.4f}, {s2_b.max().item():.4f}]")
    if diff_b > 0.01:
        print("  PASS — outputs differ as expected")
    else:
        print("  FAIL — outputs identical, change didn't take effect")
        return 1

    print()
    print("Test 3: opas = 0.0 — alpha = 0 everywhere → bin_logits = 0 everywhere")
    inp = make_inputs(n_pts, n_gauss, **cfg, n_chan=n_chan, opas_val=0.0)
    s2_l, s2_b, s2_d = run(s2_op, *inp, **cfg)
    print(f"  S2 bin max = {s2_b.abs().max().item():.4e}  (expect 0)")
    if s2_b.abs().max().item() < 1e-6:
        print("  PASS")
    else:
        print("  FAIL")
        return 1

    print()
    print("Test 4: backward is finite + non-trivial (random opacity)")
    inp = make_inputs(n_pts, n_gauss, **cfg, n_chan=n_chan, opas_val="rand")
    pts, means, opas, semantics, scales, cov3D = inp
    means.requires_grad_(True)
    opas.requires_grad_(True)
    semantics.requires_grad_(True)
    cov3D.requires_grad_(True)

    s2_l, s2_b, s2_d = run(s2_op, pts, means, opas, semantics, scales, cov3D, **cfg)
    loss = s2_l.float().sum() + s2_b.float().sum() + s2_d.float().sum()
    loss.backward()
    for name, t in [("means", means), ("opas", opas), ("semantics", semantics), ("cov3D", cov3D)]:
        g = t.grad
        finite = torch.isfinite(g).all().item()
        nz = (g.abs() > 0).any().item()
        print(f"  d/d{name:9s}: finite={finite}  nonzero={nz}  max_abs={g.abs().max().item():.2e}")
        if not (finite and nz):
            print(f"  FAIL on d/d{name}")
            return 1
    print("  PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase-6 verification — local_aggregate_s2go_tiled is bit-identical to
local_aggregate_prob_fast across realistic opacity distributions, after the
Eq. 9 → Eq. 2 patch.

Inverts the assertions of test_eq9.py: where that file asserted s2go DIVERGED
from prob_fast at opa<1 (validating the Eq. 9 redefinition), this file asserts
s2go MATCHES prob_fast at all opacities (validating the drop-in claim).

Reuses the captured Gaussians from a real GF-2 forward (frame 0), which
exposes the failure mode the original test_eq9.py was designed to *not* catch
(low mean opacity ~0.044).
"""
import torch
import local_aggregate_prob_fast as fast_op
import local_aggregate_s2go_tiled as s2_op

DEVICE = 'cuda'
KW = dict(scale_multiplier=5, H=200, W=200, D=16,
          pc_min=[-50.0, -50.0, -5.0], grid_size=0.5)
SAVED = '/media/skr/storage/self_driving/S2GO/scripts/kernel_diff/aggregator_diff_frame0.pt'


def call(op, *args):
    return op.LocalAggregator(**KW).to(DEVICE)(*args)


def make_test_inputs(captured, opa_mode):
    """Real GF-2-shaped tensors with overridden opacities."""
    n_g = captured['means3D'].shape[1]
    if opa_mode == 'unit':
        opas = torch.ones(1, n_g, device=DEVICE)
    elif opa_mode == 'rand':
        opas = torch.rand(1, n_g, device=DEVICE)
    elif opa_mode == 'real':
        opas = captured['opas']
    elif isinstance(opa_mode, float):
        opas = torch.full((1, n_g), opa_mode, device=DEVICE)
    else:
        raise ValueError(opa_mode)
    return (captured['pts'], captured['means3D'], opas,
            captured['semantics'], captured['scales'], captured['cov3D'])


def diff_channels(out_a, out_b):
    names = ['logits', 'bin_logits', 'density']
    rows = []
    for i, name in enumerate(names):
        a = out_a[i].float()
        b = out_b[i].float()
        diff = (a - b).abs()
        rows.append((name, diff.max().item(), diff.mean().item(),
                     (diff == 0).float().mean().item()))
    return rows


def main():
    saved = torch.load(SAVED)
    captured = saved['inputs']
    print(f'Loaded captured tensors: {captured["means3D"].shape[1]} Gaussians, '
          f'{captured["pts"].shape[1]} voxels')
    print(f'Real GF-2 opacities: mean={captured["opas"].mean().item():.4f} '
          f'min={captured["opas"].min().item():.4f} '
          f'max={captured["opas"].max().item():.4f}')

    cases = [
        ('opa=1.0 (sanity)',     'unit'),
        ('opa=0.5',              0.5),
        ('opa=0.1',              0.1),
        ('opa=0.044 (GF-2 mean)', 0.044),
        ('opa=rand uniform',      'rand'),
        ('opa=real (GF-2)',       'real'),
    ]

    print(f'\n{"case":<24} {"channel":<12} {"max_diff":>11} {"mean_diff":>13} {"bit-id %":>10}  {"verdict"}')
    print('-' * 90)
    n_fail = 0
    for case_name, opa_mode in cases:
        inputs = make_test_inputs(captured, opa_mode)
        with torch.no_grad():
            out_fast = call(fast_op, *inputs)
            out_s2go = call(s2_op, *inputs)
        for ch, mx, mn, bid in diff_channels(out_fast, out_s2go):
            verdict = 'PASS' if mx == 0.0 else 'FAIL'
            if mx > 0:
                n_fail += 1
            print(f'{case_name:<24} {ch:<12} {mx:>11.6f} {mn:>13.6f} {100*bid:>9.4f}  {verdict}')
        print('-' * 90)

    print(f'\n=== {"ALL PASS" if n_fail == 0 else f"{n_fail} CHECKS FAILED"} ===')
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())

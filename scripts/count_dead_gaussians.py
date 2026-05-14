"""Count "dead" gaussians (opacity below thresholds) across multiple seqs for two checkpoints.

Reuses stage1_eval's load_model + run_forward so it matches eval exactly.

Usage:
    python scripts/count_dead_gaussians.py <ckpt1> <ckpt2> [--seqs 0 100 5000 20000]
"""
import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from s2go.datasets.nusc_loader import NuScenesLoader
from s2go.tools.stage1_eval import load_model, run_forward

THRESHOLDS = [0.001, 0.005, 0.01, 0.05]


def opacities_for_seq(backbone, seg, loader, seq_idx, device):
    seq_cpu = loader[seq_idx]
    seq = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in f.items()} for f in seq_cpu]
    outputs, _ = run_forward(backbone, seg, seq, device)
    g = outputs[-1].gaussians
    return g.opacities.detach().float().cpu().flatten()


def report(label, ops):
    n = ops.numel()
    print(f"\n=== {label} ===")
    print(f"  total gaussians: {n}")
    print(f"  μ={ops.mean():.4f}  σ={ops.std():.4f}  min={ops.min():.2e}  max={ops.max():.4f}")
    for t in THRESHOLDS:
        below = (ops < t).sum().item()
        pct = 100.0 * below / n
        print(f"  opacity < {t:<6}: {below:>6}  ({pct:5.2f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+", help="one or more .pt checkpoints")
    p.add_argument("--seqs", nargs="+", type=int, default=[0, 100, 5000, 20000])
    args = p.parse_args()

    device = "cuda"
    for ckpt in args.ckpts:
        print(f"\n############### {ckpt} ###############")
        backbone, seg, cfg = load_model(ckpt, device)
        loader = NuScenesLoader(T=cfg["T_queue"], verbose=False)
        all_ops = []
        for seq_idx in args.seqs:
            ops = opacities_for_seq(backbone, seg, loader, seq_idx, device)
            report(f"{Path(ckpt).parent.name}  seq{seq_idx}", ops)
            all_ops.append(ops)
        report(f"{Path(ckpt).parent.name}  ALL SEQS pooled ({len(args.seqs)} seqs)",
               torch.cat(all_ops))
        del backbone, seg
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

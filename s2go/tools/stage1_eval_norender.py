"""Slimmed stage1_eval that skips the gsplat depth render (avoids JIT compile)
but keeps the three geometric/state checks that directly answer
'is the checkpoint degenerate or learning real structure?':

  A. BEV scatter: refined query positions vs LiDAR points (with per-query
     nearest-LiDAR distance distribution — the key number).
  C. Gaussian-state histograms.
  D. Memory-queue inspection.

Run:
    python -m s2go.tools.stage1_eval_norender \\
        --checkpoint out/s2go_small_t2_overnight_40k_cosine/best_best.pt \\
        --out-dir   out/s2go_small_t2_overnight_40k_cosine/eval_iter3455/ \\
        --seq-idx   0
"""
import argparse, os, json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .stage1_eval import (
    load_model, run_forward, check_a_bev, check_c_histograms, check_d_queue,
)
from ..datasets.nusc_loader import NuScenesLoader


def main(checkpoint: str, out_dir: str, seq_indices):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda"
    backbone, seg, cfg = load_model(checkpoint, device)

    print(f"Loading nuScenes (T={cfg['T_queue']})…")
    loader = NuScenesLoader(T=cfg['T_queue'], verbose=False)
    print(f"  pool size: {len(loader)} sequences")

    all_stats = {}
    for sidx in seq_indices:
        print(f"\n=== seq_idx {sidx} ===")
        seq_cpu = loader[sidx]
        sequence = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in f.items()} for f in seq_cpu]

        # Reset queue between sequences so previous-sequence state doesn't leak
        seg.queue.reset()

        outputs, _ = run_forward(backbone, seg, sequence, device)

        # Check A — BEV scatter + nearest-LiDAR distances
        bev_path = os.path.join(out_dir, f"A_bev_seq{sidx}.png")
        sa = check_a_bev(outputs, sequence, bev_path)
        print(f"  A: nn_dist(m)  mean={sa['nn_dist_mean']:.3f}  "
              f"median={sa['nn_dist_median']:.3f}  "
              f"min={sa['nn_dist_min']:.3f}  max={sa['nn_dist_max']:.3f}")

        # Check C — Gaussian state histograms
        hist_path = os.path.join(out_dir, f"C_histograms_seq{sidx}.png")
        sc = check_c_histograms(outputs, hist_path)
        for k, v in sc.items():
            print(f"  C: {k:>9s}  μ={v['mean']:.3f}  σ={v['std']:.3f}  "
                  f"[{v['min']:.3f}, {v['max']:.3f}]")

        # Check D — memory queue
        sd = check_d_queue(seg, outputs)
        print(f"  D: pop={sd.get('memory_populated')}/{sd.get('memory_capacity')}  "
              f"unique={sd.get('memory_unique_xyz')}  "
              f"topk_opa μ={sd.get('topk_opacity_mean')}")

        # Extra: per-query xyz spread (variance in each axis)
        refined = outputs[-1].refined_xyz[0].detach().float().cpu().numpy()
        anchors = outputs[-1].anchors_xyz[0].detach().float().cpu().numpy()
        delta = refined - anchors
        spread = {
            "refined_xyz_std": refined.std(axis=0).tolist(),
            "refined_xyz_range": (refined.max(axis=0) - refined.min(axis=0)).tolist(),
            "delta_from_anchor_norm_mean": float(np.linalg.norm(delta, axis=-1).mean()),
            "delta_from_anchor_norm_std":  float(np.linalg.norm(delta, axis=-1).std()),
            "delta_from_anchor_norm_max":  float(np.linalg.norm(delta, axis=-1).max()),
        }
        print(f"  E: refined_xyz_std       = {[f'{s:.2f}' for s in spread['refined_xyz_std']]}  (per-axis σ in m)")
        print(f"  E: refined_xyz_range     = {[f'{r:.2f}' for r in spread['refined_xyz_range']]}  (per-axis spread in m)")
        print(f"  E: |refined - anchor|    mean={spread['delta_from_anchor_norm_mean']:.3f}  "
              f"std={spread['delta_from_anchor_norm_std']:.3f}  max={spread['delta_from_anchor_norm_max']:.3f}")

        all_stats[f"seq{sidx}"] = {
            "A_bev": sa, "C_gauss": sc, "D_queue": sd, "E_spread": spread,
        }

    with open(os.path.join(out_dir, "eval_stats.json"), "w") as fp:
        json.dump({"checkpoint": checkpoint, "stats": all_stats},
                  fp, indent=2, default=float)
    print(f"\nfigures + stats saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir",    required=True)
    p.add_argument("--seq-idx",    type=int, nargs="+", default=[0])
    a = p.parse_args()
    main(a.checkpoint, a.out_dir, a.seq_idx)

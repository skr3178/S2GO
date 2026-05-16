"""Plot Stage-2 training curves from training_history.json + eval_history.json.

Usage: python scripts/plot_stage2_curves.py <run_dir>
Writes <run_dir>/curves.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_avg(x, w=200):
    """Valid moving average: returns len(x)-w+1 points."""
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    run = Path(sys.argv[1])
    hist = json.load(open(run / "training_history.json"))
    ev = json.load(open(run / "eval_history.json")) if (run / "eval_history.json").exists() else []

    it = np.array([h["iter"] for h in hist])
    total = np.array([h["total"] for h in hist], dtype=float)
    occ = np.array([h["occ_bce"] for h in hist], dtype=float)
    skl = np.array([h["sem_kl"] for h in hist], dtype=float)
    sce = np.array([h["sem_ce"] for h in hist], dtype=float)
    lov = np.array([h["sem_lovasz"] for h in hist], dtype=float)
    lr = np.array([h["lr"] for h in hist], dtype=float)
    gnorm = np.array([h["gnorm"] for h in hist], dtype=float)

    ev_it = [e["iter"] for e in ev]
    ev_miou = [e["mIoU"] * 100 for e in ev]
    ev_occ = [e.get("occupancy_IoU", e.get("occ_IoU", 0)) * 100 for e in ev]

    w = 200
    mi = it[w - 1:] if len(it) >= w else it

    fig, ax = plt.subplots(2, 2, figsize=(16, 9))

    # (0,0) total loss raw + smoothed, eval mIoU on twin axis
    a = ax[0, 0]
    a.plot(it, total, color="#ccc", lw=0.4, label="total (raw)")
    a.plot(mi, moving_avg(total, w), color="#1a5", lw=1.8, label=f"total (MA{w})")
    a.set_xlabel("micro-iter"); a.set_ylabel("loss"); a.set_title("Total loss + eval mIoU")
    a.legend(loc="upper left"); a.grid(alpha=0.3)
    a2 = a.twinx()
    a2.plot(ev_it, ev_miou, "o-", color="#c30", lw=2, ms=5, label="val mIoU %")
    a2.plot(ev_it, ev_occ, "s--", color="#06c", lw=1.2, ms=4, label="val occ-IoU %")
    a2.set_ylabel("val IoU %"); a2.legend(loc="upper right")

    # (0,1) loss components (smoothed)
    a = ax[0, 1]
    for arr, lab, col in [(occ, "occ_bce", "#e63"), (skl, "sem_kl", "#36c"),
                          (sce, "sem_ce", "#1a5"), (lov, "lovasz", "#929")]:
        a.plot(mi, moving_avg(arr, w), lw=1.5, label=lab, color=col)
    a.set_xlabel("micro-iter"); a.set_ylabel(f"loss (MA{w})")
    a.set_title("Loss components"); a.legend(); a.grid(alpha=0.3)

    # (1,0) LR schedule
    a = ax[1, 0]
    a.plot(it, lr, color="#06c", lw=1.5)
    a.set_xlabel("micro-iter"); a.set_ylabel("lr")
    a.set_title("LR schedule (warmup_cosine)"); a.grid(alpha=0.3)

    # (1,1) grad-norm (raw + smoothed, log y)
    a = ax[1, 1]
    a.plot(it, np.clip(gnorm, 1e-3, None), color="#ccc", lw=0.3, label="gnorm raw")
    a.plot(mi, moving_avg(np.clip(gnorm, 1e-3, None), w), color="#922", lw=1.5,
           label=f"gnorm MA{w}")
    a.axhline(35.0, color="k", ls="--", lw=0.8, label="clip=35")
    a.set_yscale("log"); a.set_xlabel("micro-iter"); a.set_ylabel("grad norm (log)")
    a.set_title("Gradient norm"); a.legend(); a.grid(alpha=0.3)

    fig.suptitle(f"Stage-2 curated 3-epoch (b16) — {run.name}", fontsize=14)
    fig.tight_layout()
    out = run / "curves.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")
    # quick textual summary
    print(f"final total-loss MA{w}: {moving_avg(total, w)[-1]:.3f}")
    print(f"final eval mIoU: {ev_miou[-1]:.2f}%  (peak {max(ev_miou):.2f}% @ iter {ev_it[int(np.argmax(ev_miou))]})")


if __name__ == "__main__":
    main()

"""Parse overfit.py's train.log and plot loss + lr curves.

Usage: python scripts/plot_train_log.py <run_dir> [<run_dir>...]
       (first dir is "current/foreground", overlay rest as comparisons)
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ITER_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*([\d.eE+-]+|nan)\s*\|\s*\S+\s*\|\s*([\d.eE+-]+|nan)\s*\|"
    r"\s*\S+\s*\|.*?\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)\s*\|"
)
BEST_RE = re.compile(r"\[best #\d+\] iter (\d+): smoothed L \(50-iter MA\) = ([\d.]+)")
SKIP_RE = re.compile(r"\[skip[^\]]*\] iter (\d+)")


def parse(log_path):
    iters, total, ldep, gnorm, lrs = [], [], [], [], []
    bests_iter, bests_ma = [], []
    skips = []
    for line in Path(log_path).read_text().splitlines():
        m = ITER_RE.match(line)
        if m:
            iters.append(int(m.group(1)))
            total.append(float(m.group(2)) if m.group(2) != "nan" else float("nan"))
            ldep.append(float(m.group(3)) if m.group(3) != "nan" else float("nan"))
            gnorm.append(float(m.group(4)))
            lrs.append(float(m.group(5)))
            continue
        b = BEST_RE.search(line)
        if b:
            bests_iter.append(int(b.group(1)))
            bests_ma.append(float(b.group(2)))
            continue
        s = SKIP_RE.search(line)
        if s:
            skips.append(int(s.group(1)))
    return dict(iters=iters, total=total, ldep=ldep, gnorm=gnorm, lr=lrs,
                bests_iter=bests_iter, bests_ma=bests_ma, skips=skips)


def main(run_dirs):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                              gridspec_kw=dict(height_ratios=[3, 1]))
    ax_loss, ax_lr = axes

    colors = ["C0", "C1", "C2", "C3"]
    for i, run_dir in enumerate(run_dirs):
        run_dir = Path(run_dir)
        log = run_dir / "train.log"
        if not log.exists():
            print(f"  skipping {run_dir}: no train.log")
            continue
        d = parse(log)
        c = colors[i % len(colors)]
        label = run_dir.name

        if not d["iters"]:
            print(f"  {label}: no iter rows parsed")
            continue

        # Per-iter L_depth (light)
        ax_loss.plot(d["iters"], d["ldep"], color=c, alpha=0.25, lw=0.8,
                     label=f"{label} L_dep")
        # Best-save MA (dark dots + line)
        if d["bests_iter"]:
            ax_loss.plot(d["bests_iter"], d["bests_ma"], color=c, marker="o",
                          lw=1.5, ms=5, label=f"{label} smoothed best (50-iter MA)")
        # Skips as red x marks at y=0 baseline
        if d["skips"]:
            ax_loss.scatter(d["skips"], [0.1] * len(d["skips"]),
                             color=c, marker="x", s=8, alpha=0.4,
                             label=f"{label} NaN skips ({len(d['skips'])})")

        # LR (use last col 'lr' which is backbone lr)
        ax_lr.plot(d["iters"], d["lr"], color=c, lw=1.2, label=label)

        # Console summary
        finite_ldep = [v for v in d["ldep"] if v == v]
        print(f"  {label}:")
        print(f"    iters logged       : {len(d['iters'])} (max iter {max(d['iters'])})")
        print(f"    finite L_depth     : {len(finite_ldep)}")
        print(f"    skips              : {len(d['skips'])}")
        print(f"    best-saves         : {len(d['bests_iter'])}")
        if d["bests_iter"]:
            best_idx = min(range(len(d["bests_ma"])), key=lambda i: d["bests_ma"][i])
            print(f"    best smoothed L    : {d['bests_ma'][best_idx]:.4f} m "
                  f"@ iter {d['bests_iter'][best_idx]}")

    ax_loss.set_ylabel("L_depth (m)")
    ax_loss.set_title("Stage-1 training: per-iter L_depth and 50-iter MA best-saves")
    ax_loss.set_yscale("log")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(loc="upper right", fontsize=8, ncol=2)

    ax_lr.set_ylabel("backbone lr")
    ax_lr.set_xlabel("micro-iter")
    ax_lr.set_yscale("log")
    ax_lr.grid(alpha=0.3)
    ax_lr.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    out = Path(run_dirs[0]) / "curves.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])

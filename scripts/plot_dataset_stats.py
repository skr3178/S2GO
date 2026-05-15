"""
Plot per-part and global diversity statistics for the 10 nuScenes trainval
keyframe archives. Reads:
  data/nuscenes_dl/diversity/scene_stats.csv
  data/nuscenes_dl/diversity/class_composition_per_scene.csv
Writes PNGs to <outdir>/.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = ["barrier", "bicycle", "bus", "car", "const_veh", "motorcycle",
               "pedestrian", "traffic_cone", "trailer", "truck",
               "drive_surf", "other_flat", "sidewalk", "terrain", "manmade", "vegetation"]
THING = CLASS_NAMES[:10]
STUFF = CLASS_NAMES[10:]

LOC_ORDER = ["boston-seaport", "singapore-onenorth", "singapore-queenstown", "singapore-hollandvillage"]
LOC_COLORS = ["#cc4444", "#4488cc", "#44aa66", "#cc9944"]


def load_rows(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def fig_savefig(fig, outpath):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-csv", required=True)
    ap.add_argument("--class-csv", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    scenes = load_rows(args.scene_csv)
    classes = {r["scene_token"]: r for r in load_rows(args.class_csv)}

    # Index scenes by part
    by_part = defaultdict(list)
    for r in scenes:
        by_part[int(r["part"])].append(r)
    parts = sorted(by_part)

    # ---------- Figure 1: per-part size & length ----------
    print("[fig1] per-part scenes/samples/duration")
    nscenes = [len(by_part[p]) for p in parts]
    nsamples = [sum(int(r["num_samples"]) for r in by_part[p]) for p in parts]
    dur_min = [sum(float(r["duration_s"]) for r in by_part[p]) / 60 for p in parts]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(parts, nscenes, color="#3a7", edgecolor="black", linewidth=0.5)
    axes[0].set_title("Scenes per part"); axes[0].set_xlabel("part"); axes[0].set_ylabel("# scenes")
    axes[0].set_xticks(parts); axes[0].set_ylim(0, 100); axes[0].axhline(85, color="red", linestyle="--", alpha=0.4, label="85 (uniform)")
    axes[0].legend()
    axes[1].bar(parts, nsamples, color="#37a", edgecolor="black", linewidth=0.5)
    axes[1].set_title("Keyframes (samples) per part"); axes[1].set_xlabel("part"); axes[1].set_ylabel("# samples")
    axes[1].set_xticks(parts)
    for p, n in zip(parts, nsamples): axes[1].text(p, n + 30, str(n), ha="center", fontsize=8)
    axes[2].bar(parts, dur_min, color="#a73", edgecolor="black", linewidth=0.5)
    axes[2].set_title("Recorded duration per part"); axes[2].set_xlabel("part"); axes[2].set_ylabel("minutes")
    axes[2].set_xticks(parts)
    for p, d in zip(parts, dur_min): axes[2].text(p, d + 0.1, f"{d:.1f}", ha="center", fontsize=8)
    fig.suptitle("nuScenes trainval — per-archive size & length", fontsize=13, y=1.02)
    fig_savefig(fig, outdir / "01_per_part_size.png")

    # ---------- Figure 2: location stacked bar ----------
    print("[fig2] location distribution per part")
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(parts))
    for loc, col in zip(LOC_ORDER, LOC_COLORS):
        vals = np.array([sum(1 for r in by_part[p] if r["location"] == loc) for p in parts])
        ax.bar(parts, vals, bottom=bottom, color=col, edgecolor="black", linewidth=0.4,
               label=loc.replace("singapore-", "sg-"))
        for i, (b, v) in enumerate(zip(bottom, vals)):
            if v >= 4:
                ax.text(parts[i], b + v / 2, str(v), ha="center", va="center", fontsize=8, color="white")
        bottom += vals
    ax.set_xticks(parts); ax.set_xlabel("part"); ax.set_ylabel("# scenes")
    ax.set_title("Location distribution per archive (85 scenes each)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig_savefig(fig, outdir / "02_per_part_location.png")

    # ---------- Figure 3: vehicle + day/night/rain/construction ----------
    print("[fig3] vehicle and conditions per part")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # vehicle
    n008 = [sum(1 for r in by_part[p] if r["vehicle"] == "n008") for p in parts]
    n015 = [sum(1 for r in by_part[p] if r["vehicle"] == "n015") for p in parts]
    axes[0].bar(parts, n008, label="n008", color="#5a8")
    axes[0].bar(parts, n015, bottom=n008, label="n015", color="#85a")
    axes[0].set_title("Recording vehicle per part"); axes[0].set_xlabel("part"); axes[0].set_ylabel("# scenes")
    axes[0].set_xticks(parts); axes[0].legend()
    # conditions
    night = [sum(1 for r in by_part[p] if r["night"] == "True") for p in parts]
    rain = [sum(1 for r in by_part[p] if r["rain"] == "True") for p in parts]
    cnst = [sum(1 for r in by_part[p] if r["construction"] == "True") for p in parts]
    busy = [sum(1 for r in by_part[p] if r["busy"] == "True") for p in parts]
    x = np.arange(len(parts)); w = 0.2
    axes[1].bar(x - 1.5*w, night, w, label="night", color="#222")
    axes[1].bar(x - 0.5*w, rain, w, label="rain", color="#48c")
    axes[1].bar(x + 0.5*w, cnst, w, label="construction", color="#fa3")
    axes[1].bar(x + 1.5*w, busy, w, label="busy", color="#c43")
    axes[1].set_xticks(x); axes[1].set_xticklabels(parts); axes[1].set_xlabel("part"); axes[1].set_ylabel("# scenes")
    axes[1].set_title("Conditions per part (from description tags)")
    axes[1].legend()
    fig_savefig(fig, outdir / "03_per_part_vehicle_conditions.png")

    # ---------- Figure 4: thing-class instance counts per part (heatmap) ----------
    print("[fig4] thing-class instance heatmap per part")
    mat = np.zeros((len(THING), len(parts)), dtype=np.int64)
    for j, p in enumerate(parts):
        for r in by_part[p]:
            cls_row = classes.get(r["scene_token"], {})
            for i, cn in enumerate(THING):
                mat[i, j] += int(cls_row.get(f"th_{cn}", 0))
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", norm=matplotlib.colors.LogNorm(vmin=1, vmax=mat.max()))
    ax.set_xticks(range(len(parts))); ax.set_xticklabels(parts)
    ax.set_yticks(range(len(THING))); ax.set_yticklabels(THING)
    ax.set_xlabel("part"); ax.set_title("Thing-class box counts per archive (log color)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:,}" if v > 0 else "0",
                    ha="center", va="center", fontsize=7,
                    color="white" if v < mat.max() * 0.3 else "black")
    fig.colorbar(im, ax=ax, label="# boxes (log)")
    fig_savefig(fig, outdir / "04_thing_class_per_part_heatmap.png")

    # ---------- Figure 5: stuff-class voxel counts per part (heatmap) ----------
    print("[fig5] stuff-class voxel heatmap per part")
    mat = np.zeros((len(STUFF), len(parts)), dtype=np.int64)
    for j, p in enumerate(parts):
        for r in by_part[p]:
            cls_row = classes.get(r["scene_token"], {})
            for i, cn in enumerate(STUFF):
                mat[i, j] += int(cls_row.get(f"vx_{cn}", 0))
    fig, ax = plt.subplots(figsize=(12, 3.5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", norm=matplotlib.colors.LogNorm(vmin=1, vmax=mat.max()))
    ax.set_xticks(range(len(parts))); ax.set_xticklabels(parts)
    ax.set_yticks(range(len(STUFF))); ax.set_yticklabels(STUFF)
    ax.set_xlabel("part"); ax.set_title("Stuff-class voxel counts per archive (log color)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = f"{v/1e6:.1f}M" if v >= 1e6 else (f"{v/1e3:.0f}K" if v >= 1e3 else str(v))
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if v < mat.max() * 0.3 else "black")
    fig.colorbar(im, ax=ax, label="# voxels (log)")
    fig_savefig(fig, outdir / "05_stuff_class_per_part_heatmap.png")

    # ---------- Figure 6: global class scarcity (scenes containing each class) ----------
    print("[fig6] global class scarcity (#scenes containing each class)")
    contain = []
    for cn in CLASS_NAMES:
        key = f"th_{cn}" if cn in THING else f"vx_{cn}"
        contain.append(sum(1 for r in classes.values() if int(r[key]) > 0))
    order = np.argsort(contain)
    names_s = [CLASS_NAMES[i] for i in order]
    vals_s = [contain[i] for i in order]
    cols = ["#c33" if v < 400 else ("#fa3" if v < 700 else "#3a7") for v in vals_s]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(names_s, vals_s, color=cols, edgecolor="black", linewidth=0.4)
    for y, v in enumerate(vals_s):
        ax.text(v + 5, y, f"{v}/850", va="center", fontsize=8)
    ax.set_xlim(0, 900); ax.set_xlabel("# scenes containing class (of 850)")
    ax.set_title("Class scarcity across trainval (lower = rarer)")
    ax.axvline(850, color="grey", linestyle=":", alpha=0.6)
    fig_savefig(fig, outdir / "06_class_scarcity.png")

    # ---------- Figure 7: scene-duration & sample-count distributions ----------
    print("[fig7] duration & sample-count histograms")
    durs = [float(r["duration_s"]) for r in scenes]
    nsmp = [int(r["num_samples"]) for r in scenes]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(durs, bins=40, color="#37a", edgecolor="black", linewidth=0.4)
    axes[0].set_xlabel("scene duration (s)"); axes[0].set_ylabel("# scenes")
    axes[0].set_title(f"Scene duration distribution (median={np.median(durs):.2f}s)")
    axes[0].axvline(20.0, color="red", linestyle="--", alpha=0.5, label="nominal 20s")
    axes[0].legend()
    axes[1].hist(nsmp, bins=range(min(nsmp), max(nsmp) + 2), color="#a73", edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("# samples (keyframes) per scene"); axes[1].set_ylabel("# scenes")
    axes[1].set_title(f"Samples-per-scene distribution (median={int(np.median(nsmp))})")
    fig_savefig(fig, outdir / "07_scene_dur_samples_hist.png")

    print(f"\nAll plots written to {outdir}/")


if __name__ == "__main__":
    main()

"""
Anchor + greedy submodular curation of nuScenes trainval scenes.

Phase 1: hard-include top-K scenes for each rare thing class (configurable).
Phase 2: greedy fill on a concave-coverage objective combining class counts and
categorical axes.

Outputs:
  <outdir>/curated_v2_tokens.json
  <outdir>/curated_v2_scenes.csv
  <outdir>/figures/{08..12}_*.png   (comparison vs full set)
"""
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THING_CLASSES = ["barrier", "bicycle", "bus", "car", "const_veh", "motorcycle",
                 "pedestrian", "traffic_cone", "trailer", "truck"]
STUFF_CLASSES = ["drive_surf", "other_flat", "sidewalk", "terrain", "manmade", "vegetation"]
ALL_CLASSES = THING_CLASSES + STUFF_CLASSES

LOCATIONS = ["boston-seaport", "singapore-onenorth", "singapore-queenstown", "singapore-hollandvillage"]
LOC_SHORT = {"boston-seaport": "boston", "singapore-onenorth": "sg-on",
             "singapore-queenstown": "sg-qt", "singapore-hollandvillage": "sg-hv"}
VEHICLES = ["n008", "n015"]

# Additional description-derived booleans for the categorical axis term
EXTRA_TAGS = ["intersection", "crosswalk", "jaywalker", "bus_intxn", "construction", "rain", "night", "busy"]
TAG_RE = {
    "intersection": re.compile(r"\bintersection\b", re.I),
    "crosswalk": re.compile(r"\bcrosswalk\b", re.I),
    "jaywalker": re.compile(r"\bjaywalk", re.I),
    "bus_intxn": re.compile(r"\b(bus|buses)\b", re.I),
    "construction": re.compile(r"\b(construct|cone|barrier|roadwork)", re.I),
    "rain": re.compile(r"\b(rain|wet)\b", re.I),
    "night": re.compile(r"\bnight\b", re.I),
    "busy": re.compile(r"\b(busy|heavy traffic|congest|crowd)", re.I),
}


def load_scene_rows(class_csv, scene_csv):
    """Return list of merged rows keyed by scene_token."""
    cls_rows = {r["scene_token"]: r for r in csv.DictReader(open(class_csv))}
    scn_rows = {r["scene_token"]: r for r in csv.DictReader(open(scene_csv))}
    out = []
    for tok in cls_rows:
        c = cls_rows[tok]; s = scn_rows.get(tok, {})
        rec = dict(
            scene_token=tok,
            name=c["name"],
            location=c["location"],
            vehicle=c["vehicle"],
            num_samples=int(c["num_samples"]),
            description=c["description"],
            part=int(s.get("part", 0)) if s else 0,
            night=bool(TAG_RE["night"].search(c["description"])),
            rain=bool(TAG_RE["rain"].search(c["description"])),
        )
        for cn in THING_CLASSES:
            rec[f"th_{cn}"] = int(c[f"th_{cn}"])
        for cn in STUFF_CLASSES:
            rec[f"vx_{cn}"] = int(c[f"vx_{cn}"])
        for tag, rx in TAG_RE.items():
            rec[f"tag_{tag}"] = bool(rx.search(c["description"]))
        out.append(rec)
    return out


# ---------- objective helpers ----------

def class_feat(rec):
    """Flattened, log-scaled class contribution vector (16 dims)."""
    v = np.zeros(len(ALL_CLASSES), dtype=np.float64)
    for i, cn in enumerate(THING_CLASSES):
        v[i] = math.log1p(rec[f"th_{cn}"])
    for j, cn in enumerate(STUFF_CLASSES):
        # voxel counts are 6 orders of magnitude bigger; divide by 1k first
        v[len(THING_CLASSES) + j] = math.log1p(rec[f"vx_{cn}"] / 1000.0)
    return v


def cat_indicator(rec):
    """Categorical one-hot vector across location, vehicle, and EXTRA_TAGS."""
    v = []
    for loc in LOCATIONS:
        v.append(1.0 if rec["location"] == loc else 0.0)
    for veh in VEHICLES:
        v.append(1.0 if rec["vehicle"] == veh else 0.0)
    for tag in EXTRA_TAGS:
        v.append(1.0 if rec.get(f"tag_{tag}") else 0.0)
    return np.array(v)


def gain(F_class, F_cat, f_cls, f_cat, w_cls, w_cat):
    """Concave (sqrt) marginal gain of adding (f_cls, f_cat) to current sums."""
    new_cls = F_class + f_cls
    new_cat = F_cat + f_cat
    d_cls = (np.sqrt(new_cls) - np.sqrt(F_class)) * w_cls
    d_cat = (np.sqrt(new_cat) - np.sqrt(F_cat)) * w_cat
    return d_cls.sum() + d_cat.sum()


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-csv", required=True)
    ap.add_argument("--class-csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--budget", type=int, default=255)
    ap.add_argument("--w-cat", type=float, default=2.0, help="weight on categorical axes (vs unit weight on class axes)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    rows = load_scene_rows(args.class_csv, args.scene_csv)
    print(f"[load] {len(rows)} scenes")

    # ----- Phase 1: anchors -----
    anchor_plan = [
        ("th_trailer", 6),
        ("th_const_veh", 6),
        ("th_bicycle", 5),
        ("th_motorcycle", 5),
        ("th_barrier", 4),
        ("th_bus", 4),
        ("th_traffic_cone", 3),
    ]
    picked = set()
    anchor_provenance = {}
    for col, k in anchor_plan:
        ranked = sorted(rows, key=lambda r: -r[col])
        added_for_class = 0
        for r in ranked:
            if added_for_class == k:
                break
            if r["scene_token"] in picked:
                continue
            picked.add(r["scene_token"])
            anchor_provenance[r["scene_token"]] = col.replace("th_", "")
            added_for_class += 1
    print(f"[anchors] {len(picked)} scenes locked in for rare-class coverage")

    # ----- Phase 2: greedy submodular fill -----
    cls_dim = len(ALL_CLASSES)
    cat_dim = len(LOCATIONS) + len(VEHICLES) + len(EXTRA_TAGS)
    w_cls = np.ones(cls_dim, dtype=np.float64)
    w_cat = np.full(cat_dim, args.w_cat, dtype=np.float64)

    feats_cls = np.stack([class_feat(r) for r in rows])
    feats_cat = np.stack([cat_indicator(r) for r in rows])
    F_cls = np.zeros(cls_dim, dtype=np.float64) + 1e-9
    F_cat = np.zeros(cat_dim, dtype=np.float64) + 1e-9

    tok2idx = {r["scene_token"]: i for i, r in enumerate(rows)}
    for tok in picked:
        i = tok2idx[tok]
        F_cls += feats_cls[i]
        F_cat += feats_cat[i]

    available = np.array([1.0 if rows[i]["scene_token"] not in picked else 0.0 for i in range(len(rows))])
    selection_order = list(picked)
    while len(picked) < args.budget:
        # vectorised gain across all candidates
        new_cls = F_cls[None, :] + feats_cls
        new_cat = F_cat[None, :] + feats_cat
        gain_cls = (np.sqrt(new_cls) - np.sqrt(F_cls)[None, :]).dot(w_cls)
        gain_cat = (np.sqrt(new_cat) - np.sqrt(F_cat)[None, :]).dot(w_cat)
        total_gain = (gain_cls + gain_cat) * available
        best = int(np.argmax(total_gain))
        if available[best] == 0 or total_gain[best] <= 0:
            print(f"[greedy] no positive-gain candidate left at size {len(picked)}; stopping.")
            break
        picked.add(rows[best]["scene_token"])
        selection_order.append(rows[best]["scene_token"])
        F_cls += feats_cls[best]
        F_cat += feats_cat[best]
        available[best] = 0
    print(f"[greedy] final selection: {len(picked)} scenes")

    # ----- write picks -----
    pick_rows = [r for r in rows if r["scene_token"] in picked]
    full_rows = rows
    tok_path = outdir / "curated_v2_tokens.json"
    with open(tok_path, "w") as f:
        json.dump(list(selection_order), f)
    print(f"[write] {tok_path}")
    csv_path = outdir / "curated_v2_scenes.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_token", "name", "part", "location", "vehicle", "num_samples",
                    "anchor_for", "description"])
        for r in sorted(pick_rows, key=lambda r: (r["part"], r["name"])):
            w.writerow([r["scene_token"], r["name"], r["part"], r["location"], r["vehicle"],
                        r["num_samples"], anchor_provenance.get(r["scene_token"], ""),
                        r["description"].replace(",", ";")])
    print(f"[write] {csv_path}")

    # ===================== PLOTS =====================
    def count_by(rows_, key):
        return Counter(r[key] for r in rows_)
    def tag_count(rows_, tag):
        return sum(1 for r in rows_ if r.get(f"tag_{tag}", False))
    def sum_col(rows_, col):
        return sum(int(r[col]) for r in rows_)

    # --- Fig 8: per-part composition of picks ---
    print("[fig 8] per-part composition of picks vs uniform-2/3")
    parts_full = Counter(r["part"] for r in rows)
    parts_pick = Counter(r["part"] for r in pick_rows)
    parts = sorted(parts_full)
    pick_per_part = [parts_pick.get(p, 0) for p in parts]
    full_per_part = [parts_full[p] for p in parts]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(parts)); w = 0.4
    ax.bar(x - w/2, full_per_part, w, label="full trainval (85 each)", color="#bbb", edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, pick_per_part, w, label=f"curated v2 (total {len(pick_rows)})", color="#37a", edgecolor="black", linewidth=0.4)
    for xi, v in zip(x + w/2, pick_per_part): ax.text(xi, v + 1, str(v), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(parts)
    ax.set_xlabel("archive part"); ax.set_ylabel("# scenes")
    ax.set_title("Where the curated subset is drawn from")
    ax.legend()
    fig.savefig(figdir / "08_curated_origin_per_part.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # --- Fig 9: location/vehicle composition vs full ---
    print("[fig 9] location/vehicle composition vs full")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    # location
    full_loc = count_by(full_rows, "location"); pick_loc = count_by(pick_rows, "location")
    labels = [LOC_SHORT[l] for l in LOCATIONS]
    full_pct = [100 * full_loc.get(l, 0) / len(full_rows) for l in LOCATIONS]
    pick_pct = [100 * pick_loc.get(l, 0) / len(pick_rows) for l in LOCATIONS]
    xp = np.arange(len(LOCATIONS)); w = 0.4
    axes[0].bar(xp - w/2, full_pct, w, label="full trainval", color="#bbb", edgecolor="black", linewidth=0.4)
    axes[0].bar(xp + w/2, pick_pct, w, label="curated v2", color="#37a", edgecolor="black", linewidth=0.4)
    for xi, v in zip(xp - w/2, full_pct): axes[0].text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    for xi, v in zip(xp + w/2, pick_pct): axes[0].text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    axes[0].set_xticks(xp); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("% of scenes"); axes[0].set_title("Location share: full vs curated")
    axes[0].set_ylim(0, max(max(full_pct), max(pick_pct)) * 1.2); axes[0].legend()
    # vehicle
    full_v = count_by(full_rows, "vehicle"); pick_v = count_by(pick_rows, "vehicle")
    xp = np.arange(len(VEHICLES))
    full_pct = [100 * full_v.get(v, 0) / len(full_rows) for v in VEHICLES]
    pick_pct = [100 * pick_v.get(v, 0) / len(pick_rows) for v in VEHICLES]
    axes[1].bar(xp - w/2, full_pct, w, label="full", color="#bbb", edgecolor="black", linewidth=0.4)
    axes[1].bar(xp + w/2, pick_pct, w, label="curated v2", color="#37a", edgecolor="black", linewidth=0.4)
    for xi, v in zip(xp - w/2, full_pct): axes[1].text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    for xi, v in zip(xp + w/2, pick_pct): axes[1].text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    axes[1].set_xticks(xp); axes[1].set_xticklabels(VEHICLES)
    axes[1].set_ylabel("% of scenes"); axes[1].set_title("Recording vehicle share")
    axes[1].set_ylim(0, max(max(full_pct), max(pick_pct)) * 1.2); axes[1].legend()
    fig.savefig(figdir / "09_curated_location_vehicle.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # --- Fig 10: condition coverage (% with each tag) ---
    print("[fig 10] condition coverage")
    fig, ax = plt.subplots(figsize=(12, 5))
    tags_to_show = ["night", "rain", "construction", "busy", "intersection", "crosswalk", "jaywalker", "bus_intxn"]
    full_pct = [100 * tag_count(full_rows, t) / len(full_rows) for t in tags_to_show]
    pick_pct = [100 * tag_count(pick_rows, t) / len(pick_rows) for t in tags_to_show]
    xt = np.arange(len(tags_to_show))
    ax.bar(xt - 0.2, full_pct, 0.4, label="full trainval", color="#bbb", edgecolor="black", linewidth=0.4)
    ax.bar(xt + 0.2, pick_pct, 0.4, label="curated v2", color="#37a", edgecolor="black", linewidth=0.4)
    for xi, v in zip(xt - 0.2, full_pct): ax.text(xi, v + 0.5, f"{v:.0f}%", ha="center", fontsize=8)
    for xi, v in zip(xt + 0.2, pick_pct): ax.text(xi, v + 0.5, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(xt); ax.set_xticklabels(tags_to_show, rotation=15)
    ax.set_ylabel("% of scenes with tag")
    ax.set_title("Condition coverage: full vs curated (independent tags, can sum > 100%)")
    ax.legend()
    fig.savefig(figdir / "10_curated_conditions.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # --- Fig 11: per-class instance/voxel total vs full (log y) ---
    print("[fig 11] per-class instance/voxel totals vs full")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # thing classes: box counts
    full_th = [sum_col(full_rows, f"th_{cn}") for cn in THING_CLASSES]
    pick_th = [sum_col(pick_rows, f"th_{cn}") for cn in THING_CLASSES]
    xc = np.arange(len(THING_CLASSES))
    axes[0].bar(xc - 0.2, full_th, 0.4, label="full", color="#bbb", edgecolor="black", linewidth=0.4)
    axes[0].bar(xc + 0.2, pick_th, 0.4, label="curated v2", color="#37a", edgecolor="black", linewidth=0.4)
    axes[0].set_yscale("log"); axes[0].set_xticks(xc); axes[0].set_xticklabels(THING_CLASSES, rotation=30, ha="right")
    axes[0].set_ylabel("# boxes (log)"); axes[0].set_title("Thing-class instance count: full vs curated")
    axes[0].legend()
    for xi, f, p in zip(xc, full_th, pick_th):
        pct = 100 * p / max(1, f); axes[0].text(xi + 0.2, p * 1.15, f"{pct:.0f}%", ha="center", fontsize=8)
    # stuff classes: voxel counts
    full_st = [sum_col(full_rows, f"vx_{cn}") for cn in STUFF_CLASSES]
    pick_st = [sum_col(pick_rows, f"vx_{cn}") for cn in STUFF_CLASSES]
    xc = np.arange(len(STUFF_CLASSES))
    axes[1].bar(xc - 0.2, full_st, 0.4, label="full", color="#bbb", edgecolor="black", linewidth=0.4)
    axes[1].bar(xc + 0.2, pick_st, 0.4, label="curated v2", color="#37a", edgecolor="black", linewidth=0.4)
    axes[1].set_yscale("log"); axes[1].set_xticks(xc); axes[1].set_xticklabels(STUFF_CLASSES, rotation=30, ha="right")
    axes[1].set_ylabel("# voxels (log)"); axes[1].set_title("Stuff-class voxel count: full vs curated")
    axes[1].legend()
    for xi, f, p in zip(xc, full_st, pick_st):
        pct = 100 * p / max(1, f); axes[1].text(xi + 0.2, p * 1.15, f"{pct:.0f}%", ha="center", fontsize=8)
    fig.savefig(figdir / "11_curated_class_counts.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # --- Fig 12: rare-class fraction retained (key headline plot) ---
    print("[fig 12] rare-class fraction retained")
    fig, ax = plt.subplots(figsize=(11, 5))
    cn_all = THING_CLASSES + STUFF_CLASSES
    full_v = full_th + full_st; pick_v = pick_th + pick_st
    pct = [100 * p / max(1, f) for p, f in zip(pick_v, full_v)]
    order = np.argsort(pct)
    cols = ["#3a7" if pct[i] > 50 else ("#fa3" if pct[i] > 30 else "#c33") for i in order]
    ax.barh([cn_all[i] for i in order], [pct[i] for i in order], color=cols, edgecolor="black", linewidth=0.4)
    ax.axvline(30, color="grey", linestyle="--", alpha=0.5, label="30% (= budget size)")
    ax.set_xlabel("% of instances / voxels retained in curated v2")
    ax.set_title(f"How well each class is preserved in the 255-scene curated subset (30% scene budget)")
    for y, i in enumerate(order):
        ax.text(pct[i] + 1, y, f"{pct[i]:.0f}%", va="center", fontsize=8)
    ax.legend()
    fig.savefig(figdir / "12_curated_retention_by_class.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # ----- print summary table -----
    print("\n=== Summary ===")
    print(f"Total: {len(pick_rows)} / {len(full_rows)} scenes  ({100*len(pick_rows)/len(full_rows):.1f}%)")
    print(f"Total samples (keyframes): {sum(r['num_samples'] for r in pick_rows):,} / {sum(r['num_samples'] for r in full_rows):,}")
    print(f"Anchored for rare classes: {len(anchor_provenance)}  detail:")
    anchor_cls_count = Counter(anchor_provenance.values())
    for cn, k in anchor_cls_count.items():
        print(f"    {cn}: {k}")
    print(f"\nClass retention (boxes/voxels):")
    for cn, f, p in zip(THING_CLASSES, full_th, pick_th):
        print(f"    th_{cn:<14} {p:>10,} / {f:>10,}  ({100*p/max(1,f):>5.1f}%)")
    for cn, f, p in zip(STUFF_CLASSES, full_st, pick_st):
        print(f"    vx_{cn:<14} {p:>13,} / {f:>13,}  ({100*p/max(1,f):>5.1f}%)")


if __name__ == "__main__":
    main()

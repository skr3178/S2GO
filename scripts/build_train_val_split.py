"""
Build clean train/val curated subsets using the canonical nuScenes 700/150 split:

  curated_train_270/  — 270 scenes drawn from the 700 train candidates
  curated_val_50/     —  50 scenes drawn from the 150 paper-val candidates

For each subset, runs:
  1. Anchor + greedy curation (smaller anchor counts for val)
  2. Per-class scene-count audit
  3. Full per-scene log (scene_log.csv)
  4. Comparison figures (08-12) vs the corresponding candidate pool

Disjointness train ∩ val = ∅ is guaranteed by construction.
"""
import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from nuscenes.utils.splits import train as TRAIN_NAMES, val as VAL_NAMES


# Import the curator's helpers
sys.path.insert(0, str(Path(__file__).parent))
from curate_anchor_greedy import (
    load_scene_rows, class_feat, cat_indicator,
    THING_CLASSES, STUFF_CLASSES, ALL_CLASSES,
    LOCATIONS, VEHICLES, EXTRA_TAGS,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- helpers shared with curate_anchor_greedy.py ----------

def run_curation(rows_subset, budget, anchor_plan, w_cat=2.0):
    """Return (selection_order_list, anchor_provenance_dict)."""
    picked = set()
    anchor_provenance = {}
    for col, k in anchor_plan:
        ranked = sorted(rows_subset, key=lambda r: -r[col])
        added = 0
        for r in ranked:
            if added == k: break
            if r["scene_token"] in picked: continue
            if r[col] <= 0:
                # no more scenes containing this class — stop trying
                break
            picked.add(r["scene_token"])
            anchor_provenance[r["scene_token"]] = col.replace("th_", "")
            added += 1

    cls_dim = len(ALL_CLASSES)
    cat_dim = len(LOCATIONS) + len(VEHICLES) + len(EXTRA_TAGS)
    w_cls = np.ones(cls_dim, dtype=np.float64)
    w_cat_v = np.full(cat_dim, w_cat, dtype=np.float64)

    feats_cls = np.stack([class_feat(r) for r in rows_subset])
    feats_cat = np.stack([cat_indicator(r) for r in rows_subset])
    F_cls = np.zeros(cls_dim) + 1e-9
    F_cat = np.zeros(cat_dim) + 1e-9
    tok2idx = {r["scene_token"]: i for i, r in enumerate(rows_subset)}
    for tok in picked:
        i = tok2idx[tok]
        F_cls += feats_cls[i]; F_cat += feats_cat[i]
    available = np.array([1.0 if r["scene_token"] not in picked else 0.0 for r in rows_subset])
    selection_order = list(picked)
    while len(picked) < min(budget, len(rows_subset)):
        new_cls = F_cls + feats_cls
        new_cat = F_cat + feats_cat
        g = (np.sqrt(new_cls) - np.sqrt(F_cls)).dot(w_cls) + (np.sqrt(new_cat) - np.sqrt(F_cat)).dot(w_cat_v)
        g = g * available
        b = int(np.argmax(g))
        if available[b] == 0 or g[b] <= 0:
            break
        picked.add(rows_subset[b]["scene_token"])
        selection_order.append(rows_subset[b]["scene_token"])
        F_cls += feats_cls[b]; F_cat += feats_cat[b]
        available[b] = 0
    return selection_order, anchor_provenance


def write_outputs(outdir: Path, rows_pool, selection_order, anchor_provenance,
                  candidate_label: str):
    """Write tokens.json, scenes.csv, scene_log.csv, per_class_scene_counts.csv,
    plus 5 comparison figures vs `rows_pool` (the candidate pool, not full 850)."""
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)

    pick_set = set(selection_order)
    pick_rows = [r for r in rows_pool if r["scene_token"] in pick_set]
    # ensure pick_rows in pick_order
    order_idx = {t: i for i, t in enumerate(selection_order)}
    pick_rows.sort(key=lambda r: order_idx[r["scene_token"]])

    # tokens.json (in pick order)
    (outdir / "tokens.json").write_text(json.dumps(selection_order))

    # scenes.csv (compact)
    with open(outdir / "scenes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pick_order", "scene_token", "name", "part", "location", "vehicle",
                    "num_samples", "anchor_for", "description"])
        for r in pick_rows:
            w.writerow([order_idx[r["scene_token"]], r["scene_token"], r["name"],
                        r.get("part", 0), r["location"], r["vehicle"], r["num_samples"],
                        anchor_provenance.get(r["scene_token"], ""),
                        r["description"].replace(",", ";")])

    # scene_log.csv (full per-scene)
    log_fields = ["pick_order", "scene_token", "name", "part", "location", "vehicle",
                  "num_samples", "anchor_for",
                  "night", "rain", "construction", "busy",
                  "description"] + \
                 [f"th_{c}" for c in THING_CLASSES] + \
                 [f"vx_{c}" for c in STUFF_CLASSES]
    with open(outdir / "scene_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_fields)
        w.writeheader()
        for r in pick_rows:
            row = {
                "pick_order": order_idx[r["scene_token"]],
                "scene_token": r["scene_token"],
                "name": r["name"],
                "part": r.get("part", 0),
                "location": r["location"],
                "vehicle": r["vehicle"],
                "num_samples": r["num_samples"],
                "anchor_for": anchor_provenance.get(r["scene_token"], ""),
                "night": r.get("tag_night", False),
                "rain": r.get("tag_rain", False),
                "construction": r.get("tag_construction", False),
                "busy": r.get("tag_busy", False),
                "description": r["description"],
            }
            for c in THING_CLASSES: row[f"th_{c}"] = r[f"th_{c}"]
            for c in STUFF_CLASSES: row[f"vx_{c}"] = r[f"vx_{c}"]
            w.writerow(row)

    # per_class_scene_counts.csv
    with open(outdir / "per_class_scene_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "kind", f"scenes_in_curated", "instances_in_curated",
                    f"scenes_in_{candidate_label}_pool", f"instances_in_{candidate_label}_pool",
                    "retention_pct"])
        for cn in THING_CLASSES:
            cur = [int(r[f"th_{cn}"]) for r in pick_rows]
            pool = [int(r[f"th_{cn}"]) for r in rows_pool]
            w.writerow([cn, "thing(boxes)",
                        sum(1 for v in cur if v > 0), sum(cur),
                        sum(1 for v in pool if v > 0), sum(pool),
                        f"{100*sum(cur)/max(1,sum(pool)):.1f}"])
        for cn in STUFF_CLASSES:
            cur = [int(r[f"vx_{cn}"]) for r in pick_rows]
            pool = [int(r[f"vx_{cn}"]) for r in rows_pool]
            w.writerow([cn, "stuff(voxels)",
                        sum(1 for v in cur if v > 0), sum(cur),
                        sum(1 for v in pool if v > 0), sum(pool),
                        f"{100*sum(cur)/max(1,sum(pool)):.1f}"])

    # ---------- figures ----------
    LOC_SHORT = {"boston-seaport": "boston", "singapore-onenorth": "sg-on",
                 "singapore-queenstown": "sg-qt", "singapore-hollandvillage": "sg-hv"}

    def count_by(rows_, key):
        return Counter(r[key] for r in rows_)
    def tag_count(rows_, tag):
        return sum(1 for r in rows_ if r.get(f"tag_{tag}", False))
    def sum_col(rows_, col):
        return sum(int(r[col]) for r in rows_)

    # Fig 8: per-part origin
    parts = sorted({r.get("part", 0) for r in rows_pool})
    parts_pool = Counter(r.get("part", 0) for r in rows_pool)
    parts_pick = Counter(r.get("part", 0) for r in pick_rows)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    xpos = np.arange(len(parts)); ww = 0.4
    ax.bar(xpos - ww/2, [parts_pool[p] for p in parts], ww, label=f"{candidate_label} pool", color="#bbb", edgecolor="black", linewidth=0.4)
    ax.bar(xpos + ww/2, [parts_pick.get(p, 0) for p in parts], ww, label=f"curated ({len(pick_rows)})", color="#37a", edgecolor="black", linewidth=0.4)
    for x, p in zip(xpos + ww/2, parts):
        ax.text(x, parts_pick.get(p, 0) + 0.5, str(parts_pick.get(p, 0)), ha="center", fontsize=8)
    ax.set_xticks(xpos); ax.set_xticklabels(parts); ax.set_xlabel("archive part"); ax.set_ylabel("# scenes")
    ax.set_title(f"Per-archive origin: {candidate_label} pool vs curated")
    ax.legend()
    fig.savefig(figdir / "08_curated_origin_per_part.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # Fig 9: location/vehicle composition
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    full_loc = count_by(rows_pool, "location"); pick_loc = count_by(pick_rows, "location")
    labels = [LOC_SHORT[l] for l in LOCATIONS]
    full_pct = [100 * full_loc.get(l, 0) / len(rows_pool) for l in LOCATIONS]
    pick_pct = [100 * pick_loc.get(l, 0) / len(pick_rows) for l in LOCATIONS]
    xp = np.arange(len(LOCATIONS))
    axes[0].bar(xp - 0.2, full_pct, 0.4, label=f"{candidate_label} pool", color="#bbb", edgecolor="black", linewidth=0.4)
    axes[0].bar(xp + 0.2, pick_pct, 0.4, label="curated", color="#37a", edgecolor="black", linewidth=0.4)
    for xi, v in zip(xp - 0.2, full_pct): axes[0].text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    for xi, v in zip(xp + 0.2, pick_pct): axes[0].text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    axes[0].set_xticks(xp); axes[0].set_xticklabels(labels); axes[0].set_ylabel("% of scenes")
    axes[0].set_title(f"Location share: {candidate_label} pool vs curated"); axes[0].legend()
    axes[0].set_ylim(0, max(max(full_pct), max(pick_pct)) * 1.2)
    full_v = count_by(rows_pool, "vehicle"); pick_v = count_by(pick_rows, "vehicle")
    xp = np.arange(len(VEHICLES))
    full_pct = [100 * full_v.get(v, 0) / len(rows_pool) for v in VEHICLES]
    pick_pct = [100 * pick_v.get(v, 0) / len(pick_rows) for v in VEHICLES]
    axes[1].bar(xp - 0.2, full_pct, 0.4, label="pool", color="#bbb", edgecolor="black", linewidth=0.4)
    axes[1].bar(xp + 0.2, pick_pct, 0.4, label="curated", color="#37a", edgecolor="black", linewidth=0.4)
    for xi, v in zip(xp - 0.2, full_pct): axes[1].text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    for xi, v in zip(xp + 0.2, pick_pct): axes[1].text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    axes[1].set_xticks(xp); axes[1].set_xticklabels(VEHICLES); axes[1].set_ylabel("% of scenes")
    axes[1].set_title("Recording vehicle share"); axes[1].legend()
    axes[1].set_ylim(0, max(max(full_pct), max(pick_pct)) * 1.2)
    fig.savefig(figdir / "09_curated_location_vehicle.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # Fig 10: condition coverage
    fig, ax = plt.subplots(figsize=(12, 5))
    tags = ["night", "rain", "construction", "busy", "intersection", "crosswalk", "jaywalker", "bus_intxn"]
    full_pct = [100 * tag_count(rows_pool, t) / len(rows_pool) for t in tags]
    pick_pct = [100 * tag_count(pick_rows, t) / len(pick_rows) for t in tags]
    xt = np.arange(len(tags))
    ax.bar(xt - 0.2, full_pct, 0.4, label=f"{candidate_label} pool", color="#bbb", edgecolor="black", linewidth=0.4)
    ax.bar(xt + 0.2, pick_pct, 0.4, label="curated", color="#37a", edgecolor="black", linewidth=0.4)
    for xi, v in zip(xt - 0.2, full_pct): ax.text(xi, v + 0.5, f"{v:.0f}%", ha="center", fontsize=8)
    for xi, v in zip(xt + 0.2, pick_pct): ax.text(xi, v + 0.5, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(xt); ax.set_xticklabels(tags, rotation=15); ax.set_ylabel("% of scenes with tag")
    ax.set_title(f"Condition coverage: {candidate_label} pool vs curated")
    ax.legend()
    fig.savefig(figdir / "10_curated_conditions.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # Fig 11: class counts log
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    pool_th = [sum_col(rows_pool, f"th_{cn}") for cn in THING_CLASSES]
    pick_th = [sum_col(pick_rows, f"th_{cn}") for cn in THING_CLASSES]
    xc = np.arange(len(THING_CLASSES))
    axes[0].bar(xc - 0.2, pool_th, 0.4, label="pool", color="#bbb", edgecolor="black", linewidth=0.4)
    axes[0].bar(xc + 0.2, pick_th, 0.4, label="curated", color="#37a", edgecolor="black", linewidth=0.4)
    axes[0].set_yscale("log"); axes[0].set_xticks(xc); axes[0].set_xticklabels(THING_CLASSES, rotation=30, ha="right")
    axes[0].set_ylabel("# boxes (log)"); axes[0].set_title(f"Thing-class counts: {candidate_label} pool vs curated"); axes[0].legend()
    for xi, fv, pv in zip(xc, pool_th, pick_th):
        pct = 100 * pv / max(1, fv); axes[0].text(xi + 0.2, pv * 1.15, f"{pct:.0f}%", ha="center", fontsize=8)
    pool_st = [sum_col(rows_pool, f"vx_{cn}") for cn in STUFF_CLASSES]
    pick_st = [sum_col(pick_rows, f"vx_{cn}") for cn in STUFF_CLASSES]
    xc = np.arange(len(STUFF_CLASSES))
    axes[1].bar(xc - 0.2, pool_st, 0.4, label="pool", color="#bbb", edgecolor="black", linewidth=0.4)
    axes[1].bar(xc + 0.2, pick_st, 0.4, label="curated", color="#37a", edgecolor="black", linewidth=0.4)
    axes[1].set_yscale("log"); axes[1].set_xticks(xc); axes[1].set_xticklabels(STUFF_CLASSES, rotation=30, ha="right")
    axes[1].set_ylabel("# voxels (log)"); axes[1].set_title("Stuff-class voxel counts"); axes[1].legend()
    for xi, fv, pv in zip(xc, pool_st, pick_st):
        pct = 100 * pv / max(1, fv); axes[1].text(xi + 0.2, pv * 1.15, f"{pct:.0f}%", ha="center", fontsize=8)
    fig.savefig(figdir / "11_curated_class_counts.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # Fig 12: retention by class
    fig, ax = plt.subplots(figsize=(11, 5))
    cn_all = THING_CLASSES + STUFF_CLASSES
    pool_v = pool_th + pool_st; pick_v = pick_th + pick_st
    pct = [100 * p / max(1, fv) for p, fv in zip(pick_v, pool_v)]
    order = np.argsort(pct)
    cols = ["#3a7" if pct[i] > 50 else ("#fa3" if pct[i] > 30 else "#c33") for i in order]
    ax.barh([cn_all[i] for i in order], [pct[i] for i in order], color=cols, edgecolor="black", linewidth=0.4)
    for y, i in enumerate(order):
        ax.text(pct[i] + 1, y, f"{pct[i]:.0f}%", va="center", fontsize=8)
    budget_pct = 100 * len(pick_rows) / len(rows_pool)
    ax.axvline(budget_pct, color="grey", linestyle="--", alpha=0.5, label=f"budget={budget_pct:.0f}% (uniform)")
    ax.set_xlabel(f"% of pool retained in curated")
    ax.set_title(f"Class retention in curated subset ({len(pick_rows)}/{len(rows_pool)} scenes)")
    ax.legend()
    fig.savefig(figdir / "12_curated_retention_by_class.png", dpi=130, bbox_inches="tight"); plt.close(fig)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-csv", required=True)
    ap.add_argument("--class-csv", required=True)
    ap.add_argument("--out-train", required=True, help="output dir for curated_train_270")
    ap.add_argument("--out-val", required=True, help="output dir for curated_val_50")
    ap.add_argument("--train-budget", type=int, default=270)
    ap.add_argument("--val-budget", type=int, default=50)
    args = ap.parse_args()

    rows = load_scene_rows(args.class_csv, args.scene_csv)
    name2tok = {r["name"]: r["scene_token"] for r in rows}
    train_tokens = {name2tok[n] for n in TRAIN_NAMES if n in name2tok}
    val_tokens   = {name2tok[n] for n in VAL_NAMES   if n in name2tok}
    print(f"[split] 700 train names ({len(train_tokens)} matched), 150 val names ({len(val_tokens)} matched)")
    assert train_tokens.isdisjoint(val_tokens)

    train_rows = [r for r in rows if r["scene_token"] in train_tokens]
    val_rows   = [r for r in rows if r["scene_token"] in val_tokens]

    # ----- TRAIN: 270 from 700 -----
    train_anchors = [
        ("th_trailer", 6), ("th_const_veh", 6), ("th_bicycle", 5),
        ("th_motorcycle", 5), ("th_barrier", 4), ("th_bus", 4), ("th_traffic_cone", 3),
    ]
    print(f"\n[train] anchor + greedy on 700 candidates, budget = {args.train_budget}")
    train_order, train_anchors_prov = run_curation(train_rows, args.train_budget, train_anchors)
    write_outputs(Path(args.out_train), train_rows, train_order, train_anchors_prov, "train(700)")
    print(f"[train] -> {args.out_train}  ({len(train_order)} scenes)")

    # ----- VAL: 50 from 150 -----
    # Smaller anchor counts since the pool is only 150 scenes
    val_anchors = [
        ("th_trailer", 2), ("th_const_veh", 2), ("th_bicycle", 2),
        ("th_motorcycle", 2), ("th_barrier", 2), ("th_bus", 2), ("th_traffic_cone", 2),
    ]
    print(f"\n[val] anchor + greedy on 150 candidates, budget = {args.val_budget}")
    val_order, val_anchors_prov = run_curation(val_rows, args.val_budget, val_anchors)
    write_outputs(Path(args.out_val), val_rows, val_order, val_anchors_prov, "val(150)")
    print(f"[val] -> {args.out_val}  ({len(val_order)} scenes)")

    # ----- disjointness ----
    overlap = set(train_order) & set(val_order)
    print(f"\n[verify] train ∩ val overlap: {len(overlap)} (expected 0)")
    assert len(overlap) == 0


if __name__ == "__main__":
    main()

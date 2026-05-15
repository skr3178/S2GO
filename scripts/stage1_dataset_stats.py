"""Read-only stats over the on-disk nuScenes for Stage-1 curation planning.

Mirrors the data-source conventions of scripts/make_full_stage2_splits.py
(same DATAROOT, same nuscenes-devkit usage). Emits both:

    - a stdout summary the user can paste into a chat
    - out/stage1_dataset_stats.json (machine-readable, for the follow-up
      curation script)

Run:
    cd /home/satya/skr/S2GO/S2GO
    python scripts/stage1_dataset_stats.py

Expected runtime: ~30-90 s. No GPU. <1 GB RAM. Read-only — does not touch
data/.
"""
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


DATAROOT = "/home/satya/skr/S2GO/S2GO/data/nuscenes"
OCC_ROOT = "/home/satya/skr/S2GO/S2GO/data/nuscenes_occ/nuscenes_occ/samples"
OUT_PATH = "out/stage1_dataset_stats.json"

T = 4
CAM_NAMES = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
             "CAM_BACK",  "CAM_BACK_LEFT",  "CAM_BACK_RIGHT"]
SENSORS_FOR_LOADER = CAM_NAMES + ["LIDAR_TOP"]

# Description-keyword groups. Word-boundary, case-insensitive. Keep simple
# so the stats line up with the user's exploration counts.
KEYWORD_GROUPS = {
    "rain":         r"\b(rain|raining|wet)\b",
    "night":        r"\bnight\b",
    "intersection": r"\bintersection\b",
    "turn":         r"\b(turn|turning|maneuver|maneuvering)\b",
    "ped":          r"\b(ped|peds|pedestrian|pedestrians|jaywalker|jaywalkers|crossing)\b",
    "construction": r"\b(construction|truck|trucks)\b",
    "parking":      r"\b(parking|parked)\b",
}
COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in KEYWORD_GROUPS.items()}
COOC_AXES = ["rain", "night", "intersection", "construction"]


def percentile(sorted_arr, p):
    if not sorted_arr:
        return 0
    i = max(0, min(len(sorted_arr) - 1, int(round(len(sorted_arr) * p / 100))))
    return sorted_arr[i]


def main() -> int:
    t0 = time.time()

    # ── A. On-disk reality ─────────────────────────────────────────────
    keyframes_per_sensor = {}
    for s in SENSORS_FOR_LOADER:
        d = os.path.join(DATAROOT, "samples", s)
        keyframes_per_sensor[s] = len(os.listdir(d)) if os.path.isdir(d) else 0
    print("On-disk samples per sensor:")
    for s, c in keyframes_per_sensor.items():
        print(f"  {s:20s} {c:>6d}")

    # ── Load nuScenes ──────────────────────────────────────────────────
    print(f"\nLoading NuScenes(v1.0-trainval) from {DATAROOT}…", flush=True)
    nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
    print(f"  {len(nusc.scene)} scenes, {len(nusc.sample)} samples, {len(nusc.log)} logs")

    # ── Official splits ────────────────────────────────────────────────
    official = create_splits_scenes()
    train_set = set(official["train"])
    val_set = set(official["val"])
    print(f"  official splits: train={len(train_set)} scenes, val={len(val_set)} scenes")

    # ── Pre-index Stage-2 GT npy filenames ─────────────────────────────
    if os.path.isdir(OCC_ROOT):
        occ_files = set(os.listdir(OCC_ROOT))
    else:
        occ_files = set()
    print(f"  SurroundOcc GT .npy indexed: {len(occ_files)} (root: {OCC_ROOT})")

    # ── Per-scene scan ─────────────────────────────────────────────────
    print(f"\nScanning {len(nusc.scene)} scenes (on-disk presence + GT + T={T} yield)…")
    scene_stats = []
    for sci, scene in enumerate(nusc.scene):
        if sci % 100 == 0:
            print(f"  {sci}/{len(nusc.scene)}…", flush=True)
        name = scene["name"]
        log = nusc.get("log", scene["log_token"])

        present = []  # bool per sample, in scene order — same condition the loader checks
        gt_present = 0
        n_keyframes = 0
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            n_keyframes += 1
            # Match nusc_loader.py NuScenesLoader._all_files_present(): all
            # sensor files must exist on disk for the sample to be usable.
            ok = True
            for sd_tok in sample["data"].values():
                sd = nusc.get("sample_data", sd_tok)
                if not os.path.exists(os.path.join(DATAROOT, sd["filename"])):
                    ok = False
                    break
            present.append(ok)
            # Stage-2 SurroundOcc GT presence: indexed by LIDAR_TOP filename + ".npy"
            ldata = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            if (os.path.basename(ldata["filename"]) + ".npy") in occ_files:
                gt_present += 1
            sample_token = sample["next"]

        # T contiguous-sample yield: number of i where present[i:i+T] all True
        t4_yield = 0
        if len(present) >= T:
            for i in range(len(present) - T + 1):
                if all(present[i:i + T]):
                    t4_yield += 1

        if name in train_set:
            split = "train"
        elif name in val_set:
            split = "val"
        else:
            split = "other"

        scene_stats.append({
            "name": name,
            "split": split,
            "location": log["location"],
            "date_captured": log["date_captured"],
            "description": scene["description"],
            "n_keyframes": n_keyframes,
            "n_present": sum(present),
            "fully_present": all(present),
            "t4_yield": t4_yield,
            "stage2_gt_count": gt_present,
            "stage2_gt_full": (gt_present == n_keyframes and n_keyframes > 0),
        })

    # ── Aggregate ──────────────────────────────────────────────────────
    by_split = Counter(s["split"] for s in scene_stats)

    by_location = defaultdict(Counter)
    for s in scene_stats:
        by_location[s["location"]][s["split"]] += 1

    by_date = defaultdict(Counter)
    for s in scene_stats:
        ym = (s["date_captured"] or "unknown")[:7]
        by_date[ym][s["split"]] += 1

    # Keyword scene-membership sets
    kw_match = {
        k: {s["name"] for s in scene_stats if pat.search(s["description"] or "")}
        for k, pat in COMPILED.items()
    }

    keyword_stats = {}
    name_to_scene = {s["name"]: s for s in scene_stats}
    for k, names in kw_match.items():
        kw_split = Counter()
        kw_loc = defaultdict(Counter)
        for n in names:
            s = name_to_scene[n]
            kw_split[s["split"]] += 1
            kw_loc[s["location"]][s["split"]] += 1
        keyword_stats[k] = {
            "total": len(names),
            "by_split": dict(kw_split),
            "by_location": {loc: dict(c) for loc, c in kw_loc.items()},
        }

    # Co-occurrence
    cooc = {}
    for ax in COOC_AXES:
        cooc[ax] = len(kw_match[ax])
    for i, a in enumerate(COOC_AXES):
        for b in COOC_AXES[i + 1:]:
            cooc[f"{a}&{b}"] = len(kw_match[a] & kw_match[b])
    cooc["rain&night&intersection"] = len(kw_match["rain"] & kw_match["night"] & kw_match["intersection"])
    cooc["rain&night&construction"] = len(kw_match["rain"] & kw_match["night"] & kw_match["construction"])

    # GT coverage at scene-level
    gt_full = sum(1 for s in scene_stats if s["stage2_gt_full"])
    gt_partial = sum(1 for s in scene_stats if 0 < s["stage2_gt_count"] < s["n_keyframes"])
    gt_none = sum(1 for s in scene_stats if s["stage2_gt_count"] == 0)

    # Per-scene yield distribution
    yields_pos = sorted(s["t4_yield"] for s in scene_stats if s["t4_yield"] > 0)
    n_keyframes_dist = sorted(s["n_keyframes"] for s in scene_stats)
    n_present_total = sum(s["n_present"] for s in scene_stats)
    t4_total = sum(s["t4_yield"] for s in scene_stats)

    t4_by_split = Counter()
    for s in scene_stats:
        t4_by_split[s["split"]] += s["t4_yield"]

    # ── Build JSON output ──────────────────────────────────────────────
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "T": T,
        "dataroot": DATAROOT,
        "occ_root": OCC_ROOT,
        "on_disk": {
            "keyframes_per_sensor": keyframes_per_sensor,
            "samples_with_all_sensors": n_present_total,
            "samples_total_in_metadata": len(nusc.sample),
            "scenes_fully_present": sum(1 for s in scene_stats if s["fully_present"]),
            "scenes_partially_present": sum(1 for s in scene_stats
                                            if (not s["fully_present"]) and s["n_present"] > 0),
            "scenes_absent": sum(1 for s in scene_stats if s["n_present"] == 0),
            f"T{T}_sequences_total": t4_total,
            f"T{T}_sequences_by_split": dict(t4_by_split),
        },
        "official_split": {
            "train_scenes_devkit": len(train_set),
            "val_scenes_devkit": len(val_set),
            "observed_train": by_split["train"],
            "observed_val": by_split["val"],
            "observed_other": by_split["other"],
        },
        "by_location": {loc: dict(c) for loc, c in by_location.items()},
        "by_date": {dt: dict(c) for dt, c in sorted(by_date.items())},
        "keywords": keyword_stats,
        "cooccurrence": cooc,
        "stage2_gt_coverage": {
            "scenes_full_gt": gt_full,
            "scenes_partial_gt": gt_partial,
            "scenes_no_gt": gt_none,
            "total_npy_indexed": len(occ_files),
        },
        "per_scene_n_keyframes": {
            "min": n_keyframes_dist[0] if n_keyframes_dist else 0,
            "p25": percentile(n_keyframes_dist, 25),
            "median": percentile(n_keyframes_dist, 50),
            "p75": percentile(n_keyframes_dist, 75),
            "max": n_keyframes_dist[-1] if n_keyframes_dist else 0,
            "mean": (sum(n_keyframes_dist) / len(n_keyframes_dist)) if n_keyframes_dist else 0,
        },
        f"per_scene_T{T}_yield": {
            "min": yields_pos[0] if yields_pos else 0,
            "p25": percentile(yields_pos, 25),
            "median": percentile(yields_pos, 50),
            "p75": percentile(yields_pos, 75),
            "max": yields_pos[-1] if yields_pos else 0,
            "n_scenes_with_yield_gt0": len(yields_pos),
            "total_sequences": t4_total,
        },
        # Full per-scene table at the bottom — small (~850 rows) and useful
        # for the follow-up curation script.
        "scenes": scene_stats,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    # ── stdout summary (paste-able) ────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE-1 DATASET STATISTICS")
    print("=" * 70)

    print("\nA. ON-DISK REALITY")
    print(f"  CAM_FRONT files:                   {keyframes_per_sensor.get('CAM_FRONT', 0):>6d}")
    print(f"  Samples with all {len(SENSORS_FOR_LOADER)} loader sensors:  "
          f"{n_present_total:>6d} / {len(nusc.sample)}")
    print(f"  Scenes fully extracted:            {out['on_disk']['scenes_fully_present']:>6d}"
          f" / {len(nusc.scene)}")
    print(f"  Scenes partially present:          {out['on_disk']['scenes_partially_present']:>6d}")
    print(f"  Scenes absent:                     {out['on_disk']['scenes_absent']:>6d}")
    print(f"  T={T} contiguous sequences total:  {t4_total:>6d}"
          f"  ← what NuScenesLoader actually yields")
    print(f"     train={t4_by_split['train']}, val={t4_by_split['val']}, "
          f"other={t4_by_split['other']}")

    print("\nB. OFFICIAL SPLIT (nuscenes-devkit)")
    print(f"  Train scenes: {len(train_set)}    Val scenes: {len(val_set)}    "
          f"Other (test/etc.): {by_split['other']}")

    print("\nC. SCENES BY LOCATION (train / val)")
    for loc in sorted(by_location, key=lambda l: -sum(by_location[l].values())):
        c = by_location[loc]
        print(f"  {loc:32s} train={c['train']:>3d}  val={c['val']:>3d}"
              f"  total={sum(c.values()):>3d}")

    print("\nD. SCENES BY DATE (year-month)")
    for ym in sorted(by_date):
        c = by_date[ym]
        print(f"  {ym:10s} train={c['train']:>3d}  val={c['val']:>3d}  "
              f"total={sum(c.values()):>3d}")

    print("\nE. DESCRIPTION KEYWORDS")
    print(f"  {'keyword':14s} {'total':>6s} {'train':>6s} {'val':>6s}   by-location")
    for k, ks in keyword_stats.items():
        bs = ks["by_split"]
        loc_str = ", ".join(
            f"{loc.split('-')[0][:4]}={sum(c.values())}"
            for loc, c in sorted(ks["by_location"].items())
        )
        print(f"  {k:14s} {ks['total']:>6d} {bs.get('train', 0):>6d} {bs.get('val', 0):>6d}"
              f"   {loc_str}")

    print("\nF. CO-OCCURRENCE (scene counts, all splits)")
    for k, v in cooc.items():
        print(f"  {k:32s} {v:>4d}")

    print("\nG. STAGE-2 GT (SurroundOcc) COVERAGE")
    print(f"  Scenes with FULL Stage-2 GT:      {gt_full:>4d}    ← reusable for Stage-2 fine-tune")
    print(f"  Scenes with PARTIAL Stage-2 GT:   {gt_partial:>4d}")
    print(f"  Scenes with NO   Stage-2 GT:      {gt_none:>4d}")
    print(f"  Total .npy indexed in occ_root:   {len(occ_files):>4d}")

    print(f"\nH. PER-SCENE KEYFRAME COUNT and T={T} SEQUENCE YIELD")
    psk = out["per_scene_n_keyframes"]
    pst = out[f"per_scene_T{T}_yield"]
    print(f"  n_keyframes:  min={psk['min']}, p25={psk['p25']}, median={psk['median']}, "
          f"p75={psk['p75']}, max={psk['max']}, mean={psk['mean']:.1f}")
    print(f"  T={T} yield:  min={pst['min']}, p25={pst['p25']}, median={pst['median']}, "
          f"p75={pst['p75']}, max={pst['max']}")
    print(f"  Scenes contributing >0 sequences: {pst['n_scenes_with_yield_gt0']}")

    dt = time.time() - t0
    print(f"\nWrote {OUT_PATH}  (elapsed: {dt:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

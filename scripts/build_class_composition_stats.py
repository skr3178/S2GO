"""
Build per-scene SurroundOcc 16-class composition statistics.

Combines two sources:
- Thing classes 1..10 (barrier..truck): instance counts from
  v1.0-trainval/sample_annotation.json via the nuScenes->SurroundOcc category map.
- Stuff classes 11..16 (drive.suf..vegetation): voxel counts from the per-sample
  SurroundOcc voxel GT .npy files (sparse (N,4) arrays with class_id in 0..16).

Aggregates per-sample stats into per-scene totals and writes:
- <outdir>/class_composition_per_scene.csv
- <outdir>/class_composition_summary.txt
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# ---------- SurroundOcc 16-class taxonomy ----------
CLASS_NAMES = {
    1: "barrier",
    2: "bicycle",
    3: "bus",
    4: "car",
    5: "const_veh",
    6: "motorcycle",
    7: "pedestrian",
    8: "traffic_cone",
    9: "trailer",
    10: "truck",
    11: "drive_surf",
    12: "other_flat",
    13: "sidewalk",
    14: "terrain",
    15: "manmade",
    16: "vegetation",
}
THING_CLASSES = list(range(1, 11))
STUFF_CLASSES = list(range(11, 17))

# nuScenes category -> SurroundOcc thing class id (1..10).
# Categories not listed (animal, debris, push/pull, bicycle_rack, etc.)
# are not counted toward any SurroundOcc class.
NUSC_CAT_TO_CLS = {
    "movable_object.barrier": 1,
    "vehicle.bicycle": 2,
    "vehicle.bus.bendy": 3,
    "vehicle.bus.rigid": 3,
    "vehicle.car": 4,
    "vehicle.emergency.ambulance": 4,
    "vehicle.emergency.police": 4,
    "vehicle.construction": 5,
    "vehicle.motorcycle": 6,
    "human.pedestrian.adult": 7,
    "human.pedestrian.child": 7,
    "human.pedestrian.construction_worker": 7,
    "human.pedestrian.police_officer": 7,
    "human.pedestrian.stroller": 7,
    "human.pedestrian.wheelchair": 7,
    "human.pedestrian.personal_mobility": 7,
    "movable_object.trafficcone": 8,
    "vehicle.trailer": 9,
    "vehicle.truck": 10,
}


# ---------- workers ----------

def count_voxel_classes(npy_path: str):
    """Return (sample_filename_stem, class_id -> voxel_count) for one .npy."""
    try:
        arr = np.load(npy_path)
    except Exception as e:
        return os.path.basename(npy_path), {"_error": str(e)}
    if arr.ndim != 2 or arr.shape[1] != 4:
        return os.path.basename(npy_path), {"_error": f"bad_shape:{arr.shape}"}
    cls = arr[:, 3]
    u, c = np.unique(cls, return_counts=True)
    return os.path.basename(npy_path), {int(k): int(v) for k, v in zip(u, c)}


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--voxroot", required=True,
                    help="dir containing the per-sample SurroundOcc voxel .npy files")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 4))
    args = ap.parse_args()

    dataroot = Path(args.dataroot)
    voxroot = Path(args.voxroot)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = dataroot / "v1.0-trainval"

    print(f"[load] scene.json sample.json sample_data.json sample_annotation.json instance.json category.json")
    scenes = json.loads((meta / "scene.json").read_text())
    samples = json.loads((meta / "sample.json").read_text())
    print(f"  [load] sample_data.json (1.3GB) ...", flush=True)
    sample_data = json.loads((meta / "sample_data.json").read_text())
    print(f"  [load] sample_annotation.json (557MB) ...", flush=True)
    sample_annotation = json.loads((meta / "sample_annotation.json").read_text())
    instance = json.loads((meta / "instance.json").read_text())
    category = json.loads((meta / "category.json").read_text())
    cat_name_by_tok = {c["token"]: c["name"] for c in category}
    instance_cat = {ins["token"]: cat_name_by_tok[ins["category_token"]] for ins in instance}

    # samples grouped by scene; preserve temporal order
    samples_by_scene = defaultdict(list)
    for s in samples:
        samples_by_scene[s["scene_token"]].append(s)
    for k in samples_by_scene:
        samples_by_scene[k].sort(key=lambda r: r["timestamp"])

    # sample_token -> scene_token (used to roll annotations & voxels up)
    sample_to_scene = {s["token"]: s["scene_token"] for s in samples}

    # sample_token -> LIDAR_TOP keyframe sample_data record
    print(f"[map] sample -> LIDAR_TOP keyframe ...", flush=True)
    sample_lidar = {}
    for d in sample_data:
        if d["is_key_frame"] and "/LIDAR_TOP/" in d["filename"]:
            sample_lidar[d["sample_token"]] = d
    print(f"  matched lidar keyframes: {len(sample_lidar)}")

    # ---------- THING CLASSES (1..10) from sample_annotation.json ----------
    print(f"[count] thing classes from {len(sample_annotation):,} annotations")
    thing_counts = defaultdict(lambda: {c: 0 for c in THING_CLASSES})  # scene_token -> cls -> count
    unmapped_cat_counter = defaultdict(int)
    for ann in sample_annotation:
        scene_tok = sample_to_scene.get(ann["sample_token"])
        if scene_tok is None:
            continue
        cat_name = instance_cat.get(ann["instance_token"])
        if cat_name is None:
            continue
        cls = NUSC_CAT_TO_CLS.get(cat_name)
        if cls is None:
            unmapped_cat_counter[cat_name] += 1
            continue
        thing_counts[scene_tok][cls] += 1

    print(f"  unmapped categories (not in SurroundOcc taxonomy):")
    for cat, n in sorted(unmapped_cat_counter.items(), key=lambda kv: -kv[1]):
        print(f"    {cat:50s} {n}")

    # ---------- STUFF CLASSES (11..16) from voxel .npy files ----------
    # Build list of .npy files keyed by LIDAR_TOP filename stem
    print(f"[scan] {voxroot}")
    npy_files = sorted(str(p) for p in voxroot.glob("*.npy"))
    print(f"  found {len(npy_files):,} npy files")

    # Map lidar filename stem -> sample_token using sample_lidar
    # filename example: samples/LIDAR_TOP/n008-...__LIDAR_TOP__<ts>.pcd.bin
    # npy file       : n008-...__LIDAR_TOP__<ts>.pcd.bin.npy
    lidar_basename_to_sample = {
        Path(d["filename"]).name + ".npy": st for st, d in sample_lidar.items()
    }
    print(f"  lidar-basename keys: {len(lidar_basename_to_sample):,}")

    print(f"[count] stuff classes via {args.workers} workers ...", flush=True)
    voxel_counts_per_sample = {}  # sample_token -> {cls: count}
    with Pool(args.workers) as pool:
        for i, (fname, hist) in enumerate(
            pool.imap_unordered(count_voxel_classes, npy_files, chunksize=64)
        ):
            sample_tok = lidar_basename_to_sample.get(fname)
            if sample_tok is None:
                continue
            voxel_counts_per_sample[sample_tok] = hist
            if (i + 1) % 5000 == 0:
                print(f"    processed {i+1:>6}/{len(npy_files)}", flush=True)

    # Aggregate to scene
    stuff_counts = defaultdict(lambda: {c: 0 for c in STUFF_CLASSES})
    samples_with_voxels = defaultdict(int)
    for sample_tok, hist in voxel_counts_per_sample.items():
        scene_tok = sample_to_scene.get(sample_tok)
        if scene_tok is None:
            continue
        samples_with_voxels[scene_tok] += 1
        for c in STUFF_CLASSES:
            if c in hist:
                stuff_counts[scene_tok][c] += hist[c]

    # ---------- merge & write CSV ----------
    print(f"[write] per-scene CSV")
    scene_name = {sc["token"]: sc["name"] for sc in scenes}
    # logs map (for location/vehicle)
    logs = json.loads((meta / "log.json").read_text())
    log_by_tok = {l["token"]: l for l in logs}
    scene_meta = {sc["token"]: (log_by_tok[sc["log_token"]]["location"],
                                log_by_tok[sc["log_token"]]["vehicle"],
                                sc.get("description", ""),
                                sc["nbr_samples"]) for sc in scenes}

    csv_path = outdir / "class_composition_per_scene.csv"
    fieldnames = ["scene_token", "name", "location", "vehicle", "num_samples",
                  "n_lidar_samples_seen", "description"] + \
                 [f"th_{CLASS_NAMES[c]}" for c in THING_CLASSES] + \
                 [f"vx_{CLASS_NAMES[c]}" for c in STUFF_CLASSES]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for sc in scenes:
            tok = sc["token"]
            loc, veh, desc, nsamp = scene_meta[tok]
            row = {
                "scene_token": tok,
                "name": sc["name"],
                "location": loc,
                "vehicle": veh,
                "num_samples": nsamp,
                "n_lidar_samples_seen": samples_with_voxels.get(tok, 0),
                "description": desc.replace(",", ";"),
            }
            for c in THING_CLASSES:
                row[f"th_{CLASS_NAMES[c]}"] = thing_counts[tok][c]
            for c in STUFF_CLASSES:
                row[f"vx_{CLASS_NAMES[c]}"] = stuff_counts[tok][c]
            w.writerow(row)
    print(f"  -> {csv_path}")

    # ---------- summary report ----------
    print(f"[summary]")
    n_scenes = len(scenes)

    # 1) per-class totals across all 850 scenes
    print("\n=== Global per-class totals (across all 850 scenes) ===")
    print(f"{'cls':>3} {'name':<14} {'total':>14} {'scenes_with_cls':>15} {'mean_per_scene':>14}")
    rows_summary = []
    for c in THING_CLASSES:
        col = f"th_{CLASS_NAMES[c]}"
        vals = [thing_counts[sc["token"]][c] for sc in scenes]
        total = sum(vals)
        nonzero = sum(1 for v in vals if v > 0)
        mean_pos = total / max(1, nonzero)
        rows_summary.append((c, CLASS_NAMES[c], "thing(boxes)", total, nonzero, mean_pos))
        print(f"{c:>3} {CLASS_NAMES[c]:<14} {total:>14,} {nonzero:>15} {mean_pos:>14.1f}")
    for c in STUFF_CLASSES:
        vals = [stuff_counts[sc["token"]][c] for sc in scenes]
        total = sum(vals)
        nonzero = sum(1 for v in vals if v > 0)
        mean_pos = total / max(1, nonzero)
        rows_summary.append((c, CLASS_NAMES[c], "stuff(voxels)", total, nonzero, mean_pos))
        print(f"{c:>3} {CLASS_NAMES[c]:<14} {total:>14,} {nonzero:>15} {mean_pos:>14.1f}")

    # 2) rarest scenes for each rare class (which scenes carry the bulk?)
    print("\n=== Top-10 scenes for the 5 rarest thing classes (by instance count) ===")
    rarities = sorted(
        [(c, sum(thing_counts[sc["token"]][c] for sc in scenes)) for c in THING_CLASSES],
        key=lambda kv: kv[1])
    for c, _tot in rarities[:5]:
        ranked = sorted(scenes, key=lambda sc: -thing_counts[sc["token"]][c])[:10]
        print(f"\n  Class {c:>2} ({CLASS_NAMES[c]}): top 10 scenes")
        for sc in ranked:
            v = thing_counts[sc["token"]][c]
            if v == 0:
                continue
            print(f"    {sc['name']}  count={v}  '{sc.get('description','')[:80]}'")

    # 3) write summary file
    sum_path = outdir / "class_composition_summary.txt"
    with open(sum_path, "w") as f:
        f.write("=== Global per-class totals (across all 850 scenes) ===\n")
        f.write(f"{'cls':>3} {'name':<14} {'source':>14} {'total':>14} {'scenes_with_cls':>15} {'mean_per_scene>0':>16}\n")
        for c, name, src, total, nonzero, mean_pos in rows_summary:
            f.write(f"{c:>3} {name:<14} {src:>14} {total:>14,} {nonzero:>15} {mean_pos:>16.1f}\n")
        f.write("\nThing counts are 3D box instance counts (one row per box across the scene).\n")
        f.write("Stuff counts are voxel counts summed across all lidar keyframes in the scene.\n")
    print(f"\n  -> {sum_path}")


if __name__ == "__main__":
    main()

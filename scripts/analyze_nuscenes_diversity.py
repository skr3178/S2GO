"""
Analyze nuScenes trainval (10 parts, 850 scenes) along diversity axes, group by
which keyframes archive (part 01..10) each scene came from, and propose a
~255-scene curated subset that maximises coverage while fitting the size budget
of ~3 parts (~3/10 of total).

Scene -> part mapping is recovered by stat'ing the inode change time (ctime) of
each scene's CAM_FRONT keyframes: every tarball was extracted in a separate tar
invocation, so files from the same archive share a tight ctime cluster.

Usage:
    python scripts/analyze_nuscenes_diversity.py \
        --dataroot /media/skr/storage/self_driving/S2GO/data/nuscenes \
        --outdir   /media/skr/storage/self_driving/S2GO/data/nuscenes_dl/diversity
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
import csv

# ---------- description tagging ----------

DAY_NIGHT_RE = re.compile(r"\bnight\b", re.I)
RAIN_RE = re.compile(r"\b(rain|raining|wet|after rain)\b", re.I)
CONSTRUCTION_RE = re.compile(r"\b(construct|cone|barrier|roadwork)\b", re.I)
PEDS_RE = re.compile(r"\b(ped(estrian)?s?|crowd|crossing|jaywalk)\b", re.I)
BUSY_RE = re.compile(r"\b(busy|heavy traffic|many cars|congestion|crowded)\b", re.I)
TURN_RE = re.compile(r"\b(u-?turn|turn(ing)?|intersection)\b", re.I)
PARKED_RE = re.compile(r"\bparked\b", re.I)
BIKE_RE = re.compile(r"\b(bike|bicycle|motorbike|motorcycle|scooter)\b", re.I)
TRUCK_RE = re.compile(r"\b(truck|bus|trailer|construction vehicle)\b", re.I)


def describe_tags(desc: str) -> dict:
    return dict(
        night=bool(DAY_NIGHT_RE.search(desc)),
        rain=bool(RAIN_RE.search(desc)),
        construction=bool(CONSTRUCTION_RE.search(desc)),
        pedestrians=bool(PEDS_RE.search(desc)),
        busy=bool(BUSY_RE.search(desc)),
        turn=bool(TURN_RE.search(desc)),
        parked=bool(PARKED_RE.search(desc)),
        bikes=bool(BIKE_RE.search(desc)),
        trucks=bool(TRUCK_RE.search(desc)),
    )


# ---------- IO helpers ----------

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def index_by_token(records, key="token"):
    return {r[key]: r for r in records}


# ---------- part assignment via ctime clustering ----------

def assign_parts_by_ctime(scenes, samples_by_scene, sample_data_by_sample, dataroot: Path, max_per_scene=4):
    """For each scene, sample up to `max_per_scene` CAM_FRONT keyframes,
    take the median ctime, then cluster scenes by ctime into 10 parts.
    Returns dict scene_token -> part_index (1..10) or None if unmappable."""
    scene_ctime = {}
    for sc in scenes:
        sc_token = sc["token"]
        samps = samples_by_scene.get(sc_token, [])
        if not samps:
            continue
        # pick first few samples
        picks = samps[: max_per_scene]
        cts = []
        for s in picks:
            cam_front_data = sample_data_by_sample.get(s["token"], {}).get("CAM_FRONT")
            if not cam_front_data:
                continue
            fp = dataroot / cam_front_data["filename"]
            try:
                cts.append(fp.stat().st_ctime)
            except FileNotFoundError:
                continue
        if not cts:
            continue
        scene_ctime[sc_token] = sorted(cts)[len(cts) // 2]

    if not scene_ctime:
        return {}

    # Cluster by sorted ctime into 10 contiguous chunks of equal scene count.
    # nuScenes packs scenes by log order; archives are contiguous slices of that
    # order, and each archive was extracted in a separate tar invocation so its
    # files share a tight ctime cluster relative to other archives.
    sorted_scenes = sorted(scene_ctime.items(), key=lambda kv: kv[1])
    # Find big ctime gaps -> archive boundaries (10 archives => 9 gaps)
    deltas = [
        (sorted_scenes[i + 1][1] - sorted_scenes[i][1], i)
        for i in range(len(sorted_scenes) - 1)
    ]
    deltas.sort(reverse=True)
    boundary_indices = sorted(i for _, i in deltas[:9])
    chunks = []
    start = 0
    for b in boundary_indices:
        chunks.append(sorted_scenes[start : b + 1])
        start = b + 1
    chunks.append(sorted_scenes[start:])

    scene_part = {}
    for i, chunk in enumerate(chunks, start=1):
        for tok, _ in chunk:
            scene_part[tok] = i
    return scene_part


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--target-fraction", type=float, default=0.30,
                    help="Subset size as fraction of total scenes (default 0.30 ~= 3/10)")
    args = ap.parse_args()

    dataroot = Path(args.dataroot)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = dataroot / "v1.0-trainval"

    print(f"[load] scene.json log.json sample.json sample_data.json ...")
    scenes = load_json(meta / "scene.json")
    logs = load_json(meta / "log.json")
    samples = load_json(meta / "sample.json")
    print(f"[load] sample_data.json (1.3GB) ...", flush=True)
    sample_data = load_json(meta / "sample_data.json")

    logs_by_token = index_by_token(logs)

    # samples grouped by scene
    samples_by_scene = defaultdict(list)
    for s in samples:
        samples_by_scene[s["scene_token"]].append(s)
    # ensure per-scene sample order (next pointers)
    for sc_token in samples_by_scene:
        samples_by_scene[sc_token].sort(key=lambda r: r["timestamp"])

    # CAM_FRONT keyframe per sample
    sample_data_by_sample = defaultdict(dict)
    for d in sample_data:
        if d["is_key_frame"]:
            ch = d["filename"].split("/")[1] if "/" in d["filename"] else ""
            if ch.startswith("CAM_") or ch.startswith("LIDAR_"):
                sample_data_by_sample[d["sample_token"]][ch] = d

    print(f"[map] assigning scene -> part by ctime clustering ...", flush=True)
    scene_part = assign_parts_by_ctime(scenes, samples_by_scene, sample_data_by_sample, dataroot)
    n_mapped = sum(1 for v in scene_part.values() if v)
    print(f"[map] mapped {n_mapped}/{len(scenes)} scenes to a part archive")

    # Build per-scene record
    rows = []
    for sc in scenes:
        log = logs_by_token[sc["log_token"]]
        desc = sc.get("description", "")
        tags = describe_tags(desc)
        vehicle = log["vehicle"]
        location = log["location"]
        samps = samples_by_scene[sc["token"]]
        rows.append(dict(
            scene_token=sc["token"],
            name=sc["name"],
            description=desc,
            location=location,
            vehicle=vehicle,
            num_samples=sc["nbr_samples"],
            duration_s=(samps[-1]["timestamp"] - samps[0]["timestamp"]) / 1e6 if samps else 0.0,
            part=scene_part.get(sc["token"], 0),
            **tags,
        ))

    # ---------- per-part summary ----------
    parts = sorted({r["part"] for r in rows if r["part"]})
    print(f"\n=== Per-part summary ({len(parts)} parts detected) ===")
    header = f"{'part':>4} {'#scn':>5} {'#samp':>6} | {'boston':>6} {'sg-on':>5} {'sg-qt':>5} {'sg-hv':>5} | {'n008':>5} {'n015':>5} | {'day':>4} {'nite':>4} {'rain':>4} {'cnst':>4} {'busy':>4} {'peds':>4}"
    print(header)
    print("-" * len(header))
    loc_aliases = {
        "boston-seaport": "boston",
        "singapore-onenorth": "sg-on",
        "singapore-queenstown": "sg-qt",
        "singapore-hollandvillage": "sg-hv",
    }
    for p in parts:
        prs = [r for r in rows if r["part"] == p]
        nscn = len(prs)
        nsamp = sum(r["num_samples"] for r in prs)
        loc_c = Counter(r["location"] for r in prs)
        veh_c = Counter(r["vehicle"] for r in prs)
        nnight = sum(1 for r in prs if r["night"])
        nday = nscn - nnight
        nrain = sum(1 for r in prs if r["rain"])
        ncnst = sum(1 for r in prs if r["construction"])
        nbusy = sum(1 for r in prs if r["busy"])
        npeds = sum(1 for r in prs if r["pedestrians"])
        print(f"{p:>4} {nscn:>5} {nsamp:>6} | "
              f"{loc_c.get('boston-seaport',0):>6} "
              f"{loc_c.get('singapore-onenorth',0):>5} "
              f"{loc_c.get('singapore-queenstown',0):>5} "
              f"{loc_c.get('singapore-hollandvillage',0):>5} | "
              f"{veh_c.get('n008',0):>5} {veh_c.get('n015',0):>5} | "
              f"{nday:>4} {nnight:>4} {nrain:>4} {ncnst:>4} {nbusy:>4} {npeds:>4}")
    # global
    nscn = len(rows)
    nsamp = sum(r["num_samples"] for r in rows)
    loc_c = Counter(r["location"] for r in rows)
    veh_c = Counter(r["vehicle"] for r in rows)
    nnight = sum(1 for r in rows if r["night"])
    nday = nscn - nnight
    nrain = sum(1 for r in rows if r["rain"])
    ncnst = sum(1 for r in rows if r["construction"])
    nbusy = sum(1 for r in rows if r["busy"])
    npeds = sum(1 for r in rows if r["pedestrians"])
    print("-" * len(header))
    print(f"{'all':>4} {nscn:>5} {nsamp:>6} | "
          f"{loc_c.get('boston-seaport',0):>6} "
          f"{loc_c.get('singapore-onenorth',0):>5} "
          f"{loc_c.get('singapore-queenstown',0):>5} "
          f"{loc_c.get('singapore-hollandvillage',0):>5} | "
          f"{veh_c.get('n008',0):>5} {veh_c.get('n015',0):>5} | "
          f"{nday:>4} {nnight:>4} {nrain:>4} {ncnst:>4} {nbusy:>4} {npeds:>4}")

    # Write per-scene CSV
    csv_path = outdir / "scene_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n[write] {csv_path}  ({len(rows)} scenes)")

    # ---------- curation: stratified diversity sampling ----------
    target_n = int(round(args.target_fraction * len(rows)))
    print(f"\n=== Curating {target_n} scenes ({args.target_fraction*100:.0f}% of {len(rows)}) ===")

    # Build a stratum key per scene with progressively coarser fallbacks
    def stratum(r):
        return (r["location"], r["vehicle"], r["night"], r["rain"], r["construction"])

    strata = defaultdict(list)
    for r in rows:
        strata[stratum(r)].append(r)

    print(f"strata: {len(strata)} unique cells")
    # Allocate proportional to log(1+count) so rare strata are not crushed,
    # with a minimum of 1 per non-empty stratum (caps at stratum size).
    import math
    weights = {k: math.log1p(len(v)) for k, v in strata.items()}
    wsum = sum(weights.values())
    alloc = {}
    for k, v in strata.items():
        share = max(1, int(round(target_n * weights[k] / wsum)))
        alloc[k] = min(share, len(v))
    # rebalance to exactly target_n
    diff = target_n - sum(alloc.values())
    if diff != 0:
        cells = sorted(strata.keys(), key=lambda k: len(strata[k]) - alloc[k], reverse=(diff > 0))
        i = 0
        while diff != 0 and i < len(cells) * 4:
            k = cells[i % len(cells)]
            if diff > 0 and alloc[k] < len(strata[k]):
                alloc[k] += 1
                diff -= 1
            elif diff < 0 and alloc[k] > 1:
                alloc[k] -= 1
                diff += 1
            i += 1

    # Within each stratum: prefer scenes with rich description (more tags set)
    def richness(r):
        return sum(int(r[t]) for t in ("night", "rain", "construction", "pedestrians", "busy", "turn", "parked", "bikes", "trucks"))

    picks = []
    for k, v in strata.items():
        v_sorted = sorted(v, key=lambda r: (-richness(r), -r["num_samples"], r["name"]))
        picks.extend(v_sorted[: alloc[k]])

    # Final picks summary
    print(f"\n=== Curated subset summary ({len(picks)} scenes) ===")
    loc_c = Counter(r["location"] for r in picks)
    veh_c = Counter(r["vehicle"] for r in picks)
    nnight = sum(1 for r in picks if r["night"])
    nrain = sum(1 for r in picks if r["rain"])
    ncnst = sum(1 for r in picks if r["construction"])
    nbusy = sum(1 for r in picks if r["busy"])
    npeds = sum(1 for r in picks if r["pedestrians"])
    print(f"locations: {dict(loc_c)}")
    print(f"vehicles:  {dict(veh_c)}")
    print(f"night={nnight}  rain={nrain}  construction={ncnst}  busy={nbusy}  pedestrians={npeds}")
    part_c = Counter(r["part"] for r in picks)
    print(f"parts drawn from: {dict(sorted(part_c.items()))}")

    pick_path = outdir / "curated_scenes.txt"
    with open(pick_path, "w") as f:
        f.write("# scene_token,name,part,location,vehicle,description\n")
        for r in sorted(picks, key=lambda r: (r["part"], r["name"])):
            f.write(f"{r['scene_token']},{r['name']},{r['part']},{r['location']},{r['vehicle']},{r['description'].replace(',', ';')}\n")
    print(f"[write] {pick_path}  ({len(picks)} scenes)")

    json_path = outdir / "curated_scene_tokens.json"
    with open(json_path, "w") as f:
        json.dump([r["scene_token"] for r in picks], f)
    print(f"[write] {json_path}")


if __name__ == "__main__":
    main()

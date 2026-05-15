"""Additional per-part statistics for the 10 nuScenes archive parts.

Builds on out/stage1_actual_parts.json (the ctime-derived mapping).
Adds: scene-number ranges, vehicle prefixes (n008 Boston / n015 Singapore),
unique-logs-per-part, real-world collection time spans, hour-of-day
distribution, scene-length distribution, and analysis of the 232 scenes
that were split across two adjacent .tgz files.

Run:
    cd /home/satya/skr/S2GO/S2GO
    python scripts/stage1_part_extras.py
"""
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta


PARTS_PATH = "out/stage1_actual_parts.json"
STATS_PATH = "out/stage1_dataset_stats.json"
OUT_PATH = "out/stage1_part_extras.json"

# nuScenes vehicles — well-known convention from filename prefixes
# (n008 = Renault Zoe, Boston; n015 = Renault Zoe, Singapore)
VEHICLE_LOC = {"n008": "boston", "n015": "singapore"}

# Per-location UTC offset for hour-of-day analysis
LOC_TZ_OFFSET = {
    "boston-seaport": -4,             # EDT during nuScenes 2018 collection (Aug-Nov)
    "singapore-onenorth": 8,
    "singapore-queenstown": 8,
    "singapore-hollandvillage": 8,
}


def main() -> int:
    t0 = time.time()

    # ── Load prior outputs ─────────────────────────────────────────────
    with open(PARTS_PATH) as f:
        parts_data = json.load(f)
    with open(STATS_PATH) as f:
        stats = json.load(f)

    scene_to_part = {}
    for p_str, S in parts_data["parts"].items():
        for name in S["scene_names"]:
            scene_to_part[name] = int(p_str)

    name_to_scene = {s["name"]: s for s in stats["scenes"]}

    # ── Need nuScenes for log + sample timestamps + vehicle prefix ────
    from nuscenes.nuscenes import NuScenes
    print("Loading NuScenes(v1.0-trainval)…", flush=True)
    nusc = NuScenes(version="v1.0-trainval", dataroot="/home/satya/skr/S2GO/S2GO/data/nuscenes",
                    verbose=False)

    # Build per-scene helpers
    scene_obj = {s["name"]: s for s in nusc.scene}
    log_by_token = {l["token"]: l for l in nusc.log}

    # Vehicle prefix: pull from any sample_data filename for the scene's first sample
    def scene_vehicle(scene_name):
        sc = scene_obj[scene_name]
        sample = nusc.get("sample", sc["first_sample_token"])
        sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        # filename like 'samples/CAM_FRONT/n008-2018-05-21-…jpg'
        base = os.path.basename(sd["filename"])
        prefix = base.split("-", 1)[0]
        return prefix

    # First and last sample timestamps per scene (microseconds since epoch UTC)
    def scene_ts_range(scene_name):
        sc = scene_obj[scene_name]
        first = nusc.get("sample", sc["first_sample_token"])["timestamp"]
        # walk to last
        cur = sc["first_sample_token"]
        last = first
        while cur:
            samp = nusc.get("sample", cur)
            last = samp["timestamp"]
            cur = samp["next"]
        return first, last

    # Pre-compute per-scene metadata once
    print("Annotating scenes (vehicle + timestamps)…", flush=True)
    scene_meta = {}
    for sc in nusc.scene:
        name = sc["name"]
        veh = scene_vehicle(name)
        ts_first, ts_last = scene_ts_range(name)
        scene_meta[name] = {
            "vehicle": veh,
            "log_token": sc["log_token"],
            "ts_first_us": ts_first,
            "ts_last_us": ts_last,
            "n_keyframes": name_to_scene[name]["n_keyframes"],
            "location": log_by_token[sc["log_token"]]["location"],
        }

    # ── Per-part rollups ──────────────────────────────────────────────
    per_part = defaultdict(list)
    for name, p in scene_to_part.items():
        per_part[p].append(name)

    summaries = {}
    for p in sorted(per_part):
        scenes = per_part[p]
        # Scene-number range (parse "scene-NNNN")
        nums = sorted(int(re.match(r"scene-(\d+)", n).group(1)) for n in scenes)
        # Unique logs
        logs = {scene_meta[n]["log_token"] for n in scenes}
        # Vehicle split
        veh = Counter(scene_meta[n]["vehicle"] for n in scenes)
        # Real-world collection time span (UTC)
        ts_min = min(scene_meta[n]["ts_first_us"] for n in scenes)
        ts_max = max(scene_meta[n]["ts_last_us"] for n in scenes)
        # Hour-of-day distribution (local time per scene's location)
        hours = Counter()
        for n in scenes:
            sm = scene_meta[n]
            offset = LOC_TZ_OFFSET.get(sm["location"], 0)
            local = datetime.fromtimestamp(sm["ts_first_us"] / 1e6, tz=timezone.utc) + \
                    timedelta(hours=offset)
            hours[local.hour] += 1
        # Scene-length stats (keyframes per scene)
        ks = sorted(scene_meta[n]["n_keyframes"] for n in scenes)
        # Description-length (proxy for scene complexity)
        desc_len = [len(name_to_scene[n]["description"] or "") for n in scenes]

        summaries[p] = {
            "n_scenes": len(scenes),
            "scene_num_min": nums[0],
            "scene_num_max": nums[-1],
            "scene_num_span": nums[-1] - nums[0] + 1,
            "scene_num_density": len(nums) / max(1, nums[-1] - nums[0] + 1),  # 1.0 = perfectly contiguous
            "n_unique_logs": len(logs),
            "vehicle_split": dict(veh),
            "ts_min_utc": datetime.fromtimestamp(ts_min / 1e6, tz=timezone.utc).isoformat(),
            "ts_max_utc": datetime.fromtimestamp(ts_max / 1e6, tz=timezone.utc).isoformat(),
            "collection_span_days": round((ts_max - ts_min) / 1e6 / 86400, 2),
            "hours_local": dict(sorted(hours.items())),
            "keyframes_per_scene": {
                "min": ks[0], "p25": ks[len(ks)//4], "median": ks[len(ks)//2],
                "p75": ks[3*len(ks)//4], "max": ks[-1],
                "mean": round(sum(ks) / len(ks), 1),
            },
            "description_chars": {
                "min": min(desc_len), "max": max(desc_len),
                "mean": round(sum(desc_len) / len(desc_len), 1),
            },
        }

    # ── Cross-part splits (the 232 scenes that span 2 parts) ──────────
    # In the actual_parts JSON, each scene was assigned its dominant part
    # via majority-rule on its sample-token ctimes. Re-derive the split
    # pairs here from those scene_part_counts.
    # We need to re-stat to get the sub-part counts. Cheaper: parse
    # ctimes again only for known split scenes — but they aren't listed
    # explicitly. So we'll rebuild from raw data quickly.
    print("Re-deriving cross-part scene splits…", flush=True)
    UNTAR_LOG = "/home/satya/skr/S2GO/S2GO/data/nuscenes_dl/untar.log"
    line_re = re.compile(
        r"\[([^\]]+)\]\s+(>>>|===)\s+(?:untar|done)\s+v1\.0-trainval(\d+)_keyframes\.tgz")
    starts, dones = {}, {}
    with open(UNTAR_LOG) as f:
        for line in f:
            m = line_re.search(line)
            if not m: continue
            ts_str, tag, ps = m.groups()
            ts = datetime.fromisoformat(ts_str).timestamp()
            if tag == ">>>": starts[int(ps)] = ts
            else:           dones[int(ps)] = ts
    windows = sorted([(p, starts[p], dones[p]) for p in starts])

    def find_part_ct(ct):
        for pp, t0_, t1_ in windows:
            if t0_ <= ct <= t1_: return pp
        return None

    DATAROOT = "/home/satya/skr/S2GO/S2GO/data/nuscenes"
    cam_dir = os.path.join(DATAROOT, "samples", "CAM_FRONT")
    file_to_part = {}
    for fn in os.listdir(cam_dir):
        ct = os.stat(os.path.join(cam_dir, fn)).st_ctime
        p = find_part_ct(ct)
        if p is not None:
            file_to_part[fn] = p

    fn_to_sample_tok = {}
    for sd in nusc.sample_data:
        if sd["channel"] != "CAM_FRONT" or not sd["is_key_frame"]: continue
        fn_to_sample_tok[os.path.basename(sd["filename"])] = sd["sample_token"]

    sample_to_part = {fn_to_sample_tok[fn]: p for fn, p in file_to_part.items()
                      if fn in fn_to_sample_tok}
    scene_part_counts = defaultdict(Counter)
    for samp in nusc.sample:
        if samp["token"] in sample_to_part:
            scene_part_counts[samp["scene_token"]][sample_to_part[samp["token"]]] += 1

    boundary_pairs = Counter()  # (part_a, part_b) → number of scenes split
    boundary_examples = defaultdict(list)
    for tok, c in scene_part_counts.items():
        if len(c) <= 1: continue
        ps = sorted(c)
        for i in range(len(ps) - 1):
            pair = (ps[i], ps[i+1])
            boundary_pairs[pair] += 1
            boundary_examples[pair].append(scene_obj[tok]["name"] if False else nusc.get("scene", tok)["name"])

    # ── stdout report ─────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("ADDITIONAL PER-PART STATISTICS")
    print("=" * 100)

    # 1) Scene-number ranges
    print("\nA. SCENE-NUMBER RANGES (scene-NNNN, are parts contiguous in scene numbering?)")
    print(f"  {'Part':>4}  {'min':>6}  {'max':>6}  {'span':>5}  {'density':>7}  "
          f"{'#logs':>5}  {'vehicles':<24}")
    for p, S in summaries.items():
        veh_str = ",".join(f"{k}={v}" for k, v in sorted(S["vehicle_split"].items()))
        print(f"  {p:>4d}  {S['scene_num_min']:>6d}  {S['scene_num_max']:>6d}  "
              f"{S['scene_num_span']:>5d}  {S['scene_num_density']:>6.2f}    "
              f"{S['n_unique_logs']:>5d}  {veh_str:<24}")
    print("  (density=1.0 means scene numbers are perfectly contiguous within the part;")
    print("   <1.0 means there are gaps because some scene numbers fell into other parts)")

    # 2) Real-world collection time spans
    print("\nB. REAL-WORLD COLLECTION TIME SPANS (UTC; vs. extraction time which was 30s/part)")
    print(f"  {'Part':>4}  {'first sample (UTC)':<28}  {'last sample (UTC)':<28}  {'span (days)':>10}")
    for p, S in summaries.items():
        print(f"  {p:>4d}  {S['ts_min_utc'][:19]:<28}  {S['ts_max_utc'][:19]:<28}  "
              f"{S['collection_span_days']:>10.2f}")

    # 3) Hour-of-day distribution (local time)
    print("\nC. HOUR-OF-DAY DISTRIBUTION (scene start, local time at scene's location)")
    print(f"  {'Part':>4}  hours active (counts per hour):")
    for p, S in summaries.items():
        hour_str = "  ".join(f"{h:02d}h={c}" for h, c in S["hours_local"].items())
        print(f"  {p:>4d}  {hour_str}")

    # 4) Scene-length distribution
    print("\nD. SCENE LENGTH (keyframes per scene)")
    print(f"  {'Part':>4}  {'min':>3}  {'p25':>3}  {'med':>3}  {'p75':>3}  {'max':>3}  {'mean':>5}  "
          f"{'desc.chars (min/mean/max)':>30}")
    for p, S in summaries.items():
        ks = S["keyframes_per_scene"]
        dc = S["description_chars"]
        print(f"  {p:>4d}  {ks['min']:>3d}  {ks['p25']:>3d}  {ks['median']:>3d}  "
              f"{ks['p75']:>3d}  {ks['max']:>3d}  {ks['mean']:>5.1f}  "
              f"{dc['min']:>10d}  {dc['mean']:>8.1f}  {dc['max']:>9d}")

    # 5) Cross-part scene splits
    print("\nE. SCENES SPLIT ACROSS PART BOUNDARIES (232 of 850 scenes were split)")
    print(f"  {'pair':<8}  {'#scenes':>7}   examples")
    for pair, cnt in sorted(boundary_pairs.items()):
        ex = boundary_examples[pair][:3]
        print(f"  {pair[0]:>2d}↔{pair[1]:<2d}  {cnt:>7d}   {', '.join(ex)}{'…' if len(boundary_examples[pair])>3 else ''}")
    print("  (Adjacent-pair splits dominate: nuScenes packs whole logs but apparently")
    print("   crosses log boundaries inside an archive, so a log straddling two parts")
    print("   gets its scenes split.)")

    # 6) Vehicle-summary across all parts
    print("\nF. VEHICLE FLEET (n008 = Boston Renault Zoe, n015 = Singapore Renault Zoe)")
    fleet = Counter()
    for p, S in summaries.items():
        for v, c in S["vehicle_split"].items():
            fleet[v] += c
    for v, c in fleet.items():
        loc = VEHICLE_LOC.get(v, "?")
        print(f"  {v} ({loc:<10}): {c} scenes total")

    # ── JSON ──────────────────────────────────────────────────────────
    out = {
        "per_part": {p: summaries[p] for p in sorted(summaries)},
        "boundary_pair_counts": {f"{a}-{b}": v for (a, b), v in boundary_pairs.items()},
        "fleet": dict(fleet),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}  (elapsed: {time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

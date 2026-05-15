"""Recover the literal nuScenes Part 01..10 .tgz membership from on-disk
file extraction timestamps, then re-run per-part stats.

The .tgz archives were deleted after extraction, but each extracted file's
ctime is the moment it was written to disk. untar.log records the start +
end timestamp of each part's extraction, so every file's ctime falls in
exactly one of the 10 known windows. We use CAM_FRONT (one file per
sample) as the identifier and map each sample_token to its part.

Output:
    out/stage1_actual_parts.json   — full per-part breakdown
    stdout                          — paste-able table

Run:
    cd /home/satya/skr/S2GO/S2GO
    python scripts/stage1_actual_parts.py
"""
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

DATAROOT = "/home/satya/skr/S2GO/S2GO/data/nuscenes"
UNTAR_LOG = "/home/satya/skr/S2GO/S2GO/data/nuscenes_dl/untar.log"
OUT_PATH = "out/stage1_actual_parts.json"

KEYWORDS = {
    "rain":         re.compile(r"\b(rain|raining|wet)\b", re.I),
    "night":        re.compile(r"\bnight\b", re.I),
    "intersection": re.compile(r"\bintersection\b", re.I),
    "turn":         re.compile(r"\b(turn|turning|maneuver|maneuvering)\b", re.I),
    "ped":          re.compile(r"\b(ped|peds|pedestrian|pedestrians|jaywalker|jaywalkers|crossing)\b", re.I),
    "construction": re.compile(r"\b(construction|truck|trucks)\b", re.I),
    "parking":      re.compile(r"\b(parking|parked)\b", re.I),
}


def parse_part_windows(log_path):
    """Parse untar.log → [(part_num, start_epoch, end_epoch), ...] for parts 01..10.

    Lines look like:
        [2026-05-13T12:28:34+02:00] >>> untar v1.0-trainval01_keyframes.tgz
        [2026-05-13T12:28:58+02:00] === done v1.0-trainval01_keyframes.tgz
    The [start, end] window for part N = [t_start_N, t_done_N].
    """
    starts, dones = {}, {}
    line_re = re.compile(
        r"\[([^\]]+)\]\s+(>>>|===)\s+(?:untar|done)\s+v1\.0-trainval(\d+)_keyframes\.tgz")
    with open(log_path) as f:
        for line in f:
            m = line_re.search(line)
            if not m:
                continue
            ts_str, tag, part_str = m.groups()
            # ISO8601 with timezone — fromisoformat handles +02:00 directly
            ts = datetime.fromisoformat(ts_str).timestamp()
            part = int(part_str)
            if tag == ">>>":
                starts[part] = ts
            else:
                dones[part] = ts
    windows = []
    for part in sorted(starts):
        windows.append((part, starts[part], dones[part]))
    return windows


def find_part(ctime, windows):
    """Return part number (1..10) whose [start, end] window contains ctime, else None."""
    for part, t0, t1 in windows:
        if t0 <= ctime <= t1:
            return part
    return None


def main() -> int:
    t_start = time.time()

    # ── Parse part windows ──────────────────────────────────────────────
    windows = parse_part_windows(UNTAR_LOG)
    print("Per-part extraction windows (from untar.log):")
    for p, t0, t1 in windows:
        print(f"  Part {p:02d}: {datetime.fromtimestamp(t0).isoformat()} → "
              f"{datetime.fromtimestamp(t1).isoformat()}  ({t1-t0:.0f}s)")

    # ── Stat every CAM_FRONT file → sample-token → part ────────────────
    cam_dir = os.path.join(DATAROOT, "samples", "CAM_FRONT")
    print(f"\nStatting {len(os.listdir(cam_dir))} CAM_FRONT files…", flush=True)
    file_to_part = {}  # filename → part
    unmapped = 0
    for fn in os.listdir(cam_dir):
        ct = os.stat(os.path.join(cam_dir, fn)).st_ctime
        p = find_part(ct, windows)
        if p is None:
            unmapped += 1
            continue
        file_to_part[fn] = p
    print(f"  mapped: {len(file_to_part)}, unmapped (outside windows): {unmapped}")

    # ── Load nuScenes to map filename → sample_token → scene ───────────
    from nuscenes.nuscenes import NuScenes
    print("Loading NuScenes(v1.0-trainval)…", flush=True)
    nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
    print(f"  {len(nusc.scene)} scenes, {len(nusc.sample)} samples")

    # Build CAM_FRONT-filename → sample_token map
    fn_to_sample_tok = {}
    for sd in nusc.sample_data:
        if sd["channel"] != "CAM_FRONT" or not sd["is_key_frame"]:
            continue
        fn_to_sample_tok[os.path.basename(sd["filename"])] = sd["sample_token"]

    # Sample → part. For each sample, lookup its CAM_FRONT file's part.
    sample_to_part = {}
    for fn, p in file_to_part.items():
        if fn in fn_to_sample_tok:
            sample_to_part[fn_to_sample_tok[fn]] = p

    # Scene → part(s). A scene could in principle span parts (if its samples
    # were split across archives) — count how often that happens.
    scene_part_counts = defaultdict(Counter)
    for sample in nusc.sample:
        if sample["token"] in sample_to_part:
            scene_token = sample["scene_token"]
            scene_part_counts[scene_token][sample_to_part[sample["token"]]] += 1

    intra_scene_split_count = sum(1 for c in scene_part_counts.values() if len(c) > 1)
    print(f"\nScenes whose samples span >1 part: {intra_scene_split_count} / {len(scene_part_counts)}")
    if intra_scene_split_count > 0:
        # Show a few examples
        print("  Examples (first 5):")
        examples = [(tok, dict(c)) for tok, c in scene_part_counts.items() if len(c) > 1][:5]
        for tok, c in examples:
            scene_name = nusc.get("scene", tok)["name"]
            print(f"    {scene_name}: {c}")

    # Each scene's "dominant part" = part with most samples in this scene.
    scene_to_part = {}
    for scene_token, c in scene_part_counts.items():
        scene_to_part[scene_token] = c.most_common(1)[0][0]

    # ── Build per-part scene rosters ───────────────────────────────────
    from nuscenes.utils.splits import create_splits_scenes
    official = create_splits_scenes()
    train_set = set(official["train"])
    val_set = set(official["val"])
    log_loc = {l["token"]: l["location"] for l in nusc.log}
    log_date = {l["token"]: l["date_captured"] for l in nusc.log}

    # Pre-load cached per-scene yield + keyframe count from the main stats run.
    cached_path = "out/stage1_dataset_stats.json"
    cached_by_name = {}
    if os.path.exists(cached_path):
        with open(cached_path) as f:
            cached = json.load(f)
        cached_by_name = {s["name"]: s for s in cached["scenes"]}

    per_part = defaultdict(list)
    for scene in nusc.scene:
        tok = scene["token"]
        if tok not in scene_to_part:
            continue
        part = scene_to_part[tok]
        log = nusc.get("log", scene["log_token"])
        s = {
            "name": scene["name"],
            "split": "train" if scene["name"] in train_set else "val" if scene["name"] in val_set else "other",
            "location": log["location"],
            "date_captured": log["date_captured"],
            "description": scene["description"],
            "n_keyframes": cached_by_name.get(scene["name"], {}).get("n_keyframes", 0),
            "t4_yield": cached_by_name.get(scene["name"], {}).get("t4_yield", 0),
            "samples_in_part": scene_part_counts[tok][part],
        }
        per_part[part].append(s)

    # ── Per-part summaries ────────────────────────────────────────────
    summaries = {}
    for p in sorted(per_part):
        scenes = per_part[p]
        summaries[p] = {
            "n_scenes": len(scenes),
            "n_keyframes": sum(s["n_keyframes"] for s in scenes),
            "n_samples_attributed": sum(s["samples_in_part"] for s in scenes),
            "t4_yield_estimate": sum(s["t4_yield"] for s in scenes),
            "by_split": dict(Counter(s["split"] for s in scenes)),
            "by_location": dict(Counter(s["location"] for s in scenes)),
            "by_date": dict(Counter(s["date_captured"] for s in scenes)),
            "keywords": {
                k: sum(1 for s in scenes if pat.search(s["description"] or ""))
                for k, pat in KEYWORDS.items()
            },
            "scene_names": sorted(s["name"] for s in scenes),
        }

    # ── stdout report ─────────────────────────────────────────────────
    print()
    print("=" * 96)
    print("ACTUAL .tgz PART → SCENE MAPPING (recovered from file ctime)")
    print("=" * 96)

    print(f"\n{'Part':>4} {'#scn':>4} {'#kf':>5} {'T=4':>5} "
          f"{'tr/va':>7} {'top location':<28} "
          f"{'rain':>4} {'night':>5} {'intx':>4} {'cons':>4} {'ped':>3}")
    print("      ----  ----  -----  -----  -------  ----------------------------"
          " ---- ----- ---- ---- ---")
    for p, S in summaries.items():
        loc_top = max(S["by_location"], key=S["by_location"].get)
        loc_pct = 100 * S["by_location"][loc_top] / S["n_scenes"]
        loc_str = f"{loc_top}({loc_pct:.0f}%)"
        print(f"{p:>4d} {S['n_scenes']:>4d} {S['n_keyframes']:>5d} "
              f"{S['t4_yield_estimate']:>5d}  "
              f"{S['by_split'].get('train',0):>3d}/{S['by_split'].get('val',0):>2d}  "
              f"{loc_str:<28} "
              f"{S['keywords']['rain']:>4d} {S['keywords']['night']:>5d} "
              f"{S['keywords']['intersection']:>4d} {S['keywords']['construction']:>4d} "
              f"{S['keywords']['ped']:>3d}")

    print(f"\n{'TOTAL':>4} {sum(S['n_scenes'] for S in summaries.values()):>4d} "
          f"{sum(S['n_keyframes'] for S in summaries.values()):>5d} "
          f"{sum(S['t4_yield_estimate'] for S in summaries.values()):>5d}")

    # Detailed per-part
    print("\n" + "=" * 96)
    print("PER-PART DETAIL")
    print("=" * 96)
    for p, S in summaries.items():
        print(f"\n  ── Part {p:02d} ──   {S['n_scenes']} scenes, "
              f"{S['n_keyframes']} keyframes, {S['t4_yield_estimate']} T=4 sequences "
              f"(train={S['by_split'].get('train',0)}, val={S['by_split'].get('val',0)})")
        print(f"    Locations:")
        for loc, c in sorted(S["by_location"].items(), key=lambda kv: -kv[1]):
            print(f"      {loc:30s} {c:>3d}  ({100*c/S['n_scenes']:.0f}%)")
        print(f"    Date span:")
        for d, c in sorted(S["by_date"].items()):
            print(f"      {d:12s} {c:>3d}")
        print(f"    Keyword scenes:")
        for k in ["rain", "night", "intersection", "turn", "ped", "construction", "parking"]:
            v = S["keywords"][k]
            print(f"      {k:14s} {v:>3d}  ({100*v/S['n_scenes']:.0f}%)")

    # Diversity
    print("\n" + "=" * 96)
    print("DIVERSITY: which Part(s) capture rare cases?")
    print("=" * 96)
    full_kw = {
        k: sum(1 for p in summaries for _ in [None] if True)
        for k in KEYWORDS
    }
    full_kw = {k: sum(S["keywords"][k] for S in summaries.values()) for k in KEYWORDS}
    print(f"\n  Per-part share of total RAIN scenes ({full_kw['rain']} total):")
    for p, S in summaries.items():
        if S["keywords"]["rain"] > 0:
            print(f"    Part {p:02d}: {S['keywords']['rain']:>3d}  "
                  f"({100*S['keywords']['rain']/full_kw['rain']:.0f}%)")
    print(f"\n  Per-part share of total NIGHT scenes ({full_kw['night']} total):")
    for p, S in summaries.items():
        if S["keywords"]["night"] > 0:
            print(f"    Part {p:02d}: {S['keywords']['night']:>3d}  "
                  f"({100*S['keywords']['night']/full_kw['night']:.0f}%)")
    print(f"\n  Per-part share of total CONSTRUCTION scenes ({full_kw['construction']} total):")
    for p, S in summaries.items():
        if S["keywords"]["construction"] > 0:
            print(f"    Part {p:02d}: {S['keywords']['construction']:>3d}  "
                  f"({100*S['keywords']['construction']/full_kw['construction']:.0f}%)")

    # Best 2-Part combos
    print("\n  Best 2-Part combos (by union of {rain,night,construction,intersection} scenes):")
    keep = ["rain", "night", "construction", "intersection"]
    part_to_kwscenes = {}
    for p, scenes in per_part.items():
        union = set()
        for s in scenes:
            for k in keep:
                if KEYWORDS[k].search(s["description"] or ""):
                    union.add((k, s["name"]))
        part_to_kwscenes[p] = union
    pairs = []
    for a in summaries:
        for b in summaries:
            if b <= a: continue
            cov = len(part_to_kwscenes[a] | part_to_kwscenes[b])
            sc_total = summaries[a]["n_scenes"] + summaries[b]["n_scenes"]
            t4 = summaries[a]["t4_yield_estimate"] + summaries[b]["t4_yield_estimate"]
            pairs.append((cov, t4, sc_total, a, b))
    pairs.sort(reverse=True)
    for cov, t4, sc, a, b in pairs[:6]:
        print(f"    Parts {a:>2d}+{b:>2d}: {cov:>3d} keyword-scenes  "
              f"({sc} scenes, ~{t4} T=4 seqs)")

    # ── JSON output ───────────────────────────────────────────────────
    out = {
        "method": "Recovered from file ctime (extraction time) cross-referenced with untar.log per-part windows",
        "untar_log": UNTAR_LOG,
        "windows_epoch": [{"part": p, "start": t0, "end": t1} for p, t0, t1 in windows],
        "scenes_spanning_multiple_parts": intra_scene_split_count,
        "parts": {p: summaries[p] for p in sorted(summaries)},
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}  (elapsed: {time.time() - t_start:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

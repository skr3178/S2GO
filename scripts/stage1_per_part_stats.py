"""Per-chunk breakdown of nuScenes scenes for Stage-1 curation planning.

The original 10 `.tgz` part archives have been deleted after extraction,
so the literal part->scene mapping isn't recoverable from on-disk state.
This script instead splits the 68 nuScenes logs into 10 chronological
chunks (sorted by `date_captured`, then `log_token` for tie-break) and
labels them "Slice 1 .. Slice 10". Conceptually similar to the original
parts (nuScenes packed parts roughly by collection date), but NOT
identical -- e.g. literal Part 1 had 3 376 keyframes while a pure
chronological chunk may differ.

For each slice + each location group + each scene-content keyword,
prints scene count, sample count, T=4 sequence yield, location mix,
and keyword counts.

Run:
    cd /home/satya/skr/S2GO/S2GO
    python scripts/stage1_per_part_stats.py

Reads out/stage1_dataset_stats.json (produced by stage1_dataset_stats.py).
No GPU. <100 ms.
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict


STATS_PATH = "out/stage1_dataset_stats.json"
OUT_PATH = "out/stage1_per_part_stats.json"
N_SLICES = 10

KEYWORDS = ["rain", "night", "intersection", "turn", "ped", "construction", "parking"]
COMPILED = {
    "rain":         re.compile(r"\b(rain|raining|wet)\b", re.I),
    "night":        re.compile(r"\bnight\b", re.I),
    "intersection": re.compile(r"\bintersection\b", re.I),
    "turn":         re.compile(r"\b(turn|turning|maneuver|maneuvering)\b", re.I),
    "ped":          re.compile(r"\b(ped|peds|pedestrian|pedestrians|jaywalker|jaywalkers|crossing)\b", re.I),
    "construction": re.compile(r"\b(construction|truck|trucks)\b", re.I),
    "parking":      re.compile(r"\b(parking|parked)\b", re.I),
}


def load_scenes_with_log_meta():
    """Augment scene_stats with log_token + first_sample_timestamp.

    The cached stats JSON has date_captured but not log_token (needed
    to group scenes from the same log into the same slice). Re-derive
    via nuscenes-devkit (cheap: scene + log + first_sample only).
    """
    from nuscenes.nuscenes import NuScenes
    DATAROOT = "/home/satya/skr/S2GO/S2GO/data/nuscenes"
    print(f"Loading NuScenes(v1.0-trainval) for log-token mapping…", flush=True)
    nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)

    with open(STATS_PATH) as f:
        cached = json.load(f)
    by_name = {s["name"]: s for s in cached["scenes"]}

    enriched = []
    for scene in nusc.scene:
        name = scene["name"]
        s = by_name[name]
        log_token = scene["log_token"]
        first_sample = nusc.get("sample", scene["first_sample_token"])
        enriched.append({
            **s,
            "log_token": log_token,
            "first_ts": first_sample["timestamp"],  # microseconds
        })
    return enriched, cached


def assign_slices(scenes):
    """Group scenes into N_SLICES chunks by log date, keeping each log
    intact in a single slice (mirrors how the original .tgz packaging
    worked: a tar archive contains whole logs, not partial logs).

    Returns: list[scene] with extra "slice" field (1..N_SLICES).
    """
    # Build per-log summary: date, first_ts, scene count
    per_log = defaultdict(lambda: {"date": None, "first_ts": None, "scenes": []})
    for s in scenes:
        L = per_log[s["log_token"]]
        L["scenes"].append(s["name"])
        L["date"] = s["date_captured"]
        if L["first_ts"] is None or s["first_ts"] < L["first_ts"]:
            L["first_ts"] = s["first_ts"]

    # Sort logs by date then first_ts then token
    log_order = sorted(per_log.items(),
                       key=lambda kv: (kv[1]["date"] or "", kv[1]["first_ts"] or 0, kv[0]))

    # Greedy slice assignment: target ~N_total/N_SLICES scenes per slice,
    # but never split a log across slices. Walk in date order; close a
    # slice when adding the next log would push us past target+slack.
    n_total = sum(len(v["scenes"]) for _, v in log_order)
    target = n_total / N_SLICES
    slack = target * 0.5  # allow up to ~50% over target before forcing close

    slice_idx = 1
    cur_count = 0
    log_to_slice = {}
    for log_token, L in log_order:
        # If adding this log would exceed target + slack AND we still have
        # more slices to fill AND this slice already has at least one log:
        if (cur_count > 0 and slice_idx < N_SLICES
                and cur_count + len(L["scenes"]) > target + slack):
            slice_idx += 1
            cur_count = 0
        log_to_slice[log_token] = slice_idx
        cur_count += len(L["scenes"])

    # Annotate each scene
    out = []
    for s in scenes:
        out.append({**s, "slice": log_to_slice[s["log_token"]]})
    return out, log_order, log_to_slice


def slice_summary(scenes_in_slice):
    """Compute counts for one slice."""
    n_scenes = len(scenes_in_slice)
    n_samples = sum(s["n_keyframes"] for s in scenes_in_slice)
    t4_yield = sum(s["t4_yield"] for s in scenes_in_slice)

    by_split = Counter(s["split"] for s in scenes_in_slice)
    by_loc = Counter(s["location"] for s in scenes_in_slice)
    dates = sorted({s["date_captured"] for s in scenes_in_slice if s["date_captured"]})
    date_range = (dates[0], dates[-1]) if dates else ("?", "?")

    kw_counts = {}
    for k, pat in COMPILED.items():
        kw_counts[k] = sum(1 for s in scenes_in_slice
                           if pat.search(s["description"] or ""))

    return {
        "n_scenes": n_scenes,
        "n_keyframes": n_samples,
        "t4_yield": t4_yield,
        "by_split": dict(by_split),
        "by_location": dict(by_loc),
        "date_range": date_range,
        "keywords": kw_counts,
        "scene_names": sorted(s["name"] for s in scenes_in_slice),
    }


def main() -> int:
    if not os.path.exists(STATS_PATH):
        print(f"ERROR: {STATS_PATH} not found. Run scripts/stage1_dataset_stats.py first.")
        return 1

    scenes, cached = load_scenes_with_log_meta()
    annotated, log_order, log_to_slice = assign_slices(scenes)

    # Per-slice summaries
    by_slice = defaultdict(list)
    for s in annotated:
        by_slice[s["slice"]].append(s)

    summaries = {}
    for sl in sorted(by_slice):
        summaries[sl] = slice_summary(by_slice[sl])

    # ── stdout report ─────────────────────────────────────────────────
    print()
    print("=" * 86)
    print(f"PER-CHUNK STATISTICS  (10 chronological log-chunks, NOT literal .tgz parts)")
    print("=" * 86)
    print(f"\nGrouping: 68 logs sorted by date_captured then first-sample timestamp,")
    print(f"split into 10 chunks each containing whole logs (a log never spans chunks).")
    print(f"This mirrors the nuScenes part-archive convention but does NOT exactly")
    print(f"reproduce the literal Part 01..10 .tgz file membership (those archives")
    print(f"have been deleted after extraction).")

    # Compact per-slice table
    print(f"\n{'Slice':>5} {'#scn':>4} {'#kf':>5} {'T=4 seq':>7} "
          f"{'tr/va':>7} {'date range':<24} {'top location':<22} "
          f"{'rain':>4} {'night':>5} {'intx':>4} {'cons':>4}")
    print(f"      ----  ----  -----  -------  ------- "
          f"------------------------ ---------------------- "
          f"---- ----- ---- ----")
    for sl, S in summaries.items():
        loc_top = max(S["by_location"], key=S["by_location"].get)
        loc_pct = 100 * S["by_location"][loc_top] / S["n_scenes"]
        loc_str = f"{loc_top}({loc_pct:.0f}%)"
        print(f"{sl:>5d} {S['n_scenes']:>4d} {S['n_keyframes']:>5d} "
              f"{S['t4_yield']:>7d}  "
              f"{S['by_split'].get('train',0):>3d}/{S['by_split'].get('val',0):>2d}  "
              f"{S['date_range'][0]}..{S['date_range'][1]}  "
              f"{loc_str:<22} "
              f"{S['keywords']['rain']:>4d} {S['keywords']['night']:>5d} "
              f"{S['keywords']['intersection']:>4d} {S['keywords']['construction']:>4d}")

    # Detailed per-slice breakdown
    print("\n" + "=" * 86)
    print("PER-SLICE DETAIL (location mix + all keywords)")
    print("=" * 86)
    for sl, S in summaries.items():
        print(f"\n  ── Slice {sl} ──")
        print(f"    Scenes:           {S['n_scenes']}")
        print(f"    Keyframes:        {S['n_keyframes']}")
        print(f"    T=4 sequences:    {S['t4_yield']}    "
              f"(train={S['by_split'].get('train',0)}, val={S['by_split'].get('val',0)})")
        print(f"    Date range:       {S['date_range'][0]} .. {S['date_range'][1]}")
        print(f"    Location mix:")
        for loc, c in sorted(S["by_location"].items(), key=lambda kv: -kv[1]):
            print(f"      {loc:30s} {c:>3d}  ({100*c/S['n_scenes']:.0f}%)")
        print(f"    Keyword scenes:")
        for k in KEYWORDS:
            v = S["keywords"][k]
            pct = 100 * v / S["n_scenes"]
            print(f"      {k:14s} {v:>3d}  ({pct:.0f}%)")

    # Cross-tab: location × keyword (over the full dataset, not per slice)
    print("\n" + "=" * 86)
    print("CROSS-TAB: location × keyword (full dataset, scene counts)")
    print("=" * 86)
    locs = sorted({s["location"] for s in annotated})
    print(f"\n  {'keyword':14s} " + "  ".join(f"{loc[:18]:>18s}" for loc in locs) + "  total")
    for k in KEYWORDS:
        row = []
        total = 0
        for loc in locs:
            c = sum(1 for s in annotated
                    if s["location"] == loc and COMPILED[k].search(s["description"] or ""))
            row.append(c)
            total += c
        print(f"  {k:14s} " + "  ".join(f"{c:>18d}" for c in row) + f"  {total:>5d}")

    # Diversity ranking: how different is each slice from the others?
    print("\n" + "=" * 86)
    print("DIVERSITY HINTS (which slice pairs maximize coverage)")
    print("=" * 86)
    print("\n  Single-slice keyword coverage (% of all such scenes captured by 1 slice):")
    full = {k: sum(1 for s in annotated if COMPILED[k].search(s["description"] or ""))
            for k in KEYWORDS}
    for sl, S in summaries.items():
        parts = [f"{k}={100*S['keywords'][k]/full[k]:.0f}%" if full[k] else f"{k}=0%"
                 for k in ["rain", "night", "construction"]]
        loc_top = max(S["by_location"], key=S["by_location"].get)
        print(f"    Slice {sl:>2d}: {' '.join(parts):<32s}  dominant={loc_top}")

    # Best 2-slice and 3-slice pairs by keyword + location coverage
    print("\n  Best 2-slice combos (by union of {rain,night,construction,intersection}"
          " keyword scenes):")
    keep_keys = ["rain", "night", "construction", "intersection"]
    slice_kw_sets = {}
    for sl, scenes_in in by_slice.items():
        union = set()
        for s in scenes_in:
            for k in keep_keys:
                if COMPILED[k].search(s["description"] or ""):
                    union.add((k, s["name"]))
        slice_kw_sets[sl] = union

    pairs = []
    for a in summaries:
        for b in summaries:
            if b <= a:
                continue
            union = slice_kw_sets[a] | slice_kw_sets[b]
            scn_total = summaries[a]["n_scenes"] + summaries[b]["n_scenes"]
            pairs.append((len(union), scn_total, a, b))
    pairs.sort(reverse=True)
    for cov, scn, a, b in pairs[:5]:
        print(f"    Slices {a:>2d}+{b:>2d}: {cov:>3d} keyword-scenes covered "
              f"({scn} total scenes,  ~{summaries[a]['t4_yield']+summaries[b]['t4_yield']} T=4 sequences)")

    # ── JSON output ───────────────────────────────────────────────────
    out = {
        "method": "10 chronological log-chunks (whole logs, sorted by date_captured then first-sample ts)",
        "n_slices": N_SLICES,
        "n_logs_per_slice": Counter(log_to_slice.values()),
        "slices": {sl: summaries[sl] for sl in sorted(summaries)},
        "log_to_slice": log_to_slice,
    }
    out["n_logs_per_slice"] = dict(out["n_logs_per_slice"])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

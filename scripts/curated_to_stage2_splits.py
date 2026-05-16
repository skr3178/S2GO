"""Convert curated scene-token JSONs into the Stage-2 trainer's splits format.

Inputs:
    dataset_stats/curated_train_270/tokens.json   — list[scene_token] (270)
    dataset_stats/curated_val_50/tokens.json      — list[scene_token] (50)

Output (one file):
    out/stage2_curated_splits.json
    {
      "T": 1,
      "source": {...},
      "train_scenes": [...],
      "val_scenes":   [...],
      "train_start_tokens": [...all sample tokens of the 270 train scenes...],
      "val_start_tokens":   [...all sample tokens of the 50 val scenes...],
      "train_count": ..., "val_count": ...
    }

Stage-2 uses T=1 per existing out/stage2_part1_splits.json convention (each
sample is its own sequence start).

No GPU, ~30s.
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

from nuscenes.nuscenes import NuScenes


DATAROOT = "/home/satya/skr/S2GO/S2GO/data/nuscenes"
TRAIN_TOK_JSON = "dataset_stats/curated_train_270/tokens.json"
VAL_TOK_JSON   = "dataset_stats/curated_val_50/tokens.json"
OUT_PATH       = "out/stage2_curated_splits.json"
T = 1  # Stage-2 convention


def collect_sample_tokens(nusc: NuScenes, scene_tokens: List[str]) -> Tuple[List[str], List[str]]:
    """Return (sample_tokens_in_pick_order, scene_names)."""
    sample_tokens = []
    scene_names = []
    for tok in scene_tokens:
        sc = nusc.get("scene", tok)
        scene_names.append(sc["name"])
        s_tok = sc["first_sample_token"]
        while s_tok:
            sample_tokens.append(s_tok)
            s_tok = nusc.get("sample", s_tok)["next"]
    return sample_tokens, scene_names


def main() -> int:
    train_scene_tokens = json.load(open(TRAIN_TOK_JSON))
    val_scene_tokens   = json.load(open(VAL_TOK_JSON))
    print(f"[in] train: {len(train_scene_tokens)} scene tokens")
    print(f"[in] val:   {len(val_scene_tokens)} scene tokens")
    overlap = set(train_scene_tokens) & set(val_scene_tokens)
    assert not overlap, f"train/val overlap at scene level: {len(overlap)}"

    print(f"[load] NuScenes(v1.0-trainval) from {DATAROOT}…", flush=True)
    nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)

    train_starts, train_names = collect_sample_tokens(nusc, train_scene_tokens)
    val_starts,   val_names   = collect_sample_tokens(nusc, val_scene_tokens)
    print(f"[walk] train: {len(train_starts)} sample tokens across {len(train_names)} scenes")
    print(f"[walk] val:   {len(val_starts)} sample tokens across {len(val_names)} scenes")
    sample_overlap = set(train_starts) & set(val_starts)
    assert not sample_overlap, f"train/val overlap at sample level: {len(sample_overlap)}"

    out = {
        "T": T,
        "source": {
            "train_scene_tokens_from": TRAIN_TOK_JSON,
            "val_scene_tokens_from":   VAL_TOK_JSON,
        },
        "total_sequences": len(train_starts) + len(val_starts),
        "train_count": len(train_starts),
        "val_count":   len(val_starts),
        "other_count": 0,
        "train_scenes": sorted(train_names),
        "val_scenes":   sorted(val_names),
        "train_start_tokens": train_starts,
        "val_start_tokens":   val_starts,
    }

    Path(os.path.dirname(OUT_PATH) or ".").mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[write] {OUT_PATH}  "
          f"({len(train_starts) + len(val_starts)} total sample tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

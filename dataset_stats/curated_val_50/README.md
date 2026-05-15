# curated_val_50/

**S2GO validation subset.** 50 scenes curated from the canonical nuScenes
paper-val (150 scenes) — a "miniature paper val" for fast dev-loop evaluation.
Same val set for both Stage-1 and Stage-2 (they share the SurroundOcc 16-class
mIoU/IoU protocol).

- See [summary.md](summary.md) for headline numbers, audit table, and methodology.
- Pair with [../curated_train_270/](../curated_train_270/) for training (disjoint by construction).
- For final reportable numbers, run eval on **the full 150-scene paper-val** rather than this 50-scene subset.

## Quick stats

- 50 scenes / ~2,003 keyframes / ~16 min driving
- ~14 anchored scenes (top-2 per rare thing class from within the 150 pool)
- **val ∩ curated_train_270 = ∅** (verified by script)
- Minimum class scene count: **trailer at 18** (out of 50) — sufficient for stable per-class IoU
- All other rare classes ≥ 29 scenes (const_veh 30, motorcycle 29, bicycle 32, bus 33, barrier 34)

## Files

```
curated_val_50/
├── README.md                        ← this file
├── summary.md                       ← detailed audit + plots index
├── tokens.json                      ← 50 scene tokens in pick order
├── scenes.csv                       ← compact per-scene list
├── scene_log.csv                    ← full per-scene log (24 columns)
├── per_class_scene_counts.csv       ← per-class audit table
└── figures/                         ← 5 plots vs the 150-scene val pool
    ├── 08_curated_origin_per_part.png
    ├── 09_curated_location_vehicle.png
    ├── 10_curated_conditions.png
    ├── 11_curated_class_counts.png
    └── 12_curated_retention_by_class.png
```

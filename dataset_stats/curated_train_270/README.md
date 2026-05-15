# curated_train_270/

**S2GO training subset.** 270 scenes curated from the canonical nuScenes train
split (700 scenes), with the paper-val 150 scenes excluded for a clean
train / val partition. Drop-in replacement for full-trainval training that
keeps **every class ≥ 109 scenes** and rare-class instance retention at 49–77%.

- See [summary.md](summary.md) for headline numbers, audit table, and methodology.
- Pair with [../curated_val_50/](../curated_val_50/) for evaluation.

## Quick stats

- 270 scenes / 10,776 keyframes / ~1h 28m
- 33 anchored scenes (top-K per rare thing class)
- **train ∩ paper-val = ∅** (verified by script)
- Class retention: barrier 77%, const_veh 63%, bus 62%, traffic_cone 61%, motorcycle 57%, other_flat 53%, bicycle 52%, trailer 49%

## Files

```
curated_train_270/
├── README.md                        ← this file
├── summary.md                       ← detailed audit + plots index
├── tokens.json                      ← 270 scene tokens in pick order
├── scenes.csv                       ← compact per-scene list
├── scene_log.csv                    ← full per-scene log (24 columns)
├── per_class_scene_counts.csv       ← per-class audit table
└── figures/                         ← 5 plots vs the 700-scene train pool
    ├── 08_curated_origin_per_part.png
    ├── 09_curated_location_vehicle.png
    ├── 10_curated_conditions.png
    ├── 11_curated_class_counts.png
    └── 12_curated_retention_by_class.png
```

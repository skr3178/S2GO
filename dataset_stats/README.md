# dataset_stats/

nuScenes trainval diversity / class-composition statistics for the S2GO project,
plus the curated **train + val** subsets (clean train/val split, no leakage).

## What's here

```
dataset_stats/
├── README.md                          ← this file
│
├── statistics.md                      ← global stats (1.17M boxes, 16-class composition, figs 1–7)
├── nuscenes_diversity_stats.md        ← per-archive diversity + description vocabulary
│
├── data/                              ← raw analysis outputs (per-scene CSVs)
│   ├── scene_stats.csv                  (850 rows: per-scene tags + archive part)
│   ├── class_composition_per_scene.csv  (850 rows × 16 class counts)
│   ├── class_composition_summary.txt    (global totals)
│   ├── curated_v1_scenes.txt            (description-tag stratified, 255 scenes — historical, no train/val split)
│   ├── curated_v1_tokens.json           (historical)
│   ├── curated_v2_scenes.csv            (anchor+greedy, 255 scenes, no split — historical)
│   └── curated_v2_tokens.json           (historical)
│
├── figures/                           ← per-archive plots (figs 1–7)
│   ├── 01_per_part_size.png
│   ├── 02_per_part_location.png
│   ├── 03_per_part_vehicle_conditions.png
│   ├── 04_thing_class_per_part_heatmap.png
│   ├── 05_stuff_class_per_part_heatmap.png
│   ├── 06_class_scarcity.png
│   └── 07_scene_dur_samples_hist.png
│
├── curated_train_270/                 ★ TRAIN subset (S2GO Stage-1 & Stage-2)
│   ├── README.md  summary.md
│   ├── tokens.json  scenes.csv  scene_log.csv
│   ├── per_class_scene_counts.csv
│   └── figures/  (08–12, vs 700-scene train pool)
│
└── curated_val_50/                    ★ VAL subset (same for Stage-1 & Stage-2)
    ├── README.md  summary.md
    ├── tokens.json  scenes.csv  scene_log.csv
    ├── per_class_scene_counts.csv
    └── figures/  (08–12, vs 150-scene val pool)
```

## Use this for S2GO Stage-1 / Stage-2 training

**Train on [curated_train_270/](curated_train_270/)** (270 scenes, 10,776
keyframes) — clean of the 150 paper-val scenes.
**Eval on [curated_val_50/](curated_val_50/)** for dev-loop, or on the **full
150-scene paper-val** for final reportable numbers.

The two are disjoint by construction (built from the canonical nuScenes
700 / 150 train/val split via `nuscenes.utils.splits`).

| Subset | Scenes | Keyframes | Min class scenes | Pool |
|---|---:|---:|---:|---|
| **curated_train_270** | 270 | 10,776 | trailer at 109 | 700-scene train split |
| **curated_val_50** | 50 | 2,003 | trailer at 18 | 150-scene paper-val |

## Historical curations (no train/val split)

`data/curated_v1_*` and `data/curated_v2_*` were built from all 850 scenes
**before** the clean train/val split was applied. They contain ~33% paper-val
leakage (50 of 150 val scenes were mixed in) and should not be used for
training. Kept for archival reference only — the per-scene diversity and
class-composition CSVs themselves (`scene_stats.csv`,
`class_composition_per_scene.csv`) are still valid since they cover all 850
scenes independent of any split.

## Generation scripts

| Script | Produces |
|---|---|
| [../scripts/analyze_nuscenes_diversity.py](../scripts/analyze_nuscenes_diversity.py) | `data/scene_stats.csv` |
| [../scripts/build_class_composition_stats.py](../scripts/build_class_composition_stats.py) | `data/class_composition_per_scene.csv`, summary.txt |
| [../scripts/plot_dataset_stats.py](../scripts/plot_dataset_stats.py) | `figures/01..07_*.png` |
| [../scripts/curate_anchor_greedy.py](../scripts/curate_anchor_greedy.py) | (used by build_train_val_split.py; supports `--budget`) |
| [../scripts/build_train_val_split.py](../scripts/build_train_val_split.py) | `curated_train_270/`, `curated_val_50/` |

To rebuild train + val together:

```bash
python scripts/build_train_val_split.py \
  --scene-csv dataset_stats/data/scene_stats.csv \
  --class-csv dataset_stats/data/class_composition_per_scene.csv \
  --out-train dataset_stats/curated_train_270 \
  --out-val   dataset_stats/curated_val_50 \
  --train-budget 270 --val-budget 50
```

## Headline numbers

- **Full trainval:** 850 scenes (700 train + 150 val), 34,149 keyframes,
  4h 37m, ~1.17M 3D-box annotations, all 16 SurroundOcc classes.
- **Curated train (270):** ~38.6% of train pool. Rare-class retention 49–77%
  (barrier 77%, const_veh 63%, bus 62%, traffic_cone 61%, motorcycle 57%,
  bicycle 52%, trailer 49%). Min scene count: trailer at 109.
- **Curated val (50):** ~33% of paper-val. Rare-class retention 38–68%
  (barrier 68%, traffic_cone 61%, const_veh 58%, other_flat voxels 57%,
  bus 53%, motorcycle 50%, trailer 44%, bicycle 38%). Min scene count:
  trailer at 18.
- **Disjointness:** `curated_train_270 ∩ paper-val(150) = ∅`. Verified.

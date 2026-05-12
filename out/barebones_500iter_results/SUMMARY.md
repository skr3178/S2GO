# S2GO Stage-1 barebones recipe — 500-iter results

**Date:** 2026-05-12
**Run:** `python -m s2go.tools.overfit --n-iters 500 --num-layers 6 --num-pts 13 --feedforward-channels 3072 --mixed-precision --amp-dtype bf16` (no recipe flag → barebones default)
**Checkpoint:** `out/barebones_500iter/ckpt.pt` (413 MB, 108 M params)

## Config (paper-ablation mapping)

| Knob | Value | Maps to |
|---|---|---|
| Loss | `L_depth` only at dt=0 | Table 3 row e (20.25 mIoU pretraining gain) |
| T_seq (loader) | 1 | (single-frame) |
| T_queue (memory) | 1 | Table 4 row 1 — Propagation=None (17.92 mIoU) |
| Warps (±0.5s) | off | Table 5 row 1 — Velocity=None (20.07 mIoU) |
| Query init | LiDAR (32-line) + ε | Table 8 row 2 (21.60 mIoU; tied with best) |
| Backbone | R50 + FPN, trainable, lr × 0.25 | paper §B |
| Decoder | 6 layers, num_pts=13, ffn=3072 | paper-spec T2 |
| Precision | bf16 autocast | paper §B |
| Gradient checkpointing | OFF | (memory headroom in barebones) |
| LR schedule | cosine, T_max=500, decays to **0** | ⚠ buggy for short runs — see "Known issues" |

## Training trajectory (selected iters)

| iter | L_depth | opa_mean | opa_min | topk_opa | gnorm_v | gnorm_rgb |
|---|---|---|---|---|---|---|
| 0   | 13.90 m | 0.266 | 0.122 | 0.664 | 0 | 0 |
| 10  | 4.49 m  | 0.354 | 0.057 | 0.996 | 0 | 0 |
| 50  | 3.15 m  | 0.162 | 0.006 | 0.992 | 0 | 0 |
| 100 | 3.07 m  | 0.185 | 0.006 | 0.988 | 0 | 0 |
| 200 | 2.73 m  | 0.175 | 0.001 | 0.986 | 0 | 0 |
| 300 | 2.31 m  | 0.181 | 0.001 | 0.986 | 0 | 0 |
| 400 | 2.13 m  | 0.199 | 0.001 | 0.985 | 0 | 0 |
| **499** | **1.94 m** | 0.192 | 0.001 | 0.986 | 0 | 0 |

**Headline numbers:**
- L_depth dropped 13.90 → **1.94 m** (-86 %) on the cached 4-sequence overfit
- Walltime: ~5 min for 500 iters (~0.6 s/iter), peak GPU memory 4.29 GB
- `gnorm_velocity` and `gnorm_rgb` were pinned at 0 every single iter (structural invariants of barebones)

## Per-artifact eval

### 1. Training curves — `training_curves.png`

Six panels: L_depth, LR, total gradient norm, opacity, scale+velocity, per-head gnorms.

| Panel | Reading |
|---|---|
| L_depth (log y) | Smooth monotone descent; crosses 2 m threshold around iter 320 |
| Cosine LR | Decays to ~0 by iter 500 — model effectively idle in last ~20 % of run |
| ‖grad‖ pre-clip | 5–15 range, well under clip threshold of 35 |
| Opacity | top-k (256 survivors) holds at 0.99 the entire run; all-9000 mean drops 0.27 → 0.19 |
| Scale + velocity | Scale settles at ~1.7 m; velocity drifts under no supervision (0.7 → 2.1 m/s) |
| Per-head gnorms | velocity and rgb pinned at 0; all other heads training |

### 2. BEV scatter — `eval_A_bev.png`

Refined queries (red) over LiDAR points (cyan) and FPS anchors (blue), top-down view in LIDAR_TOP frame (+y forward — see `hacks.md` H3).

| Metric | Value |
|---|---|
| Mean nearest-LiDAR distance per query | **2.21 m** |
| Median | 1.98 m |
| Min / Max | 0.09 m / 6.36 m |

**Reading:** queries drifted *further* from LiDAR than init noise (which was ±1 m). Expected for depth-only — without `L_denoise` to clamp queries to FPS anchors, the model uses `parent.offset` to push them toward visible-surface depths from cameras. This is the price of barebones; a paper recipe with `λ_denoise > 0` would pin queries near LiDAR.

### 3. Renders — `eval_B_renders.png`

3 rows (CAM_FRONT, CAM_FRONT_LEFT, CAM_BACK) × 4 cols (GT image, rendered RGB, GT depth, rendered depth).

| Column | Reading |
|---|---|
| GT image | real camera view (256 × 704) |
| Rendered RGB | muddy / random colors — **expected** (λ_rgb = 0, RGB rows gradient-pinned at 0) |
| GT depth | sparse — only ~3 k of 180 k pixels populated by LiDAR projection |
| Rendered depth | **clear horizon, ground plane, structural variation** — recognizable scene geometry |

The depth column on the right is the success signal of barebones training.

### 4. Gaussian-state histograms — `eval_C_histograms.png`

| Channel | Distribution | Interpretation |
|---|---|---|
| Opacity | Heavy spike at 0, thin tail to 0.94, μ=0.20 | Healthy sparsification: most Gaussians faded, ~256 survivors at high opacity |
| Scale L2 | **Bimodal** — peak ~0.3 m + peak ~2.5 m, μ=1.67 m | Hierarchy used as intended: parents ≈ large coarse Gaussians, children include point-like ones |
| Velocity | ~Gaussian centered at 0.7 m/s, max 1.58 | Unsupervised drift but bounded — head receives zero gradient |

### 5. Numeric eval — `eval_stats.json`

```json
"nearest_lidar_distance_per_query": {"mean": 2.21, "median": 1.98, "min": 0.09, "max": 6.36},
"rendered_depth_range": [0.00, 70.08],
"rendered_rgb_range":   [0.00, 0.69],
"memory":               {"capacity": 256, "populated": 256, "unique_xyz": 256},
"topk_opacity":         {"mean": 0.985, "std": 0.002, "min": 0.982, "max": 0.990}
```

Notable:
- Depth range **[0, 70 m]** spans full nuScenes scene (no degenerate constant prediction)
- `topk_opacity` σ = 0.002 — survivors essentially uniform at peak
- Memory `populated == unique == 256` — propagator δ-NMS deduplication working

## Verdict — Stage-2 readiness

| Dimension | Status | Comment |
|---|---|---|
| Depth pathway works | ✓ | render coherent, scalar 1.94 m below 2 m threshold |
| Opacity uses hierarchy | ✓ | sparsification + bimodal scale |
| Query positions clamp to LiDAR | ✗ | 2.2 m drift; need `λ_denoise > 0` to fix |
| Velocity supervised | n/a | barebones design (Table 5 row 1) |
| RGB pathway trained | n/a | barebones design (Table 3 row e) |

The architecture works as the recipe intends. The 2.2 m query drift is the only thing that *might* hurt Stage 2 if the semantic supervision needs queries to be in LiDAR-occupied voxels specifically. Cleanest test: train Stage 2 from this checkpoint when available, compare mIoU to paper Table 5 row 1's 20.07.

## Known issues / next steps

1. **Cosine LR decays to 0** by iter 500 — the last ~20 % of training contributed little because LR ≈ 0. Fixed in code: new `--lr-schedule constant` flag (now default) holds peak LR throughout. A re-run with this fix should land L_depth below the current 1.94 m floor.
2. **2.21 m query drift from LiDAR** is the recipe-induced cost of removing denoise loss. To recover the paper's 21.6 mIoU ceiling, add `λ_denoise > 0` back. Currently disabled in barebones to isolate the depth pathway.
3. **No streaming-data training yet** — these 500 iters cycled 4 cached frames. Full Part-1 streaming (~3,121 sequences) with multi-epoch would expose the model to real scene diversity.

## File inventory

| File | Bytes | Description |
|---|---|---|
| `SUMMARY.md` | (this file) | Per-artifact analysis + verdict |
| `training_curves.png` | 280 KB | 6-panel loss / opacity / gnorm trajectories |
| `training_history.json` | 708 KB | Raw per-iter metrics (500 iters × 40 fields) |
| `eval_A_bev.png` | 209 KB | BEV scatter (queries vs LiDAR vs FPS anchors) |
| `eval_B_renders.png` | 873 KB | GT image vs rendered RGB / depth (3 cams) |
| `eval_C_histograms.png` | 53 KB | Opacity / scale / velocity histograms |
| `eval_stats.json` | 1 KB | Numeric eval summary |

# Stage 1 Eval — Reading Guide

Visual + numeric outputs from [s2go/tools/stage1_eval.py](../../s2go/tools/stage1_eval.py),
run on the saved 50-iter T2 checkpoint at `/tmp/stage1_t2_50iter.pt`.

The checkpoint was trained with:
- Architecture: S2GO-Small (K=900, J=10, embed=768, 6 layers, 13 cross-attn pts, ffn 3072)
- Mixed precision (bf16), gradient checkpointing, RGB loss as pure L1 (no SSIM)
- ±0.5s velocity warp ON (per paper §3.3.3)
- 4 fixed overfit sequences from nuScenes Part 1, 50 iterations

**Recipe note (decided 2026-05-11, see [Stage1_design.md §11.6](../../Stage1_design.md)):** going
forward, **L_rgb is dropped** from Stage 1 first-pass runs (`--no-rgb` flag).
The figures in this folder are from a *pre-drop* checkpoint and still show the
RGB columns. New runs trained with `--no-rgb` will produce:
- Col 2 of `B_renders.png`: empty / `--` (no RGB render path)
- All `rgb_*` fields in `eval_stats.json`: `NaN`
- `gnorm_rgb` in training history: 0
The depth columns, BEV, histograms, and queue diagnostics are unaffected.

**This is qualitative.** Quantitative pass/fail thresholds (PSNR > 23 dB,
masked-depth L1 < 1 m, etc.) require full-scale training; on a 4-sample /
50-iter checkpoint we expect partial signal at best. The purpose of these
figures is **"is anything obviously broken"**, not **"is this paper-quality"**.

---

## Files in this folder

| File | What it shows |
|---|---|
| [A_bev.png](A_bev.png) | Top-down (X-Y) BEV view of refined queries vs LiDAR points and FPS anchors |
| [B_renders.png](B_renders.png) | 3 cameras × 4 columns: GT image, rendered RGB, GT depth, rendered depth |
| [C_histograms.png](C_histograms.png) | Distributions of Gaussian state: opacity, ‖scale‖, ‖velocity‖ |
| [eval_stats.json](eval_stats.json) | Numeric companion (nearest-LiDAR distances, render ranges, queue stats, etc.) |

---

## Data vs model — what comes from where

The eval mixes **dataset-derived inputs** (raw LiDAR, raw cameras, GT depth)
with **model-derived outputs** (refined queries, rendered RGB/depth, Gaussian
state distributions). To read the figures correctly, know which is which.

### A_bev.png — single panel, three overlaid layers

| Layer | Color | Source |
|---|---|---|
| LiDAR cloud | light grey dots | **Dataset** (`lidar_pts` from loader, last frame) |
| FPS anchors | blue dots | **Dataset-derived** — `FPS_K(lidar_pts)` is deterministic given input. This is the **noise-free denoise target**. |
| Refined queries | red dots | **Model checkpoint** — `init_xyz + parent.offset` (the model's prediction) |

**How to read:**
- Red dots clustered near grey LiDAR points → queries are placed sensibly.
- Red dots overlapping blue anchors exactly → denoise loss has converged.
- Red dots floating in empty regions (no nearby grey) → broken FPS init or
  uncontrolled offset → likely a bug.
- Wide red-vs-blue scatter (current state, ~0.8 m mean offset) → consistent
  with denoise loss still at the 1.5 m noise floor. Expected at 50 iters
  on 4 overfit samples.

---

### B_renders.png — 3 rows × 4 columns (12 panels)

Layout: each row = one camera (CAM_FRONT, CAM_FRONT_LEFT, CAM_BACK).
Columns are in fixed order across all rows:

| Col | Title | Source | Notes |
|---|---|---|---|
| **1** | "CAM_* GT image" | **Dataset** | Raw 256×704 photo |
| **2** | "rendered RGB" | **Model** | Gaussians → gsplat → image |
| **3** | "GT depth (LiDAR)" | **Dataset** | LiDAR projected into image plane — **VERY sparse** (~5% of pixels) |
| **4** | "rendered depth" | **Model** | Gaussians → gsplat → accumulated depth |

**Pattern: `[GT, prediction, GT, prediction]`** — left two cols compare RGB,
right two cols compare depth.

**Known visual quirk — column 3 looks mostly empty.** This is *expected*,
not a bug:

- nuScenes LiDAR has only ~30 k points per frame across 6 cameras × 256 × 704
  ≈ 1 M pixels = roughly 3 % of pixels per camera get a LiDAR return.
- In our viz, masked (zero-return) pixels are rendered as the figure
  background (white). Only the ~3 % valid pixels show viridis colors.
- At print scale, those tiny scattered dots are nearly invisible.
- The depth supervision *still works* — the rendered depth (col 4) is
  trained against just those sparse pixels via masked L1, not against a
  dense GT depth map.

**How to read:**
- **Col 2 (rendered RGB)** should look like a coarse smeary version of col 1.
  Currently: muddy grey-blue blobs, scene structure not recoverable. This
  is the RGB-head-undertrained flag we documented elsewhere.
- **Col 4 (rendered depth)** should show coherent near-far gradients matching
  the scene in col 1 — road plane as yellow/green, buildings as bands of
  green/teal, sky as dark purple. Currently this works visibly.
- Col 3's emptiness is a viz limitation; don't conclude "GT is wrong."

---

### C_histograms.png — 3 side-by-side panels (all model)

| Panel | Title | Source |
|---|---|---|
| Left | opacity | **Model** — `out.gaussians.opacities` |
| Middle | ‖scale‖ | **Model** — L2 of (sx, sy, sz) |
| Right | ‖velocity‖ | **Model** — L2 of velocity 3-vec |

All three are pure model output — no GT distribution to overlay. You read
each against priors:

- **Opacity** healthy = some distribution, not all near 0 or all near 1.
  Bimodal (spike near 0 = "ignored" + cluster at moderate values =
  "active") is common and fine.
- **Scale L2** healthy = mix of small/medium/large, not collapsed to one
  value and not piled at the upper bound. Trimodal (small children +
  medium spread + bound peak) is typical for the J=10 child structure.
- **‖velocity‖** healthy at scale = bimodal (static cluster near
  −v_ego + dynamic cluster at object speeds). A single tight peak
  (current state: 3.3 m/s ± 0.1) indicates a **lazy velocity head** —
  model has learned "everything is static" and outputs a global
  constant ≈ negative ego speed in LIDAR frame.

---

## Companion data — eval_stats.json

Schema:
```json
{
  "checkpoint":   "/path/to/checkpoint.pt",
  "seq_idx":      0,
  "A_bev":       {"nn_dist_mean": 0.77,  "nn_dist_median": 0.77, "nn_dist_min": 0.02, "nn_dist_max": 1.70},
  "B_render":    {"rgb_minmax": [0.00, 0.45], "depth_minmax": [0.0, 56.3]},
  "C_gauss":     {"opacity":  {"mean": ..., "std": ..., "min": ..., "max": ...},
                   "scale":    {...},
                   "velocity": {...}},
  "D_queue":     {"memory_capacity": 1024, "memory_populated": 1024,
                   "memory_unique_xyz": 1024,
                   "topk_opacity_min": 0.32, "topk_opacity_max": 0.35,
                   "topk_opacity_mean": 0.33, "topk_opacity_std": 0.007}
}
```

- `A_bev.nn_dist_*` — per-query distance to nearest LiDAR point, in meters.
  Should drop with training as queries snap onto surfaces.
- `B_render.*_minmax` — output ranges from the gsplat render. RGB should
  span more of [0, 1] as training progresses; depth should peak around
  50 m for nuScenes (further than that becomes sky).
- `C_gauss.*` — same stats as the histograms, in tabular form.
- `D_queue.*` — memory queue + propagator state. `memory_unique_xyz`
  should equal `memory_populated` (no degenerate duplicates).
  `topk_opacity_std` near zero means the propagator is picking from a
  very tight opacity band (currently the case — uniform opacities limit
  discrimination).

---

## Current checkpoint reading (50-iter T2 overfit)

| Check | Pass? | Reading |
|---|---|---|
| A. BEV coverage | ⚠️ partial | Queries near LiDAR (mean 0.77 m) but not snapped — consistent with L_denoise at the 1.5 m floor |
| B. Rendered depth | ✅ | Scene structure clearly visible in col 4 — depth supervision is doing real work |
| B. Rendered RGB | ❌ | Muddy blobs, RGB head essentially un-supervised due to loss weighting |
| C. Opacity | ✅ | Bimodal-ish, healthy mix of "active" and "ignored" |
| C. Scale | ✅ | Trimodal spread, neither collapsed nor exploded |
| C. Velocity | ⚠️ partial | Tight peak at ~ego speed; lazy "everything-is-static" head |
| D. Queue | ✅ | 1024/1024 unique, sane opacity spread |

**Net:** depth path works, RGB path needs either λ-tuning or scale-up;
velocity is supervised but degenerate at this scale; everything else is
structurally healthy.

**Why current images don't look paper-quality:** 4-sample overfit + 50
iters caps what these checks can reveal. The model has only seen each
scene ~12 times. Real-data scale-up is the next informative escalation,
not more iters on the same 4 samples.

---

## Re-running

```bash
cd /media/skr/storage/self_driving/S2GO
source /home/skr/miniconda3/etc/profile.d/conda.sh
conda activate /media/skr/storage/conda_envs/selfocc

# 1. Train + save a checkpoint (~2.5 min for T2 + warp + ckpt on a 3060)
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  python -m s2go.tools.overfit \
    --n-iters 50 --num-layers 6 --num-pts 13 --feedforward-channels 3072 \
    --mixed-precision --amp-dtype bf16 --rgb-ssim-weight 0 --use-checkpoint \
    --save-path /tmp/stage1_ckpt.pt --history-path /tmp/stage1_history.json

# 2. Run the 4 checks on the checkpoint (~30 s)
python -m s2go.tools.stage1_eval \
    --checkpoint /tmp/stage1_ckpt.pt \
    --out-dir out/stage1_eval/ \
    --seq-idx 0
```

`--seq-idx` selects which loader sequence to use for the visualizations
(0..N-1, where N is the number of usable T=4 sequences in Part 1 — about
3,121 in our current extraction).

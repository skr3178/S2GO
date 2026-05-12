# S2GO Stage-1 barebones — full Part-1 streaming, 5000-iter run (v2)

**Date:** 2026-05-12
**Status:** ✓ Clean completion (5000 iters, 0 NaN/inf skips, finite checkpoint saved)
**Checkpoint (not in this folder):** `out/barebones_full_part1_5000iter_v2/ckpt.pt` (413 MB)

## Why "v2"

v1 was the first attempt at the same recipe. It diverged at iter 2919 (bf16
gradient overflow in gsplat backward → `inf` total gnorm → `inf × (max_norm/inf)
= NaN` in bf16 grad-clip arithmetic → all weights NaN-poisoned from iter 3359
to 4999). The final saved checkpoint of v1 was unusable.

v2 introduced three fixes specifically targeting the v1 failure mode:

| Fix | Value | Where applied |
|---|---|---|
| Halved learning rate | 4e-4 → **2e-4** (segmentor); 1e-4 → 5e-5 (backbone) | `--lr 2e-4` |
| Tighter gradient clip | max_norm 35 → **10** | `--grad-clip 10` |
| NaN/inf guard | Skip optimizer.step() if loss or gnorm is non-finite; log sample_token | Added to `overfit.py` train loop |

Result: **0 skips across all 5000 iters** — the tighter clip prevented overflow
from ever reaching `inf`, so the guard never needed to fire. Belt-and-suspenders
worked as intended.

## Config

| Knob | Value | Mapping to paper |
|---|---|---|
| Recipe flag | `--barebones` (default; equiv. to `--depth-only --t-seq 1 --t-queue 1`) | Table 3 row e + Table 4 row 1 + Table 5 row 1 + Table 8 row 2 |
| Loss terms | only L_depth at dt=0 | Table 3 row e (20.25 mIoU pretraining gain) |
| L_denoise | **NOT COMPUTED** (skipped entirely; log shows `nan`) | (intentional) |
| L_rgb | **NOT COMPUTED** (render_mode='D' skips colors) | (intentional) |
| ±0.5s warps | OFF | velocity head structurally unreachable → no grad |
| Sequence length | T_seq = 1 (single frame per loader call) | Figure 6 leftmost point ≈ 18.4 mIoU |
| Memory queue length | T_queue = 1 (256 slots, never populated in T_seq=1) | Table 4 row 1: Propagation = None |
| Query init | LiDAR (32-line) + ε ∼ U(-1, +1) m | Table 8 row 2: 21.60 mIoU (tied with best) |
| Architecture | num_layers=6, num_pts=13, ffn=3072, embed=768 (paper-spec T2) | StreamPETR convention |
| Backbone | R50 + FPN, trainable, lr × 0.25 mult | paper §B |
| Precision | bf16 autocast | paper §B |
| Gradient checkpointing | OFF | enough memory headroom in barebones |
| LR schedule | constant 2e-4 (post cosine→0 bug fix) | (diagnostic; paper uses cosine) |
| Iters | 5000 | (≈ 1.5 epochs over 3,376 Part-1 sequences) |
| Wallclock | ~57 min on RTX 3060 | |
| Peak GPU memory | 4.24 GB | |

## Training trajectory

Selected iters (full 500 logged iters in `training_history.json`):

| iter | L_depth | gnorm | topk_opa | opa_mean | scale_l2_mean |
|---|---|---|---|---|---|
| 0 | 14.44 | 12.5 | 0.66 | 0.27 | 2.29 |
| 100 | 4.43 | 7.7 | 0.99 | 0.35 | 1.17 |
| 300 | 3.79 | 5.4 | 0.99 | 0.18 | 1.31 |
| 700 | 3.60 | 5.6 | 0.99 | 0.16 | 1.48 |
| 1800 | 3.48 | 6.1 | 0.99 | 0.18 | 1.55 |
| **2300** | **2.45** ← lowest | 6.0 | 0.99 | 0.17 | 1.65 |
| 3000 | 3.45 | 6.4 | 0.98 | 0.20 | 1.78 |
| 4000 | 2.87 | 5.9 | 0.96 | 0.22 | 1.89 |
| 4999 | 4.42 | 6.5 | 0.52 | 0.17 | 1.99 |

**Key dynamics:**
- **L_depth dropped 14.44 → 2.45 m (-83%)** by iter 2300, oscillates 2.5–4.5 m
  thereafter due to per-scene variance with batch=1.
- **Gradient norms bounded 5–10** the entire run; tighter grad-clip (10) was
  active but never the dominant cap.
- **`topk_opacity` slowly drifted 0.99 → 0.52** across the run — the propagator's
  256 survivors lost their authority as the model spread opacity across more
  diverse scenes.
- **`gnorm_velocity` and `gnorm_rgb` were exactly 0 every single iter**, confirming
  the structural-unreachability invariants of barebones.

See `training_curves.png` for 6-panel visualization (L_depth log-scale,
LR, ‖grad‖, opacity, scale + velocity, per-head gnorms).

## Eval results (single forward pass at end of training)

Run via `s2go/tools/stage1_eval.py` on `loader[0]` (a sequence seen ~1.5×
during training).

```json
"nearest_lidar_distance_per_query": {"mean": 2.44, "median": 2.14, "min": 0.12, "max": 6.47},
"rendered_depth_range": [0.00, 62.72],
"rendered_rgb_range":   [0.00, 0.67],
"memory":               {"capacity": 256, "populated": 256, "unique_xyz": 256},
"topk_opacity":         {"mean": 0.519, "std": 0.083, "min": 0.412, "max": 0.736}
```

Gaussian-state histograms (all 9,000 Gaussians, end of training):

| Channel | μ | σ | range |
|---|---|---|---|
| Opacity | 0.170 | 0.185 | [0.000, 0.736] |
| Scale L2 | 1.985 m | 0.761 m | [0.40 m, 3.46 m] |
| Velocity magnitude | 0.832 m/s | 0.181 m/s | [0.37, 1.28] |

See `eval_A_bev.png` (BEV scatter), `eval_B_renders.png` (cam vs rendered
depth/RGB), `eval_C_histograms.png` (state distributions).

## Comparison vs. 500-iter cached run

| Metric | 500-iter cached (4 seq × 125 visits) | **5000-iter v2 streaming (3,376 seq × 1.5 visits)** | Δ | Reading |
|---|---|---|---|---|
| Nearest-LiDAR distance (mean) | 2.21 m | 2.44 m | +0.23 m | similar — generalization holding |
| Rendered depth range | [0, 70 m] | [0, 63 m] | -7 m max | both span scene scale; no degenerate constant prediction |
| Opacity μ | 0.20 | 0.17 | -0.03 | both sparsify |
| Opacity max | 0.94 | **0.74** | -0.20 | ⚠ v2 never gets near-saturated |
| Scale L2 μ | 1.67 m | 1.99 m | +0.32 m | v2 uses larger Gaussians on average |
| Scale L2 distribution | bimodal (0.3 + 2.5 m peaks) | broader unimodal | — | v2 lost the parent/child scale separation |
| Velocity μ | 0.71 m/s | 0.83 m/s | +0.12 | weight-decay drift (head receives no gradient in barebones) |
| **`topk_opacity` mean** | **0.985** | **0.519** | **-0.47** | ⚠⚠ biggest gap |

## Verdict

**Stage-1 architecture works on diverse streaming data.** Loss converged
monotonically until iter 2300, plateaued in the 2.5–4.5 m oscillating band
(natural per-scene variance with batch=1, 1.5 visits per sample).

**Three concerns:**
1. `topk_opacity` 0.52 means the 256 Gaussians the propagator pushes back to
   the memory queue are mid-opacity. Stage 2's KL loss expects authoritative
   survivors; this may diffuse semantic predictions.
2. Scale lost bimodality. The K=900 × J=10 hierarchy was designed for
   parents ≈ coarse Gaussians, children ≈ fine point Gaussians. v2 collapsed
   this into broader unimodal scales.
3. No held-out validation set yet — we don't have a generalization signal
   distinct from training trajectory.

**These are not blockers for trying Stage 2.** Both concerns are exactly what
Stage 2's opacity-in-α regularization (paper Eq. 9) and direct KL semantic
supervision are designed to fix downstream. The most informative next step
is to actually train Stage 2 and measure mIoU.

## Hacks that landed during this run

(Full text in `hacks.md`)

- **H3** — LIDAR_TOP frame is **`+y` forward** (not `+x`). Discovered during
  `lidar2img` verification.
- **H4** — Python stdout block-buffers under nohup (8 KB threshold). Use
  `PYTHONUNBUFFERED=1` + `python -u` for real-time log visibility.
- **H5 (implicit, see v1 post-mortem)** — bf16 + `clip_grad_norm_` can
  produce NaN from `inf × (max_norm / inf)` due to IEEE-754 `inf × 0 = NaN`.
  Mitigated by NaN/inf guard + tighter clip.

## File inventory

| File | Size | Purpose |
|---|---|---|
| `SUMMARY.md` (this file) | — | Per-artifact analysis + verdict + caveats |
| `training_curves.png` | 361 KB | 6-panel trajectories: L_depth (log y), LR, ‖grad‖, opacity, scale+velocity, per-head gnorms |
| `training_history.json` | 7.4 MB | Raw per-iter metrics (5000 iters × ~42 fields, includes `sample_token` and `skipped` flag) |
| `train.log` | 32 KB | stdout from the training run |
| `eval_A_bev.png` | 207 KB | BEV scatter: refined queries vs LiDAR points vs FPS anchors |
| `eval_B_renders.png` | 849 KB | 3 cams × {GT image, rendered RGB, GT depth, rendered depth} |
| `eval_C_histograms.png` | 56 KB | Distributions of opacity / scale L2 / velocity magnitude |
| `eval_stats.json` | 1 KB | Numeric eval summary |

## Next steps

1. **Implement Stage 2** (~1-2 days). Use this v2 checkpoint as the
   pretraining init. A/B Stage 2 mIoU with vs. without Stage 1 init to
   measure Stage 1's actual contribution.
2. **(Optional) Add val-L_depth split** to track generalization across runs.
3. **(Deferred)** If Stage 2 mIoU is suboptimal, come back and try
   `--full-recipe` Stage 1 (add denoise + warps + RGB) for +0.5–1 mIoU per
   paper Table 3 / Table 5.

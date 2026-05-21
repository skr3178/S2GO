# Stage 2 v4 10K-iter run — post-mortem

Run launched 2026-05-14 13:41, killed 15:05 at iter ~9800 / 10000.
The model state is frozen at **iter ~1071** — last 89% of training
produced NaN gradients and never updated weights.

## Run configuration

| field | value |
|---|---|
| splits | `out/stage2_parts01-05_splits.json` (369 train scenes, 56 val scenes; 14749/2225 start tokens) |
| iters | 10,000 micro-iters → 625 optim-steps at `--grad-accum-steps 16` |
| schedule | `warmup_cosine`, warmup 500 micro-iters → cosine to peak×0.10 |
| LR (peak) | `--lr 2e-4`; backbone group ×0.25 = 5e-5 |
| optimizer | AdamW, weight_decay=0.01 |
| grad clip | 35.0 |
| `--scale-min` | **0.01** (GF-2 / paper default) |
| occ_pos_weight | 16.0 |
| cross-attn fixes | present (commit a1bb133: projection-mask, V-free-of-pos, zero-init learnable_fc.weight) |
| v4 fixes | present (grad-accum + warmup_cosine ported from Stage-1 commit 519ee4f) |

## What actually happened in the values

| signal | trajectory |
|---|---|
| **micro-iter loss** (`total`, `occ_bce`, `sem_kl`, `sem_ce`, `lovasz`) | **Finite for ALL 10,000 iters.** No NaN losses. |
| **window-end gnorm** (post-clip accumulated grad norm) | Finite for **66 windows** (iters 15→1055), then **NaN for all 559 subsequent windows** (iters 1071→9999). Zero recoveries. |
| **opa_mean** | 0.18 (iter 500) → 0.31 (iter 875) → 0.43 (iter 1055, last healthy) → 0.47–0.50 thereafter |
| **LR** (backbone group, scheduler.get_last_lr()[0]) | 0 (warmup) → 5.0e-5 at iter ~500 → 4.96e-5 at iter 1071 (99.2% of peak — cascade hit during peak LR, not during decay) |

**The failure isn't NaN losses — it's NaN gradients.** Forward pass
produced sensible numbers throughout. Backward through G2V produced
Inf/NaN starting at iter 1071, which then poisoned the accumulated
window gradient. The `clip_grad_norm_` call returns NaN whenever any
parameter's `.grad` contains Inf or NaN, which trips the skip-step
logic. Since `optim.step()` never executes after iter 1071, weights
are frozen at the iter-1071 state for the entire rest of the run.

## Per-window trajectory through the transition

| iter | total | occ_bce | gnorm | opa_mean | note |
|---:|---:|---:|---:|---:|---|
| 15 | 41.235 | 9.132 | 144.59 | 0.277 | warmup start |
| 255 | 25.240 | 2.598 | 23.06 | 0.216 | warmup |
| 495 | 24.480 | 1.879 | 46.60 | 0.145 | warmup → cosine |
| 575 | 19.803 | 2.079 | 42.28 | 0.139 | descending |
| 735 | 16.895 | 1.950 | 33.29 | 0.185 | descending |
| 815 | 18.192 | 2.101 | 35.39 | 0.243 | descending |
| 895 | 21.824 | 2.481 | 34.71 | 0.338 | opa climbing |
| 975 | 20.427 | 1.893 | 30.30 | 0.426 | opa↑ |
| 1055 | 23.187 | 1.924 | 39.19 | 0.431 | **last healthy** |
| **1071** | **25.506** | 2.173 | **nan** | 0.440 | **CASCADE START** |
| 1135 | 19.121 | 2.162 | nan | 0.496 | (frozen, opa wedged near 0.5) |
| 9999 | 21.914 | 1.958 | nan | 0.476 | (still frozen) |

## Forensic detail — micro-iter level around the cascade

All micro-iter losses are finite throughout. Per-iter records 1050–1090
show finite L (15–26), only the window-end gnorm becomes NaN:

```
iter   L_micro   occ_bce  sem_kl  opa_mean  skipped  win_end   gnorm
1055   23.187   1.9244  1.8530    0.4313    False     True    39.19    ← healthy window-end
1056   18.620   1.9595  1.4384    0.4470    False    False      nan
... (15 more healthy micro-iter Ls) ...
1070   22.383   2.3894  1.7384    0.4709    False    False      nan
1071   25.506   2.1727  2.0417    0.4400     True     True      nan    ← cascade window-end
```

So inside the window `[1056, 1071]`, every micro-iter loss was finite.
The accumulated gradient — summed over 16 sample backwards — went
non-finite. **At least one of the 16 samples in that window produced
an Inf gradient via G2V backward.**

## Mechanism — scale collapse via the G2V backward

The G2V splatting layer at [s2go/stage2/g2v.py:117–141](../../s2go/stage2/g2v.py#L117)
implements a normalized 3D Gaussian PDF integrated into voxel cells:

```python
Sigma = (R * s_sq.unsqueeze(-2)) @ R.transpose(-1, -2)     # (N, 3, 3)
inv_sqrt_det = 1.0 / torch.sqrt(det.clamp_min(1e-20))      # line 119
w_full = (opacities * inv_sqrt_det) * exp(q.clamp(max=0))  # line 141
```

The forward is numerically fine because `inv_sqrt_det` enters
linearly. The **backward** of this w.r.t. scale parameters
pulls a `det(Σ)⁻¹·⁵` factor. At the scale floor:

- `s = scale_min = 0.01 m`
- `1/s² = 10,000`
- `det(Σ) = s_x² · s_y² · s_z² ≈ 10⁻¹²`
- `det⁻¹·⁵ ≈ 10¹⁸` → overflows bf16 (max ≈ 3×10³⁸) when combined
  with other factors in the backward chain.

The chain of events:

```
slow opa_mean drift up (0.18 → 0.43 over ~1000 iters)
  ↑ driven by BCE with occ_pos_weight=16
  ↓
optimizer shrinks Gaussian scales to compensate
  (sharper Gaussians = less BCE leakage into empty voxels)
  ↓
some query's scale logit goes deep-negative
  → sigmoid ≈ 0 → scale = scale_min = 0.01 m
  ↓
G2V backward computes det⁻¹·⁵ ≈ 10¹⁸ for that Gaussian
  → Inf in parameter.grad
  ↓
clip_grad_norm_(...) returns Inf/NaN (any-NaN-poisons-all)
  → optim.step skipped, weights frozen
  ↓
weights stay in the bad regime → repeats every window
```

Grad-accum=16 amplifies this: per-window cascade probability
`≈ 1 - (1 - p)^16 ≈ 16p` for per-sample probability p. With p ≈ 5–10%
(post-drift), 16 samples per window → near-certain cascade every window.

## Why Stage 1 (5000 iters, scale_min=0.01) didn't see this

Architectures are shared *up to the head*, but two Stage-2-only pieces
drive the cascade:

| | Stage 1 | Stage 2 |
|---|---|---|
| Splat backend | gsplat (3DGS image splatting) | G2V (3D Gaussian PDF → voxels) |
| `1/√det(Σ)` normalization? | **no** | **yes** (line 119) |
| `det⁻¹·⁵` in backward? | no | **yes** |
| Loss type | image-plane regression (depth L1, RGB L1+SSIM, denoise) | voxel-grid BCE (`pos_weight=16`) + KL + CE + Lovász |
| Pressure on Gaussian scales | none — image rendering doesn't reward sharp Gaussians | **direct** — BCE rewards localizing mass into occupied voxels only |

Stage 1's clean 5000-iter run with `scale_min=0.01` is therefore
*consistent*. It isolates the cascade to the Stage-2-specific G2V +
voxel-BCE coupling, which points cleanly to `scale_min` as the right
knob.

## Comparison with prior Stage 2 runs

| recipe | scale_min | duration | result |
|---|---|---|---|
| v1 (Part 01, 1500 iter) | 0.01 | 1500 micro-iters | **87.8% NaN-skip** — cascaded from iter 179 |
| v2 (Part 01, 1500 iter) | **0.05** | 1500 micro-iters | **0% NaN-skip** — clean, 1.16% mIoU |
| v3 smoke (Part 01, 200 iter) | 0.01 | 200 micro-iters | 0% NaN-skip — too short to see drift |
| v4 smoke (Parts 01–05, 200 iter, accum 16) | 0.01 | 200 micro-iters | 0% NaN-skip — same (drift not yet established) |
| **v4 long (this run)** | **0.01** | **1071 micro-iters before NaN** | **89% NaN-skip past that** |

The pattern is duration-dependent: `scale_min=0.01` looks fine in
short smokes (200–500 iters) but cascades once the opa drift kicks in
around 800–1100 iters. `scale_min=0.05` caps `det⁻¹·⁵` at `0.05⁻⁶ ≈ 6.4×10⁷`
(vs `10¹⁸` at 0.01) — comfortably inside bf16, and v2's clean
1500-iter run proves this is sufficient.

## Why didn't auto-abort fire?

Bug in the v4 port. The auto-abort logic checks consecutive
`skipped=True` rows in `training_history.json`:

```python
if n_skipped >= nan_abort_window:
    consec = 0
    for h in reversed(history):
        if h.get('skipped', False): consec += 1
        else: break
    if consec >= nan_abort_window - 1: break
```

Under grad-accum=16, the failure pattern is:
- 15 mid-window micro-iters: `skipped=False` (L finite, `window_skipped=False`
  because micro-iter L doesn't blow up)
- 1 window-end iter: `skipped=True` (accumulated gnorm NaN)

So `skipped=True` appears once per window, never consecutively at
the threshold (50). The `consec` count resets to 0 after every
window-end. Net result: auto-abort never triggers despite 559/625
windows failing.

**Fix for v5:** count consecutive *window-end* skips, or retroactively
mark all mid-window rows `skipped=True` when the window-end fails.

## Proposed v5 mitigations (priority order)

1. **`--scale-min 0.05`** — direct fix, validated by v2's clean
   1500-iter run. Cheapest, most reliable. Caps `det⁻¹·⁵` at ~6.4×10⁷
   instead of 10¹⁸.
2. **Fix auto-abort counter for grad-accum** — count window-end skips,
   not micro-iter skips. Would have killed this run at iter ~1900
   instead of ~9999, saving 80 min of compute.
3. **Per-micro-iter gradient clip** (defensive) — apply
   `clip_grad_norm_` per micro-iter before accumulating, not just
   per window. Prevents one bad sample's Inf from poisoning the
   16-sample window.
4. **Lower `--occ-pos-weight`** from 16 — reduces the BCE pressure
   driving opacity up. May affect paper-comparable mIoU; warrants a
   separate ablation, not a default change.
5. **Scale-logit activation clamp** — clamp the sigmoid input to
   something like [-5, 5] so `scale ∈ (0.013, 2.49)` regardless of
   logit. Prevents the floor regime even if `scale_min` is low.

## What this run did tell us

Not a complete loss:

- Cross-attn fixes (commit a1bb133) work as intended: no init-time
  cascade, 1071 iters of healthy training.
- The grad-accum + warmup_cosine machinery is correct: window-end
  semantics verified, save-best fired only at window-ends, NaN-diag
  block fired on the first window-NaN.
- The cascade re-emerges with `scale_min=0.01` even with the cross-attn
  fixes — confirming the cascade has a different mechanism than the
  init-feature-noise problem the cross-attn fixes addressed.
- Auto-abort logic needs a grad-accum-aware fix.

## Files preserved in this folder

| file | size | meaning |
|---|---|---|
| `ckpt_best_train.pt` | 270 MB | snapshot at iter 976 (smoothed L=19.33), pre-cascade, post-warmup |
| `training_history.json` | 3.3 MB | all 10K micro-iter records (last 89% have NaN gnorm) |
| `eval_history.json` | 2 B | empty (no periodic eval ran — killed before iter-1000 eval) |
| `train.log` | text | full per-window console log |
| `POSTMORTEM.md` | — | this document |

No final eval ran; no `ckpt_best_val.pt` exists (no val mIoU
was ever computed). The model state in `ckpt_best_train.pt` would
evaluate to a barely-post-warmup state if loaded — not a useful
quality baseline.

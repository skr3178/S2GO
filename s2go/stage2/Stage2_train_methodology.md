# Stage 2 — Ball-park training methodology

This document records the methodology and limitations of the Stage-2
training runs done with `s2go/tools/stage2_train.py` against the Part-01
nuScenes data + SurroundOcc voxel GT. It is a **ball-park run**, not a
paper-grade reproduction.

## What this run does

- **Goal.** Produce an mIoU/IoU number on a held-out val split to check
  the Stage-2 pipeline learns something on the data we have, and to
  ball-park it against paper Table 3 row (a) (mIoU 13.02 / IoU 25.73,
  trained 24 epochs on full nuScenes-SurroundOcc train).
- **Init.** From scratch (no Stage-1 ckpt; learned-query init, paper
  §3.4.1). Matches row (a)'s Query Init = `-`, Depth/RGB/Denoise all ✗.
- **Loss objectives.** Stage-2 supervised losses only —
  occupancy BCE, semantic KL+CE+Lovász. Pretraining auxiliary objectives
  (depth/RGB/denoise) are absent because there is no Stage-1.
- **Training data.** 62 nuScenes-train scenes ∩ Part-01 keyframes →
  **2462 sequences**. Streamed (sampled with replacement) per iter; no
  caching.
- **Held-out val.** 23 nuScenes-val scenes ∩ Part-01 keyframes →
  **914 sequences**. Never seen during training (scene-disjoint, the
  standard nuScenes split applied to our subset).
- **Token manifest.** `out/stage2_part1_splits.json` — produced by
  `nuscenes.utils.splits.create_splits_scenes()` filtered to Part-01.
- **Eval metrics reported.**
  1. Raw mIoU over 17 semantic classes (excluding empty).
  2. **Restricted mIoU** — averaged only over classes with
     ≥1000 train voxels AND ≥100 val voxels (defaults).
  3. Occupancy IoU (binary, empty vs not).
  4. Per-class IoU + per-class voxel counts (train + val).

## Why the number is *not* directly comparable to paper row (a)

Even with a healthy pipeline that would hit 13.02 mIoU on full data, our
ball-park number can land anywhere from ~0.5 to ~8 mIoU **without any
pipeline bug**. Three skew sources:

1. **Class coverage skew (dominant).** With 62 train scenes vs the
   paper's 700, some classes are rare or absent in training. They get 0
   IoU. mIoU averages across 17 classes equally, so a few unlearned
   classes drag the mean down hard. Conversely, classes absent in our
   23 val scenes don't get counted at all — the denominator shrinks and
   the metric is no longer apples-to-apples to the paper's mIoU over
   all 17.

2. **Distribution shift.** 62 train scenes are not a uniform random
   sample of nuScenes diversity (city, time-of-day, weather, traffic
   density). If our 23 val scenes contain conditions absent from train,
   per-class IoUs collapse for non-pipeline reasons.

3. **Per-class IoU noise.** Rare classes have small voxel counts in
   val → noisy IoU estimates → noisy mIoU.

The **restricted mIoU** above is designed to address #1 partially —
it reports the mIoU averaged only over classes that have enough train +
val voxels to be a meaningful per-class IoU estimate. It is the more
honest "this pipeline learns" number; raw mIoU is what's comparable in
shape to paper-reported mIoU.

## To make it comparable to paper row (a)

Required to reach paper-grade methodology — not done in this run:

1. **Download nuScenes Parts 02–10 keyframes** (~38 GB) → grow train
   coverage from 62 scenes (15% of 700) to (close to) 700 scenes (100%).
2. **Train for ~24 epochs** over the full train set with appropriate
   batch size + LR schedule (paper §3.4.4 — AdamW 4e-4, cosine, batch 16,
   weight decay 0.01, grad clip 35, backbone LR ×0.25, mixed precision).
   The recipe-v3 defaults below align with paper hyperparameters where
   possible; the remaining gap is batch size (1 vs 16) and sparse-voxel
   sampling (4096 train voxels/iter) — both forced by the 12 GB 3060.
3. **Eval on the full 6019-sample SurroundOcc val** (we eval on 914 ≈
   15% of val). Subsample val mIoU is a noisy estimator of full-val
   mIoU; expect a few-percent gap purely from sampling variance.

## Outputs per run

In `--out-dir`:

| File | Purpose |
|---|---|
| `ckpt_best_train.pt` | Saved when smoothed train loss (window=50, cooldown=100) hits a new low. Tracks training fit. |
| `ckpt_best_val.pt` | Saved when val mIoU (50-sequence subset, every 500 iters) hits a new high. Tracks generalization. |
| `ckpt_periodic.pt` | Overwritten every 200 iters. Crash-recovery only. |
| `training_history.json` | Per-iter losses, gnorm, opacity stats, skip flags. |
| `eval_history.json` | Per-eval mIoU / restricted_mIoU / occ_IoU + timestamps. |
| `eval_final.json` | Final full(er)-val eval — raw mIoU, restricted mIoU, occ_IoU, per-class IoU + train/val voxel counts, restricted-class list. |

No end-of-training ckpt is saved — only the bests. The user explicitly
requested this to avoid the prior "save final but it might be worse than
mid-training" pattern.

## Recipe revisions

The defaults in `s2go/tools/stage2_train.py` have evolved as we diagnosed
failures and ported paper / fix-branch findings back in. Each row of the
table is the value actually used at that recipe revision.

| param | v1 (NaN cascade) | v2 (1500-iter ball-park) | v3 (paper-aligned, current) | v4 (grad-accum, optional) | paper §3.4.4 |
|---|---|---|---|---|---|
| `--lr` | 2e-4 | 1e-4 | **2e-4** | 2e-4 | 4e-4 (at batch 16) |
| `--grad-clip` | 10.0 | 10.0 | **35.0** | 35.0 | 35 |
| schedule | ConstantLR | ConstantLR | **CosineAnnealingLR(T=n_optim_steps)** | + `warmup_cosine` option | cosine |
| `--scale-min` (Gaussian floor, m) | 0.01 | 0.05 | **0.01** | 0.01 | 0.01 (GF-2 default) |
| weight decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| backbone LR mult | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| `--grad-accum-steps` | n/a | n/a | n/a | **opt-in, e.g. 16** | n/a (paper uses batch 16 directly) |
| effective batch | 1 | 1 | 1 (3060 limit) | grad-accum × 1 | 16 |
| arch fixes (cross-attn) | absent | absent | **present** (3 fixes from `fix/cross-attn-projection-validity-mask`) | present | n/a |
| NaN-skip rate (1500 iters) | 87.8% | **0%** | 0% (smoke) | tbd | n/a |
| val mIoU after 1500 iters | 0.53% | 1.16% | tbd | tbd | n/a (paper ran 24 epochs) |

**Rationale for the v2 → v3 changes** (all reverting v1-era defensive
workarounds that the cross-attn fixes made unnecessary):

- **LR 1e-4 → 2e-4.** v2's 1e-4 was a NaN-cascade workaround. With cross-attn
  fixes in, the sqrt-scaling-rule equivalent of paper's 4e-4 at batch 16 is
  ~1e-4; the linear-rule equivalent is 2.5e-4. 2e-4 splits the difference.
- **Grad clip 10 → 35.** v2's gnorm range was 19 → 217, so a cap of 10
  was clipping *every* iter — a hard distortion of gradient direction-vs-magnitude
  rather than a safety net. Paper's 35 lets natural magnitudes flow.
- **ConstantLR → CosineAnnealingLR.** Cosine is the paper schedule;
  ConstantLR was never deliberate, just an early placeholder.
- **scale_min 0.05 → 0.01.** v2's 0.05 was the v1-era NaN-cascade
  workaround (raises 1/s² gradient floor from 1e4 to 400). With the
  cross-attn projection-mask fix preventing the upstream cause (Gaussian
  scale collapse driven by noisy cross-attn → BCE-empty minimum), 0.01
  is expected to be safe; revalidate with a smoke before claiming so.

**NaN-guard machinery kept across all revisions:** per-iter NaN diagnostic
dump (first occurrence) + auto-abort after N consecutive NaN-skipped iters
(`--nan-abort-window`, default 50). Zero cost when training is healthy.

**v4 additions (ported from Stage 1 commit `519ee4f`):**

- `--grad-accum-steps N` — accumulate gradients over N micro-iters per
  optimizer step. Per-sample loss is scaled by `1/N` so the accumulated
  gradient is the per-sample MEAN (matches paper batch-loss semantics).
  Default `1` keeps v3 behavior bit-identical; pass `16` to approximate
  paper's batch-16 on our single-sample loader.
- `--lr-schedule warmup_cosine` — linear warmup `0 → peak` over
  `--warmup-iters` micro-iters, then cosine decay `peak → peak × 0.10`
  over the remaining iters. Scheduler ticks once per OPTIMIZER step,
  so `T_max` is in optim-step units (= `ceil(n_iters / grad_accum_steps)`).
- NaN-diag at the first **window-level** NaN (post-clip accumulated
  gradient non-finite) rather than the first micro-iter NaN — far more
  diagnostic post-accum since a single bad micro-iter no longer poisons
  weights.
- Save-best / save-periodic / eval-every are all gated on `is_accum_end`
  so they only fire after a real optimizer step, never mid-window.

**Caveat under grad-accum:** `--nan-abort-window` counts skipped
micro-iters. A fully-poisoned window of size N contributes N skipped
entries, so the wall-clock-equivalent abort threshold also scales by N
(default 50 micro-iters = ~3 windows at N=16). Scale `--nan-abort-window`
by N if you want the same per-optim-step abort behavior.

## Known limitation: crash exposure

A ~1500-iter streaming run takes ~17 min wall-clock; the final 914-seq
val eval takes another ~5 min. Both exceed the ~4.5-min window where the
PC crashed during a prior 500-iter overfit run (see chat history). The
crash-recovery ckpt is the mitigation — if the run dies, restart and
diagnose; the most recent ckpt at iter k×200 is preserved.

The crash root cause (PSU transient overload vs GPU/VRM thermal trip
vs board-level issue) is not yet diagnosed. A monitored short run with
`nvidia-smi dmon` logging power/temp/throttle reasons is the next
direct step if a repeat crash occurs.

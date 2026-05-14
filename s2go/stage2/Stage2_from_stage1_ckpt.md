# Stage 2 — Training from a Stage-1 checkpoint (paper Table 3 row e)

How to run the coupled pipeline: train Stage-1 (depth-only, on noised
LiDAR), then warm-start Stage-2 from that checkpoint instead of training
Stage-2 from scratch. This is **sequential pre-train → fine-tune** —
the two phases share no optimizer step, only a `.pt` file.

The pipeline plumbing is already in place — this doc is a runbook,
not a code change.

## TL;DR

```bash
source /home/skr/miniconda3/etc/profile.d/conda.sh
conda activate /media/skr/storage/conda_envs/selfocc

python -u -m s2go.tools.stage2_train \
    --stage1-ckpt <PATH_TO_STAGE1_CKPT> \
    --splits-json out/stage2_part1_splits.json \
    --out-dir     out/stage2_coupled_v1 \
    --iters 10000 --eval-every 500 \
    --query-init learned \
    --scale-min  0.05
```

The two non-default ingredients vs a row-(a) run are:

1. `--stage1-ckpt <path>` — points at a `.pt` saved by `overfit.py`.
2. **Don't** pass `--from-scratch` (the coalesce at
   [stage2_train.py:862-865](../tools/stage2_train.py#L862-L865) makes
   `from_scratch = True iff stage1_ckpt is None`).

Everything else can match an existing row-(a) recipe so the comparison
is clean.

## What goes through the wire

| Component | Source after coupling |
|---|---|
| R50+FPN backbone | Stage-1 ckpt (`ckpt['backbone']`) |
| Segmentor (parent refiner, decoder, lifter) | Stage-1 ckpt (`ckpt['segmentor']`) via [`load_stage1_state`](stage2_segmentor.py) |
| `child_head.head` (Linear) | Stage-1 rows 0..10 copied into Stage-2's 11-row head; Stage-1's 3 RGB rows discarded |
| `lifter.query_feat` | Stage-1 (when `--query-init learned`) |
| `lifter.query_xyz` (Stage-2 only) | Random init (no Stage-1 counterpart) |
| `SemanticHead` (per-parent classifier) | Random init (Stage-2 only) |

Architecture (K, J, embed_dims, num_layers, num_pts, feedforward_channels)
is **read from the ckpt's `config` dict** ([stage2_train.py:267-275](../tools/stage2_train.py#L267-L275)).
Manual `--num-layers / --num-pts / --ffn` overrides are asserted off
when a ckpt is supplied — the Stage-2 model must match Stage-1's shape.

## Step 1 — Inspect the Stage-1 ckpt first

Before launching anything, verify the ckpt is what `load_stage1_state`
expects. From the repo root:

```bash
python -c "
import torch, sys
ck = torch.load(sys.argv[1], map_location='cpu')
print('top-level keys:', list(ck.keys()))
print('config:', ck.get('config'))
seg = ck['segmentor']
print('child_head.head.weight:', seg['child_head.head.weight'].shape)
print('lifter keys:', [k for k in seg if k.startswith('lifter.')])
" <PATH_TO_STAGE1_CKPT>
```

Pass criteria:

- Top-level keys include `backbone`, `segmentor`, `config`.
- `config['K'] == 900`, `config['J'] == 10`.
- `child_head.head.weight.shape == (14, embed_dims)` — Stage-1 was
  trained in `child_mode='rgb'` (the default in `S2GOSegmentor`).
  A `(11, d)` ckpt is a Stage-2 ckpt and will fail the shape assertion
  at [stage2_segmentor.py:154-156](stage2_segmentor.py#L154-L156).
- `lifter.query_feat` key is present (required for
  `--query-init learned` to inherit the learned anchor embeddings).

If any check fails, **stop** — `load_stage1_state` will either assert
or silently partial-load. Re-train Stage-1 with the standard
[`overfit.py`](../tools/overfit.py) recipe.

## Step 2 — Smoke run (≈100 iters)

The 100-iter smoke is not about accuracy — it confirms (a) load
completed, (b) loss starts in a sensible range, (c) the new
`head_partial_transferred=True` signal prints, (d) no surprise key
mismatches:

```bash
python -u -m s2go.tools.stage2_train \
    --stage1-ckpt <PATH_TO_STAGE1_CKPT> \
    --splits-json out/stage2_part1_splits.json \
    --out-dir     out/stage2_coupled_smoke \
    --iters 100 --eval-every 50 \
    --batch-size 1 --grad-accum-steps 1 \
    --query-init learned \
    --scale-min  0.05 \
    2>&1 | tee out/stage2_coupled_smoke/train.log
```

Expected stdout signals (look for them in the `[2/6]` and `[3/6]` blocks):

```
[2/6] Reading Stage-1 ckpt: <PATH_TO_STAGE1_CKPT>
    arch: {K: 900, J: 10, embed_dims: 768, num_layers: 6, num_pts: 13, feedforward_channels: 3072}
[3/6] Building backbone + Stage 2 model…
    backbone + segmentor: Stage-1 weights loaded
      backbone:  missing=0, unexpected=0
      segmentor: missing=<small>, unexpected=<small>, head_partial_transferred=True
```

`missing=<small>` is normal for the segmentor — `lifter.query_xyz`
(Stage-2-only) shows up there. `unexpected=<small>` is normal too —
Stage-1 keys without a Stage-2 counterpart. Anything in the
hundreds means a wrong-arch ckpt; abort.

**`head_partial_transferred=True` is the gate** — if this prints
`False`, the 14→11 head copy was skipped and Stage-2 is effectively
training from scratch despite the ckpt. Re-check the ckpt before
proceeding to Step 3.

The first ~5 training-loss values should land noticeably below a
from-scratch run's iter-0 loss; if they look identical to row-(a)'s
early curve, the weight transfer didn't actually take.

## Step 3 — Full training run

Mirror an existing row-(a) recipe exactly, only swap in
`--stage1-ckpt`. Example matched against `out/stage2_row_a_train_v4_10k`:

```bash
python -u -m s2go.tools.stage2_train \
    --stage1-ckpt <PATH_TO_STAGE1_CKPT> \
    --splits-json out/stage2_parts01-05_splits.json \
    --out-dir     out/stage2_coupled_v1 \
    --iters 10000 \
    --eval-every 500 --eval-num-val 50 \
    --final-eval-num-val 914 \
    --save-periodic-every 200 \
    --batch-size 1 --grad-accum-steps 1 \
    --query-init learned \
    --scale-min  0.05 \
    --lr 2e-4 --lr-schedule warmup_cosine --warmup-iters 500 \
    2>&1 | tee out/stage2_coupled_v1/train.log
```

Three checkpoint files land under `--out-dir`:

- `ckpt_best_val.pt` — best held-out mIoU
- `ckpt_best_train.pt` — best smoothed train loss
- `ckpt_periodic.pt` — rolling crash-recovery save

## Step 4 — Verification

In order of cost:

1. **Load fidelity** (free, from the smoke log) —
   `grep head_partial_transferred out/stage2_coupled_smoke/train.log`
   must show `=True`.

2. **Loss trajectory** (mid-run) — compare the coupled run's
   `training_history.json` against `out/stage2_row_a_train_v4_10k/`
   at matched iter counts. The coupled curve should sit strictly
   below the row-(a) curve from iter ≈ 100 onward. The paper's
   row-(e) vs row-(a) delta on the published model is +7.2 mIoU
   (20.25 vs 13.02) — on a partial-data run, expect a smaller
   absolute lift but the same sign.

3. **End-of-training mIoU** (`eval_final.json`) — `ckpt_best_val.pt`'s
   restricted mIoU should exceed the row-(a) v4_10k baseline. Any
   measurable uplift confirms the coupling works; matching the
   paper's row-(e) number requires the paper's full nuScenes-train
   data + 12 epochs of Stage-2, not the Part-01 / 10k-iter subset.

## Common failure modes

| Symptom | Likely cause |
|---|---|
| `AssertionError: unexpected Stage-1 child_head.head shapes: w=(11, d)` | Passed a Stage-2 ckpt instead of a Stage-1 ckpt to `--stage1-ckpt`. |
| `head_partial_transferred=False` in stdout | Stage-1 ckpt was saved without a `child_head.head` entry (custom training script). Use `overfit.py` ckpts only. |
| `arch:` mismatch (e.g. `num_layers=2` when expected 6) | Stage-1 trained at T0/T1 tier, not paper-spec T2. The arch will follow the ckpt's config — to use paper-spec T2, re-train Stage-1 with `--num-layers 6 --num-pts 13 --feedforward-channels 3072`. |
| First training-loss value identical to a from-scratch run | `from_scratch` resolved to True. Check stdout for `[2/6] From-scratch arch (no Stage-1 ckpt)` — means the ckpt argument was dropped (typo in the path is the usual cause). |
| `RuntimeError` from CUDA G2V backend at iter 0 | Stage-2 backbone got a non-matching `embed_dims`. The ckpt's config is the source of truth — don't pass arch overrides. |

## Background

Paper Table 3, Pretraining ablation (SurroundOcc-nuScenes, 12 epochs
per stage unless marked):

| Row | Pretrain | Q-init | Depth | RGB | Denoise | mIoU | IoU |
|---|---|---|---|---|---|---|---|
| (a) | – | ✗ | ✗ | ✗ | ✗ | 13.02 | 25.73 |
| (a)† | – (24 ep) | ✗ | ✗ | ✗ | ✗ | 15.83 | 28.35 |
| (e) | LiDAR+ε | ✓ | ✓ | ✗ | ✗ | **20.25** | **32.44** |
| (f) | LiDAR+ε | ✓ | ✓ | ✓ | ✓ | **21.60** | **33.91** |

Row (a) is the row this codebase has been reproducing — Stage-2 from
scratch, no Stage-1 weights. Until commit `e18f73c` ([stage2_fixes.md](stage2_fixes.md)
Tier-A #1) the `--from-scratch` argparse default was permanently True,
so even runs that passed `--stage1-ckpt` silently fell back to row (a).

Row (e) is what this runbook achieves: depth-only Stage-1 pre-training
(the same recipe `overfit.py --depth-only` runs) → load into Stage-2 →
fine-tune. The cross-attn projection-mask fix and TSA/Deformable
zero-init landed on `main` in commits `73c26eb` and the absorbed
`a81e729` — both are baked into any Stage-1 ckpt produced after
those commits, so the warm-start Stage-2 inherits them for free.

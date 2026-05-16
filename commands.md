# S2GO run commands

## Stage-1 (overfit.py) — curated_train_270, depth-only single-frame, 6-epoch plan

Recipe: **`--depth-only`** = pure `L_depth` at dt=0, single-frame
(`T_seq=1`, `T_queue=1`), no denoise grad, no RGB, no ±0.5s warps.
Paper Table 3 ablation 1 (LiDAR+ε, depth-only, dt=0).

LR-curve logic:
- `--total-iters` = full planned horizon, shapes the LR curve. **Keep constant
  across the initial run AND every resume.**
- `--n-iters`     = where THIS invocation stops (loop is `range(start_iter, n_iters)`).
- Cosine is sampled at the absolute optim-step vs `--total-iters`, so a planned
  stop + later `--resume-from` follow ONE continuous curve (no warm-restart jump).
- Omit `--total-iters` → defaults to `--n-iters` → legacy single-shot behaviour.

Single-frame epoch math: curated_train_270 = 10,776 keyframes
(`summary.md`); single-frame ≈ 1 sequence/keyframe, grad_accum=16 →
1 epoch ≈ 10,776 micro-iters; 3 epochs ≈ 32,328 ; 6 epochs ≈ 64,656.
(Estimate — the run prints the exact single-frame sequence count /
epoch estimate at startup; the master curve only needs a fixed
--total-iters, so a small discrepancy just shifts the epoch label.)

### Run 1 — auto-stops cleanly at epoch 3 (curve = 6 epochs)

```bash
python -m s2go.tools.overfit --depth-only \
  --train-tokens-json dataset_stats/curated_train_270/tokens.json \
  --val-tokens-json   dataset_stats/curated_val_50/tokens.json \
  --n-iters 32328 --total-iters 64656 \
  --lr 4e-4 --lr-schedule warmup_cosine --warmup-iters 500 --grad-accum-steps 16 \
  --num-layers 6 --num-pts 13 --feedforward-channels 3072 \
  --mixed-precision --amp-dtype bf16 --use-checkpoint \
  --save-periodic-every 250 --save-best --eval-every 500 --num-workers 0 \
  --run-name stage1_curated_270_depthonly --out-root out
```

### Run 2 — resume to finish epochs 3→6 (same --total-iters)

```bash
python -m s2go.tools.overfit --depth-only \
  --train-tokens-json dataset_stats/curated_train_270/tokens.json \
  --val-tokens-json   dataset_stats/curated_val_50/tokens.json \
  --n-iters 64656 --total-iters 64656 \
  --lr 4e-4 --lr-schedule warmup_cosine --warmup-iters 500 --grad-accum-steps 16 \
  --num-layers 6 --num-pts 13 --feedforward-channels 3072 \
  --mixed-precision --amp-dtype bf16 --use-checkpoint \
  --save-periodic-every 250 --save-best --eval-every 500 --num-workers 0 \
  --resume-from out/stage1_curated_270_depthonly-<TIMESTAMP>/ckpt_last.pt \
  --out-root out
```

Rule of thumb: `--total-iters` is the plan (constant); change only `--n-iters`
to pick where a run halts.

Notes:
- `--use-checkpoint` mostly targets temporal-decoder layers → near-no-op at
  single-frame; harmless to keep, or drop it to save ~30% wall-time.
- Stage-1 eval is BEV nn_dist (no mIoU). `--save-best` tracks the
  moving-avg `L_total`, which here = `L_depth` only.

### setsid-detached launch (survives IDE/agent teardown)

Prepend the command with full process detachment + logging:

```bash
source /home/skr/miniconda3/etc/profile.d/conda.sh
conda activate /media/skr/storage/conda_envs/selfocc
cd /media/skr/storage/self_driving/S2GO
setsid nohup python -m s2go.tools.overfit --depth-only ...flags... \
  > out/stage1_curated_270_depthonly.log 2>&1 < /dev/null &
```

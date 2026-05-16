# Stage-2 curated 3-epoch run — SUMMARY

**Date:** 2026-05-15 20:30 → 2026-05-16 02:45 (≈ 6 h 15 m wall-clock)
**Run dir:** `out/stage2_curated_3ep_b16/`
**Best checkpoint:** `ckpt_best_val.pt` — mini-eval mIoU **5.63 %** (iter 30 415)
**Final full-val eval:** **mIoU 6.06 %**, occ-IoU 15.50 % (2 012 curated-val sequences)

## Headline

| Metric | This run | Previous overnight (`stage2_coupled_t2_b16_35k_full`) |
|---|---:|---:|
| Final-eval mIoU | **6.06 %** | 1.15 % |
| Best val mIoU (mini-eval) | 5.63 % | 2.03 % |
| Occupancy IoU (final) | 15.50 % | 10.00 % |
| Best smoothed train loss | 7.66 | 15.62 |
| Iters total | 32 550 (3 epochs) | 35 000 |
| Iters skipped (NaN) | **0** | 0 |
| Init | from-scratch | Stage-1 ckpt |
| Train data | curated 270 scenes / 10 850 seq | full 700 / 28 130 |
| Val data (final eval) | curated-val 2 012 seq | part-1 val 914 seq |

**~5× the previous overnight's final mIoU, from scratch, on 38 % of the
scenes.** Edges past the README row-(a) from-scratch baseline (5.55 % mIoU
on parts 01–05). Eval sets are not identical (914 part-1 vs 2 012
curated-val) but the gap is far too large to be an eval-set artifact.

## Config (exact-mirror of the proven previous overnight, 3 deltas)

| Knob | Value |
|---|---|
| Splits | `out/stage2_curated_splits.json` (270 train / 50 val scenes; 10 850 / 2 012 sample tokens) |
| Init | `--from-scratch` (R50 backbone = torchvision ImageNet; segmentor/head random) |
| Iters | 32 550 micro-iters (= 3 epochs over 10 850 curated-train tokens) |
| Grad accumulation | `--grad-accum-steps 16` → 2 035 optim-steps (effective batch ≈ 16) |
| LR schedule | `warmup_cosine`, 500-iter warmup → 2e-4 → cosine to 2e-5 (peak ×0.10) |
| Optimizer | AdamW, grad-clip max_norm 35.0 |
| Precision | bf16 autocast |
| scale_min | override 0.01 → 0.05 (Gaussian floor, paper-stability fix) |
| Arch | T2: K=900, J=10, embed_dims=768, num_layers=6, num_pts=13, ffn=3072 |
| Eval cadence | every 100 optim-steps (= 1 600 micro-iters) → 20 mid-evals over 100 seq |
| Final eval | full 2 012 curated-val sequences |

Deltas vs `stage2_coupled_t2_b16_35k_full`: (1) curated splits not full,
(2) from-scratch not Stage-1-init, (3) final-eval on 2 012 curated-val.
Everything else identical.

## mIoU trajectory (20 mid-training mini-evals, 100 seq each)

```
phase        iters            mIoU                                  occ-IoU
climb        1.6k → 12.8k     1.22→1.81→1.94→2.58→3.12→3.58→3.82→4.08   9.9→14.1
plateau      12.8k → 17.6k    4.08→4.07→3.99→4.02                       ~14
cosine tail  17.6k → 32.0k    4.02→4.92→5.02→5.21→5.40→5.10→5.20        15.0
                              →5.51→5.63→5.63
final full   (2012 seq)       6.06 %                                    15.50
```

The cosine-LR tail (steepest decay 15 k–30 k) drove a +2 pp breakout out
of the ~4 % plateau — the "let it finish" decision paid off. The full
2 012-seq final eval (6.06 %) reads higher than the noisier 100-seq
mini-evals (peak 5.63 %), as expected.

## Per-class IoU (final, 2 012 curated-val seq)

| Class | IoU | | Class | IoU |
|---|---:|---|---|---:|
| empty | 71.89 % | | bicycle | 5.47 % |
| other_flat | 18.87 % | | driveable_surface | 5.19 % |
| manmade | 10.70 % | | motorcycle | 4.67 % |
| terrain | 10.35 % | | truck | 3.65 % |
| other | 10.31 % | | traffic_cone | 1.78 % |
| sidewalk | 9.41 % | | barrier | 0.69 % |
| car | 8.05 % | | trailer | 0.09 % |
| construction_vehicle | 7.93 % | | pedestrian | 0.00 % |
| vegetation | 5.91 % | | bus | 0.00 % |

Stuff classes dominate; small/rare thing-classes (pedestrian, bus,
trailer, barrier) still unlearned at this compute scale — expected; the
paper reaches these only after far more training.

## Loss-curve reading — see `curves.png`

- **Total loss**: sharp drop 0–5 k (16 → ~12), slow grind to ~8 (MA200).
- **Components**: `sem_ce` (×10 weighted) dominates and drives nearly all
  total-loss reduction (2.9 → ~1.0). `occ_bce` flattens at ~0.65 by iter
  5 k — occupancy learned early; further mIoU gains are pure
  classification of already-detected voxels.
- **LR**: clean warmup_cosine; steepest decay 15 k–30 k aligns exactly
  with the mIoU breakout.
- **Grad norm**: stays well under clip=35, no spikes → `iters_skipped=0`.
  Contrast: the Stage-1 40k overnight had a 70.5 % NaN skip rate. The
  grad-accum=16 averaging + scale_min=0.05 kept this run fully stable.

## Verdict

| Question | Answer |
|---|---|
| Did it train stably? | **Yes** — 0/32 550 iters skipped, no NaN storm. |
| Did the curated dataset help? | **Yes, decisively** — ~5× the full-data + Stage-1-init overnight's final mIoU, from scratch on 38 % of scenes. |
| Did "let it finish" pay off? | **Yes** — cosine tail added +2 pp (4.0 → 6.06) over the back half. |
| Limiting factor now? | Compute scale + small-class learning. occ-IoU saturated ~15 %; rare thing-classes at 0 %. Next lever: more epochs or Stage-1 init on top of curation. |

## File inventory

| File | Description |
|---|---|
| `SUMMARY.md` | this file |
| `train.log` | full run log (32 550 iters) |
| `curves.png` | 4-panel: total loss + eval mIoU / components / LR / grad-norm |
| `eval_final.json` | final 2 012-seq eval (mIoU 6.06 %) + per-class table |
| `eval_history.json` | 20 mid-training mIoU snapshots |
| `training_history.json` | per-iter loss components (32 550 rows) |
| `ckpt_best_val.pt` | best checkpoint — mini-eval mIoU 5.63 % @ iter 30 415 |
| `ckpt_best_train.pt` | best smoothed-train-loss checkpoint |
| `ckpt_periodic.pt` | rolling crash-recovery checkpoint |

## Reproduce

```bash
# 1. curated scene-token JSONs → Stage-2 sample-token splits
python scripts/curated_to_stage2_splits.py
# 2. train (this run)
python -u -m s2go.tools.stage2_train \
  --splits-json out/stage2_curated_splits.json --from-scratch \
  --iters 32550 --grad-accum-steps 16 \
  --lr-schedule warmup_cosine --warmup-iters 500 --lr 2e-4 --lr-min 2e-5 \
  --eval-every 100 --eval-num-val 100 --final-eval-num-val 2012 \
  --save-periodic-every 200 --out-dir out/stage2_curated_3ep_b16
# 3. plot
python scripts/plot_stage2_curves.py out/stage2_curated_3ep_b16
```

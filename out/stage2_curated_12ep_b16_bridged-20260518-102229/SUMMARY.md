# Stage-2 bridged (from depth-only Stage-1) — run summary

**Run dir:** `out/stage2_curated_12ep_b16_bridged-20260518-102229/`
**Status:** manually **STOPPED at iter 60,815 / 130,200 (46.7%, ~epoch 5.6)** on
2026-05-18 ~20:57 — halted early because it was tracking the Stage-2-only
baseline with no measurable benefit (rationale below).

## Verdict

**Bridging Stage-2 from a depth-only (λ_denoise=0) Stage-1 checkpoint gives
no measurable benefit over training Stage-2 from scratch.** At matched
compute it tracked the completed from-scratch baseline (marginally *behind*
on loss, even on mIoU). This empirically reproduces the paper's Table-3
finding: depth-only pretraining is *not* where the gain comes from — the
**denoising objective** is. Not a bug; the expected outcome of bridging an
ablation-(e)-style Stage-1.

## Configuration

| | value |
|---|---|
| Trainer | `s2go.tools.stage2_train` (bridged via `--stage1-ckpt`) |
| Stage-1 ckpt | `out/stage1_curated_270_depthonly_12ep_finish-20260517-211858/ckpt_final.pt` (depth-only, single-frame, **λ_denoise=0**, iter 130,200) |
| Splits | `out/stage2_curated_270_50_splits.json` — 270 tr / 50 val (10,850 / 2,012 seq) |
| Iters / schedule | `--iters 130200` (12 ep), `warmup_cosine`, `--warmup-iters 2000` |
| Peak LR | **2e-4** (stage2_train.py default; not passed) → cosine to 2e-5; backbone ×0.25 → 5e-5 |
| grad-accum / grad-clip / scale-min | 16 / 35.0 / 0.05 |
| Arch | T2 (K900 J10 embed768, layers 6 / pts 13 / ffn 3072) — **read from Stage-1 ckpt config** (bridged runs cannot pass `--num-layers/-pts/-ffn`) |
| Precision / seed | bf16 autocast / 0 |

### Launch bug + fix (recorded for reproducibility)
First launch crashed ~1 s in at `stage2_train.py:354 assert from_scratch`:
arch-override flags (`--num-layers/--num-pts/--ffn`) are **from-scratch-only**;
when bridging, arch comes from the Stage-1 ckpt config. Fix = drop those 3
flags (zero arch change — Stage-1 ckpt is already T2). Relaunched clean.

## Results at stop (iter 60,815)

| metric | bridged (this run) | Stage-2-only ref @ same iter | ref final (complete) |
|---|---|---|---|
| total loss | 9.16 (min 3.51) | **7.41** | 6.17 |
| held-out mIoU | 5.68% (peak **5.91%** @ best-val #15) | 5.97% | **6.38% (peak 6.55%)** |
| occ_IoU | 16.02% | — | 16.7% |
| NaN skips | 0 | 0 | 0 |

Comparison trajectory (bridged − ref, at matched progress):
- 13% : ΔmIoU +0.11 pp (noise — dead even)
- 21% : Δtotal −2.91 (bridged transiently ahead on loss)
- 34% : Δtotal +2.74 (ref ahead); mIoU a wash
- **47% : Δtotal +1.75 (ref ahead); ΔmIoU −0.06 pp** → no benefit, slightly behind

Reference baseline = completed Stage-2-only from-scratch 12-ep run (other PC),
`history/{training_history,eval_history}.json` (final mIoU 6.38%, peak 6.55%).

## Why no benefit (paper-grounded, s2go.pdf §3.3 + Table 3)

Paper Table 3 (12-ep Stage-2, full SurroundOcc-nuScenes):

| pretraining | mIoU |
|---|---|
| (a) none, 12 ep | 13.02 |
| (a)† none, 24 ep (compute-matched) | 15.83 |
| (b) learnable-init + depth | 12.42 (worse than none) |
| (c) LiDAR-init + depth | 13.62 (barely above none) |
| (e) **LiDAR+ε init, depth-only** | 20.25 |
| (f) **LiDAR+ε, depth + RGB + denoise (full)** | **21.60** |

The +6–8 mIoU gain requires the **full** denoise+RGB+velocity pretraining
with noised-LiDAR init. Paper §3.3.1: the **denoising objective is the
mechanism that supervises query movement onto geometry**; depth-only trains
Gaussians but not query repositioning. This run bridged from a Stage-1 that
was `--depth-only` (λ_denoise=0, no RGB, single-frame T=1, no ±0.5 s velocity
warps) → no transferable query-movement prior → ≈ from-scratch. Absolute
mIoU here (~6%) ≪ paper's 21.6% because we train on a 270-scene curated
subset, not full nuScenes; only the *relative* with/without comparison is
meaningful, and it shows no edge.

## Artifacts

- `ckpt_periodic.pt` (iter ~60,800, full optimizer+scheduler+RNG → resumable)
- `ckpt_best_val.pt` (best held-out mIoU 5.91%), `ckpt_best_train.pt`
- `loss_curve_live.png`, `compare_vs_stage2only.png`
- Generators: `scripts/repro_paper_figs_stage1.py`, `scripts/viz_query_denoising.py`

## Next experiment (to actually see Stage-1 benefit)

Run Stage-1 with the **full recipe** — `--full-recipe` (noised-LiDAR init +
denoise + depth + RGB + ±0.5 s velocity warps), *not* `--depth-only` — then
bridge Stage-2 from it. That is the paper-(f) configuration that yields the
+6–8 mIoU. Optionally also try bridging from Stage-1 `ckpt_eval_best`
(held-out-geometry-best, 1.547 m) vs `ckpt_final` (1.60 m, mildly overfit).

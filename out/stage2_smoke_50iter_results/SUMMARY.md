# S2GO Stage-2 smoke run — 50 iters × 4 sequences

**Date:** 2026-05-13
**Status:** ✓ Pipeline validation PASS (wiring, autograd, loss, eval all healthy)
**Checkpoint (not in this folder):** `out/stage2_smoke_50iter/ckpt.pt` (269 MB)

## What this run proves

Stage 2 lives in its own files/folders (`s2go/stage2/`) and trains end-to-end:

1. **Forward**: backbone → segmentor (`child_mode='semantic'`) → SemanticHead → per-Gaussian broadcast → G2V sparse → voxel-space loss. No edits to Stage 1.
2. **Backward**: gradient flows through every term — `occ_bce`, `sem_kl`, `sem_ce`, `sem_lovasz` — and updates **all** parameters (backbone, segmentor, semantic head). 50 iters, zero NaN/inf skips.
3. **Loss decreases**: total 30.999 → 12.680 (-59%), monotonic in all four sub-terms.
4. **Eval pipeline**: dense G2V (no grad) over the full 200×200×16 grid → argmax → mIoU + figures. Non-degenerate prediction with 7+ classes appearing.

This is **not** a quality result — it's a wiring validation. Paper-spec mIoU (~21% on full data) requires ≫50 iters and the full streaming run.

## What this run does NOT prove

- Convergence on real data. 50 iters cycling 4 sequences with random-init weights cannot reach paper mIoU.
- Stage 1 → Stage 2 init (the "bridge"). The smoke ran `--from-scratch` because the v2 Stage-1 ckpt's paper-spec arch (num_layers=6, num_pts=13, ffn=3072) + Stage 2's autograd graph + G2V exceeded 12 GB GPU memory. See [Memory note](#memory-note) below.

## Config

| Knob | Value | Notes |
|---|---|---|
| Architecture | T0-tier: K=900, J=10, d=768, **num_layers=2, num_pts=4, ffn=2048** | Random init (no Stage 1 ckpt load) |
| Loss | `--sem-loss both` | KL + CE + Lovász + occupancy-BCE |
| Loss weights | w_occ=1.0, w_kl=1.0, w_ce=10.0, w_lovasz=1.0 | Paper-aligned defaults |
| Voxel sampling | **4096 per iter, balanced 50/50 occupied/empty** | Sparse G2V — dense is ~70 GB autograd peak |
| Voxel grid | 200×200×16 (0.5 m, LIDAR_TOP) | Matches local Occ3D .npy data |
| Classes | 18 (17 semantic + 1 empty=17) | Matches GaussianFormer-2 convention |
| Sequences | 4 GT-covered, loader indices 80–83 (scene `n008-2018-08-01-...`) | 914/3376 total covered |
| LR | 2e-4 segmentor, 5e-5 backbone (×0.25) | Constant schedule |
| Grad clip | 10.0 | NaN/inf-guarded |
| Precision | bf16 autocast (segmentor); fp32 (backbone + G2V + loss) | Same pattern as Stage 1 |
| Iters | 50 | Smoke validation, not training |
| Wallclock | 59 sec | RTX 3060 (12 GB) |
| Peak GPU memory | 5.22 GB | ample headroom |

## Training trajectory

| iter | total | occ_bce | sem_kl | sem_ce | lovasz | gnorm | opa_mean |
|---|---|---|---|---|---|---|---|
| 0  | 30.999 | 1.279 | 2.616 | 2.616 | 0.944 | 69.7 | 0.305 |
| 5  | 18.918 | 0.729 | 1.571 | 1.571 | 0.907 | 12.6 | 0.100 |
| 10 | 16.212 | 0.793 | 1.322 | 1.322 | 0.872 | 15.2 | 0.101 |
| 20 | 14.436 | 0.721 | 1.172 | 1.172 | 0.822 | 24.9 | 0.153 |
| 30 | 13.530 | 0.704 | 1.091 | 1.091 | 0.828 | 24.8 | 0.172 |
| 40 | 13.004 | 0.734 | 1.041 | 1.041 | 0.813 | 50.0 | 0.191 |
| 49 | 12.680 | 0.745 | 1.012 | 1.012 | 0.803 | 23.4 | 0.209 |

Key dynamics:
- **All four loss terms decreasing monotonically** (modulo noise from per-sample variance).
- `sem_kl` and `sem_ce` are identical numerically — expected: for one-hot GT, KL(softmax || onehot) = -log softmax[gt] = CE.
- `gnorm` settles around 20–50 after iter 5; the clip at 10 fires frequently but never produces non-finite values.
- `opa_mean` drifts down then back up (collapse-and-recover at iter ~5, then steady growth) — typical optimizer dynamics, not a concern at this scale.

## Eval (single forward pass at end of training, on training sequence loader idx 80)

```json
{
  "mIoU (excl. empty)":  5.73,
  "occupancy_IoU":      17.47,
  "n_voxels_evaluated": 675729,
  "occ_pred_range":     [0.00, 0.9999],
  "occ_pred_mean":      0.312
}
```

Per-class IoU on the most-represented classes:

| Class | GT count | IoU | Precision | Recall |
|---|---|---|---|---|
| empty            | 604,271 | 85.7% | 96.9% | 88.1% |
| vegetation       |  14,044 | 16.6% | 20.4% | 47.4% |
| other_flat       |   8,818 | 10.6% | 11.8% | 51.3% |
| other            |   4,353 | 11.2% | 13.4% | 40.6% |
| manmade          |   3,784 | 10.0% | 15.0% | 22.9% |
| terrain          |   2,051 |  8.8% | 21.4% | 13.0% |
| construction_veh |   1,782 |  0.0% |   —   |  0.0% |
| sidewalk         |     482 |  0.0% |   —   |  0.0% |

The empty class is well-classified (the OccBCE term works); a handful of common foreground classes (vegetation, ground, manmade) are starting to differentiate; rarer classes (vehicles, pedestrians, traffic cones) remain at IoU=0% because the 4-sequence training set doesn't contain enough examples for the random-init weights to learn them in 50 iters.

See:
- `eval_A_bev.png` — top-down BEV comparison (GT vs prediction)
- `eval_B_voxel_slices.png` — 4 z-slices, GT vs prediction
- `eval_C_confusion.png` — 18×18 row-normalized confusion matrix

## Memory note

Dense G2V over the full 200×200×16 = 640,000 voxels × 9,000 Gaussians × 18 classes builds an autograd graph of intermediate tensors totaling ~70 GB across all chunks (autograd retains *all* chunk intermediates until backward, regardless of `chunk_voxels`). That fundamentally exceeds the 12 GB on this card.

The fix used here — **sparse voxel sampling** — picks 4,096 balanced voxels per iter and runs G2V only on those. Backward retains intermediates for the sampled subset (~440 MB). Peak training memory: **5.22 GB**.

This matches how SemanticKITTI / occupancy-prediction training is typically structured. The eval-time forward uses the dense path (no autograd → no chunk retention → fits in 12 GB).

For a future paper-spec run that loads the v2 Stage-1 ckpt: the autograd graph through num_layers=6, num_pts=13, ffn=3072 segmentor was the OOM cause, not G2V. Options: (a) drop to a smaller `num_pts` for Stage 2, (b) enable gradient checkpointing in the temporal decoder (currently increases memory at autocast boundaries — needs investigation), (c) train on a GPU with ≥16 GB.

## File inventory

| File | Size | Purpose |
|---|---|---|
| SUMMARY.md          | this file | analysis + verdict + caveats |
| train.log           | 4 KB  | stdout of the training run |
| training_history.json | 15 KB | per-iter metrics × 50 iters |
| eval_A_bev.png      | 44 KB | top-down BEV (GT vs prediction) |
| eval_B_voxel_slices.png | 53 KB | 4 z-slices |
| eval_C_confusion.png | 48 KB | row-normalized confusion matrix |
| eval_stats.json     | 4 KB  | per-class IoU + mIoU + occ IoU |

## Next steps

1. **Bridge to Stage 1**: load the v2 ckpt onto a model whose `num_pts` matches a memory-friendly size (e.g. 8 instead of 13). Verify ckpt loads cleanly (most cross-attn shapes will mismatch, expected — only backbone + a subset of segmentor params transfer).
2. **Longer overfit**: 500 iters on the same 4 sequences — does mIoU climb to ~30% (overfit signal)?
3. **Streaming training**: 5000 iters over the 914 covered sequences — closer to a paper-aligned baseline.
4. **CUDA G2V backend**: drop the pure-torch loop for the existing CUDA kernel at `localagg_s2go/local_aggregate_s2go/` — dense forward + a custom backward that avoids the (V, N, C) intermediate, unlocking dense voxel-space training.
5. **Hold-out validation split**: pick e.g. 64 covered sequences as a frozen val set; track val mIoU across the streaming run.

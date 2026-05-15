

## Results (current baseline)

Trained on **SurroundOcc-nuScenes parts 01-05** (369 train scenes, 56 val
scenes — about half of the full archive) for **35 000 micro-iters / 2 188
optimizer steps** at effective batch 16 (`--grad-accum-steps 16`).
Architecture is paper-spec T2 (`num_layers=6`, `num_pts=13`,
`feedforward_channels=3072`, `K=900`, `J=10`, `embed_dims=768`).

### Comparison vs paper Table 1 (% IoU)

| Method | IoU | mIoU | barrier | bicycle | bus | car | const. veh. | motorcycle | pedestrian | traffic cone | trailer | truck | drive. suf. | other flat | sidewalk | terrain | manmade | vegetation | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S2GO-Small (paper) | 34.3 | 22.1 | 20.8 | 13.1 | 27.5 | 30.3 | 14.5 | 16.5 | 11.7 | 10.9 | 13.5 | 23.3 | 46.3 | 29.4 | 29.7 | 28.4 | 13.0 | 25.1 | 26.1 |
| S2GO-Base (paper) | 35.5 | 22.7 | 21.9 | 13.4 | 27.5 | 32.1 | 14.9 | 15.3 | 12.9 | 11.8 | 13.4 | 24.0 | 46.9 | 29.1 | 30.3 | 29.1 | 14.7 | 26.4 | 19.6 |
| **ours (row-(a) baseline, this repo)** | **14.89** | **5.55** | 0.25 | 4.87 | 0.00 | 9.60 | 7.57 | 3.18 | 0.00 | 1.29 | 0.00 | 3.23 | 5.89 | 16.72 | 8.91 | 10.04 | 10.70 | 6.55 | n/m |

mIoU computed over the 16 paper-standard classes (barrier through vegetation),
evaluated on all 2 225 val sequences covered by parts 01-05 (56 scene-disjoint
val scenes). The repo's `MeanIoU` also reports a 17-class average including the
extra "other" semantic class at 10.78 %; that 17-class value is **5.86 %**.

### Why our number is well below S2GO-Small / -Base

| | Paper (S2GO-Small / -Base) | This run |
|---|---|---|
| Stage-1 pretraining | full (depth + RGB + denoise) | **none — row (a) ablation from scratch** |
| Training data | 700 train scenes / ~28 130 sequences | 369 train scenes / 14 749 sequences (~52 %) |
| Training duration | 24 epochs ≈ 42 000 optim-steps at batch 16 | 2 188 optim-steps (~5 % of paper) |
| Hardware / precision | RTX 4090, fp32 | RTX 3060, bf16 autocast + `scale_min=0.05` |
| Val coverage | full 6 019 val sequences | 2 225 (parts 01-05 val, ~37 % of paper val) |

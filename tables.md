# Tables from S2GO Paper

## Table 1: 3D occupancy estimation results on the nuScenes-SurroundOcc validation set (Wei et al. 2023)

All baselines leverage 900×1600 resolution while S2GO uses 256×704 resolution images for efficiency. All methods are benchmarked on the 4090 GPU. *GaussianWorld's paper results over-weight intermediate frames during evaluation; we re-evaluate released checkpoints under the standard setting.

| Method | IoU | mIoU | barrier | bicycle | bus | car | const. veh. | motorcycle | pedestrian | traffic cone | trailer | truck | drive. surf. | other flat | sidewalk | terrain | manmade | vegetation | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MonoScene (Cao & de Charette, 2022) | 24.0 | 7.3 | 4.0 | 0.4 | 8.0 | 8.0 | 2.9 | 0.3 | 1.2 | 0.7 | 4.0 | 4.4 | 27.7 | 5.2 | 15.1 | 11.3 | 9.0 | 14.9 | – |
| Atlas (Murez et al., 2020) | 28.7 | 15.0 | 10.6 | 5.7 | 19.7 | 24.9 | 8.9 | 8.8 | 6.5 | 3.3 | 10.4 | 16.2 | 34.9 | 15.5 | 21.9 | 21.0 | 11.2 | 20.5 | – |
| BEVFormer (Li et al., 2022b) | 30.5 | 16.6 | 14.2 | 6.6 | 23.5 | 28.3 | 8.7 | 10.8 | 6.6 | 4.1 | 11.2 | 17.8 | 37.3 | 18.0 | 22.9 | 22.2 | 13.8 | 22.2 | 3.1 |
| TPVFormer (Huang et al., 2023) | 30.9 | 17.1 | 16.0 | 5.3 | 23.9 | 27.3 | 9.8 | 8.7 | 7.1 | 5.2 | 11.0 | 13.1 | 34.2 | 13.2 | 23.1 | 23.2 | 11.7 | 20.8 | 2.9 |
| OccFormer (Zhang et al., 2023) | 31.4 | 19.0 | 18.7 | 10.4 | 23.9 | 30.1 | 10.3 | 14.2 | 13.6 | 10.1 | 12.5 | 20.8 | 38.8 | 19.8 | 24.2 | 22.2 | 13.5 | 21.4 | – |
| SurroundOcc (Wei et al., 2023) | 31.5 | 20.3 | 20.6 | 11.7 | 28.1 | 30.9 | 10.7 | 15.1 | 14.1 | 12.1 | 14.4 | 22.3 | 37.3 | 23.7 | 24.5 | 22.8 | 14.9 | 21.9 | 2.7 |
| GaussianFormer (Huang et al., 2024) | 29.8 | 19.1 | 19.5 | 11.3 | 26.1 | 29.8 | 10.5 | 13.8 | 12.6 | 8.7 | 12.7 | 21.6 | 39.6 | 23.3 | 24.5 | 23.0 | 9.6 | 19.1 | 2.7 |
| GaussianFormer-2 (Huang et al., 2025) | 31.7 | 20.8 | 21.4 | 13.4 | 28.5 | 30.8 | 10.9 | 15.8 | 13.6 | 10.5 | 13.6 | 22.0 | 40.6 | 24.4 | 26.1 | 24.3 | 13.0 | 21.8 | 2.6 |
| QuadricFormer (Zuo et al., 2025a) | 31.2 | 20.1 | 19.6 | 13.2 | 27.1 | 29.6 | 11.3 | 16.3 | 12.7 | 9.2 | 12.5 | 21.4 | 40.2 | 24.3 | 25.7 | 24.2 | 13.0 | 21.9 | 6.2 |
| GaussianWorld* (Zuo et al., 2025b) | 32.8 | 21.8 | 21.6 | 13.4 | 27.9 | 32.7 | 12.1 | 18.9 | 16.6 | 11.5 | 15.3 | 23.3 | 41.9 | 24.3 | 28.4 | 26.6 | 15.9 | 22.5 | 5.4 |
| ALOcc-mini-GF (Chen et al., 2025b) | 34.6 | 23.1 | 22.2 | 16.0 | 27.8 | 32.7 | 12.1 | 18.9 | 19.6 | 17.5 | 15.3 | 23.5 | 43.6 | 28.2 | 29.7 | 31.2 | 19.2 | 25.0 | 0.9 |
| ALOcc-GF (Chen et al., 2025b) | 38.2 | 25.5 | 24.3 | 18.8 | 29.8 | 36.3 | 17.9 | 19.5 | 17.5 | 15.5 | 16.5 | 26.5 | 47.6 | 29.9 | 31.2 | 29.2 | 20.0 | 29.5 | 0.9 |
| **S2GO-Small** | 34.3 | 22.1 | 20.8 | 13.3 | 27.5 | 32.1 | 14.9 | 15.3 | 14.0 | 13.4 | 13.5 | 23.5 | 46.3 | 29.2 | 29.7 | 28.4 | 13.0 | 25.1 | **26.1** |
| **S2GO-Base** | 35.5 | 22.7 | 21.9 | 13.4 | 27.5 | 32.1 | 14.9 | 15.3 | 14.0 | 13.4 | 13.5 | 24.0 | 46.9 | 29.1 | 30.3 | 29.1 | 14.7 | 26.4 | 19.6 |

---

## Table 2: Results on the SSCBench-KITTI-360 test set (Geiger et al. 2012) with a monocular camera

S2GO achieves new state-of-the-art, achieving strong performance in all categories.

| Method | Input | IoU | mIoU | car | bicycle | motorcycle | truck | other-veh. | person | road | parking | sidewalk | other-grnd | building | fence | vegetation | terrain | pole | traf.-sign | other-struct. | other-object |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LMSCNet (Roldao et al., 2020) | L | 47.5 | 13.7 | 20.9 | 0.0 | 0.0 | 0.3 | 0.0 | 0.0 | 63.1 | 0.5 | 21.5 | 0.0 | 24.5 | 8.5 | 43.0 | 26.8 | 0.0 | 0.0 | 3.6 | 0.0 |
| SSCNet (Song et al., 2017) | L | 53.6 | 17.0 | 32.0 | 0.0 | 0.0 | 0.3 | 0.0 | 0.0 | 65.7 | 17.3 | 41.2 | 3.2 | 44.4 | 6.8 | 43.0 | 28.9 | 0.8 | 0.8 | 8.6 | 0.0 |
| MonoScene (Cao & de Charette, 2022) | C | 37.9 | 12.3 | 19.3 | 0.4 | 0.6 | 8.0 | 2.0 | 0.9 | 48.4 | 11.4 | 28.1 | 3.2 | 32.9 | 3.5 | 26.2 | 16.8 | 6.9 | 5.7 | 4.2 | 3.1 |
| Voxformer (Li et al., 2023a) | C | 38.8 | 11.9 | 17.8 | 1.2 | 0.9 | 4.6 | 2.1 | 1.6 | 47.4 | 9.7 | 27.2 | 2.9 | 31.2 | 4.6 | 28.7 | 14.7 | 6.5 | 6.9 | 3.2 | 2.4 |
| TPVFormer (Huang et al., 2023) | C | 40.2 | 13.6 | 21.6 | 1.1 | 1.4 | 8.1 | 2.6 | 1.9 | 52.9 | 5.9 | 33.0 | 3.1 | 34.8 | 4.8 | 30.1 | 17.5 | 7.5 | 5.5 | 5.2 | 2.7 |
| OccFormer (Zhang et al., 2023) | C | 40.3 | 13.8 | 22.6 | 0.7 | 0.9 | 9.9 | 3.8 | 2.8 | 54.3 | 13.4 | 31.5 | 3.6 | 36.4 | 4.8 | 31.0 | 19.5 | 7.8 | 8.5 | 7.0 | 4.6 |
| GaussianFormer (Huang et al., 2024) | C | 35.4 | 12.9 | 18.9 | 1.0 | 4.6 | 18.1 | 7.6 | 3.4 | 45.9 | 10.8 | 25.0 | 5.3 | 28.4 | 5.7 | 29.5 | 8.6 | 2.0 | 2.3 | 5.5 | 1.4 |
| GaussianFormer-2 (Huang et al., 2025) | C | 38.4 | 15.5 | 21.1 | 2.6 | 4.2 | 12.4 | 5.7 | 1.6 | 56.3 | 13.3 | 31.5 | 4.5 | 38.0 | 7.2 | 31.2 | 21.1 | 6.4 | 6.5 | 6.0 | 4.2 |
| **S2GO-Base (ours)** | C | **40.8** | **15.1** | 22.7 | 1.3 | 1.7 | 15.5 | 5.1 | 2.1 | 53.8 | 13.3 | 33.4 | 8.8 | 35.3 | 7.2 | 31.2 | 21.1 | 6.4 | 6.5 | 6.0 | 4.2 |

(Header columns reproduced from the published table; some lower-precision per-class entries match the published font-color encoding.)

---

## Table 3: Ablation on pretraining

We ablate pretraining query initialization and the depth, RGB, and query denoising loss terms.
- LiDAR + ε indicates queries are initialized from noised LiDAR.
- Pretraining with all objectives is essential.
- † Trained for 24 epochs (no pretraining baseline at equal compute budget).

| Row | Query Init | Depth | RGB | Denoise | mIoU | IoU |
|---|---|---|---|---|---|---|
| (a) | – | ✗ | ✗ | ✗ | 13.02 | 25.73 |
| (a)† | – | ✗ | ✗ | ✗ | 15.83 | 28.35 |
| (b) | Learnable | ✓ | ✓ | ✗ | 12.42 | 26.64 |
| (c) | LiDAR | ✓ | ✓ | ✗ | 13.62 | 27.08 |
| (d) | LiDAR+ε | ✓ | ✓ | ✗ | 20.55 | 32.68 |
| (e) | LiDAR+ε | ✓ | ✗ | ✗ | 20.25 | 32.44 |
| (d) | LiDAR+ε | ✓ | ✓ | ✗ | 20.55 | 32.68 |
| (f) | LiDAR+ε | ✓ | ✓ | ✓ | **21.60** | **33.91** |

---

## Table 4: Ablation on query propagation strategies

"None" indicates no temporal information is used.

| Propagation Type | mIoU | IoU |
|---|---|---|
| None | 17.92 | 29.24 |
| top-k opacity | 19.94 | 32.03 |
| δ-dist top-k opacity | **20.51** | **32.51** |

---

## Table 5: Ablation on using velocity modeling in each stage

| Pretrain. | Occ. Est. | mIoU | IoU |
|---|---|---|---|
| ✗ | ✗ | 20.07 | 31.87 |
| ✗ | ✓ | 20.15 | 31.94 |
| ✓ | ✗ | 20.50 | 32.62 |
| ✓ | ✓ | **20.55** | **32.68** |

---

## Table 6: Ablation on Gaussian-to-Voxel Splatting (G2V)

GPU training hours are calculated for training 12 epochs on a single GPU. All benchmarks are on a single 4090 GPU.

| Opacity in α | Efficient G2V | mIoU | IoU | Train GPU hours | Infer GPU Mem. | Infer. FPS |
|---|---|---|---|---|---|---|
| ✗ | ✗ | 16.97 | 28.75 | 55 h | 2436 MB | 25.2 |
| ✓ | ✗ | 20.13 | 32.28 | 129 h | 7043 MB | 20.4 |
| ✓ | ✓ | **20.55** | **32.68** | **28 h** | **2448 MB** | **25.3** |

---

## Table 7: Ablation on the number of queries and Gaussians

FPS is measured on a 4090 GPU.

| # Query | # Gauss./Query | # Gauss. | mIoU | IoU | FPS |
|---|---|---|---|---|---|
| 900 | 10 | 9 000 | 21.60 | 33.91 | **26.1** |
| 1 260 | 14 | 17 640 | 21.78 | 34.15 | 22.7 |
| 1 800 | 20 | 36 000 | **21.84** | **34.51** | 19.6 |

---

## Table 8: Ablation on pretraining with various depth sources

| Pretraining Query Initialization | mIoU | IoU |
|---|---|---|
| Occupied voxel locations | 21.61 | 33.75 |
| LiDAR (32-line) | **21.60** | **33.91** |
| LiDAR (16-line) | 21.16 | 33.39 |
| Zero-shot RGB depth pred. (Yin et al. 2023) | 20.99 | 33.57 |

---

## Table 9: Quantitative comparisons with QuadricFormer (Zuo et al. 2025a) at 256×704 resolution

All methods benchmarked on an **A6000 GPU**.

| Method | IoU | mIoU | barrier | bicycle | bus | car | const. veh. | motorcycle | pedestrian | traffic cone | trailer | truck | drive. surf. | other flat | sidewalk | terrain | manmade | vegetation | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QuadricFormer-3.2k | 28.9 | 18.5 | 18.0 | 11.0 | 25.7 | 28.0 | 11.0 | 13.3 | 11.7 | 9.1 | 11.5 | 19.2 | 39.0 | 23.1 | 24.5 | 22.9 | 19.2 | 22.3 | – |
| QuadricFormer-12.8k | 30.3 | 18.9 | 18.2 | 10.6 | 25.3 | 28.4 | 11.0 | 12.6 | 11.6 | 9.4 | 12.3 | 19.8 | 39.9 | 22.6 | 25.6 | 23.7 | 10.8 | 19.8 | 13.9 |
| **S2GO-Small** | 34.3 | 22.1 | 20.8 | 13.1 | 27.5 | 30.3 | 14.5 | 16.5 | 11.7 | 13.5 | 13.5 | 23.5 | 46.3 | 29.2 | 29.7 | 28.4 | 13.0 | 25.1 | **32.2** |
| **S2GO-Base** | **35.5** | **22.7** | 21.9 | 13.4 | 27.5 | 32.1 | 14.9 | 15.3 | 12.9 | 11.8 | 13.4 | 24.0 | 46.9 | 29.1 | 30.3 | 29.1 | 14.7 | 26.4 | 23.6 |

---

## Table 10: 3D occupancy performance on the Occ3D-nuScenes validation set (Tian et al. 2023)

FPS is benchmarked on both an A100 and on a 4090. A100 numbers for prior work are sourced from SparseOcc (Liu et al. 2024) and original papers, while 4090 numbers are sourced from ALOcc (Chen et al. 2025a). † ProtoOcc benchmarks on a 3090 GPU.

| Method | Backbone | Mask | Input Size | Epoch | RayIoU | RayIoU (1m, 2m, 4m) | mIoU | FPS (A100) | FPS (4090) |
|---|---|---|---|---|---|---|---|---|---|
| BEVFormer (Li et al., 2022b) | R101 | ✓ | 1600×900 | 24 | 32.4 | 26.1 / 32.9 / 38.0 | 39.2 | 3.0 | – |
| RenderOcc (Pan et al., 2023) | Swin-B | ✓ | 1408×512 | 12 | 19.5 | 13.4 / 19.6 / 25.5 | 24.4 | – | – |
| SimpleOcc (Gan et al., 2024) | R101 | ✓ | 672×336 | 12 | 22.5 | 17.0 / 22.7 / 27.9 | 31.8 | 9.7 | – |
| BEVDet-Occ (Huang & Huang, 2021) | R50 | ✓ | 704×256 | 90 | 29.6 | 23.6 / 30.0 / 35.1 | 36.1 | 2.6 | – |
| BEVDet-Occ-Long (Huang & Huang, 2021) | R50 | ✓ | 704×384 | 90 | 32.6 | 26.6 / 33.1 / 38.2 | 39.3 | 0.8 | – |
| FB-Occ (Li et al., 2023b) | R50 | ✓ | 704×256 | 90 | 33.5 | 26.7 / 34.1 / 39.7 | 39.1 | 10.3 | – |
| ProtoOcc (Kim et al., 2025) | R50 | ✓ | 704×256 | 24 | – | – | 39.6 | – | 12.8† |
| ProtoOcc (Oh et al., 2025) | R50 | ✓ | 800×432 | 12 | – | – | 39.0 | – | – |
| BEVFormer (Li et al., 2022b) | R101 | ✗ | 1600×900 | 24 | 33.7 | – | 23.7 | 3.0 | 4.4 |
| FB-Occ (Liu et al., 2023a) | R50 | ✗ | 704×256 | 90 | 35.6 | – | 27.9 | 10.3 | – |
| SparseOcc (Liu et al., 2024) | R50 | ✗ | 704×256 | 24 | 36.1 | 30.2 / 36.8 / 41.2 | 30.9 | 12.5 | – |
| GSD-Occ (Tian et al., 2024) | R50 | ✗ | 704×256 | 48 | 38.9 | – | – | 20.0 | – |
| StreamOcc (Moon et al., 2025) | R50 | ✗ | 704×256 | 24 | 41.1 | 34.2 / 41.9 / 47.1 | – | – | 12.0 |
| OPUS-L (Wang et al., 2024a) | R50 | ✗ | 704×256 | 100 | 41.2 | 34.7 / 42.1 / 46.7 | 36.2 | 7.2 | 8.2 |
| STCOcc (Liao et al., 2025) | R50 | ✗ | 704×256 | 36 | 42.1 | 36.9 / 42.8 / 46.7 | – | – | – |
| ALOcc-3D (Chen et al., 2025b) | R50 | ✗ | 704×256 | 54 | 43.7 | 37.8 / 44.7 / 48.8 | 38.0 | – | 6.0 |
| ALOcc-GF (Chen et al., 2025b) | R50 | ✗ | 704×256 | 24 | **44.1** | – | – | – | 6.2 |
| **S2GO-Small (ours)** | R50 | ✗ | 704×256 | 24 | 37.2 | 31.3 / 38.1 / 42.2 | 30.8 | 20.8 | **25.3** |
| **S2GO-Base (ours)** | R50 | ✗ | 704×256 | 24 | 39.1 | 33.1 / 40.0 / 44.1 | 31.2 | 14.5 | 20.9 |

---

## BibTeX (from project site https://jindapark.github.io/projects/s2go/)

```bibtex
@inproceedings{Park2026S2GO,
  title={S2GO: Streaming Sparse Gaussian Occupancy},
  author={Jinhyung Park and Chensheng Peng and Yihan Hu and Wenzhao Zheng and Kris Kitani and Wei Zhan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

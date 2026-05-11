# Background papers + reference repos

These are the papers cited at the top of S2GO §2 (Related Work) and §3.2 (Architecture) — the streaming-perception lineage S2GO inherits from. PDFs are in [papers/](papers/), code is in [reference_code/](reference_code/).

| # | Citation | Paper | arXiv | PDF | GitHub repo | Cloned at |
|---|---|---|---|---|---|---|
| 1 | Carion et al., 2020 | End-to-End Object Detection with Transformers (**DETR**) | [2005.12872](https://arxiv.org/abs/2005.12872) | ✅ [papers/1_DETR_Carion2020.pdf](papers/1_DETR_Carion2020.pdf) | [facebookresearch/detr](https://github.com/facebookresearch/detr) | ✅ [reference_code/DETR/](reference_code/DETR/) |
| 2 | Wang et al., 2020 | DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries | [2110.06922](https://arxiv.org/abs/2110.06922) | ✅ [papers/2_DETR3D_Wang2020.pdf](papers/2_DETR3D_Wang2020.pdf) | [WangYueFt/detr3d](https://github.com/WangYueFt/detr3d) | ✅ [reference_code/DETR3D/](reference_code/DETR3D/) |
| 3 | Wang et al., 2023 | Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection (**StreamPETR**) | [2303.11926](https://arxiv.org/abs/2303.11926) | ✅ [papers/3_StreamPETR_Wang2023.pdf](papers/3_StreamPETR_Wang2023.pdf) | [exiawsh/StreamPETR](https://github.com/exiawsh/StreamPETR) | ✅ [reference_code/StreamPETR/](reference_code/StreamPETR/) |
| 4 | Lin et al., 2022 | Sparse4D: Multi-view 3D Object Detection with Sparse Spatial-Temporal Fusion | [2211.10581](https://arxiv.org/abs/2211.10581) | ✅ [papers/4_Sparse4D_Lin2022.pdf](papers/4_Sparse4D_Lin2022.pdf) | [HorizonRobotics/Sparse4D](https://github.com/HorizonRobotics/Sparse4D) | ✅ [reference_code/Sparse4D/](reference_code/Sparse4D/) |
| 5 | Yuan et al., 2024 | StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction | [2308.12570](https://arxiv.org/abs/2308.12570) | ✅ [papers/5_StreamMapNet_Yuan2024.pdf](papers/5_StreamMapNet_Yuan2024.pdf) | [yuantianyuan01/StreamMapNet](https://github.com/yuantianyuan01/StreamMapNet) | ✅ [reference_code/StreamMapNet/](reference_code/StreamMapNet/) |

## Suggested read order (time-bound)

**DETR (1) → StreamPETR (3) → Sparse4D (4)**

- **DETR** establishes the matching idea (direct Hungarian) that S2GO contrasts with in §1 (occupancy can't use it).
- **StreamPETR** is the closest-pattern paper — S2GO's queue, ego-motion compensation, and `PETRTemporalDecoderLayer` derive almost verbatim from here.
- **Sparse4D** offers an alternative streaming design (sparse 4D feature sampling) that informs the propagation choices.

DETR3D and StreamMapNet are useful background but not on the critical path for S2GO implementation.

## Other reference repos already in [reference_code/](reference_code/)

| Repo | Origin | Role in S2GO impl |
|---|---|---|
| [GaussianFormer/](reference_code/GaussianFormer/) | huang-yh/GaussianFormer (merged GF + GF-2) | Gaussian primitives, v2 refiner, `localagg_prob_fast` CUDA splatter |
| [SurroundOcc/](reference_code/SurroundOcc/) | weiyithu/SurroundOcc | NuScenes occupancy dataloader, eval metrics, GT pipeline |

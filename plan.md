# S2GO Full Reproduction — Implementation Plan

## Context

S2GO (ICLR 2026, Park et al.) is a streaming sparse-Gaussian framework for camera-only 3D semantic
occupancy estimation. It targets the same accuracy as dense Gaussian methods (GaussianFormer-2,
GaussianWorld) while running 4–6× faster (26 FPS on a 4090) by maintaining only ~1k temporally
propagated 3D queries that decode into ~9k–36k Gaussians per frame.

The user wants a **full reproduction** (training + evaluation) of S2GO at
[/media/skr/storage/self_driving/S2GO/s2go/](/media/skr/storage/self_driving/S2GO/s2go/), reusing
the surveyed reference repos under [reference_code/](/media/skr/storage/self_driving/S2GO/reference_code/).
The goal is to land Table 1 numbers (S2GO-Small: 22.1 mIoU / 34.3 IoU / 26 FPS at 256×704) on
nuScenes-SurroundOcc as the primary milestone, then add the Table 6 ablation matrix.

### Current state (2026-05-08, after sub-phase 0)

- **Datasets**: SurroundOcc GT (`train.zip` 2.99 GB / `val.zip` 627 MB) and pickles fully downloaded; **`val.zip` extracted** (6 019 .npy at `data/nuscenes_occ/nuscenes_occ/samples/`); `train.zip` not extracted yet. nuScenes raw is keyframes-part-1 only (3 376 LiDAR + 6×3 376 cams, **extracted** at `data/nuscenes/{samples,v1.0-trainval}/`); LiDAR is bundled in the keyframes archive. **Part-1 covers 15.2 % of val (914 / 6 019)** — enough for sub-phase 1 dataloader testing, **not** enough for Phase H/J full eval (need parts 2–10, ~38 GB more).
- **Environment**: conda env at `/media/skr/storage/conda_envs/selfocc` validated. Stack is **torch 2.0.0+cu118 / mmcv 2.0.1 / mmdet 3.0.0 / mmdet3d 1.1.1 / mmengine 0.10.7 / spconv 2.3.6 / gsplat 1.5.3 / torch_cluster 1.6.3+pt20cu118 / flash_attn 2.6.3** on RTX 3060 (sm_86). CUDA toolchain auto-routes to env-local nvcc 11.8 via `activate.d/cuda118.sh`. See §2 for full table.
- **Reference repos**: GaussianFormer (ECCV 2024) and GaussianFormer-2 (CVPR 2025) merged in [reference_code/GaussianFormer/](reference_code/GaussianFormer/) — GF-2 lives under `config/prob/`, `localagg_prob_fast/`, `refine_module_v2.py`. The GF-2 prob splatter already implements S2GO Eqs. 1–4; only Eq. 9 + tiling diff is left. DETR / DETR3D / StreamPETR / Sparse4D / StreamMapNet also cloned for cross-reference (see [download.md](download.md)).
- **Code**: no `s2go/` package on disk yet. Configs and modules referenced throughout this plan are targets, not files. The closest existing templates are GF-2's `config/prob/nuscenes_gs6400.py` (model arch + occ loss + AdamW), GF's `_base_/{model,surroundocc,misc}.py` (dataset + runtime), and StreamPETR's `stream_petr_r50_flash_704_bs2_seq_24e.py` (streaming sampler + memory queue).
- **Conceptual background**: see the "Conceptual background" section of [Implementation.md](Implementation.md) for the online-framework definition, DETR Hungarian-matching context, full Stage-1 / Stage-2 training pseudocode, Eq. 8 term-by-term breakdown, and the §3.4.3 efficient-G2V plain-English summary. Not duplicated here.
- **Execution sequence**: Stage-1-first (sidesteps the CUDA kernel work in Phases D + E, which are Stage-2 only). Sub-phases: 0 env+unpack ✅ → 1 dataloader+depth-projection → 2 FPS+ε init → 3 model skeleton (no CUDA) → 4 4-sample overfit. Each sub-phase ~1 day; total Stage-1 path ~600 LOC of PyTorch.

User-confirmed scope decisions:
- **Full train + eval reproduction** (Stage 1 pretraining + Stage 2 occupancy).
- **Variant order: S2GO-Small first, then S2GO-Base.** Small is the easier target (fewer queries, fewer Gaussians, ImageNet1k init — no dependency on the nuImages pretrained checkpoint), and it is the FPS-leading variant the paper recommends for real-time use. S2GO-Base reuses the exact same code path, only swapping `K=900→1800`, `J=10→20`, and the backbone init from torchvision ImageNet1k → mmdet3d nuImages. Once Small reproduces (Phase J), Base is essentially a config swap + retrain.
- **Framework**: mmengine + mmcv (matches GaussianFormer / StreamPETR).
- **CUDA kernel**: modify GaussianFormer's `localagg_prob_fast` rather than rewrite. Two changes
  (Eq. 9 opacity weighting + 4×4×4 voxel-block tiling).
- **Datasets**: defer multi-dataset choice; build dataloader interface dataset-agnostic so
  Occ3D-nuScenes and KITTI-360 can be added later. Primary target = nuScenes-SurroundOcc.

### Variant differences (paper Appendix B)

| | S2GO-Small | S2GO-Base |
|---|---|---|
| # queries (K) | 900 | 1800 |
| # Gaussians per query (J) | 10 | 20 |
| Total Gaussians | 9 000 | 36 000 |
| Backbone init | ImageNet1k (torchvision) | nuImages-pretrained R50 |
| Image resolution | 256×704 | 256×704 |
| Stage-1 pretrain epochs (nuScenes) | 12 | 12 |
| Stage-2 occupancy epochs (nuScenes) | **24** | **24** |
| KITTI Stage-2 epochs (if added later) | 12 | 12 |
| Headline (Table 1, 4090) | 22.1 mIoU / 34.3 IoU / 26.1 FPS | 22.7 mIoU / 35.5 IoU / 19.6 FPS |

Everything else — architecture, hyperparameters, optimizer, loss weights, queue length, embed dim, training schedule shape — is identical between the two. Implementing Small ⇒ Base is a config diff, not a code diff.

## What to build vs. what to reuse

### Reused largely as-is (~80% of foundation)

**Important**: the local [GaussianFormer repo](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/) holds the merged codebases of **both GaussianFormer (ECCV 2024) and GaussianFormer-2 (CVPR 2025)** — the latter is in `config/prob/` + `localagg_prob_fast/` + `refine_module_v2.py` + `localagg_prob/`. This recovers the probabilistic mixture-of-Gaussians formulation S2GO inherits, plus the v2 refiner that's the right base for the S2GO refiner.

- **Gaussian primitives** from [GaussianFormer/model/encoder/gaussian_encoder/utils.py:62-69](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/model/encoder/gaussian_encoder/utils.py#L62-L69) (`GaussianPrediction` namedtuple).
- **GaussianFormer-2 v2 refiner** from [refine_module_v2.py](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/model/encoder/gaussian_encoder/refine_module_v2.py) (`SparseGaussian3DRefinementModuleV2`) — base for our parent-query refiner; we add the J-children head on top.
- **GaussianFormer-2 prob splatter** from [model/head/localagg_prob_fast/](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/model/head/localagg_prob_fast/) — already implements the probabilistic mixture (Eq. 4 in S2GO). We fork + apply the Eq. 9 + 4×4×4 tile diffs (§5). Three kernel variants exist (`localagg/`, `localagg_prob/`, `localagg_prob_fast/`); we target the third.
- **GF-2 distribution-init infra** from [config/prob/](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/config/prob/) — `PixelDistributionLoss` and the loss-multi wiring. Useful as a working ablation baseline; S2GO replaces the init itself with FPS+ε but keeps the loss-stack pattern.
- **Encoder skeleton** from [gaussian_encoder.py](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/model/encoder/gaussian_encoder/gaussian_encoder.py) — the `operation_order` orchestration is reusable verbatim with our op list inserted.
- **Streaming queue + temporal decoder** from [StreamPETR/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py:312-365](/media/skr/storage/self_driving/S2GO/reference_code/StreamPETR/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py#L312-L365) and [StreamPETR/.../utils/petr_transformer.py:513-797](/media/skr/storage/self_driving/S2GO/reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/petr_transformer.py#L513-L797) (`PETRTemporalDecoderLayer`, `PETRMultiheadFlashAttention`).
- **Ego-motion utilities** from [StreamPETR/.../utils/misc.py](/media/skr/storage/self_driving/S2GO/reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/misc.py) (`transform_reference_points`, `memory_refresh`, `topk_gather`, `MLN`).
- **Deformable cross-attention to images** from [StreamPETR/.../utils/detr3d_transformer.py:480-562](/media/skr/storage/self_driving/S2GO/reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/detr3d_transformer.py#L480-L562) (`DeformableFeatureAggregationCuda`). GaussianFormer also has [its own deformable cross-attn](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/model/encoder/gaussian_encoder/ops/) — choose whichever integrates more cleanly with the GF-2 v2 refiner.
- **NuScenes loader + GT pipeline** from [SurroundOcc/projects/mmdet3d_plugin/datasets/nuscenes_occupancy_dataset.py](/media/skr/storage/self_driving/S2GO/reference_code/SurroundOcc/projects/mmdet3d_plugin/datasets/nuscenes_occupancy_dataset.py); eval metrics in [evaluation_metrics.py](/media/skr/storage/self_driving/S2GO/reference_code/SurroundOcc/projects/mmdet3d_plugin/datasets/evaluation_metrics.py).
- **Loss infra** (CE + Lovász + focal + multi-loss wrapper) from [GaussianFormer/loss/](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/loss/).
- **Training entrypoint pattern** from [GaussianFormer/train.py](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/train.py) (MMEngine config-driven, TIMM cosine, DDP, AMP).

### Built from scratch (genuinely novel S2GO contributions)

After accounting for the GF/GF-2 merge plus reuse from external repos (gsplat, BEVDepth-style depth projection, torch_cluster FPS), the truly-from-scratch list is small:

1. **Hierarchical query→Gaussian decomposition (Eq. 6)** — K parent queries + J child Gaussians with shared parent attributes (p, v, a) and child attributes (offset, r, s, a). Built on top of the GF-2 v2 refiner: query head emits parent params, new children head emits J × {o, r, s, a}.
2. **δ-distance top-k opacity propagator** — ~20-line greedy NMS-by-distance over query opacities. No prior public implementation.
3. **G2V kernel diff for Eq. 9** — opacity-in-occupancy modification of `localagg_prob_fast` forward/backward (~30-line patch, drops `(2π)^-1.5·|Σ|^0.5` factor; multiplies opacity into `bin_logit`). The 4×4×4 tile rewrite (§3.4.3) is a larger surgery, but the per-voxel fallback covers correctness.

### Built mostly from external open-source (no reimplementation)

These were originally on the "from scratch" list but have public reference implementations to fork:

4. **Two-stage train pipeline** — start from **GaussianPretrain** (Yang 2024, arXiv:2411.12452) if its code is public; otherwise scaffold using GF's MMEngine train loop with a stage flag.
5. **Gaussian rendering for depth/RGB pretrain loss** — **gsplat** (Apache-2.0) natively supports depth output, or **diff-gaussian-rasterization** with the public depth patch. **DesireGS** (Peng 2024, S2GO co-author Chensheng Peng) likely has the closest-style wrapper.
6. **FPS+ε query initialization** — `torch_cluster.fps` or `mmdet3d.ops.furthest_point_sample`. The +ε is one line.
7. **LiDAR→camera depth-map projection** — **BEVDepth** data prep already projects LiDAR pts via `lidar2img` to make sparse depth GT.
8. **3D velocity prediction per query** — trivial extension of StreamPETR's `code_size=10` to `code_size=11` with z-velocity. **Sparse4D-v3** does 3D velocity natively.
9. **Temporal decoder for Gaussian queries** — StreamPETR's `PETRTemporalDecoderLayer` minus its detection-specific refinement, plus our refiner from #1.
10. **Streaming sequence sampler** — StreamPETR's sampler + ~50 lines wiring to SurroundOcc's `CustomNuScenesOccDataset`.

## 1. Repository Layout

Root: `/media/skr/storage/self_driving/S2GO/s2go/`

```
s2go/
├── setup.py                      # editable install + builds CUDA exts
├── environment.yml               # conda env (see §2)
├── s2go/
│   ├── datasets/
│   │   ├── nusc_surroundocc.py   # S2GONuScenesOccDataset (forked; streaming-aware)
│   │   ├── pipelines/
│   │   │   ├── loading.py        # LoadMultiViewImage, LoadOccGT, LoadLiDAR
│   │   │   ├── lidar_depth.py    # ProjectLiDARToCamDepth (sparse depth maps)
│   │   │   ├── fps_init.py       # SampleFPSQueryInit (Stage-1 only)
│   │   │   └── transforms.py     # ResizeCropFlipImage 256×704, normalize
│   │   ├── samplers/seq_group_sampler.py  # ported from StreamPETR
│   │   └── utils.py              # ego_pose math, lidar2img helpers
│   ├── models/
│   │   ├── detectors/s2go.py     # top-level S2GO detector
│   │   ├── backbone/resnet.py    # mmdet ResNet-50 wrapper
│   │   ├── neck/cp_fpn.py        # FPN multi-scale
│   │   ├── lifter/s2go_lifter.py # learnable + LiDAR-FPS+ε modes
│   │   ├── encoder/
│   │   │   ├── temporal_decoder.py
│   │   │   ├── decoder_layer.py
│   │   │   ├── flash_self_attn.py     # ported PETRMultiheadFlashAttention
│   │   │   ├── def_cross_attn.py      # ported DeformableFeatureAggregationCuda
│   │   │   ├── refiner.py             # emits Δp, Δa, v + (J × child params)
│   │   │   ├── propagator.py          # δ-dist top-k opacity
│   │   │   └── memory_queue.py        # 4-frame queue + ego-pose updates
│   │   ├── head/
│   │   │   ├── splat_head.py     # calls modified CUDA kernel
│   │   │   └── render_head.py    # gsplat depth/RGB
│   │   └── splatting/
│   │       ├── gaussian_assembly.py   # Eq. 6 query→K·J Gaussians
│   │       └── localagg_wrap.py       # autograd Function around s2go_localagg ext
│   ├── losses/
│   │   ├── occ_loss.py           # CE + Lovász (port GF/loss/occupancy_loss.py)
│   │   ├── pretrain_loss.py      # denoise + L_depth + L_rgb (Eq. 8)
│   │   └── lovasz.py             # port GF/loss/utils/lovasz_softmax.py
│   ├── render/
│   │   ├── gsplat_wrapper.py     # render_depth_rgb(gaussians, cams)
│   │   └── camera_utils.py
│   ├── cuda/
│   │   └── s2go_localagg/        # MODIFIED fork of localagg_prob_fast
│   │       ├── setup.py
│   │       ├── ext.cpp
│   │       ├── src/{forward.cu, backward.cu, aggregator_impl.cu, auxiliary.h, config.h}
│   │       └── tests/test_kernel.py
│   ├── configs/
│   │   ├── _base_/{default_runtime.py, nusc_surroundocc.py}
│   │   ├── s2go_small_pretrain.py     # K=900, J=10, 12 ep, ImageNet1k init  ← START HERE
│   │   ├── s2go_small_occ.py          # K=900, J=10, 24 ep
│   │   ├── s2go_base_pretrain.py      # K=1800, J=20, 12 ep, nuImages init
│   │   └── s2go_base_occ.py           # K=1800, J=20, 24 ep
│   ├── tools/{train.py, eval.py, prep_lidar_depth.py, vis.py}
│   └── utils/{safe_ops.py, geom.py, ego_motion.py, distributed.py}
└── tests/{test_dataset.py, test_lifter.py, test_temporal_decoder.py,
         test_kernel_grad.py, test_e2e_smoke.py}
```

## 2. Environment

The conda env at `/media/skr/storage/conda_envs/selfocc` (built per [scripts/setup_gaussianformer_env.sh](scripts/setup_gaussianformer_env.sh) following GaussianFormer-2's recipe) is **already validated** (sub-phase 0 of execution, 2026-05-08). Actual installed stack:

| Package | Version | Notes |
|---|---|---|
| Python | 3.8.16 | |
| torch / torchvision | 2.0.0+cu118 / 0.15.1+cu118 | newer than originally planned (was 1.13.1) — works fine |
| mmcv | 2.0.1 | mmcv 2.x API (not 1.x) |
| mmdet / mmdet3d / mmsegmentation / mmengine | 3.0.0 / 1.1.1 / (auto) / 0.10.7 | matches GaussianFormer-2 release |
| spconv-cu117 | 2.3.6 | |
| timm / numpy | 1.0.26 / 1.24.4 | |
| nuscenes-devkit | local checkout at `/media/skr/storage/self_driving/S2GO/nuscenes/nuscenes-devkit/python-sdk` | importable; no pip install needed |
| **gsplat** | 1.5.3 | Apache-2.0; supports torch 2.0; depth output native |
| **torch_cluster** | 1.6.3+pt20cu118 | from PyG wheel index; provides `fps()` for query init |
| **flash_attn** | 2.6.3 | v2 API — note: StreamPETR uses v1 API (`flash_attn_unpadded_kvpacked_func`); needs ~4-line port to `flash_attn_varlen_kvpacked_func`. v2.7+ dropped torch 2.0/cp38 wheels — 2.6.3 is the ceiling for our combo |

**GPU**: RTX 3060 (Ampere, sm_86), CUDA 11.8 from env-local nvcc.

### CUDA toolchain auto-export

The host has CUDA 12.8 at `/usr/local/cuda` which mismatches the torch 2.0+cu118 wheel. Fix: a `conda activate.d` script auto-points `CUDA_HOME` / `PATH` / `LD_LIBRARY_PATH` at the env-local nvcc 11.8 whenever the env is activated. Files:
- [conda_envs/selfocc/etc/conda/activate.d/cuda118.sh](file:///media/skr/storage/conda_envs/selfocc/etc/conda/activate.d/cuda118.sh) — exports
- [conda_envs/selfocc/etc/conda/deactivate.d/cuda118.sh](file:///media/skr/storage/conda_envs/selfocc/etc/conda/deactivate.d/cuda118.sh) — restores

Without this, building any CUDA extension (torch_cluster, flash_attn, our `s2go_localagg`) fails with `detected CUDA version (12.8) mismatches torch (11.8)`.

### Wheels and references

- flash_attn 2.6.3 wheel: https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
- torch_cluster wheel index: https://data.pyg.org/whl/torch-2.0.0+cu118.html
- gsplat: `pip install gsplat` (PyPI wheels exist for our combo).

### Plan-time vs reality deltas

The original plan targeted mmcv 1.6.2 / mmdet 2.28 / mmdet3d 1.0.0rc6 — actual env is **mmcv 2.x / mmdet 3.x / mmdet3d 1.1.1**. Two practical implications:

1. The mmengine 2.x registry / config API differs from 1.x — code ported from StreamPETR (which uses 1.x conventions) needs minor adapters (`@MODELS.register_module()` from `mmengine.registry` instead of `mmcv.runner`).
2. The GaussianFormer / GaussianFormer-2 reference repo is **already on this 2.x stack**, so its modules drop in directly with no API porting.

## 3. Phase-by-Phase Implementation Order

Each phase ends with a concrete test that must pass before proceeding. Each entry annotates **R** = reused (forked / wrapped from a reference repo), **N** = net-new code we author.

### Sub-phase mapping (Stage-1-first execution sequence)

The user-greenlit execution path runs **Stage 1 first** to validate the streaming + rendering plumbing before committing to the riskier Stage-2 CUDA kernel work. This re-cuts the lettered phases into a tactical sequence; the lettered phases below are the architectural contract, the sub-phases are what we actually do this week.

| Sub-phase | What | Maps to lettered phase(s) | Status |
|---|---|---|---|
| **0** | Env + data unpack + smoke read | part of A | ✅ done (2026-05-08) |
| **1** | `S2GONuScenesOccDataset` + streaming sampler + `ProjectLiDARToCamDepth` | rest of A | ⏭️ next |
| **2** | FPS+ε query init pipeline stage | piece of H prep | pending |
| **3** | Stage-1 model skeleton (no CUDA): backbone + lifter pretrain mode + temporal decoder + parent+J refiner + gsplat render head | B + C + F + G | pending |
| **4** | Eq. 8 loss + 4-sample overfit run | H (toy) | pending |
| (later) | CUDA splatter, voxel CE+Lovász, full Stage-2 train, eval, ablations | D + E + I + J + K | deferred until Stage 1 validates |

Phase headers below carry their sub-phase tag inline (e.g., "Phase A `[sub-0, sub-1]`") so the two views stay in sync.

### Phase A `[sub-0 ✅, sub-1 ⏭️]` — Env + dataloader smoke
- **R** SurroundOcc's [`CustomNuScenesOccDataset`](/media/skr/storage/self_driving/S2GO/reference_code/SurroundOcc/projects/mmdet3d_plugin/datasets/nuscenes_occupancy_dataset.py), GT pipeline, [`evaluation_metrics.py`](/media/skr/storage/self_driving/S2GO/reference_code/SurroundOcc/projects/mmdet3d_plugin/datasets/evaluation_metrics.py); StreamPETR's `InfiniteGroupEachSampleInBatchSampler` and `prev_exists` flag.
- **N** thin `S2GONuScenesOccDataset` subclass wiring streaming sampler + multi-frame collation; `pipelines/lidar_depth.py` (LiDAR→cam depth, BEVDepth-style).
- **Exit test**: `pytest tests/test_dataset.py` — 4-frame batch with correct shapes.
- **Status (2026-05-08)**: sub-phase 0 PASSED — env validated, all CUDA exts importable, nuScenes meta + part-1 keyframes extracted, SurroundOcc val GT extracted (6 019 .npy), end-to-end smoke read of one matched sample (LiDAR + 6 cams + occ GT) confirms shapes/value-ranges. Sub-phase 1 (`S2GONuScenesOccDataset` + streaming sampler + `ProjectLiDARToCamDepth`) is the next milestone.

### Phase B `[sub-3]` — Backbone + neck (1 frame)
- **R** mmdet `ResNet` + `FPN` (or GF's [`backbone/`](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/model/backbone/) and [`neck/`](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/model/neck/)).
- **N** thin `s2go/models/detectors/s2go.py` skeleton (forks GF's `segmentor/`).
- **Exit test**: forward gives multi-scale feats `[(B*6, 256, h, w)]_{l=0..3}`.

### Phase C `[sub-3]` — Lifter + decoder (1 frame, no memory)
- **R** GF [`gaussian_lifter.py`](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/model/lifter/gaussian_lifter.py) (learnable mode); GF-2 [`refine_module_v2.py`](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/model/encoder/gaussian_encoder/refine_module_v2.py); StreamPETR `PETRMultiheadFlashAttention`, `DeformableFeatureAggregationCuda`; GF [`gaussian_encoder.py`](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/model/encoder/gaussian_encoder/gaussian_encoder.py) `operation_order` orchestration.
- **N** `refiner.py` parent-query head + J-children head (Tier-1 item #1, ~150 lines); `decoder_layer.py` glue; `s2go_lifter.py` mode-flag wrapper.
- **Exit test**: `tests/test_lifter.py` — produces K·J Gaussians with paper-spec shapes.

### Phase D `[deferred — Stage-2 only]` — Python-prototype G2V splat + voxel CE
- **R** GF [`loss/occupancy_loss.py`](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/loss/occupancy_loss.py) (CE + Lovász + multi-loss wrapper).
- **N** `splatting/gaussian_assembly.py` (Eq. 6 flatten K·J Gaussians); naive PyTorch O(K·J·V) splat as numerical reference for Phase E.
- **Exit test**: matches Phase E within 1e-4 on a 50³ subsampled grid.

### Phase E `[deferred — Stage-2 only]` — Modified CUDA kernel
- **R** GF-2 [`localagg_prob_fast/`](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/model/head/localagg_prob_fast/) — verbatim base, including the autograd Function pattern from `ext.cpp`.
- **N** Eq. 9 diff in `forward.cu` + `backward.cu` (~30-line patch, Tier-1 item #3); `localagg_wrap.py` autograd Function; **optional** 4×4×4 tile rewrite (~200 lines, Tier-1 item #4 — splittable into a follow-up).
- **Exit test**: `tests/test_kernel_grad.py` `gradcheck` passes (1e-3 atol vs Phase-D Python reference).

### Phase F `[sub-3]` — Streaming queue + temporal transformer
- **R** StreamPETR `pre_update_memory`, `post_update_memory`, [`memory_refresh`](/media/skr/storage/self_driving/S2GO/reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/misc.py), `transform_reference_points`, `MLN` modulation; `PETRTemporalDecoderLayer` re-enabled with `temp_memory`.
- **N** `memory_queue.py` 4-frame queue wrapper; `propagator.py` δ-distance top-k opacity (~30 lines, Tier-1 item #2); ~20 lines wiring 3D-velocity ego-compensation (Tier-2 item #5).
- **Exit test**: 4-frame forward with `prev_exists` toggle; queue grows then refreshes; no NaN across frames.

### Phase G `[sub-3]` — Gaussian rendering pipeline
- **R** `gsplat` library (or `diff-gaussian-rasterization` fallback); if **GaussianPretrain** or **DesireGS** code is public, fork their wrapper directly.
- **N** `gsplat_wrapper.py` thin adapter (~150 lines, Tier-3 item #9); `render_head.py` (RGB MLP from query embed); `pretrain_loss.py` (L1 depth on sparse mask + L1+SSIM RGB).
- **Exit test**: 4-sample overfit run — render loss decreases monotonically; visual sanity via `tools/vis.py`.

### Phase H `[sub-2, sub-4 toy → full]` — Stage-1 pretraining (**Small first**)
- **R** GF [`train.py`](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/train.py) — MMEngine harness, optimizer, TIMM cosine, AMP, DDP, checkpoint resume.
- **N** `configs/s2go_small_pretrain.py` (K=900, J=10, ImageNet1k init); `pipelines/fps_init.py` (`torch_cluster.fps` + ε noise, Tier-2 item #6); two-stage scaffold flag (Tier-3 item #10).
- **Exit test**: full 12-epoch run on 4×4090 converges; denoise loss < 0.1 m by epoch 6.

### Phase I `[deferred — after Stage 1 validates]` — Stage-2 occupancy training (**Small first**)
- **R** GF `train.py` (same harness as Phase H); CE + Lovász loss from Phase D.
- **N** `configs/s2go_small_occ.py` (loads Stage-1 checkpoint, `lifter.mode='occ'`, `loss.stage='occ'`); δ-distance training schedule (`delta ~ U(0, 3m)` per iter).
- **Exit test**: full 24-epoch run; val mIoU climbs through ≥18 by epoch 12.

### Phase J `[deferred — after I]` — Final eval Small (Table 1 row "S2GO-Small")
- **R** SurroundOcc [`evaluation_metrics.py`](/media/skr/storage/self_driving/S2GO/reference_code/SurroundOcc/projects/mmdet3d_plugin/datasets/evaluation_metrics.py) `evaluation_semantic()`; GF eval loop pattern.
- **N** `tools/eval.py` wrapper that aggregates per-class TP/P/G across val set + measures FPS (warm 200-sample median).
- **Exit test**: Small **22.1 mIoU / 34.3 IoU / ≥24 FPS** (±0.5 mIoU acceptance window). Once this lands, the implementation is validated end-to-end.

### Phase J.5 `[deferred — after J]` — S2GO-Base (config diff only)
- **R** all training + eval code from Phases A–J unchanged.
- **N** `configs/s2go_base_pretrain.py` and `s2go_base_occ.py` — three-line diff from Small configs: `num_query=1800`, `J=20`, backbone init swapped to nuImages-pretrained R50 (download from https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/...). Same 12 + 24 epoch schedule.
- **Train**: Stage 1 pretrain (12 ep) → Stage 2 occ (24 ep). Larger memory + slower per-step due to 4× Gaussian count; expect ~36–48 hr per stage on 4×4090 vs Small's ~24–28 hr.
- **Exit test**: Base **22.7 mIoU / 35.5 IoU / ≥18 FPS** (±0.5 mIoU acceptance).

### Phase K `[deferred — after J]` — Ablations (Tables 3–7)
- **R** everything from Phases A–J — kernel toggles via constructor flags, propagation strategies via `propagator.py` mode flag, etc.
- **N** config variants on **Small** (cheaper; matches paper's ablation choice — all ablations Tables 3–7 are on S2GO-Small): `s2go_small_no_pretrain.py`, `s2go_small_no_denoise.py`, `s2go_small_topk_only.py`, `s2go_small_no_velo.py`, `s2go_small_dense_g2v.py`.
- **Exit test**: reproduce Table 6 (16.97 → 20.13 → 20.55 mIoU, ±0.3) and Table 4 propagation ladder.

### Phase summary

| Phase | Mostly | New LOC est. | Sequencing |
|---|---|---|---|
| A | R | ~150 | parallel with B |
| B | R | ~50 | parallel with A |
| C | R + N | ~250 | after A, B |
| D | N | ~200 | after C |
| E | N (CUDA) | ~250 (or ~50 w/o tiling) | after D, gates F |
| F | R + N | ~150 | after E |
| G | N (wrapper) | ~250 | parallel with F |
| H | N (config, Small) | ~80 | after F + G |
| I | N (config, Small) | ~50 | after H |
| J | N (script) | ~100 | after I — **Small target hit here** |
| J.5 | N (config, Base) | ~30 | after J — config diff only, retrain |
| K | N (configs) | ~300 (config only) | after J — uses Small for cost |
| **Total** | | **~1830 LOC net-new** | |

## 4. Per-Component Design

### 4.1 `S2GOLifter` ([s2go/models/lifter/s2go_lifter.py](/media/skr/storage/self_driving/S2GO/s2go/s2go/models/lifter/s2go_lifter.py))
Two modes via constructor flag. Output: `{p:(B,K,3), embed:(B,K,D), a:(B,K,1), v:(B,K,3)}`.
- `mode='occ'`: learnable `nn.Parameter` for p (init U(pc_range)), embed (xavier), a (sigmoid⁻¹0.5).
  Forks [GaussianFormer/model/lifter/gaussian_lifter.py:30-52](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/model/lifter/gaussian_lifter.py).
- `mode='pretrain'`: per-batch `p = torch_cluster.fps(metas['lidar_pts'], ratio=K/N) + ε`,
  ε ~ U(-1,1)^3 (Eq. 7, e=1m).

### 4.2 `S2GOTemporalEncoder` ([encoder/temporal_decoder.py](/media/skr/storage/self_driving/S2GO/s2go/s2go/models/encoder/temporal_decoder.py))
Stack of 6 `S2GODecoderLayer` blocks, op order `('self_attn','norm','cross_attn','norm','ffn','norm','refine')`.
Memory queue length 4 (= 2s @ 2Hz keyframes). Embed dim 768, 8 heads, FFN 2048. Self-attn over
{current ∪ memory queries}; cross-attn to multi-scale image features via deformable.

### 4.3 `S2GOQueryRefiner` ([encoder/refiner.py](/media/skr/storage/self_driving/S2GO/s2go/s2go/models/encoder/refiner.py))
Fork [refine_module_v2.py](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/model/encoder/gaussian_encoder/refine_module_v2.py) (`SparseGaussian3DRefinementModuleV2`). Replace its single per-Gaussian output head with **two parallel heads** off each layer's `tgt`:
- **Query head** (parent attributes): `Linear(D, 8) → (Δp:3, Δa:1, v:3, gate:1)`. Δp clipped via `tanh·δ_p_max`.
- **Children head** (J Gaussians): `Linear(D, J·11) → (B,K,J,{o:3, r:4, s:3, a:1})`.
- Eq. 6 assembly in `splatting/gaussian_assembly.py`: flatten K·J Gaussians,
  `mean = p^i + o^i + o_j^i`, `opacity = a^i · a_j^i`, semantics shared per parent query.
- The v2 module already exposes `cartesian` / `reverse_cartesian` and the `pc_range` / `scale_range` activation logic — reuse verbatim.

### 4.4 `S2GOQueryPropagator` ([encoder/propagator.py](/media/skr/storage/self_driving/S2GO/s2go/s2go/models/encoder/propagator.py))
δ-distance top-k by opacity, greedy O(K·k):
```
sort queries by opacity desc; iterate, keep if dist to all kept ≥ δ; stop at k
```
Training δ ~ U(0, 3m); inference δ = 1.6m. Ego-motion compensation: `p ← egopose_inv · (p_world + v·dt)`.

### 4.5 `S2GOSplatHead` ([head/splat_head.py](/media/skr/storage/self_driving/S2GO/s2go/s2go/models/head/splat_head.py))
Wraps `s2go_localagg.LocalAggregator`. Computes `Cov = R·S·Sᵀ·Rᵀ` then `CovInv` on GPU
(replace GF's `.cpu().inverse().cuda()` with `torch.linalg.inv` for AMP).
Returns per-voxel logits `(B, V, C)`.

### 4.6 `S2GORenderHead` ([head/render_head.py](/media/skr/storage/self_driving/S2GO/s2go/s2go/models/head/render_head.py))
Wraps gsplat. Inputs: `(means, scales, rotations, opacities, colors)` + per-cam `(K, c2w, H, W)`.
RGB from `Linear(D, 3)` on query embed; broadcast to all J children + sigmoid. Returns
`depth_per_cam (B, T, 6, H, W)` and `rgb_per_cam`.

### 4.7 `S2GOLoss` ([losses/{occ_loss,pretrain_loss}.py](/media/skr/storage/self_driving/S2GO/s2go/s2go/losses/))
- **Stage 1** (`PretrainLoss`): `L = λ1·L1(FPS_K(pts)−(p+o)) + λ2·L_depth + λ3·L_rgb`.
  Initial λ1=10, λ2=1, λ3=1.
- **Stage 2** (`OccLoss`): voxel CE (class-weighted, ignore=255) + 0.25·Lovász + 0.1·(L_depth+L_rgb)
  on neighbor frame at ±0.5s (paper keeps render aux during Stage 2).

## 5. CUDA Kernel Modifications

Fork [GaussianFormer/model/head/localagg_prob_fast/](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/model/head/localagg_prob_fast/) (this is the GaussianFormer-2 fast prob splatter) into
`s2go/cuda/s2go_localagg/`, rename module to avoid clobbering GF. The GF-2 kernel already implements the probabilistic mixture (S2GO Eq. 4) and computes `bin_logit = 1 − ∏(1 − power)` (Eq. 1). The remaining gap S2GO closes is opacity-in-occupancy (Eq. 9).

### 5.1 Eq. 9 — opacity-weighted α (essential for Table 6 gain)
[forward.cu:78](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/model/head/localagg_prob_fast/src/forward.cu#L78), inside per-voxel Gaussian loop. In GF-2's current form, `opa` enters only via `prob` (the semantic-mixture weight), but `bin_logit` is computed from `power` alone — exactly the issue S2GO §3.4.2 calls out.
- **Replace**: `prob = (2π)^-1.5 · |Σ|^0.5 · power · opa`
- **With**: `alpha = opa · power; prob = alpha; bin_logit *= (1 − alpha); density += alpha`
- Drop the `(2π)^-1.5 · |Σ|^0.5` density prefactor entirely.

[backward.cu:79](/media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer/model/head/localagg_prob_fast/src/backward.cu#L79):
- Drop the same prefactor; chain rule:
  `power_grad += prob_grad·opa`,
  `opa_grad += prob_grad·power`,
  `power_grad += (1−bin_logits)/(1−alpha+1e-9) · bin_logits_grad · opa`.
- Drop all `deter_grad·(...)` clauses (no determinant in Eq. 9). Net: ~40% fewer FLOPs in
  backward, partially explaining 129h → 28h training drop in Table 6.

### 5.2 4×4×4 voxel-block tiling (forward)
Currently 1 thread = 1 voxel; range lookup is per-voxel. New scheme:
- `config.h`: `#define TILE 4`. Re-bin into 4×4×4=64-voxel tiles (50×50×4 = 10 000 tiles for 200×200×16).
- `aggregator_impl.cu`: preprocess against tile-space rectangles instead of voxels; sort
  produces `(tile_id, gaussian_id)` pairs.
- `forward.cu` new kernel: launch `<<<num_tiles, 64>>>`. Threads cooperatively load batches of
  N=256 Gaussians into shared mem (`__shared__ float s_means[256][3], s_cov[256][6], s_opa[256], s_sem[256][C]`),
  `__syncthreads()`, each thread iterates its own voxel against the shared batch, accumulates in
  registers, writes once at end. Use `__launch_bounds__(64, 8)`.
- Fall back to per-voxel kernel for `range.y - range.x > 4096` (rare).

### 5.3 Backward stays Gaussian-thread-tied
GF's backward already has 1 thread = 1 Gaussian (no atomics) — only addition is a parallel
`gaussian → list of (tile_id, voxel2pts)` index built in preprocess.

### 5.4 Build hygiene
`setup.py` registers extension as `s2go_localagg`. Compile `-O3`, arch list `8.0;8.6;8.9`.
Document forward-vs-backward duality in `cuda/s2go_localagg/README.md`.

## 6. Dataset Preparation

SurroundOcc GT (`train.zip` 2.99 GB / `val.zip` 627 MB / `nuscenes_infos_{train,val}.pkl`) and nuScenes part-1 keyframes (4.3 GB, includes LiDAR_TOP) are **already on disk** under [data/](data/) — see [dataset.md](dataset.md). Parts 2–10 of nuScenes keyframes (~38 GB) deferred until Phase H. Tasks:
1. **Extract**: unzip SurroundOcc `{train,val}.zip` to `data/nuscenes_occ/`; untar nuScenes archives to `data/nuscenes/{samples,v1.0-trainval}/`.
2. **Build pkl**: `tools/create_pkl.py` adapts [SurroundOcc/tools/create_data.py](/media/skr/storage/self_driving/S2GO/reference_code/SurroundOcc/tools/) to write `s2go_nusc_infos_{train,val}.pkl` with `occ_path` field. (SurroundOcc's released `nuscenes_infos_{train,val}.pkl` may already work; verify the `occ_path` key first before regenerating.)
3. **LiDAR depth**: pipeline `ProjectLiDARToCamDepth` (in dataloader workers) projects via
   `lidar2img`, rasterizes 256×704 sparse depth. Optional offline cache via `tools/prep_lidar_depth.py`.
4. **Streaming sampler**: `seq_group_sampler.py` ported from StreamPETR with `seq_split_num=2,
   queue_length=4, num_frame_losses=1`. First-frame-of-chunk gets `prev_exists=0`.

> **Smoke-test path note**: with only part-1 keyframes available (3 377 of ~28 130 trainval keyframes), Phases A–G can run end-to-end on the part-1 subset. Phase H pretraining converges meaningfully only with the full trainval set; queue parts 2–10 download once Phases A–G pass.

## 7. Training Configs

Key entries from paper Appendix B for `configs/s2go_small_occ.py`:

```python
model = dict(
    type='S2GO',
    backbone=dict(type='ResNet', depth=50,
                  init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(type='CPFPN', in_channels=[256,512,1024,2048], out_channels=256, num_outs=4),
    lifter=dict(type='S2GOLifter', num_query=900, embed_dims=768, mode='occ',
                pc_range=[-50,-50,-5,50,50,3]),
    encoder=dict(type='S2GOTemporalEncoder', num_layers=6, embed_dims=768, num_heads=8,
                 ffn_dims=2048, J=10, memory_len=4, num_propagated=256,
                 propagator=dict(delta_train=(0.,3.), delta_test=1.6)),
    splat_head=dict(type='S2GOSplatHead', grid=(200,200,16),
                    pc_range=[-50,-50,-5,50,50,3], num_classes=17),
    render_head=dict(type='S2GORenderHead', img_hw=(256,704), aux_weight=0.1),
    loss=dict(type='S2GOLoss', stage='occ',
              weights=dict(ce=1.0, lovasz=0.25, depth=0.1, rgb=0.1)),
)
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=4e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.25)}),
    clip_grad=dict(max_norm=35., norm_type=2))
param_scheduler = dict(type='CosineAnnealingLR', T_max=24)
train_cfg = dict(by_epoch=True, max_epochs=24, val_interval=4)
load_from = './work_dirs/s2go_small_pretrain/epoch_12.pth'
train_dataloader = dict(batch_size=4, num_workers=4,
    sampler=dict(type='SeqGroupSampler', seq_split_num=2, queue_length=4),
    dataset=dict(type='S2GONuScenesOccDataset', ann_file='data/s2go_nusc_infos_train.pkl',
                 occ_size=[200,200,16], pc_range=[-50,-50,-5,50,50,3],
                 use_semantic=True, num_frame_losses=1))
```

`s2go_small_pretrain.py`: `lifter.mode='pretrain'`, `loss.stage='pretrain'`,
weights `{denoise:10, depth:1, rgb:1}`, `max_epochs=12`, no `load_from`.

`s2go_base_*.py`: `num_query=1800, J=20`, backbone init from `nuimages_pretrained_r50.pth`,
**same** `max_epochs=24` for Stage 2 (paper Appendix B: "trained for 24 epochs for 3D semantic
occupancy estimation" applies to both Small and Base on nuScenes). KITTI configs use 12 epochs
Stage 2 instead.

## 8. Verification

| Test | Command | Expected |
|---|---|---|
| Smoke forward | `pytest tests/test_e2e_smoke.py` | shapes ok, no NaN, peak mem < 18 GB |
| Kernel gradcheck | `pytest tests/test_kernel_grad.py` | passes 1e-3 atol vs Python ref |
| 4-sample overfit (Stage 1) | `tools/train.py configs/s2go_small_pretrain.py --indices 0-3 --epochs 200` | denoise loss < 0.05m by epoch 100 |
| 4-sample overfit (Stage 2) | analogous | mIoU on those 4 > 80% by epoch 200 |
| Table 6 ablation | run 3 configs `dense_g2v_no_opa`, `dense_g2v_opa`, `efficient_g2v_opa` | 16.97 → 20.13 → 20.55 mIoU (±0.3) |
| Final Table 1 Small (Phase J) | `tools/eval.py work_dirs/s2go_small_occ/last.pth` | 22.1 mIoU / 34.3 IoU / ≥24 FPS |
| Final Table 1 Base (Phase J.5, after Small lands) | `tools/eval.py work_dirs/s2go_base_occ/last.pth` | 22.7 mIoU / 35.5 IoU / ≥18 FPS |

## 9. Risk Register

1. **mmcv 1.4 build on RTX-4090 + CUDA 12 driver fails**. Mitigation: bumped pin (torch 1.13.1 +
   mmcv 1.6.2 + mmdet3d 1.0.0rc6). 3-5 days first time.
2. **Custom CUDA kernel numerical drift vs paper FPS**. Mitigation: Phase-D Python reference
   gates Phase E; profile with `nsys`. If shared-mem hurts occupancy, fall back to global-mem
   variant (still gets the opacity-in-α gain alone, ~20.13 mIoU).
3. **gsplat ↔ mmengine version conflict**. Mitigation: thin render-head interface; fallback
   `diff-gaussian-rasterization`.
4. **FPS+ε pretrain instability**. Symptoms: denoise loss explodes early. Mitigation: 1k-iter LR
   warmup 1e-6 → 4e-4; clip query-position update to ±1m per layer via `tanh·1.0`; start λ1=20
   anneal to 10.
5. **24GB VRAM overflow with 4-frame BPTT + flash-attn**. Mitigation: detach `temp_memory`
   (StreamPETR already does, lines 363-365); AMP fp16 in encoder; gradient checkpointing
   `with_cp=True`. Worst case: drop queue 4→2.
6. **δ-dist top-k is O(K²)**. K=900 fine (<0.5ms). At K=1800 add voxel-hash bucketing.
7. **Stage 1 → Stage 2 distribution shift**. Risk that Stage-1 query positions drift away from
   learnable prior. Mitigation: warm-start with `load_from`, freeze backbone first 2 epochs.

## 10. Out of Scope (First Pass)

- Occ3D-nuScenes (Table 10) and KITTI-360 (Table 2) — placeholders only; add after Table 1 lands.
- Multi-task heads, world-model finetuning, large-scale pretraining mentioned in Conclusion.
- Mixed Stage-2 with depth pretraining Table 8 ablations beyond LiDAR-32-line default.
- Visualization tooling beyond a basic `tools/vis.py` mesh dump.

## Critical Files to Modify / Create

- [s2go/cuda/s2go_localagg/src/forward.cu](/media/skr/storage/self_driving/S2GO/s2go/s2go/cuda/s2go_localagg/src/forward.cu) — Eq. 9 + 4×4×4 tiling
- [s2go/cuda/s2go_localagg/src/backward.cu](/media/skr/storage/self_driving/S2GO/s2go/s2go/cuda/s2go_localagg/src/backward.cu) — drop `deter_grad`, opacity chain
- [s2go/models/encoder/temporal_decoder.py](/media/skr/storage/self_driving/S2GO/s2go/s2go/models/encoder/temporal_decoder.py)
- [s2go/models/encoder/refiner.py](/media/skr/storage/self_driving/S2GO/s2go/s2go/models/encoder/refiner.py)
- [s2go/models/encoder/propagator.py](/media/skr/storage/self_driving/S2GO/s2go/s2go/models/encoder/propagator.py)
- [s2go/models/lifter/s2go_lifter.py](/media/skr/storage/self_driving/S2GO/s2go/s2go/models/lifter/s2go_lifter.py)
- [s2go/models/head/render_head.py](/media/skr/storage/self_driving/S2GO/s2go/s2go/models/head/render_head.py)
- [s2go/datasets/nusc_surroundocc.py](/media/skr/storage/self_driving/S2GO/s2go/s2go/datasets/nusc_surroundocc.py)
- [s2go/configs/s2go_small_pretrain.py](/media/skr/storage/self_driving/S2GO/s2go/s2go/configs/s2go_small_pretrain.py)
- [s2go/configs/s2go_small_occ.py](/media/skr/storage/self_driving/S2GO/s2go/s2go/configs/s2go_small_occ.py)

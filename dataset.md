# S2GO — Dataset Download Checklist

> **Legend:** ✅ done · 🔄 in progress · ⏳ queued · ❌ not started · ➖ not needed for current goal

---

## 📊 Current status (snapshot)

| Item | Status | Local path / location |
|---|---|---|
| GF-2 model weights (Prob-128, 12 800 Gaussians) | ✅ | [reference_code/GaussianFormer/out/prob/nuscenes_gs12800/state_dict.pth](reference_code/GaussianFormer/out/prob/nuscenes_gs12800/state_dict.pth) — 470 MB |
| R101-DCN-FCOS3D image backbone | ✅ | [reference_code/GaussianFormer/ckpts/r101_dcn_fcos3d_pretrain.pth](reference_code/GaussianFormer/ckpts/r101_dcn_fcos3d_pretrain.pth) — 215 MB |
| R50 nuImages-pretrained (S2GO-Base) | ✅ | [data/ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth](data/ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth) — 294 MB |
| R50 ImageNet1k (S2GO-Small) | ✅ | torchvision cache `~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth` — 102 MB |
| `nuscenes_infos_val.pkl` | ✅ | [data/surroundocc_dl/7059038762_-/](data/surroundocc_dl/7059038762_-/) — 85 MB |
| `nuscenes_infos_train.pkl` | ✅ | [data/surroundocc_dl/7059038762_-/](data/surroundocc_dl/7059038762_-/) — 411 MB |
| SurroundOcc `val.zip` (200×200×16 GT, val) | ✅ | [data/surroundocc_dl/7059038762_-/val.zip](data/surroundocc_dl/7059038762_-/val.zip) — 627 MB (finished 2026-05-08 00:57) |
| SurroundOcc `train.zip` (200×200×16 GT, train) | ✅ | [data/surroundocc_dl/7059038762_-/train.zip](data/surroundocc_dl/7059038762_-/train.zip) — 3.0 GB (finished 2026-05-08 12:47, 28 130 .npy files) |
| nuScenes `v1.0-trainval_meta.tgz` | ✅ | [data/nuscenes_dl/v1.0-trainval_meta.tgz](data/nuscenes_dl/v1.0-trainval_meta.tgz) — 441 MB |
| nuScenes `v1.0-trainval01_keyframes.tgz` (part 1) | ✅ | [data/nuscenes_dl/v1.0-trainval01_keyframes.tgz](data/nuscenes_dl/v1.0-trainval01_keyframes.tgz) — 4.3 GB (finished 2026-05-08 01:12) |
| nuScenes `v1.0-mini` (10 scenes) | ✅ | `/media/skr/storage/self_driving/CoPilot4D/data/nuScenes/` (pre-existing) |
| nuScenes-lidarseg-mini | ✅ | same path, `lidarseg/v1.0-mini/` (404 .bin files) |
| nuScenes parts 2–10 keyframes | ❌ | needed only for **full** Table 1 reproduction (after smoke test) |
| Occ3D-nuScenes labels | ❌ | optional (Table 10) |
| KITTI-360 + SSCBench labels | ❌ | optional (Table 2) |

---

S2GO is evaluated on **three benchmarks**. Each is independent — a model trained on one is **not** used for another.

| Benchmark | Reported in | Required for |
|---|---|---|
| **SurroundOcc-nuScenes** | Table 1, Tables 3–9 (main + all ablations) | Reproducing the paper's headline numbers |
| **Occ3D-nuScenes** | Table 10 (Appendix J) | Optional secondary nuScenes benchmark |
| **SSCBench-KITTI-360** | Table 2 (monocular setting) | KITTI-360 experiment |

The minimum to reproduce the **main paper results** (Table 1 + ablations) is the SurroundOcc-nuScenes set. Occ3D and KITTI are additive.

---

## 1. nuScenes-SurroundOcc (REQUIRED for main results)

### 1a. Raw nuScenes v1.0-trainval

The pkl references `samples/CAM_*/...` and `samples/LIDAR_TOP/...` only — `sweeps: []` for every val entry. Use the **keyframes-only** sub-archives, not the full blobs.

| Item | Size | Status | Source |
|---|---|---|---|
| `v1.0-trainval_meta.tgz` (metadata json files) | 0.43 GB | ✅ | https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz |
| `v1.0-trainval01_keyframes.tgz` (keyframes part 1) | 4.22 GB | ✅ | https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval01_keyframes.tgz |
| `v1.0-trainval{02..10}_keyframes.tgz` (parts 2–10) | ~38 GB total | ❌ | same CDN, increment the part number |
| `v1.0-test_*.tgz` (test set) | ~50 GB | ➖ | not needed (eval is on val) |
| Sweeps (full blobs minus keyframes) | ~250 GB | ➖ | not referenced by val.pkl |
| Lidarseg / maps / panoptic / CAN bus | ~1 GB | ➖ | not needed for inference |

> ⚠️ **About the "Lidar / Camera / Radar blobs only" download options on the nuScenes site (and why we did not download them).** The download page lists, per part, both a **keyframes** archive and separate **Lidar / Camera / Radar blobs only** archives. These are *orthogonal axes*, not mutually exclusive sets:
>
> | Axis | Cuts on… | Example sizes (part 1) |
> |---|---|---|
> | Keyframes vs sweeps | temporal sampling rate (2 Hz keyframes vs 20 Hz/12 Hz sweeps) | 4.22 GB (keyframes-only, all 12 sensors) |
> | By-sensor | which sensor type | 12.80 GB Lidar + 16.39 GB Camera + 0.22 GB Radar = 29.41 GB ("File blobs of 85 scenes, part 1") |
>
> **Our `v1.0-trainval01_keyframes.tgz` already contains the keyframe LiDAR.** Verified by listing the archive: 3 377 keyframes × 12 sensors = `samples/LIDAR_TOP/*.pcd.bin` (3 377 files) + 6×3 377 camera JPGs + 5×3 377 radar PCDs. The 12.80 GB "Lidar blobs only" download would have given us the same keyframes plus the 20 Hz inter-keyframe sweeps — which **S2GO does not use** (paper Appendix B operates on 2 Hz keyframes only; Eq. 8's "+/- 0.5s neighbor frame" supervision is exactly the adjacent keyframe). Do not download "Lidar blobs only" unless we later add a workflow that needs sweep-rate LiDAR (e.g., regenerating SurroundOcc GT from scratch — but we already have the precomputed `train.zip`/`val.zip`).

**Already on disk (legacy):** `v1.0-mini` (10 scenes, 9.8 GB) at `/media/skr/storage/self_driving/CoPilot4D/data/nuScenes/`. Useful for sanity-checking but not aligned with the SurroundOcc val pkl.

> **Part-1 coverage measurement (2026-05-08, after extraction):** the part-1 keyframe archive contains 3 376 keyframes covering **914 of 6 019 SurroundOcc val samples (15.2 %)** — verified by checking `os.path.exists(...)` for every `lidar_path` in `nuscenes_infos_val.pkl`. Sufficient for sub-phase 1 dataloader smoke tests (any 4-frame sequence pulls cleanly) and 4-sample overfit runs. **Not** sufficient for Phase H pretraining or Phase J full-val eval — both require parts 2–10 (~38 GB more) for the full ~28k trainval / 6019 val coverage.

### 1b. SurroundOcc occupancy ground truth + info pickles

Author's Baidu Pan share at `https://pan.baidu.com/s/1swouSVNwYqVRQC2x6jJmeA?pwd=afs2`. Already transferred into your Baidu drive at `/data/`.

| File | Size | Status | Purpose |
|---|---|---|---|
| `nuscenes_infos_val.pkl` | 85 MB | ✅ | Val split metadata (6019 entries, 150 scenes) |
| `nuscenes_infos_train.pkl` | 411 MB | ✅ | Train split metadata (~28k entries, 700 scenes) |
| `val.zip` | 598 MB (got 627) | ✅ | 200×200×16 dense voxel GT for val (6019 .npy files) — finished 2026-05-08 00:57 |
| `train.zip` | 2.99 GB | ✅ | 200×200×16 dense voxel GT for train (28 130 .npy files) — finished 2026-05-08 12:47 |

**Eval-only minimum: `nuscenes_infos_val.pkl` + `val.zip`** (~683 MB). Each `.npy` is `(n, 4)` — xyz voxel index + semantic label, in LiDAR coord frame. Empty = label 0, ignore = label 255.

### 1c. Baidu Pan setup (already done)

✅ Logged into `pan.baidu.com` via Chrome (cookies extracted), share transferred to drive root, BaiduPCS-Go authenticated, detached download running with logs in [data/surroundocc_dl/](data/surroundocc_dl/).

| Option | Cost / Effort | Notes |
|---|---|---|
| ✅ **BaiduPCS-Go** (current path) | Free + Baidu account | Throttled to ~75 KB/s without VIP — full 4 GB ETA ~12 hours |
| **Erranium proxy** (paid) | ~$4.21 | Skips throttle entirely (https://baidu.erranium.com/downloader) |
| **¥6 Super VIP day-pass** | ~$0.85 | Lifts throttle to ~10 MB/s — finishes in ~10 min |

---

## 2. nuScenes-Occ3D (optional, for Table 10)

| Item | Size | Status | Source |
|---|---|---|---|
| Raw nuScenes v1.0-trainval | covered by 1a | partial | shared with SurroundOcc |
| Occ3D-nuScenes occupancy labels (200×200×16, ego frame) | ~50 GB | ❌ | https://github.com/Tsinghua-MARS-Lab/Occ3D (gdrive/baidu) |
| Occ3D info pickles | small | ❌ | same |

Different label format and coord frame than SurroundOcc — not interchangeable.

---

## 3. SSCBench-KITTI-360 (REQUIRED for Table 2 only)

S2GO uses the **monocular** camera setting on KITTI-360, with 7/1/1 sequence train/val/test split.

| Item | Size | Status | Source |
|---|---|---|---|
| KITTI-360 raw perspective images (left camera) | ~80 GB | ❌ | https://www.cvlibs.net/datasets/kitti-360/ |
| KITTI-360 calibrations + poses | ~1 GB | ❌ | same |
| SSCBench-KITTI-360 dense semantic labels | ~30 GB | ❌ | https://github.com/ai4ce/SSCBench (Google Drive — no Baidu wall) |
| SSCBench split files (7/1/1) | small | ❌ | same SSCBench repo |

⚠️ **Code-availability blocker:** the `huang-yh/GaussianFormer` repo (latest commit March 2025) does **not** include KITTI-360 dataloader / config / training code. Even with the data, no released model can be run on Table 2 today — would require implementing the dataloader from scratch or waiting for S2GO code release.

**Already on disk (not reusable):** `/media/skr/storage/self_driving/CoPilot4D/data/kitti/` is **KITTI odometry**, not KITTI-360 — different sequences, different format.

---

## 4. Pretrained model checkpoints + backbones

| Item | Size | Status | Source |
|---|---|---|---|
| GF-2 **Prob-128** state_dict (12 800 Gaussians, mIoU 20.08) | 470 MB | ✅ | https://cloud.tsinghua.edu.cn/f/b6038dca93574244ad57/?dl=1 |
| GF-2 Prob-64 state_dict | ~200 MB | ❌ | https://cloud.tsinghua.edu.cn/f/d041974bd900419fb141/?dl=1 (skip — Prob-128 is the sweet spot) |
| GF-2 Prob-256 state_dict | ~1 GB | ❌ | https://cloud.tsinghua.edu.cn/f/e30c9c92e4344783a7de/?dl=1 (skip — diminishing returns) |
| R101-DCN-FCOS3D image backbone (used by GF / GF-2 nuScenes configs) | 215 MB | ✅ | https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth |
| ResNet-50 ImageNet1k (S2GO-Small + KITTI variant) | ~100 MB | ❌ | torchvision auto-download on first use, or https://download.pytorch.org/models/resnet50-0676ba61.pth |
| ResNet-50 nuImages-pretrained (S2GO-Base) | 294 MB | ✅ | [data/ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth](data/ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth) (mmdet3d v0.1.0 nuImages model zoo; downloaded 2026-05-08) |

S2GO does **not** require LiDAR semantic labels (lidarseg) for training itself — raw LiDAR points are used for query initialization (Eq. 7) and depth supervision. Lidarseg is only needed if regenerating SurroundOcc GT yourself.

---

## Summary: Minimum Disk Required

| Goal | Datasets needed | Total size | Status |
|---|---|---|---|
| **Smoke test on mini split** (no Baidu) | regenerate GT from existing `nuScenes-mini` + lidarseg-mini | <500 MB extra | ➖ alternate path |
| **Smoke test on val part 1** ← *current plan* | nuScenes meta + part 1 keyframes + val.zip + val.pkl + GF-2 weights + R101 backbone | ~5.6 GB | 🔄 ~70% downloaded |
| **Eval-only on full SurroundOcc val** | + nuScenes parts 2–10 keyframes | +38 GB | ❌ |
| **Reproduce Table 1 + ablations** | + train.zip + train.pkl + nuImages backbone | +3.5 GB | partly ⏳ |
| **+ Reproduce Table 10 (Occ3D)** | + Occ3D-nuScenes labels | +50 GB | ❌ |
| **+ Reproduce Table 2 (KITTI)** | + KITTI-360 raw + SSCBench labels | +110 GB | ❌ (also code-blocked) |
| **All three benchmarks** | everything above | **~225 GB** (keyframes-only path) | ❌ |

---

## Folder Structure (target, per SurroundOcc convention)

```
S2GO/
└── data/
    ├── nuscenes_dl/                          # ✅ raw nuScenes downloads (current)
    │   ├── v1.0-trainval_meta.tgz            # ✅ 441 MB
    │   └── v1.0-trainval01_keyframes.tgz     # 🔄 4.22 GB
    ├── surroundocc_dl/                       # SurroundOcc Baidu downloads
    │   └── 7059038762_-/
    │       ├── nuscenes_infos_val.pkl        # ✅ 85 MB
    │       ├── nuscenes_infos_train.pkl      # ✅ 411 MB
    │       ├── val.zip                       # 🔄 ~76%
    │       └── train.zip                     # ⏳ queued
    │
    └── (after extraction — target layout for the loader:)
        nuscenes/                             # extract trainval_meta + part 1 here
        ├── samples/
        │   ├── CAM_FRONT/
        │   ├── CAM_FRONT_LEFT/
        │   ├── CAM_FRONT_RIGHT/
        │   ├── CAM_BACK/
        │   ├── CAM_BACK_LEFT/
        │   ├── CAM_BACK_RIGHT/
        │   └── LIDAR_TOP/
        └── v1.0-trainval/                    # *.json metadata
        nuscenes_occ/                         # extract val.zip here
        └── samples/
            └── <token>.pcd.bin.npy
        nuscenes_infos_val.pkl                # move from surroundocc_dl/
        nuscenes_infos_train.pkl              # move from surroundocc_dl/
```

---

## 📦 By-stage usage of on-disk data (2026-05-11 snapshot)

Distinguishes (a) what's actually being read by Stage 1 today, (b) what's downloaded but idle, and (c) what's missing for a full paper-recipe scale-up. Sources: `nuscenes.org` = official mirror (free account); **Baidu** = SurroundOcc team's Baidu Netdisk distribution.

### Currently in use (Stage 1 pretraining)

| Folder | Source | Size | Stage 1 | Stage 2 | Notes |
|---|---|---|---|---|---|
| `data/nuscenes/samples/` (cameras + LiDAR, **Part 1 only**) | nuscenes.org | 7.8 GB | ✅ in use | ✅ in use | 3,121 keyframe sequences (~11% of full trainval) |
| `data/nuscenes/v1.0-trainval/` (JSON metadata) | nuscenes.org | (in above) | ✅ in use | ✅ in use | sample.json, ego_pose.json, calibrated_sensor.json, etc. — read via nuscenes-devkit |
| `data/nuscenes_dl/` (raw .tgz archives, Part 1 + meta) | nuscenes.org | 4.7 GB | source only | source only | Already extracted into `data/nuscenes/`; safe to delete if disk is tight |
| `nuscenes/nuscenes-devkit/` (library) | github.com/nutonomy | — | ✅ in use | ✅ in use | Driver for all nuScenes I/O |

### Downloaded but sitting idle on disk

| Folder | Source | Size | Stage 1 | Stage 2 | Notes |
|---|---|---|---|---|---|
| `data/surroundocc_dl/train.zip` (occupancy GT) | **Baidu** | ~3 GB | ➖ unused | ✅ required | Voxel occupancy labels — Stage 2 supervision target |
| `data/surroundocc_dl/val.zip` (occupancy GT) | **Baidu** | ~627 MB | ➖ unused | ✅ for eval | Stage 2 validation mIoU |
| `data/surroundocc_dl/nuscenes_infos_train.pkl` | **Baidu** | 411 MB | ➖ unused | ✅ required | Sample-ordering index used by the SurroundOcc loader |
| `data/surroundocc_dl/nuscenes_infos_val.pkl` | **Baidu** | 85 MB | ➖ unused | ✅ for eval | Same, val split |
| `data/nuscenes_occ/` (alt occupancy GT, likely Occ3D format) | not yet identified | 7.2 GB | ➖ unused | optional alt-GT | Probably for Occ3D benchmark (Table 10); only relevant if we benchmark on Occ3D instead of/alongside SurroundOcc |
| `data/gf2_pkls/` (GaussianFormer-2 cam pickles) | upstream GF-2 | 658 MB | ➖ unused | reference only | Likely only relevant if running the GF-2 baseline for comparison |
| `data/ckpts/cascade_mask_rcnn_…pth` | mmdet pretrained zoo | 295 MB | ➖ unused | unused for us | nuImages-pretrained backbone — only for paper's **S2GO-Base**; we target S2GO-Small with ImageNet1k via torchvision |

### Missing for full paper-recipe scale-up

| Stage | What's missing | Where to get it | Cost |
|---|---|---|---|
| **Stage 1 full data** | nuScenes-trainval **Parts 2–10** (~25,000 more keyframes) | **nuscenes.org** (registered account) — **NOT on Baidu** | ~50 GB compressed, ~150 GB extracted; rate-limited |
| **Stage 1 (current scope)** | nothing | — | already have Part 1 sufficient for mini-epoch + medium runs |
| **Stage 2** | nothing extra — just extract `surroundocc_dl/train.zip` + `val.zip` | already on disk | ~3 GB extracted |

### One-line per stage

- **Stage 1 (now):** nuScenes Part 1 only (cameras + LiDAR + poses). Used.
- **Stage 1 scale-up:** would need nuScenes Parts 2–10 from **nuscenes.org**. The Baidu downloads do **not** contain these — Baidu only hosted SurroundOcc artefacts.
- **Stage 2 (later):** extract SurroundOcc zips already on disk + reuse same nuScenes images. No new downloads needed unless we also benchmark on Occ3D.

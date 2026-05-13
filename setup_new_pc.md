# Setting up a second machine for Stage-1 training

End-to-end checklist for bringing up this repo on a fresh Linux + NVIDIA GPU machine and running Stage-1 (`L_depth` + `L_rgb` + `L_FPS·K`). Stage 2 is **not** covered here — it needs an additional ~80 GB of SurroundOcc occupancy GT that Stage 1 doesn't touch.

## Canonical environment spec

The same env [CLAUDE.md](CLAUDE.md) pins, recreated bit-for-bit on the second PC:

- **Location (path-based, no name):** `/media/skr/storage/conda_envs/selfocc` — **literal, same on every machine**
- **Version stack:** Python 3.8.16 / PyTorch 2.0.0+cu118 / mmcv 2.0.1 / mmdet 3.0.0 / mmseg 1.0.0 / mmdet3d 1.1.1 / spconv-cu117 + 4 custom CUDA ops (compiled against env-local nvcc 11.8)
- **Activation (target path is canonical; the `source` line points at *your* miniconda install):**
  ```bash
  source "$(conda info --base)/etc/profile.d/conda.sh"   # or: source <YOUR_MINICONDA>/etc/profile.d/conda.sh
  conda activate /media/skr/storage/conda_envs/selfocc
  ```
- **Built by:** [scripts/setup_gaussianformer_env.sh](scripts/setup_gaussianformer_env.sh)

The env *target* (`/media/skr/storage/conda_envs/selfocc`) stays literal on every machine so [CLAUDE.md](CLAUDE.md) and every in-repo activation snippet copy-paste verbatim. The miniconda *install location* is per-PC (it lives wherever the user ran the miniconda installer, e.g. `/home/<your-user>/miniconda3` or `/opt/miniconda3`) — §3 shows how to handle both.

## 0. Host prerequisites

| Requirement | Why |
|---|---|
| Linux x86_64 | The CUDA ops are only built/tested on Linux |
| NVIDIA GPU + working driver | Needs to support **CUDA 11.8 runtime** — driver ≥ 520.61 is enough. The driver does *not* have to be 11.8; the env ships its own toolkit. |
| Miniconda or Anaconda installed | Path-based conda envs are how this project pins CUDA 11.8 without touching `/usr/local` |
| ~50 GB free disk for env + repo | Env is ~6 GB; repo + cache ~5 GB; rest goes to data |
| ~50 GB free for nuScenes keyframes | Or ~5 GB for a part-1-only smoke setup |

`nvidia-smi` should print the GPU and a CUDA version ≥ 11.8. That's the only host-CUDA check needed.

## 1. Clone the repo

```bash
mkdir -p ~/self_driving && cd ~/self_driving
git clone https://github.com/skr3178/S2GO.git
cd S2GO
```

The repo's `.gitignore` already excludes the heavy stuff (`data/`, `out/`, `reference_code/`, `*.pt`). So a clone gets only the source — you'll need to bring in `reference_code/GaussianFormer` separately (next step) and download nuScenes.

## 2. Bring in `reference_code/GaussianFormer`

The setup script compiles four custom CUDA ops from this upstream repo. It's not in this repo (vendored locally, `.gitignore`-d to avoid redistribution). Get it from the original source:

```bash
mkdir -p reference_code && cd reference_code
git clone https://github.com/huang-yh/GaussianFormer.git
cd ..
```

We don't pin a specific upstream commit; any recent main should work — the four `model/{encoder,head}/.../ops/` and `model/head/localagg*/` directories are what the env build script uses ([scripts/setup_gaussianformer_env.sh:84-87](scripts/setup_gaussianformer_env.sh#L84-L87)).

## 3. Recreate the conda env at the canonical path

The env lives at the literal path `/media/skr/storage/conda_envs/selfocc` — same on every machine, so [CLAUDE.md](CLAUDE.md) and every activation snippet stay valid without edits.

### 3a. Make the path resolvable on the new PC

The host doesn't need a drive *named* `storage` mounted under `/media/skr/`. The simplest portable approach is a symlink:

```bash
# As root, one-time per machine:
sudo mkdir -p /media/skr
sudo ln -s /your/actual/data/disk /media/skr/storage
sudo chown -h $USER:$USER /media/skr/storage    # so the path is writable as your user
```

After that, `/media/skr/storage/...` resolves to whatever real disk you have space on. If the second PC's user is not `skr`, the symlink target name doesn't matter — only the path `/media/skr/storage/...` must resolve.

(If you prefer an actual mount, label the disk `storage` and add the appropriate `/etc/fstab` entry under `/media/skr/storage`. Same end result.)

### 3b. Edit only the two non-path-canonical lines of the script

In [scripts/setup_gaussianformer_env.sh](scripts/setup_gaussianformer_env.sh):

```bash
# Line 21: REPO_DIR     → /media/skr/storage/self_driving/S2GO/reference_code/GaussianFormer
#                        (only changes if you cloned the S2GO repo somewhere else;
#                         simplest: clone at /media/skr/storage/self_driving/S2GO/ too)
# Line 22: ENV_PREFIX   → /media/skr/storage/conda_envs/selfocc   (LEAVE AS-IS — this is canonical)
# Line 23: CONDA_BASE   → wherever your miniconda is installed on this PC
#                        (find it with: conda info --base)
```

Easiest reproduction: clone the repo to `/media/skr/storage/self_driving/S2GO` on the new PC too — then *zero* edits to the script are required.

### 3c. Run it

```bash
bash scripts/setup_gaussianformer_env.sh
```

What the script does ([scripts/setup_gaussianformer_env.sh](scripts/setup_gaussianformer_env.sh)):

1. Creates a fresh Python 3.8.16 env at `/media/skr/storage/conda_envs/selfocc`
2. Installs the **CUDA 11.8 toolkit *inside the env*** via `nvidia/label/cuda-11.8.0` — gives a local `nvcc 11.8` regardless of what's in `/usr/local/cuda`
3. Installs PyTorch 2.0.0 + cu118 wheels
4. Installs MMLab stack via `openmim`: mmcv 2.0.1 / mmdet 3.0.0 / mmseg 1.0.0 / mmdet3d 1.1.1 (plus a `scikit-image==0.21.0` pin and a Cython pre-install for an mmdet3d sub-dep)
5. Installs `spconv-cu117` + `timm`
6. Compiles the four CUDA extensions (`model/encoder/gaussian_encoder/ops`, `localagg`, `localagg_prob`, `localagg_prob_fast`) in-place via `pip install -e .`
7. Runs a smoke import: prints torch / mmcv / mmdet / mmdet3d / mmseg / spconv / timm versions and confirms `torch.cuda.is_available()`

Re-runnable: yes — skips env creation if the prefix already exists.

If step 6 fails to compile, the most common cause is `CUDA_HOME` not pointing at the env's nvcc. The script exports it during its own run; if you ever rebuild manually, use:

```bash
export CUDA_HOME=/media/skr/storage/conda_envs/selfocc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

These exports are **only** required at compile time. For runtime use of the pre-compiled extensions, you don't need them.

## 4. Activation

The `conda activate` target is canonical (same on every machine). The `source` line that initializes conda itself is per-PC — it points at *your* miniconda install, which may not live at the same location as on the first PC.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"      # autodetects your miniconda
conda activate /media/skr/storage/conda_envs/selfocc      # literal, identical everywhere
```

The `conda info --base` trick works because `conda` is on PATH after a normal miniconda install (`.bashrc` modification by the installer). If you prefer to hardcode it, use whichever path applies on the new PC — common ones are `/home/<user>/miniconda3`, `/opt/miniconda3`, or `/opt/anaconda3`. The path that's on *this* (first) PC is `/home/skr/miniconda3`, but do not assume that holds on the second PC.

[CLAUDE.md](CLAUDE.md) currently has the first-PC `source` line hardcoded. Either replace it with the `conda info --base` form above on the second PC, or just remember that line is the one host-specific bit and edit as needed.

## 5. nuScenes data

Stage-1 reads **only nuScenes raw data** — no SurroundOcc occupancy, no auxiliary pseudo-labels. Three things to pull:

### 5a. Raw v1.0-trainval keyframes

From the nuScenes website (after signup): download the **v1.0-trainval blob parts**. For a smoke setup you can start with **part 1 only** (~5 GB, covers 15 % of val) — sufficient for 4-frame overfit runs. For real training you need all 10 parts (~38 GB total). Sweeps and maps are **not** needed for Stage 1 — skip them.

Untar into:

```
data/nuscenes/
├── samples/CAM_FRONT/  CAM_BACK/  CAM_FRONT_LEFT/  CAM_FRONT_RIGHT/  CAM_BACK_LEFT/  CAM_BACK_RIGHT/
├── samples/LIDAR_TOP/
└── v1.0-trainval/                  # JSON metadata
```

The path is currently hardcoded at [s2go/datasets/nusc_loader.py:56](s2go/datasets/nusc_loader.py#L56) — keep the same `data/nuscenes` layout under the repo root, or edit that one line.

### 5b. The two info `.pkl` files

The standard mmdet3d-format sample index. Available in the SurroundOcc release (Baidu link per [dataset.md:206-207](dataset.md#L206-L207)):

```
data/nuscenes/
├── nuscenes_infos_val.pkl          # 85 MB  — required
└── nuscenes_infos_train.pkl        # 411 MB — required for full train split
```

Easiest path: just `scp` these two files from the first PC (~500 MB total) instead of re-fetching from Baidu.

### 5c. ImageNet ResNet50 weights

Auto-downloaded by torchvision the first time the backbone is constructed (~100 MB, cached in `~/.cache/torch/hub/checkpoints/`). Requires internet on first run only.

## 6. Per-loss data dependencies — sanity check

Confirm the answer to the prerequisite question:

| Loss | Reads from the batch | Needs anything else? |
|---|---|---|
| **L_depth** | `lidar_depth` (LiDAR projected to each camera inside the loader at [nusc_loader.py:104-212](s2go/datasets/nusc_loader.py#L104-L212)) | No |
| **L_rgb** | `imgs` (the 6 surround cameras) compared against gsplat renders | No |
| **L_FPS·K** | `lidar_pts` (35 k points/frame); FPS anchors computed on the fly | No |

All three derive their targets from raw nuScenes sensor data. No external depth GT, no semantic labels, no auxiliary models — so the data requirement is the same whether you enable one term or all three.

## 7. Smoke test the install

Before training, verify the metric port works and the loaders pull cleanly:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /media/skr/storage/conda_envs/selfocc

# Metric correctness — should print "9/9 cases passed"
python tests/test_miou.py --no-viz
```

If that passes, the env is healthy enough to do real Stage-1 work. The metric tests don't need nuScenes — they catch env-side problems (torch import, CUDA presence) independently of data.

## 8. Run Stage-1

The barebones Stage-1 recipe is in [s2go/tools/overfit.py](s2go/tools/overfit.py). Two reference recipes documented in [architecture.md](architecture.md):

- **5000-iter scale-up:** `python -m s2go.tools.overfit --barebones --iters 5000 …` — produces the v2 results in [out/barebones_full_part1_5000iter_v2](out/barebones_full_part1_5000iter_v2) (~3 h on a 3060)
- **Quick smoke:** `--iters 50` — minutes, validates the loop end-to-end

The exact CLI flags and which loss terms each recipe enables are listed in [out/barebones_500iter/curves.png](out/barebones_500iter/curves.png) and the corresponding `train.log` files on the first PC — copy a working command line over rather than rederiving from CLI help.

## 9. What you do *not* need for Stage 1

Explicitly absent from the requirement list (these are Stage-2-only):

- `data/nuscenes_occ/` — SurroundOcc 200×200×16 occupancy `.npy` files (~80 GB)
- `data/surroundocc_dl/` — anything beyond the two `.pkl` info files
- nuImages, nuScenes maps, nuScenes sweeps
- Any third-party depth predictor (Stage 1 supervises against LiDAR-projected depth, not a monodepth pseudo-label)
- nuScenes can-bus / radar

Skip these on the second machine until Stage 2 work begins.

## 10. Common failure modes

| Symptom | Likely cause |
|---|---|
| `nvcc: command not found` during step 6 of the env script | `CUDA_HOME` not set, or pointing at host `/usr/local/cuda-12.x`. Re-export the three env vars from §3. |
| `RuntimeError: CUDA error: no kernel image is available` at training time | The four CUDA ops were compiled for a different SM than the GPU you're using. Recompile with `TORCH_CUDA_ARCH_LIST` matching your GPU (e.g. `"8.6"` for RTX 3060). |
| `FileNotFoundError: …/samples/CAM_FRONT/...` | nuScenes layout mismatch. Verify the directory tree in §5a; the `samples/` prefix is required. |
| `KeyError: 'lidar_path'` from the loader | Wrong info pkl (e.g. an mmdet3d v0.x format pickle). Use the SurroundOcc-released ones referenced in §5b. |
| Resnet50 weights download fails | First-run torchvision pull needs internet. Either bring up internet for one run, or copy `~/.cache/torch/hub/checkpoints/resnet50-*.pth` from the first PC. |

---

**One-line summary:** symlink `/media/skr/storage` → real disk → clone repo to `/media/skr/storage/self_driving/S2GO` → run the env script (zero or one edit to `CONDA_BASE`) → drop nuScenes keyframes + the two info pkls into `data/nuscenes/` → `python tests/test_miou.py --no-viz` to verify, then training is unblocked. No data beyond raw nuScenes is needed for any Stage-1 loss.

#!/usr/bin/env bash
# ===========================================================================
# Setup script for GaussianFormer / GaussianFormer-2 reference code.
#
# Recipe (per reference_code/GaussianFormer/docs/installation.md):
#   - Python 3.8.16
#   - PyTorch 2.0.0 + CUDA 11.8
#   - mmcv 2.0.1 / mmdet 3.0.0 / mmseg 1.0.0 / mmdet3d 1.1.1
#   - Four custom CUDA op extensions compiled from source
#
# Why conda (and not uv) for THIS project specifically:
#   The pinned recipe needs nvcc 11.8 to compile the CUDA ops, but the host
#   only has nvcc 12.8 in /usr/local/cuda-12.8. Conda installs cuda-toolkit
#   11.8 INSIDE the env (no sudo, no /usr/local pollution) — uv cannot.
#
# Re-runnable: yes. Skips env creation if it already exists.
# ===========================================================================
set -euo pipefail

# ---- paths ----------------------------------------------------------------
REPO_DIR="/home/satya/skr/S2GO/reference_code/GaussianFormer"
ENV_PREFIX="/home/satya/conda_envs/selfocc"          # local path on this PC (no /media symlink)
CONDA_BASE="/home/satya/anaconda3"

# ---- 1. enable `conda` in this shell -------------------------------------
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---- 2. create the env ----------------------------------------------------
if [ ! -d "${ENV_PREFIX}" ]; then
    echo "==> Creating conda env at ${ENV_PREFIX} (python 3.8.16)"
    conda create -y -p "${ENV_PREFIX}" python=3.8.16
else
    echo "==> Env already exists at ${ENV_PREFIX}, skipping creation"
fi
conda activate "${ENV_PREFIX}"

# ---- 3. CUDA 11.8 toolkit inside the env ---------------------------------
# The nvidia/label/cuda-11.8.0 channel pins everything to 11.8.0 release —
# safer than `cuda-toolkit=11.8` which can pull mismatched sub-packages.
if ! command -v nvcc >/dev/null 2>&1 || ! nvcc --version | grep -q "release 11.8"; then
    echo "==> Installing CUDA 11.8 toolkit from nvidia channel"
    conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
else
    echo "==> CUDA 11.8 toolkit already present in env"
fi

# Point all subsequent extension builds at the env's nvcc.
export CUDA_HOME="${ENV_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "==> nvcc in use:   $(which nvcc)"
nvcc --version | tail -2

# ---- 4. PyTorch 2.0.0 + cu118 --------------------------------------------
echo "==> Installing PyTorch 2.0.0+cu118"
pip install --no-cache-dir \
    torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu118

# ---- 5. MMLab stack via openmim ------------------------------------------
# mim fetches prebuilt mmcv wheels from the MMLab CDN matched to torch+cuda.
# Plain `pip install mmcv==2.0.1` would try to compile from source and fail.
echo "==> Installing MMLab packages via openmim"
pip install openmim
mim install mmcv==2.0.1
mim install mmdet==3.0.0
mim install mmsegmentation==1.0.0

# mmdet3d -> lyft-dataset-sdk pins scikit-image to ~0.14.x, which has no py38
# wheel and source-builds via numpy.distutils + Cython. Pre-install Cython so
# that source build can succeed; pin scikit-image to the last py38 wheel
# release (0.21.0) so the resolver ideally doesn't even try to downgrade.
pip install Cython
pip install "scikit-image==0.21.0"
mim install mmdet3d==1.1.1

# ---- 6. Other Python deps ------------------------------------------------
echo "==> Installing spconv-cu117, timm"
pip install spconv-cu117 timm

# ---- 7. Compile the four custom CUDA ops ---------------------------------
echo "==> Building CUDA extensions"
( cd "${REPO_DIR}/model/encoder/gaussian_encoder/ops" && pip install -e . )
( cd "${REPO_DIR}/model/head/localagg"            && pip install -e . )
( cd "${REPO_DIR}/model/head/localagg_prob"       && pip install -e . )
( cd "${REPO_DIR}/model/head/localagg_prob_fast"  && pip install -e . )

# ---- 8. Smoke test --------------------------------------------------------
echo "==> Verifying installation"
python - <<'PY'
import torch
print(f"torch:        {torch.__version__}  (CUDA build: {torch.version.cuda})")
print(f"CUDA avail:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
import mmcv, mmdet, mmdet3d, mmseg
print(f"mmcv:         {mmcv.__version__}")
print(f"mmdet:        {mmdet.__version__}")
print(f"mmseg:        {mmseg.__version__}")
print(f"mmdet3d:      {mmdet3d.__version__}")
import spconv, timm
print(f"spconv:       {spconv.__version__}")
print(f"timm:         {timm.__version__}")
PY

# ---- 9. Done -------------------------------------------------------------
cat <<EOF

================================================================
Setup complete.

To activate this env in a new shell:

    source ${CONDA_BASE}/etc/profile.d/conda.sh
    conda activate ${ENV_PREFIX}
    export CUDA_HOME=${ENV_PREFIX}
    export PATH=\$CUDA_HOME/bin:\$PATH
    export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

(The CUDA_HOME exports are only required if you re-compile CUDA ops;
not needed for runtime use of pre-compiled extensions.)
================================================================
EOF

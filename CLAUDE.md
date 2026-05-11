# S2GO project notes

## Default Python env

All Python work in this repo (training, eval, inference, exploratory scripts) should run inside the conda env at:

```
/media/skr/storage/conda_envs/selfocc
```

This is a **path-based** conda env (no name). Activate it with:

```bash
source /home/skr/miniconda3/etc/profile.d/conda.sh
conda activate /media/skr/storage/conda_envs/selfocc
```

The env was built by [scripts/setup_gaussianformer_env.sh](scripts/setup_gaussianformer_env.sh) following the GaussianFormer / GaussianFormer-2 recipe (Python 3.8.16, PyTorch 2.0.0+cu118, mmcv 2.0.1 / mmdet 3.0.0 / mmseg 1.0.0 / mmdet3d 1.1.1, spconv-cu117, plus four custom CUDA ops compiled against the env-local nvcc 11.8). The `selfocc` path is reused from an existing convention; it is not specific to the SelfOcc reference code.

When recompiling CUDA ops, also export:

```bash
export CUDA_HOME=/media/skr/storage/conda_envs/selfocc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

These are not needed for runtime use of pre-compiled extensions.

## Reference code layout

[reference_code/](reference_code/) holds vendored upstream repos used as references — do not modify them as part of unrelated work:

- [reference_code/GaussianFormer/](reference_code/GaussianFormer/) — GaussianFormer / GaussianFormer-2 (this is what the env is configured for)
- [reference_code/StreamPETR/](reference_code/StreamPETR/)
- [reference_code/SurroundOcc/](reference_code/SurroundOcc/)

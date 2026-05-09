# S2GO Kernel — Compressed Implementation Plan

**Strategy:** compose the three repos already on disk. Don't reinvent any wheels. The 2D→3D adaptation, tile blocking, prefilter, and probabilistic occupancy math are already in GF-2. The atomic-free backward is already in Taming-3DGS. **S2GO is the composition** — not a new design.

This is the *focused, fast-path* plan. The conservative version with full optionality lives in [../Implementation.md](../Implementation.md).

---

## What S2GO actually changes vs. existing code

| Component | Source we keep | What S2GO changes |
|---|---|---|
| Forward tile blocking + cooperative shared-mem load | GF-2 [localagg_prob_fast/src/forward.cu](GaussianFormer/model/head/localagg_prob_fast/src/forward.cu) | nothing structural — already 3DGS-style |
| Tile-Gaussian intersection prefilter (radix sort etc.) | GF-2 [localagg_prob_fast/src/aggregator_impl.cu](GaussianFormer/model/head/localagg_prob_fast/src/aggregator_impl.cu) | nothing |
| Probabilistic occupancy math (Eqs. 1, 4) | GF-2 same | nothing — Eqs are GF-2's |
| **Backward pass** | GF-2's `backward.cu` is the *baseline being replaced* | **Atomic-free pattern from Taming-3DGS — port to GF-2's 3D layout** |
| **Eq. 9 opacity weighting** (S2GO §3.4.2) | n/a | **New: `α(x; G) = a · exp(...)` — small change to forward + matching grad term in backward** |

So the genuine new CUDA work is: **(a) atomic-free backward port + (b) ~5-line Eq. 9 patch on forward**. Everything else is reuse.

---

## Phases

### Phase 0 — Env + benchmark harness
- Run [setup_gaussianformer_env.sh](../scripts/setup_gaussianformer_env.sh)
- Verify all 4 GF-2 ops compile and import in the env
- Write `kernel/tests/benchmark_g2v.py` with synthetic inputs
- Run baseline timings for `localagg_prob_fast` forward + backward at `(N_gauss = 9k, 12.8k, 25.6k)`, voxels = 200×200×16
- Save baseline JSON → `kernel/tests/baseline_3060.json`

**Gate:** ops import + baseline numbers captured.

### Phase 1 — Characterize the atomic-free patch
- Generate diff between Taming's modified backward and vanilla 3DGS backward to isolate exactly what changed:
  ```bash
  diff -u \
      kernel/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu \
      kernel/taming-3dgs/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu \
      > kernel/taming_atomic_free_diff.txt
  ```
- Read GF-2's [localagg_prob_fast/src/backward.cu](GaussianFormer/model/head/localagg_prob_fast/src/backward.cu) end-to-end (~182 lines). Identify all `atomicAdd` calls on Gaussian-parameter gradients.
- Map: each Taming-3DGS atomic-free trick → corresponding location in GF-2's backward. Document in `kernel/Kernel_diff.md`.

**Gate:** clear annotated mapping showing which lines in GF-2's backward.cu need which Taming-3DGS-style change.

### Phase 2 — Clone GF-2 op as a working copy
- `cp -r kernel/GaussianFormer/model/head/localagg_prob_fast kernel/localagg_s2go`
- Update `setup.py` package name to `localagg_s2go`, rename `local_aggregate_prob_fast` symbols
- `cd kernel/localagg_s2go && pip install -e .`
- Verify the cloned op produces identical outputs to the original on a fixed seed

**Gate:** fresh op compiles and is bitwise identical to the original (so future diffs are isolated to S2GO changes).

### Phase 3 — Forward: Eq. 9 opacity tweak
- In `kernel/localagg_s2go/src/forward.cu`, locate the alpha computation
- Change `α = exp(-½ ...)` → `α = a * exp(-½ ...)` per Eq. 9
- Update Python wrapper to ensure opacity tensor `a` is passed through (it already is — opacity is already an input)
- Write a tiny PyTorch reference `kernel/localagg_s2go/python/reference.py` for verification
- Forward output matches PyTorch reference within `1e-4`

**Gate:** forward gradcheck passes; output reflects Eq. 9 weighting.

### Phase 4 — Backward: atomic-free port (the dominant work)
This is where the real engineering happens. Sub-steps:

1. **Apply per-Gaussian thread mapping** to GF-2's backward — each thread accumulates its own Gaussian's `∂L/∂(p, s, r, a, c)` in registers, scanning over voxels in tiles the Gaussian touches. Pattern is from Taming-3DGS.
2. **Eliminate atomicAdd on Gaussian-param gradients.** Cross-Gaussian races still possible on `∂L/∂voxel_input` (the gradient flowing back into the head's MLP) — keep atomics there if needed; the paper's claim is specifically about Gaussian-param gradients.
3. **Add the Eq. 9 derivative term** (the `∂L/∂a` contribution flowing through the new `a *` factor).
4. **Tile size guard:** keep 4×4×4 = 64 voxels per tile per the paper. If shared-memory budget is tight on sm_86 (RTX 3060: 100 KB/SM), fall back to 2×2×2 = 8 voxels per tile.
5. **gradcheck loop:** every change → `torch.autograd.gradcheck` on small synthetic inputs. Iteratively narrow down any drift sources.

**Gate:** all of the following pass:
- `gradcheck` within `1e-3` relative tolerance against PyTorch autograd reference
- `nvcc --ptxas-options=-v` shows no register spilling
- Nsight Compute confirms zero atomics on Gaussian-param gradient writes

### Phase 5 — Benchmark & validate
- Run `kernel/tests/benchmark_g2v.py` against the new `localagg_s2go`
- Compare to Phase 0 baseline numbers
- Targets on RTX 3060 (allow 50% margin vs. paper's A100 ratios):
  - Forward: ≤ baseline (any speedup OK; paper claims 1.5×)
  - **Backward: ≥ 10×** faster than baseline (paper claims 20.4×)
  - **Memory: ≥ 2×** smaller backward peak (paper claims 3.3×)
- Document numbers in `kernel/tests/results.md`

**Gate:** backward speedup ≥ 10×, memory ≥ 2× — otherwise return to Phase 4 for tuning.

### Phase 6 — Integration smoke test
- Add `use_localagg_s2go: bool` to GF-2's [model/head/gaussian_head.py](GaussianFormer/model/head/gaussian_head.py) head config
- Run a single forward+backward pass on synthetic data through the full GF-2 head — both with `localagg_prob_fast` and with `localagg_s2go`
- Verify outputs match within `1e-3`
- *(Deferred until val data extracted)*: full GF-2 Prob-128 inference on one real val frame, compare voxel mIoU

**Gate:** `localagg_s2go` is a drop-in replacement; downstream model output unchanged.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Taming's backward pattern doesn't fit RTX 3060's shared-memory budget at 4×4×4 tiles | medium | Drop to 2×2×2 = 8-voxel tiles → 8× smaller per-tile load; rerun |
| GF-2's existing forward isn't actually at parity with paper's claim — i.e. paper's "1.5×" comes from forward changes too, not just backward | low | If Phase 5 shows the cloned forward is meaningfully slower than paper, also port Taming's forward as bonus phase |
| Numerical drift causes gradcheck failure on edge cases (degenerate Gaussians, single-voxel tiles, near-zero opacity) | medium | Bracket failing cases → either widen tolerance for those or fix; document deviations |
| Cross-Gaussian gradient races on the `∂L/∂voxel_input` side | medium | Keep atomics on input-side gradients (paper's "no atomics" is on Gaussian-param gradients only); verify this matches the paper's claim |
| 2D Taming patterns assume image-plane bounding boxes; GF-2 already does 3D voxel adaptation, so most of this is already solved | low | Diff isolates 2D-only assumptions — port carefully |

---

## Definition of success

| Criterion | Status |
|---|---|
| `localagg_s2go` compiles on RTX 3060 with nvcc 11.8 | ⏳ |
| Forward output matches `localagg_prob_fast` within `1e-4` (modulo Eq. 9 difference) | ⏳ |
| Backward gradients pass `torch.autograd.gradcheck` within `1e-3` rel tolerance | ⏳ |
| Backward speedup ≥ 10× on RTX 3060 vs. `localagg_prob_fast` baseline | ⏳ |
| Memory savings ≥ 2× on backward peak | ⏳ |
| Drop-in replacement in GF-2 `gaussian_head.py` produces ≤ 1% mIoU drift on val | ⏳ |

---

## Concrete first commands

```bash
# Phase 0 prerequisite
bash /media/skr/storage/self_driving/S2GO/scripts/setup_gaussianformer_env.sh

# Phase 1 first action (no code yet)
cd /media/skr/storage/self_driving/S2GO/kernel
diff -u \
    gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu \
    taming-3dgs/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu \
    > taming_atomic_free_diff.txt
wc -l taming_atomic_free_diff.txt   # rough size of the patch
```

---

## What's deliberately out of scope

- Streaming temporal transformer (use `StreamPETR` verbatim per [../architecture.md](../architecture.md))
- Stage 1 denoising pretraining
- Velocity head
- nuScenes data preparation (Phase 6 only needs synthetic data; real val data is bonus)
- Multi-GPU or mixed precision

---

## Reference checklist

| Resource | Local path |
|---|---|
| 3DGS paper | [3dgs_2308.04079.pdf](3dgs_2308.04079.pdf) |
| Taming-3DGS paper | [taming_3dgs_2406.15643.pdf](taming_3dgs_2406.15643.pdf) |
| GaussianFormer paper | [gaussianformer_2405.17429.pdf](gaussianformer_2405.17429.pdf) |
| 3DGS source | [gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/](gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/) |
| Taming-3DGS source | [taming-3dgs/submodules/diff-gaussian-rasterization/cuda_rasterizer/](taming-3dgs/submodules/diff-gaussian-rasterization/cuda_rasterizer/) |
| GF-2 baseline op | [GaussianFormer/model/head/localagg_prob_fast/](GaussianFormer/model/head/localagg_prob_fast/) (symlink) |
| S2GO paper §3.4.3 (the spec) | [../s2go.pdf](../s2go.pdf) |
| Conservative plan | [../Implementation.md](../Implementation.md) |

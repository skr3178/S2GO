# S2GO CUDA kernel — implementation summary

A from-scratch reimplementation of S2GO §3.4.3's "efficient Gaussian-to-voxel
splatting" kernel as a drop-in replacement for GaussianFormer-2's
`localagg_prob_fast`. Built on RTX 3060 (sm_86, nvcc 11.8 via conda).

---

## TL;DR — speedup numbers on RTX 3060

| | Forward | Backward | Memory |
|---|---|---|---|
| Prob-64 (6 400 Gauss)   | 2.07× | 1.13× | 1.77× |
| Prob-128 (12 800 Gauss) | 3.18× | 3.71× | 2.40× |
| **Prob-256 (25 600 Gauss)** | **5.00×** | **5.48×** | **3.55×** |

Combined fwd+bwd at Prob-128: **3.65× total speedup** per training step.

Mechanism (from Nsight Compute):
- **Backward L2 hit rate: 17% → 64%** (the design's headline win)
- **DRAM traffic: 10× reduction** on backward
- **Wave parallelism: 100× more on backward** (0.89 → 89.29 waves/SM)

Outputs are **bit-identical** to the Phase 3 oracle on forward; gradients differ
only by FP32 reordering noise (relative diff ~1e-5 across all four parameters).

---

## What was built

| Artifact | Path | Purpose |
|---|---|---|
| Phase 3 oracle — Eq. 9 patched, per-voxel structure | [localagg_s2go/](localagg_s2go/) | Correctness reference for all subsequent kernels |
| Phase 4 result — tile-blocked forward + backward | [localagg_s2go_tiled/](localagg_s2go_tiled/) | The working S2GO kernel |
| Microbenchmark suite | [tests/benchmark_g2v.py](tests/benchmark_g2v.py), [tests/baseline_3060.json](tests/baseline_3060.json) | Speedup measurement |
| Eq. 9 unit tests | [tests/test_eq9.py](tests/test_eq9.py) | 4 boundary cases (opa=0, opa=1, varied, gradcheck) |
| ncu profile reports | [tests/profile_baseline.ncu-rep](tests/profile_baseline.ncu-rep), [tests/profile_tiled.ncu-rep](tests/profile_tiled.ncu-rep) | Hardware counter before/after |
| Profile launcher targets | [tests/profile_target.py](tests/profile_target.py), [tests/profile_target_tiled.py](tests/profile_target_tiled.py) | Single-purpose CUDA workloads for ncu |

---

## Why this kernel exists

S2GO §3.4.3 claims, on A100 with 9k Gaussians:

| Metric | GF baseline | S2GO | Speedup |
|---|---|---|---|
| Forward | 1.29 ms | 0.87 ms | 1.5× |
| Backward | 116 ms | 5.7 ms | 20.4× |
| Memory | 2079 MB | 633 MB | 3.3× |

The S2GO source was not released, so the kernel had to be reimplemented from the
two-paragraph description: 4×4×4 voxel tile blocking with shared-memory
cooperative Gaussian load (forward), thread-per-Gaussian register accumulation
(backward, "atomic-free" pattern from Taming-3DGS).

---

## Phased implementation

| Phase | What | Outcome |
|---|---|---|
| **0** | env + benchmark harness | Conda env (`selfocc`) with all 4 GF-2 ops compiled. Baseline numbers captured. |
| **1** | characterize the gap vs. existing GF-2 kernel | [Kernel_diff.md](Kernel_diff.md). Found: GF-2 already register-accumulates (zero atomicAdds), the *real* gap is missing tile-based cache locality. |
| **2** | clone GF-2's `localagg_prob_fast` → working copy | bit-identical to source after rebuild |
| **3** | add S2GO Eq. 9 (`α = a · exp(...)`) to forward + matching gradient term in backward | 4/4 boundary tests pass. Identical to S2GO Eq. 9 = 3DGS Eq. 1 = standard opacity-in-alpha. |
| **4a** | profile the bottleneck with ncu | [Phase4a_profile_findings.md](Phase4a_profile_findings.md). Confirmed memory-bandwidth-bound. Backward L2 hit 17%, only 50 blocks for 28 SMs (0.89 waves/SM). |
| **4b** | design memo before any CUDA code | [Phase4b_design.md](Phase4b_design.md). Six concrete decisions: tile dim 4³, block 64 threads, CHUNK_SIZE 64, lex tile-id encoding, etc. |
| **4c** | write tile-blocked forward kernel | After fixing voxel-AABB filter: bit-identical to Phase 3 oracle, 1.96–5.00× speedup. |
| **4d** | rewrite prefilter for 4³ tile coords (rolled into 4c) | `getTileRect` produces tile-coord AABB; sort/range-find logic dimension-agnostic. |
| **4e** | write tile-blocked backward kernel (same pattern as forward, plus per-Gaussian register accumulators + atomicAdd at end) | Gradients match oracle within FP32 reordering noise (rel ~1e-5). 1.13–5.48× speedup. |
| **5** | re-profile with ncu | [Phase5_findings.md](Phase5_findings.md). Backward L2 hit 17% → 64% confirmed. Wave parallelism 0.89 → 89.29 confirmed. |

---

## What's NOT done yet

| Phase | Reason |
|---|---|
| **6 — integration with GF-2 model** | Deferred. Requires extracting `v1.0-trainval01_keyframes.tgz`, arranging the data layout per [GaussianFormer/config/_base_/surroundocc.py](GaussianFormer/config/_base_/surroundocc.py), and verifying our scanline-order pts assumption holds for GF-2's actual inference path. The GF-2-specific pkl (`nuscenes_infos_val_sweeps_occ.pkl`) is downloaded; the data archives are downloaded but not extracted. |
| **7 — final docs / report** | Most of the writeable material is already in the per-phase findings docs. A consolidated report is appropriate after Phase 6 yields end-to-end mIoU numbers. |

---

## Key design decisions

| Decision | Value | Why |
|---|---|---|
| Tile dimension | 4×4×4 = 64 voxels | Matches paper. Maps to 2 warps. Shared mem fits comfortably. |
| Block size | 64 threads (one per voxel-in-tile in forward, one per Gaussian-in-chunk in backward) | Same threads serve dual purpose. |
| Grid | `(H/4, W/4, D/4)` = 50×50×4 = 10 000 blocks | 200× more parallelism than the oracle's 50 blocks. |
| Shared mem per block (forward) | ~7 KB (Gaussian state for 64-Gaussian chunk) | Fits with margin. |
| Shared mem per block (backward) | ~17 KB (Gaussian state + per-voxel state cooperatively pre-loaded) | Backward also pre-loads per-voxel state once per tile. |
| Voxel-AABB inner-loop filter | yes | Restores exact equivalence with Phase 3 oracle (each voxel sees the same set of Gaussians as the per-voxel kernel). |
| pts ordering | scanline (`pts[vx*W*D + vy*D + vz]`) | Allows kernel to compute voxel position from blockIdx/threadIdx without per-voxel coord lookup. |
| Eq. 9 opacity-in-alpha | baked into kernel | No separate patch needed; matches 3DGS standard. |

---

## Why the speedup is hardware-dependent

The paper ran on an **A100 (80 MB L2, 1.5 TB/s HBM)**; we ran on an **RTX 3060
(3 MB L2, 360 GB/s GDDR6)**.

The smaller cache and narrower bandwidth mean the oracle's random-access pattern
hurts more on consumer GPUs. Tile-blocking helps proportionally more. So we see
3–5× forward speedups where the paper saw 1.5×.

The backward speedup is smaller relative to paper (3.71× vs paper's 20.4×) for a
different reason: our oracle (`localagg_prob_fast`) is GF-2's already-optimized
kernel with register accumulation, not the slow per-voxel-atomicAdd kernel the
paper compared against.

So:
- **vs. `localagg_prob_fast` (our baseline):** 3.65× combined fwd+bwd speedup.
- **vs. paper's "GaussianFormer baseline" (the older slow kernel):** the same kernel would presumably show >5× if measured against that baseline.

---

## Files in this folder

```
kernel/
├── SUMMARY.md                       ← this file
├── Plan.md                          ← original phase plan (no time estimates)
├── Kernel_diff.md                   ← Phase 1 finding: real gap = tile blocking, not atomics
├── Phase4a_profile_findings.md      ← memory-bandwidth-bound diagnosis
├── Phase4b_design.md                ← six concrete decisions before coding
├── Phase5_findings.md               ← ncu before/after with cache hit rates
├── Kernel_pseudocode.md             ← user notes / scratchpad
├── 3dgs_2308.04079.pdf              ← Kerbl 2023 (3DGS) reference paper
├── taming_3dgs_2406.15643.pdf       ← Mallick 2024 (Taming) reference paper
├── gaussianformer_2405.17429.pdf    ← Huang 2024 (GaussianFormer) reference paper
├── taming_atomic_free_diff.txt      ← diff of Taming vs vanilla 3DGS backward.cu
├── localagg_s2go/                   ← Phase 3 oracle (Eq. 9 + per-voxel structure)
├── localagg_s2go_tiled/             ← Phase 4 result (tile-blocked, the actual S2GO kernel)
├── tests/                           ← benchmarks, ncu profiles, unit tests
├── gaussian-splatting/              ← Kerbl 2023 source code (reference)
├── taming-3dgs/                     ← Mallick 2024 source code (reference)
├── pmpp/                            ← Programming Massively Parallel Processors code (reference)
├── kernels/                         ← HuggingFace kernels lib (distribution layer)
├── cuda-course/                     ← general CUDA learning material
├── cuda-samples/                    ← NVIDIA CUDA samples
├── CUDALibrarySamples/              ← NVIDIA CUDA library samples
└── GaussianFormer                   ← symlink → ../reference_code/GaussianFormer/
```

---

## Status

**Functionally complete and validated.** The kernel:
- compiles cleanly with the project's conda env
- produces forward outputs bit-identical to the Phase 3 oracle
- produces backward gradients matching within FP32 reordering tolerance
- runs 1.96–5.00× faster on forward, 1.13–5.48× faster on backward, with
  1.77–3.55× memory savings, depending on Gaussian count
- has its mechanism (cache locality) confirmed by Nsight Compute hardware
  counters

**Ready for** integration into the GF-2 inference pipeline (Phase 6) when the
data extraction and folder-layout setup work is undertaken.

**Push target** for the eventual git push: https://github.com/skr3178/S2GO.git
(local repo initialized, nothing pushed yet).

# Phase 1 finding — what S2GO actually changes vs. GF-2's `localagg_prob_fast`

## TL;DR

S2GO §3.4.3's "atomic-free backward" wording is misleading on first read. The real differences relative to GF-2's `localagg_prob_fast` are:

| Aspect | GF-2 baseline | S2GO change |
|---|---|---|
| Per-Gaussian backward atomicAdds | already 0 (uses register accumulation) | unchanged |
| Tile-based shared-memory cooperative Gaussian load **(forward)** | ❌ missing | ✅ add |
| Tile-based shared-memory cooperative Gaussian load **(backward)** | ❌ missing | ✅ add |
| Voxel iteration order | random per-voxel access into `point_list[]` | tile-coherent (4×4×4 blocks) |
| Eq. 9 opacity weighting `α = a·exp(...)` | ❌ missing | ✅ add (~5 lines + grad term) |

**The dominant work is the tile-based blocking — not removing atomics.** The atomic-free framing in the paper refers to the *property* the new kernel preserves, not a delta over GF-2.

---

## Evidence

### 1. GF-2 forward already lacks tile structure
[GaussianFormer/model/head/localagg_prob_fast/src/forward.cu:51](GaussianFormer/model/head/localagg_prob_fast/src/forward.cu#L51):
```cuda
auto idx = cg::this_grid().thread_rank();   // thread = single voxel
const int voxel_idx = point_int[0] * grid.y * grid.z + ...;
uint2 range = ranges[voxel_idx];
for (int i = range.x; i < range.y; i++) {
    int gs_idx = point_list[i];
    float3 cov1 = { cov3D[gs_idx * 6 + 0], ... };  // random global read
    ...
}
```
- Thread per voxel (640k threads for a 200×200×16 grid)
- For each voxel, random Gaussian fetches from global memory
- No `__shared__`, no `__syncthreads`, no cooperation between neighboring voxels
- Comment at line 31 ("Collaboratively works on one tile per block, each thread treats one pixel") is inherited from 3DGS but **does not match the actual implementation**

### 2. GF-2 backward also lacks tile structure (and has 0 atomicAdds)
[GaussianFormer/model/head/localagg_prob_fast/src/backward.cu](GaussianFormer/model/head/localagg_prob_fast/src/backward.cu):
```cuda
auto idx = cg::this_grid().thread_rank();   // thread = single Gaussian
uint32_t start = (idx == 0) ? 0 : offsets[idx - 1];
for (int i = start; i < end; i++) {
    int voxel_idx = point_list_keys_unsorted[i];   // random voxel access
    int pts_idx = voxel2pts[voxel_idx];
    ...  // accumulate in registers (power_grad, deter_grad, prob_grad — line 80-82)
}
// final write — direct, no atomicAdd
```
- Thread per Gaussian (P threads — typically ~9k–25k)
- Each thread scans **all** voxels its Gaussian touches via random memory access
- Already register-accumulates (no inner-loop atomics — paper's "atomic-free" property)
- 640k voxels potentially touched per Gaussian → cache-thrashing

### 3. 3DGS vs Taming-3DGS diff (the reference for what S2GO inherits)
- [taming_atomic_free_diff.txt](taming_atomic_free_diff.txt) — 758 lines
- Both files use atomicAdd; Taming has *more* (14 vs 8) at the file level
- Difference is *where* atomicAdds fire: vanilla 3DGS atomics fire **inside the per-pixel inner loop**; Taming atomics fire **once per thread per Gaussian**, after register accumulation
- Both 3DGS variants use 16×16 image-plane tile blocking with `__shared__` memory cooperative load — this is the structural pattern S2GO ports to 3D voxel grids

---

## Implication for Phase 3 and Phase 4 of [Plan.md](Plan.md)

### Phase 3 (forward) — bigger than originally framed
Not just an Eq. 9 opacity tweak. Also need to add:
- 4×4×4 voxel tile blocking
- `__shared__` memory chunk for Gaussian data per tile
- Cooperative load by all threads in the block, then all 64 tile voxels splat against the shared cache
- Reference: [gaussian-splatting/.../forward.cu](gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu) `renderCUDA` — same pattern, 16×16 image-plane tile

### Phase 4 (backward) — same shape, plus thread-per-Gaussian within the tile
- 4×4×4 voxel tile blocking (same as forward)
- Within each tile-block, threads cooperatively load nearby Gaussians into `__shared__`
- After load, **redistribute work**: each thread now takes responsibility for one Gaussian, scans the 64 voxels in the tile, accumulates that Gaussian's gradient in registers
- Final write: direct (already non-atomic in GF-2's pattern, just preserved)
- Reference: [taming-3dgs/.../backward.cu:597-606](taming-3dgs/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L597) — the `if (valid_splat) { atomicAdd(...Register_dL_d... ); }` pattern after the inner loop

### Eq. 9 opacity tweak (independent, simple)
Both forward and backward gain a single multiply/division by `a`. The corresponding `∂L/∂a` term in backward is already partially there (GF-2 outputs an opacity gradient); needs the new factor added.

---

## What this means for time estimate

The "compose existing repos" strategy still applies, but the forward isn't free (1–2 days, not 0.5). The backward is closer to the original plan estimate (5–7 days). Total kernel work: still ~2 calendar weeks for an experienced CUDA dev, with the rewrite concentrated in `forward.cu` + `backward.cu` of the new op (the prefilter, sort, wrapper, and Python autograd glue all stay verbatim from GF-2).

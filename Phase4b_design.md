# Phase 4b — Design memo for the tile-blocked S2GO kernel

This memo answers six concrete design questions before any CUDA code gets written. It supersedes the high-level pseudocode in [Plan.md](Plan.md) with specific data-structure and parallelism choices informed by the [Phase 4a profile findings](Phase4a_profile_findings.md).

**Target operation:**

```
inputs:
  pts        : (M, 3)  voxel-grid points (output sampling locations)
  means3D    : (P, 3)  Gaussian centers
  cov3D      : (P, 6)  symmetric covariance entries (a,b,c,d,e,f)
  opas       : (P,)    opacity per Gaussian
  semantic   : (P, C)  per-Gaussian class logits
  scales     : (P, 3)  used only for radius / tile-AABB
  H, W, D, grid_size, pc_min  : voxel grid spec

outputs:
  out_logits     : (M, C)  opacity-weighted class mixture (Eq. 4)
  out_bin_logits : (M,)    binary occupancy (1 - ∏(1 - α·o))   (Eq. 9 wrapped via Eq. 1)
  out_density    : (M,)    Σ α (no opacity) — diagnostic
  out_probability: (M,)    Σ α·o normalisation
```

The grid is fixed at **200 × 200 × 16 = 640 000 voxels**. Number of Gaussians per call: typically 6 400 / 12 800 / 25 600 (Prob-64 / Prob-128 / Prob-256). Channel count `C = 18` for nuScenes-SurroundOcc.

---

## Decision 1 — tile dimensions

**Choice: 4 × 4 × 4 = 64 voxels per tile.**

| Reason | Detail |
|---|---|
| Paper specifies it | "we block voxels into 4×4×4 grids" |
| 64 = exact warp pair | maps to `blockDim.x = 64` (2 warps) — clean cooperative load |
| Shared-memory budget fits | per-tile state = ~3 KB out of 100 KB/SM (see Decision 3) |
| 200 × 200 × 16 evenly divisible | grid = (50, 50, 4) tiles = **10 000 blocks** ← matches the 200× parallelism increase predicted in 4a |

Alternatives considered: 2³=8 too few voxels per tile (cooperative load wasted on a quarter-warp); 8³=512 too many — shared mem budget for Gaussian chunk would cap at much smaller chunk size.

## Decision 2 — block / grid layout

**Forward kernel:**
```
Grid  = (Tx, Ty, Tz)    where Tx = H/4 = 50, Ty = W/4 = 50, Tz = D/4 = 4
                         => 10 000 blocks
Block = 64 threads      // one thread per voxel in the tile
```

Each thread holds the per-voxel accumulator state for **its one voxel**:
- `bin_logit` (scalar) — partial product
- `C[18]` (mixture numerator)
- `density` (scalar)
- `prob_sum` (scalar)
- ≈ 22 floats = 22 registers per thread → comfortable headroom on sm_86 (256 regs / thread max)

**Backward kernel:** same 10 000 blocks, 64 threads. After cooperative load (see Decision 3), the 64 threads in the block split work as follows:

> Each chunk of 64 Gaussians is loaded into shared mem. Then thread `i` is responsible for **Gaussian `i` of the chunk**, scanning the tile's 64 voxels and accumulating that Gaussian's gradients in registers. Once chunk done, sync, the same 64 threads atomicAdd one final value per Gaussian to global memory.

This is the Taming-3DGS pattern (per-Gaussian register accumulators, atomicAdd once at the end) — the part of the backward where atomics actually matter.

## Decision 3 — shared-memory layout

**Per-block static `__shared__` buffer**, holding a chunk of `CHUNK_SIZE = 64` Gaussians:

```cuda
__shared__ struct {
    float3 means      [CHUNK_SIZE];   // 12 B × 64 = 768 B
    float  cov_lower6 [CHUNK_SIZE][6];// 24 B × 64 = 1 536 B
    float  opa        [CHUNK_SIZE];   //  4 B × 64 =   256 B
    float  semantic   [CHUNK_SIZE][18];// 72 B × 64 = 4 608 B
    int    gauss_id   [CHUNK_SIZE];   //  4 B × 64 =   256 B (for backward atomicAdd target)
} chunk;
```

**Total ≈ 7.4 KB / block.** sm_86 budget is 100 KB / SM, so up to ~13 blocks can co-reside on one SM (well above the 16 raised by `Block Limit SM` ceiling — won't be the bottleneck).

**Cooperative load:** all 64 threads each load 1 Gaussian's state in lockstep, then `__syncthreads()`. Linear access → fully coalesced.

```cuda
int g = chunk_start + threadIdx.x;     // each thread loads one Gaussian
if (g < chunk_end) {
    chunk.means[threadIdx.x]    = means3D_global[g];
    // ... etc.
}
__syncthreads();
```

For semantic (18 floats), threads loop 18 times to load it. Or: do a parallel load where each thread loads a different field. Either works.

## Decision 4 — tile-Gaussian intersection in 3D

**Approach: 3DGS-style "duplicate per overlapping tile + radix sort by tile id".**

This happens in the **prefilter pass** (currently in `aggregator_impl.cu`). Algorithm:

```
preprocess pass (one thread per Gaussian):
  for each Gaussian g:
    radius_g = scale_multiplier · max(scales_g)            // 3-σ AABB inflation
    tile_min_xyz = floor((mean_g - radius - pc_min) / (grid_size · 4))   // 4 = tile dim
    tile_max_xyz = floor((mean_g + radius - pc_min) / (grid_size · 4)) + 1
    n_tiles_g    = product(tile_max - tile_min)
    tiles_touched[g] = n_tiles_g

inclusive scan tiles_touched -> offsets

duplicate pass (one thread per Gaussian, writes n_tiles_g entries):
  for each (tile_x, tile_y, tile_z) in [tile_min, tile_max):
    tile_id = tile_x + tile_y * Tx + tile_z * Tx * Ty
    keys[off]   = tile_id
    values[off] = g
    off++

sort (keys, values) by keys — cub::DeviceRadixSort

compute_ranges pass (one thread per (g, tile) pair):
  if tile_id changes between consecutive entries:
    ranges[prev_tile].y = idx
    ranges[curr_tile].x = idx
```

This is the existing GF-2 pattern, just with 3D AABBs instead of 2D. **Key change vs. existing code:** `tile_min_xyz` / `tile_max_xyz` are 3D; the loop over touched tiles is triple-nested.

**Tile-id encoding:** linear lexicographic
```
tile_id = tile_x + tile_y * Tx + tile_z * (Tx * Ty)
        = 0..(Tx * Ty * Tz - 1)
        = 0..9999  for the 200×200×16 grid
```
fits in 16 bits (we have 32 — comfortable).

**Why not Morton:** lexicographic is simpler, sorted ranges are contiguous in memory, and at 10 000 tiles total there's no L1/L2 pressure benefit from Morton ordering. Keep simple.

## Decision 5 — backward per-thread register accumulator

In backward, after the cooperative Gaussian load, each thread takes one Gaussian and scans all 64 voxels. Per-thread register state:

```cuda
// per-Gaussian gradient accumulators (28 floats = 28 regs)
float means_grad[3]   = {0, 0, 0};
float cov_grad[6]     = {0, 0, 0, 0, 0, 0};
float opa_grad        = 0;
float sem_grad[18]    = {0, ... 0};
```

**Total: 28 floats = 28 registers per thread**, vs. current 107. Much better. Should let the compiler hit ≥40% theoretical occupancy (vs. current 33%).

**At end of inner loop:** atomicAdd once per gradient component to global memory. ~28 atomics per thread × 64 threads × 10 000 blocks ≈ 18 M atomics total — but each is uncontended (different Gaussian × different gradient component), so cost is low.

Compare to current pattern: ~640 000 atomicAdd's per Gaussian (one per voxel touched). Even at zero contention this is ~10 000× fewer atomic operations.

## Decision 6 — inner-loop math (forward)

After cooperative load of 64-Gaussian chunk into shared mem:

```cuda
// thread t handles voxel idx t in the tile
float3 voxel_pos = compute_voxel_pos(t);
for (int g = 0; g < CHUNK_SIZE; ++g) {
    if (chunk.gauss_id[g] >= P) break;             // tail of tile

    float3 d = chunk.means[g] - voxel_pos;
    float power = compute_quadratic(d, chunk.cov_lower6[g]);  // -½ dᵀΣ⁻¹d
    power = expf(power);
    if (power < 1e-9f) continue;                   // negligible contribution

    float opa = chunk.opa[g];
    float alpha = power * opa;                     // ← Eq. 9 baked in
    bin_logit *= (1.0f - alpha);

    float deter = compute_det(chunk.cov_lower6[g]);
    float prob  = (2π)^-1.5 * sqrtf(deter) * alpha;   // = power * opa
    for (int c = 0; c < 18; ++c) {
        C[c] += chunk.semantic[g][c] * prob;
    }
    density  += power;
    prob_sum += prob;
}
```

**Note:** opacity-in-alpha is built-in via `alpha = power * opa`. No separate Eq. 9 patch needed in this kernel — the new design uses the same opacity-weighted form 3DGS has always used.

## Backward inner-loop math

After cooperative load, thread `t` handles **Gaussian `t` of the chunk**. Scan all 64 voxels in the tile:

```cuda
int g_chunk = t;
if (chunk.gauss_id[g_chunk] >= P) return;          // tail
int g_global = chunk.gauss_id[g_chunk];

float opa = chunk.opa[g_chunk];
float3 mean = chunk.means[g_chunk];
float cov[6]; #pragma unroll for ... cov[i] = chunk.cov_lower6[g_chunk][i];
float sem[18]; #pragma unroll for ... sem[c] = chunk.semantic[g_chunk][c];

for (int v = 0; v < 64; ++v) {                     // 64 voxels in this tile
    float3 voxel_pos = compute_voxel_pos(v);
    float3 d = mean - voxel_pos;
    float power = expf(-0.5f * quadratic(d, cov));
    if (power < 1e-9f) continue;
    float alpha = power * opa;

    // gradients into per-thread registers (no atomics yet)
    accumulate_means_grad(means_grad, ...);
    accumulate_cov_grad(cov_grad, ...);
    opa_grad += bin_factor * power + ...;
    accumulate_sem_grad(sem_grad, ...);
}

// at end: 28 atomicAdds per thread, all on uncontended Gaussian-param locations
atomicAdd(&means3D_grad[g_global * 3 + 0], means_grad[0]);
// ... etc.
```

**Cross-tile gradient accumulation:** different tiles will write to the same Gaussian's gradients (a Gaussian can touch multiple tiles). The final atomicAdd handles this — uncontended by tile, but shared across tiles. Still ~10 000× fewer atomics than current per-voxel pattern.

---

## Outputs of this phase

| Decision | Value |
|---|---|
| Tile dim | 4×4×4 = 64 voxels |
| Block size | 64 threads |
| Grid size | (50, 50, 4) = 10 000 blocks |
| `__shared__` per block | 7.4 KB (CHUNK_SIZE = 64 Gaussians) |
| Tile-id encoding | linear: `tx + ty·50 + tz·2500` |
| Per-thread regs (backward) | 28 (down from 107) |
| Forward output formulation | `α = o · exp(...)` baked in (Eq. 9 = 3DGS Eq. 1) |
| Backward atomicAdd reduction | ~10 000× fewer atomic events |

---

## Success criteria for Phase 4c–4e

The new kernel passes Phase 4 if:

1. ✅ Forward output matches our Phase 3 oracle (`localagg_s2go` with Eq. 9 patch) within `1e-4` on random inputs.
2. ✅ Backward gradients pass `torch.autograd.gradcheck` against the same oracle within `1e-3` relative tolerance.
3. ✅ Backward time on RTX 3060 ≤ **half** of current (target: ≥ 2× speedup; stretch goal 3-6× per Phase 4a estimate).
4. ✅ L2 hit rate ≥ **60%** (currently 17%) on backward, measured via Nsight Compute.
5. ✅ Achieved occupancy on backward ≥ **40%** (currently 30%).
6. ✅ Forward time stays within ±10% of current (i.e. doesn't regress; speedup welcome).

If 1+2 pass but 3 doesn't, we still have a correct kernel and have learned something about the perf model. If 1 fails, return to Phase 4c-4e to fix correctness before perf tuning.

---

## Implementation sub-steps (the actual coding work)

### 4c — port forward.cu (target: ~150 LOC)
1. Rewrite `renderCUDA` from `<thread per voxel + random gauss read>` → `<block per tile + cooperative shared-mem load>`
2. Reuse the inner-loop math from existing GF-2 forward (the math is correct, the parallelism is wrong)
3. Validate against Phase 3 oracle: forward outputs match within `1e-4`

### 4d — port aggregator_impl.cu prefilter (target: ~100 LOC change)
1. Update `tiles_touched` per-Gaussian computation: 2D AABB → 3D AABB
2. Update duplicate pass to write 3D tile_id keys
3. Sort and range-find logic stays as-is (works in any dimension)
4. Validate via test: `out_bin_logits` shape matches expected; nonzero at expected locations

### 4e — port backward.cu (target: ~200 LOC)
1. Same block/tile structure as forward
2. After cooperative load, redistribute work: thread `t` ↔ Gaussian `t`
3. Per-thread register accumulators: means_grad / cov_grad / opa_grad / sem_grad
4. AtomicAdd at end of chunk
5. Validate: gradcheck against Phase 3 oracle within `1e-3` rel tolerance

### Build and compose
- The setup.py / ext.cpp / Python `LocalAggregator` wrapper stay essentially the same — same input/output contract
- `cub::DeviceRadixSort` already present, no header changes needed

---

## Open questions for Phase 4c kickoff

1. **Tile_id sorting:** GF-2's existing `aggregator_impl.cu` uses radix sort over a 32-bit key. With 10 000 tiles that fits in 14 bits. Should we reduce sort key width? (Probably not — sort is fast either way; one less moving part.)
2. **Semantic gradient:** at C=18, sem_grad[18] in registers may push register count higher than we'd like. If profile shows register spilling in 4e, reduce CHUNK_SIZE or split semantic accumulation into shared mem reduction.
3. **Tail handling:** when a tile has fewer than CHUNK_SIZE Gaussians remaining, threads with `g_chunk ≥ remaining` need to no-op. Use an explicit branch + `__syncthreads_count` or sentinel `gauss_id = -1`.

These are tactical — none block starting Phase 4c.

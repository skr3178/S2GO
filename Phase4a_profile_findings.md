# Phase 4a — Profile findings

Hardware: NVIDIA RTX 3060, sm_86, 28 SMs, 360 GB/s peak DRAM
Kernel: `renderCUDA<18>` from `localagg_s2go` (Eq. 9-patched)
Workload: 12 800 Gaussians, 200×200×16 = 640 000 voxels, 18 channels
Tool: Nsight Compute (`ncu --section MemoryWorkloadAnalysis ComputeWorkloadAnalysis Occupancy SpeedOfLight`)

Raw report: [kernel/tests/profile_baseline.ncu-rep](kernel/tests/profile_baseline.ncu-rep)

---

## Forward kernel — already reasonable

| Metric | Value | Verdict |
|---|---|---|
| Duration (per launch) | 5.13 ms | matches benchmark fwd ≈ 8.8 ms (different sample) |
| Grid × block | 2 500 × 256 = 640 000 threads | one thread per voxel ✓ |
| Memory Throughput (% peak BW) | **79.5%** | bandwidth-saturated |
| DRAM Throughput | 34.3% | latency-tolerant; ~120 GB/s achieved |
| L1/TEX Hit Rate | 71.85% | moderate |
| L2 Hit Rate | 80.73% | good |
| Compute (SM) Throughput | 15.5% | low — kernel is waiting on memory |
| Theoretical Occupancy | 83.3% | high |
| Achieved Occupancy | 73.8% | close to theoretical |
| Registers/thread | 48 | moderate |
| L1/TEX Cache Throughput | 98.5% | the L1 cache pipe is saturated |

**Diagnosis:** memory-bandwidth-bound, but well-utilized. Forward is already close to what tile-blocking can achieve. Realistic forward speedup target on this HW: **1.2×–1.5×** (matching paper's 1.5× claim).

---

## Backward kernel — broken in multiple ways simultaneously

| Metric | Value | Verdict |
|---|---|---|
| Duration (per launch) | **27.09 ms** | matches benchmark bwd ≈ 29.6 ms |
| Grid × block | **50 × 256 = 12 800 threads** | one thread per Gaussian — **way too few blocks** |
| Memory Throughput (% peak) | 27.64% | underutilized |
| **L2 Hit Rate** | **17.22%** | **CATASTROPHIC** — 83% of L2 accesses miss → DRAM thrashing |
| L1/TEX Hit Rate | 62.0% | OK at L1, but L2 is collapsed |
| DRAM Throughput | 27.6% | latency-bound, not bandwidth-bound |
| Compute (SM) Throughput | 18.8% | mostly idle, waiting on memory |
| **Theoretical Occupancy** | **33.3%** | **register-pressure-limited** (107 regs/thread) |
| Achieved Occupancy | 29.5% | hits the theoretical ceiling |
| **Waves Per SM** | **0.89** | **under-loaded** — work doesn't even fill the GPU once |
| Block Limit (Registers) | 2 | only 2 blocks/SM due to register pressure |

**Diagnosis:** the backward kernel is a textbook bad CUDA pattern.

| Problem | Why | What tile-blocking does about it |
|---|---|---|
| L2 hit rate **17%** vs forward's 80% | Each thread (per-Gaussian) randomly walks through 640 000 voxels via `point_list_keys_unsorted[i]` → `voxel2pts[voxel_idx]` — no locality | Tile-blocked threads cooperatively load Gaussian state into `__shared__` memory once per tile, then all 64 voxels in the tile read from cache → expected L2 hit rate **70–95%** |
| Only **50 blocks** for 28 SMs | Grid sized = number of Gaussians = 12 800 / 256 = 50 blocks | 200×200×16 / 64 voxels-per-tile = **10 000 blocks** = 200× more parallelism, fully fills the GPU |
| **107 registers per thread** | Per-Gaussian thread holds means_grad[3] + opa_grad + sem_grad[18] + cov_grad[6] = 28 registers + intermediate working state pushes to 107 | Tile-block threads each handle one Gaussian within the tile (per-thread accumulators stay), but blocks cooperate so register pressure is amortized differently |
| 0.89 waves/SM | Too few blocks for full GPU utilization | Solved automatically by 200× more blocks |

---

## Predicted speedup on RTX 3060

This isn't a 1.5× / 20.4× story like A100. On the 3060 the backward bottleneck stack is different:

| Lever | Estimated effect |
|---|---|
| L2 hit 17% → 70% | **3–4×** reduction in DRAM traffic (was the main stall source) |
| 50 → 10 000 blocks | Fixes wave-per-SM under-utilization (currently 0.89, target 4+) |
| Register pressure relief | Marginal — keeps achieved occupancy at ~30% but more blocks help latency hiding |
| Combined | **~3–6× backward speedup is realistic** on the 3060 |

The paper's 20.4× was on A100 where the broken backward had even worse cache behavior at higher voxel counts. On consumer-class HW the gain is smaller in absolute terms but still substantial.

---

## Conclusion

**Phase 4a verdict: ✅ tile-blocking is unambiguously the right intervention.**

The profile data shows the backward kernel is broken across at least three orthogonal axes:
1. Cache locality (L2 hit 17%)
2. Wave parallelism (0.89 waves/SM)
3. Register pressure (107/thread)

Tile-blocking with shared-memory cooperative load addresses all three simultaneously:
- (1) by sharing Gaussian state across the tile's 64 voxels
- (2) by using ~200× more blocks (10 000 vs 50)
- (3) indirectly by enabling more blocks/SM despite register cost

Expected outcome: **3–6× backward speedup** on RTX 3060, modest **1.2–1.5× forward speedup**, and meaningful peak-memory reduction.

Proceeding to **Phase 4b — design memo** with these numbers as concrete success criteria.

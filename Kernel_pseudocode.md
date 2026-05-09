# S2GO Gaussian-to-Voxel (G2V) Kernel — Pseudocode

Pseudocode for the efficient G2V splatting algorithm described in S2GO §3.4.3.
Adapted from 3DGS (Kerbl et al., 2023) tile-based forward + Taming 3DGS
(Mallick et al., 2024) per-Gaussian backward, with the key simplification that
the occupancy product (Eq. 1) is **order-independent** — so no depth sort and
no transmittance pipelining are required.

---

## Algorithm 1 — G2V Setup (Preprocess + Binning)

*P*: number of Gaussians; *B* = (4, 4, 4): voxel block size

```
for all Gaussians g = (μ, r, s, a, c) in parallel do
    Σ ← R(r) · diag(s)² · R(r)ᵀ                ▷ 3D covariance, S2GO Eq. 3
    AABB ← (μ − 3√diag(Σ),  μ + 3√diag(Σ))     ▷ 3-σ bound
    n_g ← number of B-blocks intersecting AABB
end for
offsets ← PrefixSum(n_g)                       ▷ key write positions
for all Gaussians g in parallel do
    for all B-blocks (bx, by, bz) intersecting AABBg do
        keys[offsets[g]++] ← (Flatten(bx,by,bz), g)
    end for
end for
SortByKey(keys, key = tile_id)                 ▷ no depth tiebreak (Eq. 1 commutes)
for all keys k in parallel do                  ▷ build per-tile ranges
    if k = 0 or keys[k].tid ≠ keys[k−1].tid then ranges[keys[k].tid].start ← k
    if k = N−1 or keys[k].tid ≠ keys[k+1].tid then ranges[keys[k].tid].end  ← k + 1
end for
```

---

## Algorithm 2 — G2V Forward

*One block per voxel-tile, one thread per voxel.*
**input:** *keys, ranges* from Alg. 1
**output:** α(x), e(x), and *T*ₛₐᵥₑ, *W*ₛₐᵥₑ for backward

```
(bx, by, bz) ← UnflattenTile(blockIdx)
x ← VoxelCenter(bx·Bx + tx,  by·By + ty,  bz·Bz + tz)        ▷ this thread's voxel
log_T ← 0;  W ← 0;  E ← 0^C                                   ▷ register accumulators
(s, e) ← ranges[blockIdx]
for r ← 0 to ⌈(e − s) / 64⌉ − 1 do
    p ← r·64 + LinearTid()
    if s + p < e then
        cache[LinearTid()] ← LoadGaussian(keys[s + p].g)      ▷ cooperative shared-mem load
    end if
    SyncBlock()
    for j ← 0 to min(64, e − s − r·64) − 1 do
        g ← cache[j]
        d ← x − g.μ;   q ← −½ dᵀ g.Σ⁻¹ d
        if q > 0 then continue                                 ▷ out of support
        α ← g.a · exp(q)                                       ▷ S2GO Eq. 9
        log_T ← log_T + log(1 − min(α, 0.9999))                ▷ S2GO Eq. 1
        w ← g.a · exp(q) / √det(g.Σ);   W ← W + w              ▷ S2GO Eq. 4
        E ← E + w · g.c
    end for
    SyncBlock()
end for
Occ[x] ← 1 − exp(log_T)                                        ▷ occupancy output
Class[x] ← E / max(W, ε)                                       ▷ semantic mixture
Tsave[x] ← exp(log_T);   Wsave[x] ← W                          ▷ checkpoint for backward
```

---

## Algorithm 3 — G2V Backward

*One warp per bucket of 32 Gaussians, one thread per Gaussian.*
**input:** *keys, ranges, Tsave, Wsave, Class* from forward; ∇Occ, ∇Class from upstream
**output:** ∂*L*/∂μ, ∂*L*/∂Σ⁻¹, ∂*L*/∂a, ∂*L*/∂c

```
bid ← blockIdx · WarpsPerBlock + WarpId()                      ▷ global bucket index
tid ← bucket_to_tile[bid]
(s, e) ← ranges[tid]
slot ← (bid − bucket_offset[tid])·32 + LaneId()
if slot ≥ e − s then return
g ← LoadGaussian(keys[s + slot].g)                             ▷ this thread's Gaussian
∇μ ← 0;  ∇Σ⁻¹ ← 0;  ∇a ← 0;  ∇c ← 0^C                          ▷ all in registers
for all voxels x in tile tid do
    d ← x − g.μ;   q ← −½ dᵀ g.Σ⁻¹ d
    if q > 0 then continue
    α  ← min(g.a · exp(q), 0.9999)
    p  ← exp(q) / √det(g.Σ)
    Tx ← Tsave[x];   Wx ← Wsave[x];   ex ← Class[x]
    gα ← ∇Occ[x] · Tx / max(1 − α, ε)                          ▷ ∂α_total/∂α_i = (1−α_total)/(1−α_i)
    gq ← gα · α                                                ▷ ∂α/∂q = α
    ∇a ← ∇a + gα · exp(q)
    gw ← Σ_ch ∇Class[x][ch] · (g.c[ch] − ex[ch]) / max(Wx, ε)  ▷ ∂e/∂w_i, S2GO Eq. 4
    ∇c ← ∇c + (p · g.a / max(Wx, ε)) · ∇Class[x]
    ∇a ← ∇a + gw · p
    gq ← gq + gw · g.a · p
    ∇μ   ← ∇μ   + gq · (g.Σ⁻¹ d)                                ▷ ∂q/∂μ = Σ⁻¹d
    ∇Σ⁻¹ ← ∇Σ⁻¹ − ½ gq · (d ⊗ d)                                ▷ ∂q/∂Σ⁻¹ = −½ d⊗d
end for
AtomicAdd(∂L/∂μ[g],   ∇μ)                                      ▷ one atomic per Gaussian,
AtomicAdd(∂L/∂a[g],   ∇a)                                      ▷  not per (Gaussian, voxel)
AtomicAdd(∂L/∂Σ⁻¹[g], ∇Σ⁻¹)                                    ▷  → 20× fewer atomics
AtomicAdd(∂L/∂c[g],   ∇c)
```

---

## Algorithm 4 — G2V Training Step (top-level)

Mirrors 3DGS Alg. 1.

```
M, S, C, A ← InitQueries()                                     ▷ S2GO sparse 3D queries
i ← 0
while not converged do
    Î_t, V_t ← SampleTrainingFrame()                           ▷ images + voxel GT
    Q_t ← TemporalTransformer(Past_Q, CNN(Î_t))                ▷ refine queries
    G_t ← DecodeGaussians(Q_t)                                 ▷ per-query Gaussians
    Occ, Class ← G2V_Forward(G_t, V_t)                         ▷ Alg. 1 + Alg. 2
    L ← Loss(Occ, Class, OccGT_t, ClassGT_t)
    ∇G_t ← G2V_Backward(∇Occ, ∇Class, …)                       ▷ Alg. 3
    M, S, C, A ← Adam(∇L)                                      ▷ backprop further & step
    Past_Q ← PropagateQueries(Q_t)                             ▷ δ-distance top-k opacity
    i ← i + 1
end while
```

---

## What's missing vs. 3DGS + Taming 3DGS (and why)

| 3DGS / Taming step | Skipped in S2GO | Reason |
|---|---|---|
| `computeCov2DCUDA` (project Σ → 2D conic) | ✓ skipped | No projection — voxels and Gaussians both 3D |
| Sort keys by `tile_id ‖ depth` | only `tile_id` | Occupancy product is order-independent (Eq. 1) |
| `sampled_T` per-bucket transmittance checkpoint | replaced by 1 scalar per voxel (`Tsave`) | Recoverable from saved final α |
| `__shfl_up` warp-pipelining of T between threads | ✓ skipped | No sequential dependency between Gaussians |
| Two render passes (forward + backward) of α-blend loop | ✓ skipped | Backward recomputes per-Gaussian α directly |

## Atomic budget per Gaussian

`3 (μ) + 1 (a) + 6 (Σ⁻¹ symmetric) + C (class)` ≈ **10 + C atomics per Gaussian**,
regardless of how many voxels it covers. Naive G2V (GaussianFormer-2) does
*O*(voxels-touched) atomics per Gaussian — typically hundreds. This is the
source of S2GO's reported 20.4× backward speedup.

## References

- Kerbl et al., 2023. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* arXiv:2308.04079
- Mallick et al., 2024. *Taming 3DGS: High-Quality Radiance Fields with Limited Resources.* arXiv:2406.15643
- S2GO (this paper) §3.4.3 — Efficient Gaussian-to-Voxel Splatting

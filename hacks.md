# Hacks, gotchas, and discoveries

Loose notes on non-obvious behaviour we hit while building S2GO. Save anything
that would have wasted an hour the first time we encountered it.

---

## H1 — `torch_cluster.fps` is non-deterministic by default

**TL;DR:** Two back-to-back `torch_cluster.fps(pts, ratio=K/M)` calls on the
same input produce ~36 % overlap, not 100 %. The default
`random_start=True` re-picks the seed point on every call.

### Experiment (real nuScenes LiDAR, M=34,720 pts, K=900)

| Config | Run-vs-run overlap | Identical? |
|---|---|---|
| Default (`random_start=True`, no seed) | **321 / 900 = 35.7 %** | ❌ |
| `torch.manual_seed(42)` before each call | **900 / 900 = 100 %** | ✅ |
| `random_start=False` (use first input point) | **900 / 900 = 100 %** | ✅ |
| Different seeds (1 vs 2) | **327 / 900 = 36.3 %** | ❌ |

Mean nearest-neighbour spacing is essentially identical across all configs
(2.65–2.66 m on this scene) — so the **uniformity** is deterministic even
when the **specific point selection** isn't. The starting-point choice
shuffles *which* points get picked but not *how evenly* they cover the
scene.

### Where this bites us

[s2go/models/lifter/s2go_lifter.py:68-69](s2go/models/lifter/s2go_lifter.py):

```python
global_idx = fps(flat, batch=batch_idx, ratio=self.K / M)
# ↑ no random_start=False, no manual_seed
```

→ Every Stage-1 training iter on the same scene picks a different 900-pt
FPS subset. The denoise loss target `anchors_xyz` changes between iters.

### Implications

- **For training**: this is essentially a free augmentation on top of the
  ε ~ U(−1,+1)m noise already in Eq. 7. The paper doesn't say either way;
  arguably consistent with the stochastic-ε spirit.
- **For eval / reproducibility**: hand-pinned seeding required. Without
  it, the same val frame produces different anchors → different
  `L_denoise` → noisy metrics.
- **For unit tests**: `_self_test()` results that depend on FPS specifics
  must seed via `torch.manual_seed()` before the call, OR set
  `random_start=False`.

### Recommended fix (when we get to S1.7 eval path)

In `S2GOLifter.forward`, gate the determinism on the `add_noise` arg
(which already proxies for train/eval mode):

```python
global_idx = fps(flat,
                 batch=batch_idx,
                 ratio=self.K / M,
                 random_start=add_noise)         # ← train: random, eval: deterministic
```

Or, more explicit: take a separate `seed: int | None = None` kwarg, and if
provided do `torch.manual_seed(seed)` before the fps call.

### Verification trail

Discovered 2026-05-11 while writing the FPS visualisation; experimental
script saved inline in chat log. The 36 % overlap is *not* zero because
the FPS pattern is highly constrained — even from different start points,
the algorithm tends to lock onto similar high-spread "skeleton" points
around the periphery of the cloud.

---

## H2 — Stage-1 architecture overfit is **slow**, not stuck

**TL;DR:** A 108 M-param S2GO-Small architecture takes ~500 iters to drive
L_denoise from 2.4 m → 0.2 m on a single fixed frame. Mid-run snapshots
(50–200 iters) look "stuck at the 1.5 m noise floor" but the model is just
in the asymptotic descent phase. **Don't conclude "stuck" before iter 500.**

### Diagnostic script

[s2go/tools/denoise_only_check.py](s2go/tools/denoise_only_check.py) — single
frame from `NuScenesLoader[0]`, T=1, only L_denoise active (no depth/RGB/warp),
FPS seeded for stable target, backbone in `eval()` mode (frozen BN), dropout=0.

### Test 1 — full architecture, 500 iters

```
iter   0:  2.43 m   ← init
iter   1:  6.31 m   ← spike (offset gets scrambled)
iter   5:  3.36 m
iter  10:  1.58 m   ← past the spike
iter  50:  1.46 m
iter 100:  1.36 m
iter 150:  1.08 m   ← breaks 1 m
iter 200:  0.82 m
iter 300:  0.38 m
iter 499:  0.20 m   ← still descending; linear extrapolation says ~0.01 m at iter ~800
```

### Test 2 — same, but cross-attention bypassed (set to `_Zero()`)

```
iter 499:  1.05 m   ← 5× higher than Test 1
```

**Cross-attention is helpful, not harmful.** Image features provide
information the parent_offset head uses to refine query positions beyond
what L_denoise alone can teach. Removing it cuts convergence speed ~5×.

### Train-mode stochasticity contribution

Earlier 80-iter run with `backbone.train()` (BN running stats updating)
and decoder `dropout=0.1` plateaued at 2.40 m. Same setup with
`backbone.eval()` + dropout=0 plateaued at 1.42 m at iter 80 (-1 m).
Train-mode noise was costing ~1 m of denoise convergence on top of the
intrinsic slowness.

### Implications for real training

- **50- and 200-iter overfit runs are massively undertrained** for
  L_denoise. They land mid-asymptote, not at a real floor.
- **Mini-epoch on real data (~1,750 iters at batch=1) is borderline.**
  Per the single-sample timing, the model needs ~500–1000 iters per
  unique sample to converge. With one pass over diverse data, each sample
  is seen once — so per-sample convergence depth is shallow. Multiple
  epochs likely needed for L_denoise to drop sharply.
- **The "is denoise stuck at the floor?" diagnostic from §11.7 needs a
  revised threshold:** below 1.5 m in <200 iters is impossible regardless
  of recipe; the meaningful test is "below 0.5 m by iter 1000 on real data."

### What this does NOT prove

- Doesn't validate the implementation handles diverse data well —
  single-sample overfit is a *necessary* not *sufficient* test.
- Doesn't say anything about the **velocity** head or **warps** (they
  were inactive in this test).
- Doesn't prove the ego-pose composition in `MemoryQueue.transform_reference_points`
  is correct (still flagged as a HIGH-suspicion item in `Stage1_design.md`).

### Verification trail

Discovered 2026-05-11. Two runs in chat session: `python -m s2go.tools.denoise_only_check`
(default, x-attn ON) and `DISABLE_XATTN=1 python -m s2go.tools.denoise_only_check`
(Test 2). ~3 min each on the 3060.

---

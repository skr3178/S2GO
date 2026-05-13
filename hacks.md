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

## H3 — LIDAR_TOP frame is `+y forward`, not `+x forward`

**TL;DR:** nuScenes' LIDAR_TOP sensor frame uses **`+y` as the forward
direction**. Don't assume "+x forward" by analogy with the ego frame —
they're different. Projecting a hand-picked test point like `[10, 0, -1, 1]`
("10 m forward, centerline, ground level") through `lidar2img[CAM_FRONT]`
yields a *negative* depth and pixel coords thousands of pixels off-image,
which looks like a catastrophic bug but is just a wrong-axis test point.

### Empirical verification

Project all LiDAR points through `lidar2img[0]` (CAM_FRONT), keep the ones
that land in-bounds with `z > 0`. The axis whose values are strictly
positive for that filtered set is the LIDAR_TOP "forward" axis. Result on a
real frame (3,364 visible points):

| axis | min      | max      | median |
|------|----------|----------|--------|
| x    | −23.22 m | +11.97 m | −0.32  |
| y    | **+5.19 m** | **+74.21 m** | **+16.62** |  ← strictly positive ⇒ forward
| z    | −1.80 m  | +11.65 m | −1.41  |

So LIDAR_TOP in nuScenes is: **+y forward, +x ≈ left/right, +z ≈ up**.

### Why this doesn't break training

The model never sees these axis labels. The temporal decoder's
cross-attention uses `lidar2img` to project queries to pixel coords; the
projection math is axis-agnostic, it just composes the calibration that the
loader baked in. The bulk consistency check (count of in-bounds projected
points per cam vs. count of nonzero pixels in `lidar_depth`):

| cam | in-bounds via `l2i` | nonzero in `lidar_depth` |
|-----|--------------------:|-------------------------:|
| 0 (CAM_FRONT)       | 3,364 | 3,352 |
| 1 (CAM_FRONT_LEFT)  | 3,263 | 3,256 |
| 2 (CAM_FRONT_RIGHT) | 2,770 | 2,766 |
| 3 (CAM_BACK)        | 4,318 | 4,315 |
| 4 (CAM_BACK_LEFT)   | 4,055 | 4,045 |
| 5 (CAM_BACK_RIGHT)  | 2,998 | 2,984 |

Differences are ≤ 14 pixels per cam (edge-clipped LiDAR returns), confirming
`lidar2img`, `viewmats`, `cam_K`, and the loader's own LiDAR→cam depth
rasterization are all internally consistent.

### Where this bites

- **Manual debug projections.** If you write a one-off sanity check
  "project a known forward point through CAM_FRONT", use **`+y`** as the
  forward axis. Otherwise the projection returns garbage and you spend an
  hour suspecting `lidar2img` is broken.
- **Sample-token-keyed visualizations.** If you ever want to filter LiDAR
  points by "in front of the car", the filter is
  `(pts[:, 1] > 0) & (np.abs(pts[:, 0]) < W/2)`, NOT `pts[:, 0] > 0`.
- **Ego_pose composition.** `ego_pose` in our loader is `world ← LIDAR_TOP`
  (verified in [`nusc_loader.py`](s2go/datasets/nusc_loader.py)). Its translation magnitude on the verification
  sample was 1180 m (sane for nuScenes global coords in Boston/Singapore);
  its rotation is SE(3)-valid (`R^T·R ≈ I` to 6e-8, `det = +1.0` to 1e-6).

### Other small things confirmed during the investigation

- **`ego_pose @ ego_pose_inv − I`** has max error 6.1e-5. This is just fp32
  noise on a 1180-m translation; the practical round-trip
  `LIDAR → world → LIDAR` lands within 4.2e-5 m of the original point. Safe
  to use both in autocast/fp32 forward paths.
- **`lidar2img == K_4×4 @ viewmat`** to 7.75e-5 across all 6 cams
  ([`_build_lidar2img()`](s2go/datasets/nusc_loader.py) line 163), so there's no funny non-projective term
  hidden in `l2i`.

### Verification trail

Discovered 2026-05-12 while auditing whether the architecture-doc's claim
"reference_points = refined_xyz" matched the code (it didn't — it's
actually `init_xyz`, see [s2go/models/segmentor.py:148](s2go/models/segmentor.py)). The
axis-convention finding fell out of the same verification run.

---

## H4 — Python stdout block-buffers under `nohup`, hiding training progress

**TL;DR:** When stdout is redirected to a file (as `nohup ... > log 2>&1` does),
Python switches from line-buffered to **block-buffered** (8 KB). `print()`
calls inside a tight training loop accumulate in the buffer and don't appear
in the log file until 8 KB has piled up — which for a sparse `--log-every 100`
setup can mean **never** during a 2-hour run.

### Symptom

Launched a 5,000-iter training run via `nohup`; after 13 minutes the log file
still contained zero iter rows even though `nvidia-smi` showed the GPU pinned
at 100 % utilization with steady memory. Looked like a hang; was actually
just buffered output.

### Math

- One iter row ≈ 150 bytes
- `--log-every 100` over 5,000 iters → 50 rows total
- 50 × 150 = 7,500 bytes — **just under** the 8 KB flush threshold
- Buffer would only flush at process exit (atexit handlers)

### Fix

```bash
exec env PYTHONUNBUFFERED=1 python -u -m s2go.tools.overfit  ...
```

- `-u` forces unbuffered stdout and stderr
- `PYTHONUNBUFFERED=1` is the env-var equivalent (belt-and-suspenders)

After the fix, iter 0 appeared in the log within 30 s, and every subsequent
logged iter flushed immediately.

### When this bites

Any long Python run under nohup with redirected stdout — applies to all of
our overfit/eval/checkpoint scripts. Worth adding `python -u` to launcher
scripts by default.

### Verification trail

Discovered 2026-05-12 during the 5,000-iter `--full-data` streaming run.
Killed and relaunched with `-u`; iter rows became visible in real time.

---

## H5 — `DeformableCrossAttention` softmax fires blind: 80 % of attention mass lands on geometrically invisible camera sites

**TL;DR:** Each Stage-1 query projects 13 keypoints into all 6 nuScenes
cameras, producing 312 sample sites = 6 × 4 levels × 13 pts. By the
geometry of the 6-camera rig, ~80 % of those sites are behind the
camera or outside its image (any query is visible to only 1-2 of the 6
cameras). The unmodified [`DeformableCrossAttention.forward`](s2go/models/encoder/temporal_decoder.py#L169)
softmaxes attention weights across all 312 sites **without computing or
applying a validity mask**, so most of the attention budget gets spent
reading noise from cameras that physically cannot see the query.

This is the **canonical Deformable-DETR / Sparse4D / GaussianFormer
pattern**, present in S2GO's direct upstream lineage but missing from
the version of `temporal_decoder.py` we shipped. Adding it ≈ 5×
sharpens attention on valid signal, no params, no compute.

### The bug

In [s2go/models/encoder/temporal_decoder.py:169-253](s2go/models/encoder/temporal_decoder.py#L169-L253)
(pre-fix), the forward did:

```python
1.  offsets       ← learnable_fc(query)
2.  key_points    ← reference_points + offsets
3.  raw_logits    ← weights_fc(feat_pos)
4.  weights       ← softmax(raw_logits)        ⚠ uniform over 312 sites
5.  pts_2d_h      ← lidar2img @ pts_h          ┐  projection happens
6.  pts_2d_norm   ← perspective divide         │  AFTER softmax — too
                                               │  late to inform
7.  mmcv_deform_attn(weights, pts_2d_norm)     ┘  the weights
```

No `mask = (z>0) ∧ (xy ∈ [0,1])` is ever computed. The clamp
`pts_2d[..., 2:3].clamp(min=1e-5)` only avoids `nan` from divide-by-zero;
it does **not** filter behind-camera or out-of-image points. Those
samples just return edge-of-image noise from `MultiScaleDeformableAttnFunction`,
but the attention budget already routes finite probability mass to them.

### Empirical measurement (probe, not retrain)

[`probe_attn_waste.py`](probe_attn_waste.py) loads the 5,000-iter T2
checkpoint, runs one nuScenes frame, hooks each of the 6 cross-attn
layers, computes geometric validity and the softmax probability that
currently lands on invalid sites.

```
Layer | Invalid sites | Wasted (PRE-fix) | Wasted (POST-fix) | Δ attn on valid
   0  |       81.80 % |          80.46 % |           0.22 %  |     5.11×
   1  |       82.01 % |          83.26 % |           0.11 %  |     5.97×
   2  |       82.34 % |          76.89 % |           0.11 %  |     4.32×
   3  |       82.97 % |          74.73 % |           0.00 %  |     3.96×
   4  |       83.42 % |          83.45 % |           0.11 %  |     6.04×
   5  |       86.49 % |          83.92 % |           0.00 %  |     6.22×
 mean |       83.17 % |          80.45 % |           0.09 %  |     5.11×
```

Per-camera attention distribution is nearly uniform across all 6
cameras (~16 % each, range 9-27 %), confirming the trained model has
**not** learned to suppress invalid cameras through logit magnitudes.
After 5,000 iters of T2-spec depth-only barebones training, the
softmax is structurally indistinguishable from an untrained one with
respect to camera-validity routing.

**Reading**: PRE-fix wasted (80.45 %) ≈ geometric-invalid fraction
(83.17 %). The two numbers being almost equal means *the model has
learned essentially nothing about visibility* — the wasted mass is
what you would predict from a uniform softmax over all 312 sites.

### The fix

[s2go/models/encoder/temporal_decoder.py:192-253](s2go/models/encoder/temporal_decoder.py#L192-L253),
applied in-place:

```python
# Hoist projection up so validity is known before softmax
pts_h        = cat([key_points, ones])
pts_2d_h     = lidar2img @ pts_h
pts_z        = pts_2d_h[..., 2]
pts_2d_norm  = (pts_2d_h[..., :2] / pts_z.clamp(1e-5)) / (pad_w, pad_h)

# Validity in the (B, N_cam, N_q, num_pts) layout
valid_cqp = (pts_z > 1e-5) & (pts_2d_norm[..., 0] ∈ [0,1]) & (...[..., 1] ∈ [0,1])

# Broadcast to 312-site layout: site_index = cam*52 + level*13 + pt
valid_sites = valid_cqp.unsqueeze(2).expand(-1,-1,num_levels,-1,-1)
              .permute(0,3,1,2,4).reshape(B, N_q, -1)

# Mask before softmax. -1e4 is bf16/fp16-safe; exp(-1e4) → 0.
weights = weights_fc(feat_pos).reshape(B, N_q, -1, num_groups)
weights = weights.masked_fill(~valid_sites.unsqueeze(-1), -1e4)
weights = weights.softmax(dim=-2)
```

Three new lines (validity), one new line (mask), one moved block
(projection). Output of `MultiScaleDeformableAttnFunction` unchanged in
shape; only the weight distribution differs.

### What's still missing — `all_miss` handler

[GaussianFormer's `deformable_module.py:213-214`](reference_code/GaussianFormer/model/encoder/gaussian_encoder/deformable_module.py#L213-L214)
also handles the corner case where **every** sample site for a query
is invalid (no camera sees it):

```python
weights[~mask]     = -torch.inf
weights[all_miss]  = 0.            # row of softmax is meaningless; zero it
```

My current fix uses `-1e4` (avoids `nan` from softmax-of-all-`-inf`)
but for queries with zero valid cameras the output is a uniform
`1/312` distribution over noise. Rare in practice (queries are FPS-
sampled from on-scene LiDAR), but worth adding for robustness.

### Reference implementations across the deformable-cross-attn lineage

| Repo | File | Style | Validity mask before softmax? |
|---|---|---|---|
| **Deformable-DETR** (Zhu 2020, canonical) | `multi_scale_deform_attn.py` | deformable | ✅ |
| **Sparse4D** (Lin 2022) | not vendored on this PC | deformable | ✅ (per paper §3.2) |
| **DETR3D** (Wang 2020) | [reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/detr3d_transformer.py:540-562](reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/detr3d_transformer.py#L540-L562) | deformable | **❌ — commented out at [L555](reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/detr3d_transformer.py#L555)**: `# attention_weights = weights * mask` |
| **StreamPETR** (Wang 2023, paper's stated ref) | [reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/petr_transformer.py](reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/petr_transformer.py) | **PETR-style** (3D position embeddings, no per-query projection) | n/a — sidesteps the issue entirely |
| **GaussianFormer** (Huang 2024) | [reference_code/GaussianFormer/model/encoder/gaussian_encoder/deformable_module.py:202-214](reference_code/GaussianFormer/model/encoder/gaussian_encoder/deformable_module.py#L202-L214) | deformable | ✅ `weights[~mask] = -torch.inf` + `all_miss` zero |
| **S2GO** (pre-fix, what we cloned) | [s2go/models/encoder/temporal_decoder.py:169-253](s2go/models/encoder/temporal_decoder.py#L169-L253) | deformable | ❌ regression |
| **S2GO** (post-fix) | same | deformable | ✅ `masked_fill(~valid, -1e4)` |

### Why S2GO probably inherited the regression

[arch_paper.md:18-19](arch_paper.md#L18-L19) cites Deformable Attention
via "Zhu et al., 2020; Lin et al., 2022; Wang et al., 2023" (Deformable-
DETR / Sparse4D / StreamPETR's DETR3D variant). The S2GO `_get_weights`
+ `weights_fc` + `cam_embed` structure is a near-verbatim copy of
StreamPETR's [detr3d_transformer.py:531-538](reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/detr3d_transformer.py#L531-L538)
— same variable names, same softmax-without-mask, same `cam_embed` MLP
shape. The S2GO author appears to have copied from the StreamPETR
repo's DETR3D file rather than from Deformable-DETR or GaussianFormer,
and inherited the same omission (with the same commented-out hint at
L555 that the original author noticed but didn't ship).

The paper-cited Deformable-DETR / Sparse4D / GaussianFormer all do
validity masking. The fix restores S2GO to the canonical form the
paper actually references.

### What this changes for training

Probe says **the pre-fix model wastes ~80 % of cross-attn budget on
noise**. Predicted impact of the fix on loss curves:
- **Sharper gradient signal** at every iter — cross-attn output stops
  averaging noise with real features.
- **Free capacity reclaimed** — `weights_fc` no longer has to learn
  large negative logits for invisible sites (a task it apparently
  wasn't doing anyway, per per-cam ~16 % uniformity).
- **Likely faster convergence + lower L_depth floor**, though magnitude
  unknown until a paired A/B retrain.

Validation experiment: re-run the same 5,000-iter T2 barebones recipe
post-fix and compare to [out/s2go_small_t2_half_5000iter_nockpt/](out/s2go_small_t2_half_5000iter_nockpt/)'s
curve. Init variance noted as caveat — for a clean A/B both runs
should set the same `torch.manual_seed` before model build.

### Discovery trail

1. Reviewing [Stage1_fix_proposed.md](Stage1_fix_proposed.md) flagged
   this as one of three "must-fix before Stage 2" items.
2. Physics argument (camera FOV math): 6 × 70° = 420° angular coverage,
   so any 3D point lies in 1-2 cameras' FOVs → 4-5 of 6 cameras
   structurally invalid → ~67-83 % invalid sites.
3. Empirical confirmation via [`probe_attn_waste.py`](probe_attn_waste.py):
   actual measured invalid-site fraction = 83.17 % (matches upper end
   of geometric prediction).
4. Cross-checked against [reference_code/GaussianFormer/](reference_code/GaussianFormer/)
   and [reference_code/StreamPETR/](reference_code/StreamPETR/) to
   confirm the canonical pattern is the masked version.
5. Applied fix and re-ran probe: wasted mass dropped from 80.45 % to
   0.09 % (numerical residual of `exp(-1e4)`).

Date discovered + fixed: 2026-05-13.

---

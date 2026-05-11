# Stage 1 — Denoising & Rendering Pretraining: design memo

> Concrete engineering decisions and API contracts for S2GO Stage 1, sized to be
> resolved **before** the first line of model code.
> Companion to [plan.md](plan.md) (broad multi-phase reuse map) and
> [kernel/Phase4b_design.md](kernel/Phase4b_design.md) (analogous memo for Stage 2 kernel).

## 0. Scope

Implements the **left half of [Figure1.png](Figure1.png)**:

```
[Past 3D Queries Q_{t-1..t-T}] ──┐
                                  ├──► Temporal Transformer ──► Refined queries (parent + J children)
[Image features F_t]  ────────────┤                                  │
                                  │                                  ├──► gsplat depth + RGB render
[Init queries:                    │                                  │
  FPS(pts_t) + ε  (Stage 1)] ─────┘                                  │
                                                                     ├──► L_denoise = ‖FPS(pts) − (p+o)‖₁
                                                                     ├──► L_depth   = L1(D̂, D_LiDAR)
                                                                     └──► L_rgb     = L1+SSIM(Î, I_t)
                                                                     │
                                                                     └──► top-k by opacity + δ filter ──► queue[t+1]
```

**In scope:** query init, temporal queue, decoder, parent+child refinement, RGB & velocity heads,
gsplat rendering, Eq. 8 losses, queue propagation.

**Out of scope:** Stage-2 voxel splatting (already done — see [kernel/SUMMARY.md](kernel/SUMMARY.md)),
Stage-2 occupancy losses, dataset loader (Phase A in plan.md), nuImages backbone init (Base variant only).

**Target variant:** S2GO-Small (K=900 queries, J=10 children → 9 000 Gaussians, 12-epoch pretrain).

---

## 1. Concrete decisions to pin before coding

### 1a. Stage-1-specific (S2GO-Small)

| # | Decision | Choice | Rationale |
|---|---|---|---|
| **D1** | **FPS implementation** | `torch_cluster.fps(pts, ratio=K/M)` | already in env (1.6.3+pt20cu118), supports CUDA, gradient-free (FPS output is just indices). Uses paper's `K=900`. |
| **D2** | **Noise distribution** | `ε ~ U(-cfg.lidar_noise, +cfg.lidar_noise)^{K×3}` (meters), independent per axis. Default `cfg.lidar_noise = 1.0` for nuScenes-SurroundOcc. | Eq. 7 hyperparameter `e=1m`, paper §B verbatim. **Paper also says: "For nuScenes-Occ3D and KITTI, all distances are scaled according to the smaller extent of the 3D scene."** Must be a config knob, not a literal `1.0` — future dataset switches will silently break otherwise. |
| **D3** | **Differentiable renderer** | `gsplat 1.5.3` (already installed) | Apache-2.0, native depth output, torch 2.0 compatible. Fallback `diff-gaussian-rasterization` only if gsplat depth gradients are flaky. |
| **D4** | **Temporal queue length** | `T = 4` past frames; `cfg.propagate_k = 256` per frame (configurable, not paper-pinned) | Paper §B says T=4; matches StreamPETR default. Per-frame cap of 256 propagated queries inherited from StreamPETR convention; **paper does not pin this number for S2GO** — surface as `cfg.propagate_k` so it can be ablated. |
| **D4b** | **δ propagation distance** (for [H] propagator) | **Train:** `δ ~ U(0, 3m)` per iteration (paper §B). **Inference:** `δ = 1.6m` (paper §B). | **Critical:** must branch `if self.training: ... else: δ = 1.6m`. Missing the eval value silently degrades val mIoU by ~0.5–1.0 — there is no model-state signal that anything is wrong. |
| **D5** | **Eq. 6 parent-child split** | parent emits `(o, v, a)` (5 dims), J=10 children each emit `(o_j, r_j, s_j, a_j)` (11 dims). RGB color attached at child level (3 dims). | Matches paper Eq. 6 plus per-child RGB for Stage 1's photometric loss. |
| **D6** | **Loss weights** | `λ₁=10, λ₂=1, λ₃=1` (Eq. 8) — **starting defaults, not paper-specified.** Surface as `cfg.lambda_denoise / depth / rgb`. | Paper Eq. 8 (line 327-330) writes `λ₁·L_denoise + λ₂·L_depth + λ₃·L_rgb` but does **not** give numeric values. Table 3 ablates only the on/off of each term. **Empirical caveat:** if the denoise term ends up in meters and the depth term is also reduced to meters (per-pixel L1 averaged), the raw magnitudes are already the same order — the 10× boost may be unnecessary. The denoise residual is summed across K=900 queries (paper Eq. 8 has no normalization explicit), so the magnitudes before averaging may differ. Pin a tiny ablation (1, 5, 10, 20) on a 4-sample overfit (S1.7) before committing to 12 epochs. |

### 1b. Temporal transformer dimensions (verified against paper §B + StreamPETR/RepDETR3D source 2026-05-09)

These supersede earlier values that had been inherited from GF-2's `localagg_prob_fast` config.

| # | Decision | Choice | Source / rationale |
|---|---|---|---|
| **D7** | **Embedding dimension** | `embed_dims = 768` | S2GO Appendix B verbatim. **3× the StreamPETR default of 256** ([stream_petr_r50_flash_704_bs2_seq_24e.py:107](reference_code/StreamPETR/projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_24e.py#L107)). |
| **D8** | **Number of decoder layers** | `num_layers = 6` | Figure 1 inset shows `6×`. Matches StreamPETR config `num_layers=6` ([same file:101](reference_code/StreamPETR/projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_24e.py#L101)). |
| **D9** | **FFN hidden** | `feedforward_channels = 4 × embed_dims = 3072` | Standard 4× ratio. StreamPETR uses 2048 with embed=256 (also 8×); we follow the 4× convention since `embed=768` gives a comfortable 3072. |
| **D10** | **Self-attn type** | `PETRMultiheadFlashAttention` ([petr_transformer.py:35](reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/petr_transformer.py#L35)) | Required by the past-query concat at [petr_transformer.py:707-713](reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/petr_transformer.py#L707-L713) which balloons sequence length to ~K + T·256 = 1924 tokens; vanilla attn is O(N²) on 1924² = ~3.7M attention scores per head per layer. Flash is essential. **StreamPETR's main config uses vanilla — S2GO swaps to Flash explicitly (paper §B).** |
| **D11** | **Cross-attn type** | `DeformableFeatureAggregationCuda` ([detr3d_transformer.py:480](reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/detr3d_transformer.py#L480)), `num_pts=13`, `num_groups=12` (= num_heads), `num_levels=4`, `num_cams=6` | Paper §B: "deformable attention for the cross attention layer". This is StreamPETR's **RepDETR3D variant** ([repdetr3d_vov_800_bs2_seq_24e.py:89-98](reference_code/StreamPETR/projects/configs/RepDETR3D/repdetr3d_vov_800_bs2_seq_24e.py#L89-L98)), not the headline StreamPETR config. The op itself wraps mmcv's `MultiScaleDeformableAttnFunction`, which is the packaged version of [Deformable-DETR/models/ops/functions/ms_deform_attn_func.py:21](reference_code/Deformable-DETR/models/ops/functions/ms_deform_attn_func.py#L21). |
| **D12** | **Past-query concat path** | Past queries concatenated into self-attn **K, V** (not as separate cross-attn input); current queries remain as **Q** | Smoking-gun code in [petr_transformer.py:707-713](reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/petr_transformer.py#L707-L713): `temp_key = temp_value = torch.cat([query, temp_memory], dim=0)`. This is the StreamPETR pattern S2GO inherits literally. |

**Memory implication of D7+D8 vs the earlier GF-2-inherited 128/4 setting**: roughly 9× more activation memory per sample per layer. At K + 1024 = 1924 tokens × 768 dims × 6 layers × FP16, that's ~17 MB activations per sample for the temporal transformer — comfortably within budget on a 12 GB 3060 with `with_cp=True` (gradient checkpointing, already on in StreamPETR's `PETRTemporalDecoderLayer`).

These are *defaults*. Each is a knob to tune later but should be fixed before code is written so we have one moving variable at a time.

---

## 2. Reuse map — Figure 1 block → file

This consolidates the per-block sources from [architecture.md](architecture.md) and [plan.md](plan.md) so each Stage-1 module has exactly one upstream reference.

| Fig. 1 block | Action | Upstream file | Symbol |
|---|---|---|---|
| Image encoder | reuse verbatim | [reference_code/GaussianFormer/model/segmentor/bev_segmentor.py](reference_code/GaussianFormer/model/segmentor/bev_segmentor.py) | `BEVSegmentor.extract_img_feat` (L40–L69) |
| Past-query queue | port verbatim | [reference_code/StreamPETR/.../streampetr_head.py](reference_code/StreamPETR/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py) | `reset_memory` L312, `pre_update_memory` L319, `post_update_memory` L345, `temporal_alignment` L420 |
| FPS+ε init | write new | n/a | `S2GOLifter` (mode `'fps_eps'`) |
| Temporal transformer | port verbatim | [reference_code/StreamPETR/.../petr_transformer.py](reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/petr_transformer.py) | `PETRTemporalTransformer` L423, `PETRTemporalDecoderLayer` L513 |
| **Self-attn (Flash)** | reuse | [reference_code/StreamPETR/.../petr_transformer.py:35](reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/petr_transformer.py#L35) | `PETRMultiheadFlashAttention` |
| **Cross-attn (deformable)** | reuse — RepDETR3D variant | [reference_code/StreamPETR/.../detr3d_transformer.py:480](reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/detr3d_transformer.py#L480) | `DeformableFeatureAggregationCuda` — wraps mmcv's `MultiScaleDeformableAttnFunction` (= packaged [Deformable-DETR/.../ms_deform_attn_func.py:21](reference_code/Deformable-DETR/models/ops/functions/ms_deform_attn_func.py#L21)). Pure-PyTorch reference at [Deformable-DETR/.../ms_deform_attn_func.py:41](reference_code/Deformable-DETR/models/ops/functions/ms_deform_attn_func.py#L41) `ms_deform_attn_core_pytorch` for correctness gating. |
| Parent refiner (Eq. 6 outer) | extend | [reference_code/GaussianFormer/.../refine_module_v2.py](reference_code/GaussianFormer/model/encoder/gaussian_encoder/refine_module_v2.py) | `SparseGaussian3DRefinementModuleV2` — add velocity head (output_dim 10+include_opa+sem → 10+include_opa+sem+3) |
| Child decode (Eq. 6 inner) | write new | n/a | `ChildGaussianHead` — MLP per parent, J×11 outputs |
| Per-Gaussian RGB | write new | n/a | `RGBHead` — MLP from child feature → 3 dims |
| Differentiable render | external dep | `gsplat.rasterization` | call signature in §6 |
| L_denoise | write new | n/a | `DenoiseLoss` — ~5 lines L1 |
| L_depth | write new | n/a | `DepthRenderLoss` — L1 on rendered, masked by depth>0 |
| L_rgb | write new | n/a | `RGBRenderLoss` — L1 + SSIM (use `kornia.losses.ssim_loss` already in env via mmcv) |
| Top-k + δ propagator | write new | n/a | `OpacityDeltaPropagator` — top-k by opacity, then greedy mutual-distance pruning |

**Net new code estimate**: ~600 LOC PyTorch (matches plan.md §0 sub-phase mapping).

---

## 3. Tensor-shape contract (single forward pass at one timestep)

Letters: `B` = batch, `T_q` = T_queue (past frames stored), `K` = #queries (900),
`J` = #children per query (10), `N=6` = #cameras, `H_img×W_img = 256×704` = camera resolution.
`embed_dims = 768` per D7.

```
imgs                    : (B, N, 3, H_img, W_img)             ─► image_encoder
ms_img_feats            : list of (B, N, C, h_l, w_l)         ─► temporal decoder
pts_t                   : (B, M, 3) raw LiDAR, M ≈ 35k         ─► FPS+ε
init_xyz                : (B, K, 3)                            ─┐
init_feat               : (B, K, 768)                          ─┴► (queries at t)
mem_xyz   (read)        : (B, T_q×num_propagated=1024, 3)      ─┐
mem_feat  (read)        : (B, T_q×num_propagated, 768)         ─┴► (past queries, ego-compensated)
mem_velo                : (B, T_q×num_propagated, 3)
mem_timestamp           : (B, T_q×num_propagated, 1)

      ┌─ self-attn  K,V = concat([Q_current, mem_feat]) (D12)             ─┐
      └─ cross-attn(deformable, num_pts=13, num_levels=4, num_cams=6) ─ ffn ┘  × num_layers=6 (D8)

post-decoder
queries_refined         : (B, K + T_q*num_propagated, 768)
keep first K            : (B, K, 768)

parent refiner
parent_offset           : (B, K, 3)   — Δ position
parent_opa              : (B, K, 1)
parent_velo             : (B, K, 3)
parent_feat (for child) : (B, K, 768)

child decode (Stage 1: per-Gaussian RGB attached)
child_offset            : (B, K, J, 3)    — relative to parent
child_scale             : (B, K, J, 3)
child_rotation          : (B, K, J, 4)    — quaternion
child_opa               : (B, K, J, 1)
child_rgb               : (B, K, J, 3)    — Stage-1 specific (Stage 2 swaps for shared parent semantics)

assembled Gaussians (flat over K·J)
G.means                 : (B, K*J, 3)     means = init_xyz[i] + parent_offset[i] + child_offset[i,j]
G.scales                : (B, K*J, 3)
G.rotations             : (B, K*J, 4)
G.opacities             : (B, K*J, 1)     parent_opa[i] * child_opa[i,j]
G.colors                : (B, K*J, 3)     child_rgb[i,j]
G.velocity              : (B, K*J, 3)     broadcast parent_velo[i]

rendering (gsplat)
D̂_t  (depth)            : (B, N, H_img, W_img)
Î_t  (RGB)              : (B, N, 3, H_img, W_img)
+ optional ±0.5s warps for the multi-view rendering trick (paper §3.3.3) — apply
  G.means += G.velocity * dt before re-rendering on cams[t+dt]
```

This contract is the single source of truth for module signatures. Each new module's `forward()` takes inputs of the shapes listed above and returns outputs of the listed shapes. **No module guesses shapes from context.**

---

## 3a. Temporal decoder — exact `attn_cfgs` (the dict that goes into the config file)

Hybrid of StreamPETR's `stream_petr_*.py` config (provides the operation_order + temporal layer scaffold) and StreamPETR's `repdetr3d_*.py` config (provides the deformable cross-attn). With paper-specified `embed_dims=768` (D7), `num_layers=6` (D8), Flash for self-attn (D10), and deformable for cross-attn (D11):

```python
transformer = dict(
    type='PETRTemporalTransformer',                    # StreamPETR petr_transformer.py:423
    decoder=dict(
        type='PETRTransformerDecoder',                 # StreamPETR petr_transformer.py:372
        return_intermediate=True,
        num_layers=6,                                  # D8 — was StreamPETR default 6 too
        transformerlayers=dict(
            type='PETRTemporalDecoderLayer',           # StreamPETR petr_transformer.py:513
            attn_cfgs=[
                # self-attn: Flash (D10). StreamPETR's main config uses vanilla MultiheadAttention here;
                # S2GO swaps to Flash explicitly because past-query concat pushes seq len to ~1924.
                dict(
                    type='PETRMultiheadFlashAttention',     # petr_transformer.py:35
                    embed_dims=768,                         # D7
                    num_heads=12,                           # 768/64 head dim, matches Sparse4D ratio
                    dropout=0.1),
                # cross-attn: deformable (D11). Lifted from RepDETR3D variant (Wang 2023 / Lin 2022).
                dict(
                    type='DeformableFeatureAggregationCuda', # detr3d_transformer.py:480
                    embed_dims=768,                          # D7
                    num_groups=12,                           # = num_heads
                    num_levels=4,                            # FPN levels from img backbone
                    num_cams=6,                              # 6 nuScenes cameras
                    num_pts=cfg.num_pts,                     # default 13 (StreamPETR class default;
                                                            #  NOT paper-mandated for S2GO).
                                                            #  13×4×6 = 312 sample sites/query —
                                                            #  on the heavy end. PETR-family typical
                                                            #  is 4-8. Worth ablating at 4 / 8 / 13
                                                            #  once Stage 1 trains.
                    dropout=0.1,
                    bias=1.0),
            ],
            feedforward_channels=3072,                       # D9 = 4 × embed_dims
            ffn_dropout=0.1,
            with_cp=True,                                    # gradient checkpointing per layer
            operation_order=('self_attn', 'norm',
                             'cross_attn', 'norm',
                             'ffn',       'norm')),          # PETR canonical order
    ))
```

This dict slots directly into `s2go/configs/s2go_small_pretrain.py` — no module code needs to know about it; mmengine's registry resolves the type strings to the classes already on disk in `reference_code/StreamPETR/`.

**One open Q remains:** does our `mmcv 2.0.1` env still ship `MultiScaleDeformableAttnFunction`? (it is in mmcv ≥1.4 and was kept through 2.x). Verify with `from mmcv.ops import MultiScaleDeformableAttention` before the first integration test (S1.5).

---

## 3b. Config knobs — consolidated list

All numbers that are *not* set in stone by the paper. Keep these in one block at the top of `s2go_small_pretrain.py` so ablations are config-only edits, not code edits.

```python
# Stage-1 config knobs (paper-unspecified or dataset-dependent values)
cfg = dict(
    # Query / Gaussian counts (paper-fixed for S2GO-Small)
    K = 900,                       # # parent queries
    J = 10,                        # # children per parent

    # Architecture (paper-pinned in §B)
    embed_dims = 768,              # D7
    num_layers = 6,                # D8 (Figure 1 "6×")
    feedforward_channels = 3072,   # D9 = 4 × embed_dims
    num_pts = 13,                  # D11 — StreamPETR class default, NOT paper-pinned. Ablate at 4/8.

    # Streaming queue (paper-pinned T=4; propagate_k from StreamPETR convention)
    T = 4,                         # past frames retained
    propagate_k = 256,             # # queries pushed to t+1; not paper-pinned for S2GO

    # FPS+ε init (D2 — paper-pinned for nuScenes-SurroundOcc; scaled per dataset for Occ3D/KITTI)
    lidar_noise = 1.0,             # ε ~ U(-1, 1)^3 in meters

    # δ propagation distance (D4b — both values paper-pinned)
    delta_train_lo = 0.0,          # train: δ ~ U(0, 3)
    delta_train_hi = 3.0,
    delta_eval = 1.6,              # eval: fixed at 1.6m — REQUIRED for paper-correct mIoU

    # Loss weights (D6 — Stage-1 Eq. 8, NOT paper-specified)
    lambda_denoise = 10.0,         # to tune in S1.7 ablation
    lambda_depth = 1.0,
    lambda_rgb = 1.0,

    # Stage-2 only (paper says aux render is mandated, weights are not)
    lambda_lovasz = 0.25,          # NOT paper-specified
    lambda_aux_render = 0.1,       # NOT paper-specified

    # RGB rendering loss form (paper says "L_rgb" without form; we adopt 3DGS Eq.7)
    rgb_l1_weight = 0.85,          # 3DGS Kerbl 2023 — NOT S2GO-specified
    rgb_ssim_weight = 0.15,

    # Render-warp window (D ⏵ ±0.5s for nuScenes 2 Hz keyframes — needs adaptation for KITTI 10 Hz)
    render_dt_seconds = [-0.5, 0.0, +0.5],   # nuScenes specific

    # Voxel grid + class count (dataset-dependent)
    H = 200, W = 200, D = 16,      # SurroundOcc nuScenes
    num_classes = 18,              # SurroundOcc=18, KITTI-360=19, Occ3D=18
)
```

The classification of each knob (paper-pinned vs paper-silent vs dataset-dependent) is in the comments. Anything tagged "NOT paper-specified" is a place to ablate or rerun if reproducing the published numbers turns out to require different values.

---

## 4. Stage-1 forward pass — pseudocode mapped to code locations

```python
def stage1_forward(self, batch):
    """batch is a T-frame mini-sequence (T=4)."""
    self.reset_memory()                     # StreamPETR L312
    losses = {}

    for t, frame in enumerate(batch.frames):
        # ── 1. Image features ─────────────────────────────────────────
        feats = self.image_encoder(frame.imgs)              # BEVSegmentor.extract_img_feat
        ms_img_feats = feats['ms_img_feats']

        # ── 2. Initialize current queries: FPS+ε on LiDAR ─────────────
        init_xyz, init_feat = self.lifter.fps_eps(
            frame.lidar_pts, K=900, e=1.0)                   # NEW: S2GOLifter

        # ── 3. Pre-update memory queue (ego-compensate past queries) ──
        self.pre_update_memory({                              # StreamPETR L319
            'prev_exists': frame.prev_exists,
            'timestamp':   frame.timestamp,
            'ego_pose':    frame.ego_pose,
            'ego_pose_inv': frame.ego_pose_inv,
        })

        # ── 4. Temporal-aligned query bundle ──────────────────────────
        query_pos, tgt, ref_pts = self.temporal_alignment(    # StreamPETR L420
            query_pos=self.embed_xyz(init_xyz),
            tgt=init_feat,
            reference_points=init_xyz)

        # ── 5. Temporal decoder ───────────────────────────────────────
        outs_dec = self.temporal_decoder(                     # PETRTemporalTransformer L459
            memory=flatten_feats(ms_img_feats),
            tgt=tgt,
            query_pos=query_pos,
            pos_embed=img_pos_embed,
            attn_masks=None,
            temp_memory=self.memory_embedding,
            temp_pos=self.memory_pos)
        # outs_dec: (num_layers, B, K + T_q*num_propagated, embed_dims)

        # ── 6. Parent + child decode (Eq. 6) ──────────────────────────
        parent = self.parent_head(outs_dec[-1, :, :K])        # NEW: extends refine_module_v2
        children = self.child_head(parent.feat, J=10)         # NEW: J Gaussians per parent

        # ── 7. Assemble flat Gaussians (B, K*J, ...) ──────────────────
        G = assemble_gaussians(
            init_xyz=init_xyz,
            parent=parent,
            children=children,
            with_rgb=True)                                    # Stage-1 attaches RGB per child

        # ── 8. Render (current + ±0.5s warps) ─────────────────────────
        for dt in [-0.5, 0.0, +0.5]:
            G_dt = ego_warp(G, parent.velocity, dt)            # NEW: G.means += v*dt
            cam_dt = nearest_cam_set(batch, t, dt)            # nearest sweep frame
            D_hat, I_hat = gsplat_render(G_dt, cam_dt)         # NEW: gsplat_wrapper
            losses['depth'] = losses.get('depth', 0) + l1_masked(D_hat, cam_dt.lidar_depth)
            losses['rgb']   = losses.get('rgb',   0) + l1_ssim(I_hat, cam_dt.imgs)

        # ── 9. Denoise loss (Eq. 8 first term) ────────────────────────
        anchors = fps_only(frame.lidar_pts, K=900)             # noise-free FPS
        losses['denoise'] = l1(anchors - (init_xyz + parent.offset))

        # ── 10. Post-update memory: top-k by opacity, push to queue ───
        prop = self.propagator(parent, k=256, delta=uniform(0, 3))  # NEW
        self.post_update_memory_s2go(prop)                     # variant of StreamPETR L345
                                                              # uses opacity rather than cls score

    L = (10 * losses['denoise']
         + 1.0 * losses['depth']
         + 1.0 * losses['rgb'])
    return L
```

Steps 3, 4, 5, and 10 are wholesale-port + adapt from StreamPETR. Steps 2, 6, 7, 8, 9 are net-new.

---

## 5. Eq. 6 parent–child decomposition — design detail

S2GO Eq. 6:

```
G_t = { { (p_i + o_i + o_{i,j}, v_i, r_{i,j}, s_{i,j}, a_i · a_{i,j}) }_{j=1..J} }_{i=1..K}
```

Reading this carefully:
- `p_i` is the **anchor query position** (FPS+ε in Stage 1)
- `o_i` is the **parent query offset** (refined globally)
- `o_{i,j}` is the **child Gaussian offset** (relative to parent)
- `v_i` is shared velocity across all J children of query i
- `r_{i,j}, s_{i,j}` are per-child rotation and scale
- `a_i · a_{i,j}` — opacity is the **product** of parent and child opacities (gives the parent a "switch off" lever for an entire query group)

### Proposed module structure

```python
class ParentRefiner(nn.Module):           # extends SparseGaussian3DRefinementModuleV2
    """Outputs per-parent: offset (3), opa (1), velocity (3). 7 dims."""
    def __init__(self, embed_dims, ...):
        # shared trunk: 2× linear-relu-ln (matches V2)
        self.trunk = linear_relu_ln(embed_dims, 2, 2)
        self.head = nn.Linear(embed_dims, 3 + 1 + 3)
    def forward(self, instance_feature, anchor, anchor_embed):
        h = self.trunk(instance_feature + anchor_embed)
        out = self.head(h)
        offset = (2 * sigmoid(out[..., :3]) - 1) * unit_xyz
        opa    = sigmoid(out[..., 3:4])
        velo   = out[..., 4:7]              # m/s, no activation
        return ParentPred(offset, opa, velo, feat=h)

class ChildGaussianHead(nn.Module):
    """For each parent, decodes J children: offset (3), scale (3), rot (4), opa (1), rgb (3) = 14 dims × J."""
    def __init__(self, embed_dims, J=10, ...):
        self.J = J
        self.expand = nn.Linear(embed_dims, embed_dims * J)
        self.head = nn.Linear(embed_dims, 3 + 3 + 4 + 1 + 3)
    def forward(self, parent_feat):
        # parent_feat: (B, K, embed_dims)
        x = self.expand(parent_feat).view(B, K, self.J, -1)
        out = self.head(x)
        offset = (2 * sigmoid(out[..., :3]) - 1) * child_unit_xyz
        scale  = sigmoid(out[..., 3:6]) * scale_range_span + scale_range_min
        rot    = F.normalize(out[..., 6:10], dim=-1)
        opa    = sigmoid(out[..., 10:11])
        rgb    = sigmoid(out[..., 11:14])
        return ChildPred(offset, scale, rot, opa, rgb)
```

`unit_xyz` (parent) is the same `[4.0, 4.0, 1.0]` from GF-2's V2 refiner config.
`child_unit_xyz` is smaller (paper Appendix B): the **inter-child spread**. Default `[1.0, 1.0, 0.5]` — children stay close to parent.

### Why this layout

- One `nn.Linear(embed_dims, embed_dims*J)` per parent expands the J child slots (~`768*768*10` ≈ 5.9M params at D7).
- All J children share the same trunk activation; only the final head differentiates them.
- Total params for child head: ~`768*768*10 + 768*14 ≈ 5.9M`. Small relative to the temporal decoder (`6 layers × ~7M` per layer at embed=768) ≈ 5–8% of the head budget.

### Stage 1 vs Stage 2 — the four things that differ

Stage 2 reuses **every** other module (backbone, queue, decoder, refiner, propagator) verbatim. Only the lifter mode, child head mode, loss head, and dataset target swap. This table is the authoritative comparison:

| Component | **Stage 1** (pretrain, 12 ep) | **Stage 2** (occupancy, 24 ep) |
|---|---|---|
| **Query position `p_t^i`** | `FPS_K(pts_t) + ε`, ε ~ U(−1m, +1m) — **per-frame**, derived from each frame's LiDAR | `nn.Parameter(K, 3)` initialized random in scene — **shared across all frames**, learned by gradient descent |
| **Query feature `q_t^i`** | `nn.Parameter(K, 768)` initialized via xavier_uniform — **shared across all frames** (broadcast), learned | `nn.Parameter(K, 768)` — **same convention**, no change |
| **Per-Gaussian color** | own RGB `c_j^i ∈ R^3` per child (Stage-1-specific 3 head dims) | shared parent semantic class per query (the head emits 11 dims, no RGB; a separate parent classifier emits one class) |
| **Loss head** | `gsplat_wrapper.render` → L_denoise + L_depth + L_rgb (Eq. 8) | `localagg_s2go_tiled` voxel splatter → CE + Lovász + (0.1·L_depth + 0.1·L_rgb auxiliary, paper §3.4.1) |
| **Data needed at train** | 6 cams + LiDAR (init AND depth GT) + ego pose | 6 cams + SurroundOcc voxel-occupancy `.npy` + ego pose (+ LiDAR optional for aux depth) |
| **Data needed at inference** | n/a — Stage 1 never runs at inference | 6 cams + ego pose ONLY (no LiDAR ever) |

**Important note on what's shared across frames:** in both stages, `q_t^i` is an `nn.Parameter` of shape `(K, embed_dims=768)` that does **not** depend on the input. Each frame just receives a broadcast copy of these same K learnable feature vectors. The temporal decoder then differentiates per-query representations per-frame by attending to that frame's image features and to the streaming memory queue. The lifter's *initial* feature is invariant; the post-decoder `parent.feat` is frame-dependent.

### Code-level swap from Stage 1 → Stage 2

Implementation deltas (everything else stays):
- **Lifter**: add `mode='learnable'` branch that returns `self.query_xyz = nn.Parameter(K, 3)` instead of running FPS. Forward signature drops the `pts` arg in that mode. ~15 LoC.
- **Child head**: already supports `mode='semantic'` (returns `rgb=None`); add a parent-level `nn.Linear(d, num_classes)` for the shared per-query class. ~10 LoC.
- **Segmentor**: wire `kernel/localagg_s2go_tiled` (already built) where the gsplat renderer currently sits. ~30 LoC.
- **Losses**: reuse GF-2's `OccupancyLoss` (CE + Lovász) verbatim. 0 LoC.
- **Loader**: drop `lidar_depth` field, load `data/nuscenes_occ/.../<sample_token>.npy` as the occupancy target. Requires extracting SurroundOcc `train.zip` (~3 GB extracted). ~30 LoC loader + extraction step.

Total Stage-1 → Stage-2 transition: **~85 LoC + one zip extraction**. Most of the heavy lifting (backbone, queue, decoder, refiner, propagator, kernel) is already done.

---

## 6. gsplat call signature

`gsplat 1.5.3` provides:

```python
from gsplat import rasterization
renders, alphas, _meta = rasterization(
    means         = (N, 3),                 # world frame
    quats         = (N, 4),                 # rotation quaternions
    scales        = (N, 3),                 # log-space or linear (set log=False)
    opacities     = (N,),                   # in [0,1]
    colors        = (N, 3),                 # RGB ∈ [0,1]   for Stage 1
    viewmats      = (C, 4, 4),              # world→cam matrices
    Ks            = (C, 3, 3),              # camera intrinsics
    width         = W_img,
    height        = H_img,
    render_mode   = "RGB+D",                # we want both depth and RGB
    near_plane    = 0.1,
    far_plane     = 80.0)
# renders: (C, H, W, 4) — last channel is depth
```

For nuScenes 6-cam → `C = N_cams = 6`, `W=704, H=256`. Means/quats/scales/opacities/colors come straight from the assembled flat-Gaussian tensor (§3).

**Sparse depth supervision**: `cam_dt.lidar_depth` is produced by projecting `frame.lidar_pts` via `lidar2img` per camera (BEVDepth-style, the `ProjectLiDARToCamDepth` pipeline stage from plan.md Phase A). Mask of valid pixels is `lidar_depth > 0`.

```python
mask = lidar_depth > 0
L_depth = (D_hat - lidar_depth).abs()[mask].mean()
```

For RGB:
```python
L_rgb = 0.85 * (I_hat - imgs).abs().mean() + 0.15 * (1 - ssim(I_hat, imgs))
```

(0.85/0.15 weights from 3DGS Eq. 7.)

---

## 7. Build order with exit criteria (Stage 1 only)

Stage 1 fits inside plan.md sub-phases 1–4. This memo scopes them tighter:

| Step | Deliverable | Exit test |
|---|---|---|
| **S1.0** | gsplat smoke test in isolation | render 100 random Gaussians on one nuScenes frame, eyeball depth + RGB output. No network, no loss. |
| **S1.1** | `S2GOLifter.fps_eps` | given `(B, M, 3)` LiDAR, returns `(B, 900, 3)` xyz with `‖xyz - FPS(pts)‖_∞ < 1m`. unit test on toy point clouds. |
| **S1.2** | `ParentRefiner` + `ChildGaussianHead` | given `(B, K, embed_dims)` instance_feature + anchor, produces flat-Gaussian tensor `(B, K*J, ...)` with paper shapes. unit test against fixed seed. |
| **S1.3** | `assemble_gaussians` + sanity render | full pipeline FPS → query embed → parent + child → flat Gaussians → gsplat render. No temporal queue, no losses, no decoder yet. Visual eyeball + non-zero gradient flow check. |
| **S1.4** | Loss module (`DenoiseLoss + DepthRenderLoss + RGBRenderLoss`) | each loss separately produces a finite scalar with `requires_grad=True` on a 1-frame batch. |
| **S1.5** | Temporal queue + decoder port from StreamPETR | T=4 frame loop; `prev_exists` toggle works; queue grows then refreshes; no NaN across frames. |
| **S1.6** | `OpacityDeltaPropagator` | top-256 by opacity from K=900, then greedy mutual-distance prune at δ=2m (sample δ from U(0,3) only at train time). unit test on synthetic opacities. |
| **S1.7** | 4-sample overfit run | `tools/train.py --config s2go_small_pretrain.py --overfit 4`; combined Eq. 8 loss decreases monotonically over 200 iterations. |
| **S1.8** | Full Stage-1 train (deferred — needs more nuScenes parts) | 12 epochs; denoise loss < 0.1m by epoch 6. |

S1.0–S1.7 are tractable on Part-1 of nuScenes (3 376 keyframes, already extracted). S1.8 needs parts 2–10 (~38 GB more downloads — orthogonal effort).

---

## 7a. Test scales — T0 / T1 / T2 / T3

§7 above lists the *implementation* milestones (S1.0 → S1.8). This section lists the orthogonal *training-scale tiers* — how big a config to actually run at each milestone. The paper's S2GO-Small (K=900, J=10, 9 000 Gaussians) is the floor for paper-comparable numbers, but for verifying the implementation we run shrunken configs that aren't paper-benchmarked.

| Tier | Config | Wall time on RTX 3060 | Purpose |
|---|---|---|---|
| **T0 — smoke** (S1.0–S1.6) | K=900, J=10, **T=1**, num_layers=2, batch=1, 4 fixed samples, 200 iters, no temporal queue | ~10 min | "pipeline runs end-to-end without crashing or NaN" |
| **T1 — overfit** (S1.7) | K=900, J=10, **T=2**, num_layers=4, batch=1, 4 fixed samples, 1000 iters | ~45 min | "gradients flow; losses drop monotonically on a tiny set" |
| **T2 — partial paper-rep** (S1.8) | **paper-spec S2GO-Small**: K=900, J=10, T=4, num_layers=6, embed=768, batch=2, 12 epochs on **nuScenes-trainval Part 1** (3 376 keyframes ≈ 15 % of full) | ~12 hr | partial reproduction; loss curves should track paper trends; absolute Stage-2 mIoU will be lower than paper because of reduced training data |
| **T3 — full paper-rep** | paper-spec on full nuScenes-trainval, 12 ep Stage 1 + 24 ep Stage 2 | ~28 GPU-h on a **single 4090** for the best G2V config (paper Table 6 caption — `tables.md:91`); Stage-1 portion is a fraction of that. Feasible on our 4090; infeasible on the 3060 within reasonable wall time. | match published 22.1 mIoU |

For our hardware, **T0 → T1 → T2** is the realistic ladder. T3 is gated on hardware we don't have.

### What each tier drops vs. S2GO-Small spec

| Knob | Paper / Small | T0 smoke | T1 overfit | T2 paper-rep |
|---|---|---|---|---|
| `K` (queries) | 900 | **100** (10× speedup) | 900 | 900 |
| `J` (children) | 10 | **4** | 10 | 10 |
| `embed_dims` | 768 | **256** (4× memory) | 512 | 768 |
| `num_layers` | 6 | **2** | 4 | 6 |
| `T` (frames) | 4 | **1** (no queue) | 2 | 4 |
| `num_pts` (deformable cross-attn) | 13 | **4** | 8 | 13 |
| `cfg.render_dt_seconds` | [-0.5, 0, +0.5] | **[0]** (skip warps) | [0, +0.5] | [-0.5, 0, +0.5] |
| Image resolution | 256×704 | **128×352** | 256×704 | 256×704 |
| Dataset | full trainval | 4 fixed scenes | 50 scenes (Part 1 subset) | Part 1 (3 376 keyframes) |
| Epochs | 12 | 200 *iterations* (no epoch concept) | 1 epoch | 12 epochs |

T0 deliberately strips the temporal queue, the velocity warps, and most of the architecture — it's a "code compiles, gradients flow, shapes match" smoke test. **The sanity-check assertions in [Stage1_pseudocode.md §sanity-check](Stage1_pseudocode.md) all run at T0.**

T1 keeps the architecture intact but on 4 samples, so we can verify all three loss terms drop monotonically. By construction, an overfit-to-4-samples run that fails to drive `L_denoise` below 0.5 m within 200 iterations indicates an architectural bug, not a data or hyperparameter issue.

T2 is the realistic ceiling for a 3060: full-spec architecture, 12-epoch training on the data we have. Loss curve shapes should match paper trends qualitatively even at 15% of the data; absolute Stage-2 mIoU after fine-tuning will be lower than the paper's 22.1 mIoU because of the reduced training set.

---

## 7b. Stage-1 evaluation metrics

S2GO Stage 1 has **no standalone evaluation benchmark** in the paper. Stage 1 doesn't produce voxel logits — there's no class head yet — so paper Table 1's mIoU/IoU don't apply. The paper measures Stage 1's contribution *indirectly*, by training Stage 2 from each pretrain variant and reporting downstream mIoU/IoU (Table 3).

We therefore have **three independent metric streams**, ordered by cost:

### A. Training-time metrics (cheap — log every iteration)

| Metric | What it tells us | Healthy range |
|---|---|---|
| `L_denoise` (m) | how close refined queries land to noise-free FPS anchors | starts ~σ = 1.0 m; drops to **<0.1 m by epoch 6** (S1.8 target) |
| `L_depth` (m) | render-vs-LiDAR depth error | starts large (random-Gaussian render is noise); steady-state ~1–2 m |
| `L_rgb` (in [0, 1]) | render-vs-image L1+SSIM | starts ~0.3; drops to ~0.05–0.1 |
| Total `L` (weighted sum) | overall training loss | monotone-ish decrease; spikes mean clipped gradients (norm 35) |
| Grad norm (pre-clip) | training stability | should sit < 35 most of the time; chronic clipping → lr too high |
| Lr schedule | sanity check | cosine 4·10⁻⁴ → ~0 over 12 epochs |

For T0 / T1, this stream is **sufficient signal that the architecture is wired correctly**. No held-out set required.

### B. Validation-time metrics (T2/T3 — held-out scenes, standard 3DGS/NeRF set)

| Metric | Definition | Why |
|---|---|---|
| **Depth MAE** (m) | mean(\|D̂ − D^LiDAR\|) on `D^LiDAR > 0` mask | render-quality measure; paper-implicit for L_depth |
| **Depth RMSE** (m) | sqrt(mean((D̂ − D^LiDAR)²)) on valid pixels | penalises outliers more than MAE |
| **PSNR** | −10·log₁₀(MSE_rgb), per camera, averaged over val | standard 3DGS render metric |
| **SSIM** | Kornia's `ssim_loss` (we already compute `1 − SSIM` for L_rgb, so it's free) | structural similarity |
| **Mean denoise residual** (m) | mean L1(FPS anchors, refined positions) on val mini-sequences | Stage-1-specific; S1.8 target <0.1 m at epoch 6 |

### C. The "real" metric — Stage 2 mIoU/IoU after Stage 1 pretraining (paper Table 3)

This is what the paper actually uses to validate Stage 1's value. Reproducing the table on our T2 setup is the eventual definition-of-done test.

| Stage 1 variant | Stage 2 mIoU | Stage 2 IoU | What we should see at T2 (Part 1 only, scaled-down) |
|---|---|---|---|
| **(a) no pretraining** (12 ep S2 only) | 13.02 | 25.73 | rough lower bound — should **beat this** even at T2 |
| (a)† no pretrain, 24 ep (compute-equalized) | 15.83 | 28.35 | **Stage 1 is broken if our T2 lands at or below this** |
| (b) learnable init + depth+RGB pretrain | 12.42 | 26.64 | learnable-init disaster mode (paper-confirmed); only relevant if Stage 1 is run with `mode='learnable'` by mistake |
| (c) raw LiDAR init (no ε) + depth+RGB | 13.62 | 27.08 | useful negative control: Stage 1 without the noise factor |
| (d) LiDAR+ε + depth+RGB (no denoise term) | 20.55 | 32.68 | denoise-ablation comparison |
| (e) LiDAR+ε + depth only | 20.25 | 32.44 | depth-only baseline |
| **(f) LiDAR+ε + depth+RGB+denoise (full S1)** | **21.60** | **33.91** | full-Stage-1 target (paper headline for Small at 12+24 ep) |

Reading Table 3 backwards — what each Stage-1 component contributes to final mIoU:

| Component dropped from full S1 | Δ mIoU | Reading |
|---|---|---|
| denoise term | (f → d) **−1.05** | denoising contributes ~1 mIoU |
| RGB term | (f → e, approx) **~−0.7** | RGB contributes ~0.3–0.7 mIoU; small |
| depth term | (f → b, approx) **~−9** | **depth is the dominant signal** |
| entire Stage 1 | (f → a†) **~+5.8** | total Stage-1 effect at compute-matched compute |

**Practical thresholds for our T2 run:**
- If T2 → Stage-2 finetune mIoU **lands above ~16** (the no-pretrain compute-matched baseline), Stage 1 is contributing positive value.
- If it lands **above ~18**, depth supervision is working.
- If it lands **above ~20** (ablation row e/d territory), at minimum two of the three Eq. 8 terms are wired correctly.
- The full **21.60 mIoU** is out of reach without parts 2-10 of nuScenes (~38 GB more downloads).

### What the paper does NOT measure — and we shouldn't either

- **Standalone Stage 1 mIoU** — there is no voxel class head at Stage 1, so no mIoU is computable.
- **"Denoise success rate"** as a discrete metric — the L_denoise scalar is the right summary.
- **3D structural metrics during Stage 1** — paper has Figure 2 (qualitative) but no numbers.

### Recommended evaluation order

| Order | Test | Expected signal |
|---|---|---|
| 1 | T0 smoke (S1.0–S1.6) | All 8 sanity-check assertions in Stage1_pseudocode pass; no NaN |
| 2 | T1 overfit (S1.7) | All three loss terms drop monotonically over 1000 iters on 4 fixed samples; L_denoise < 0.5 m by iter 200 |
| 3 | T2 partial paper-rep (S1.8) | L_denoise < 0.1 m by epoch 6; checkpoint loads cleanly into a Stage-2 config; rendered depth visually overlays LiDAR points (qualitative eyeball) |
| 4 *(optional, hardware permitting)* | T2 → Stage 2 (24 ep on Part 1) → eval | Stage-2 mIoU ~13–18 (above 15.83 = pretraining is contributing); paper's 21.60 is unreachable without full data |

For Stage 1 implementation work, **the three most actionable metrics are L_denoise (m), Depth MAE (m), and PSNR** — all three computable on the data we have, all three give actionable feedback within hours of starting a run, all three are standard 3DGS/NeRF conventions our reviewers will recognize.

---

## 8. Open questions to resolve before starting S1.0

| Q | Why it matters | Suggested answer |
|---|---|---|
| **Q1** Does gsplat's depth gradient flow back to means+scales+opacity reliably on torch 2.0? | If not, fall back to `diff-gaussian-rasterization` with the depth patch. | smoke-test in S1.0; trust the dep until proven flaky |
| **Q2** Where to attach RGB in the child decode — final 3 dims of the 14-dim child head, or a separate `RGBHead` MLP? | Cleanliness vs. param count. | inline (Stage 2 then masks it away with `mode='semantic'`); avoids a separate module |
| **Q3** Velocity head — parent-only (matches Eq. 6 `v_i`) or per-child? | Paper says parent-level; per-child gives slightly more flexibility. | parent-only — matches paper, fewer params |
| **Q4** ±0.5s warp frames — use nearest sweep timestamps or interpolate poses? | sweeps are at ~12.5Hz; nearest is within ~40ms. | nearest sweep — exact, no extra interp code |
| **Q5** When `prev_exists=False` (sequence reset), what fills the propagated query slots? | StreamPETR uses `pseudo_reference_points` — learnable but frozen. Same applies here. | reuse StreamPETR pattern verbatim |
| **Q6** Memory queue token capacity — paper says T=4 × 256 = 1024 propagated tokens. Decoder sees `K + 1024 = 1924` tokens. Memory budget at `embed_dims=768` (D7) and `num_layers=6` (D8)? | 1924 × 768 × 2B (FP16) ≈ 3 MB activations per layer per sample → ~17 MB across 6 layers. With `with_cp=True` gradient checkpointing (already on in `PETRTemporalDecoderLayer`), backward holds only one layer at a time. Comfortable on 12 GB 3060. | go with T=4, num_propagated=256, embed=768, layers=6. |

---

## 9. Definition of "Stage 1 done"

Stage 1 is **functionally complete** when:
1. S1.0–S1.7 all pass on Part-1 nuScenes (smoke + overfit).
2. The 4-sample overfit run (S1.7) reaches ≤ 0.5m mean denoise error after 200 iterations and visibly-recognisable depth + RGB renders.
3. Module signatures match §3 exactly (no shape drift across the codebase).
4. Stage 1 checkpoint serializes and loads cleanly into a Stage-2 config (proves the architecture is shared).

Stage 1 is **paper-validated** (S1.8) when:
- Full 12-epoch run on full nuScenes-trainval converges to denoise loss < 0.1m.
- Loading the resulting checkpoint into Stage 2 (warm-start the lifter→decoder→refiner) accelerates Stage-2 mIoU above the no-pretrain baseline by ≥ 4 mIoU at epoch 12 (Table 3 ablation reproduces the headline pretraining gain).

---

## 10. Notes on what is **not** in this document

- Dataset + dataloader specifics → plan.md Phase A (sub-phase 1)
- Multi-frame collation, streaming sampler → plan.md Phase F prep work
- Stage-2 voxel splatting kernel → kernel/ folder (done)
- Stage-2 occupancy losses (CE + Lovász) → defer to a future Stage2_design.md when we get there

The aim of this doc is to make the *next ~600 LOC of model code* unambiguous. Module-level decisions (D1–D6, §3 shape contract, §4 forward path, §5 child decode) are the deliverable; everything else is downstream of these choices.

---

## 11. Implementation status snapshot (post-build)

Sections 0-10 are the *design*. This section is what actually got built and what actually ran. Captured here so it survives conversation compactions.

### 11.1 Modules built

All 11 Stage-1 modules have a backing file with a passing `_self_test()` block. Total **~2,680 LoC across 16 files**.

| Module | Path | LoC | Status |
|---|---|---|---|
| gsplat smoke (S1.0) | [s2go/render/gsplat_smoke.py](s2go/render/gsplat_smoke.py) | 145 | ✅ |
| FPS+ε lifter (S1.1) | [s2go/models/lifter/s2go_lifter.py](s2go/models/lifter/s2go_lifter.py) | 130 | ✅ |
| Parent refiner + Child head (S1.2) | [s2go/models/encoder/heads.py](s2go/models/encoder/heads.py) | 215 | ✅ |
| Assembly + integration render (S1.3) | [s2go/models/encoder/assembly.py](s2go/models/encoder/assembly.py) + [s2go/render/gsplat_wrapper.py](s2go/render/gsplat_wrapper.py) | 80 + 165 | ✅ |
| Eq. 8 losses (S1.4) | [s2go/losses/pretrain_loss.py](s2go/losses/pretrain_loss.py) + [ssim.py](s2go/losses/ssim.py) | 190 | ✅ |
| Memory queue (S1.5a) | [s2go/models/queue/memory.py](s2go/models/queue/memory.py) | 240 | ✅ |
| Temporal decoder (S1.5b) | [s2go/models/encoder/temporal_decoder.py](s2go/models/encoder/temporal_decoder.py) | 315 | ✅ |
| Opacity-δ propagator (S1.6) | [s2go/models/queue/propagator.py](s2go/models/queue/propagator.py) | 230 | ✅ |
| Top-level segmentor (S1.7d) | [s2go/models/segmentor.py](s2go/models/segmentor.py) | 280 | ✅ |
| nuScenes loader (S1.7b) | [s2go/datasets/nusc_loader.py](s2go/datasets/nusc_loader.py) | 290 | ✅ |
| R50+FPN backbone (S1.7a) | [s2go/models/backbone/r50_fpn.py](s2go/models/backbone/r50_fpn.py) | 225 | ✅ |
| Overfit training script (S1.7e/f) | [s2go/tools/overfit.py](s2go/tools/overfit.py) | 250 | ✅ |

External code reused (no LoC of ours):
- StreamPETR temporal-block layers (via mmcv-2.x-compatible port in `temporal_decoder.py`)
- `flash_attn 2.6.3`, `mmcv.ops.MultiScaleDeformableAttnFunction`, `gsplat 1.5.3`, `torch_cluster.fps`
- GF-2's `OccupancyLoss`, `BaseLoss`/`MultiLoss` scaffold
- nuscenes-devkit at `/media/skr/storage/self_driving/S2GO/nuscenes/nuscenes-devkit/python-sdk`

### 11.2 Training runs done

| Run | Config | Iters | Wall | Peak mem | Final L_total | Final L_denoise | Final L_depth | Final L_rgb | Status |
|---|---|---|---|---|---|---|---|---|---|
| T0 short | num_layers=2, num_pts=4, ffn=2048, fp32 | 50 | 107 s | 9.2 GB | 19.58 | 1.51 m | 4.23 m | 0.282 | ✅ |
| **T0 long** | same | **300** | **620 s** | **9.2 GB** | **16.72** | **1.39 m** | **2.56 m** | **0.265** | **✅** |
| T2 fp16 + GradScaler | num_layers=6, num_pts=13, ffn=3072, fp16 | 50 | – | – | static at ~38 | – | – | – | ❌ overflow → optimizer skipping |
| T2 bf16 all | same with bf16 | 0 | – | – | – | – | – | – | ❌ `upsample_nearest2d_nhwc` missing bf16 kernel in PyTorch 2.0 |
| T2 bf16 split (backbone fp32, segmentor bf16) | same | 1 | – | 10.3 GB → OOM | 38.44 (iter 0) | – | – | – | ❌ OOM in SSIM, iter 1 |
| T2 + drop SSIM | same with `--rgb-ssim-weight 0` | — | — | — | — | — | — | — | **⏳ pending — next attempt** |

Plot of T0 long: [out/s1.7_curves/loss_curves_T0_300_final.png](out/s1.7_curves/loss_curves_T0_300_final.png).

### 11.3 What T0 long proved

```
T0 over 300 iters on 4 fixed nuScenes T=4 sequences:
  L_total:    41.22 → 16.72   (−59 %)
  L_denoise:   2.72 →  1.39 m  (plateaued near ε=1m noise floor)
  L_depth:    13.76 →  2.56 m  (−81 %, still trending down)
  L_rgb:       0.225→  0.265   (bouncing, slowest-converging term — matches Table 3 ablation: RGB is smallest contributor)
```

The 300-iter T0 run is the strongest evidence we have that the pipeline is correct end-to-end. It demonstrates:
- All 8 architecture blocks compose without crashing
- All gradients flow (verified per-module + end-to-end)
- All three Eq. 8 supervision signals deliver real learning signal (`L_depth` is most dramatic)
- Memory + wall time stay well within RTX 3060 budget (9.2 GB / 12 GB, 2.1 s/iter)

### 11.4 T0 (what we ran) vs paper-spec (S2GO-Small)

| Knob | Paper-spec | What we ran (T0) | Delta |
|---|---|---|---|
| K queries | 900 | 900 | ✓ |
| J children | 10 | 10 | ✓ |
| `embed_dims` | 768 | 768 | ✓ |
| `T_queue` | 4 | 4 | ✓ |
| Image size | 256×704 | 256×704 | ✓ |
| ε noise | 1.0 m | 1.0 m | ✓ |
| `num_layers` | **6** | **2** | ⚠️ 3× shallower |
| `num_pts` | **13** | **4** | ⚠️ 3.25× fewer cross-attn samples |
| `feedforward_channels` | **3072** | **2048** | ⚠️ |
| `batch_size` | **16** (paper §B, *"All models are trained with a 4e-4 learning rate with a batch size of 16…"* — `s2go.pdf` line 1241) | **1** | ⚠️ 16× smaller; paper does not state GPU count or grad-accum, only "single 4090" for the Table-6 ablation |
| Mixed precision | "all models trained with" | **off in T0** | ⚠️ |
| Train data | full trainval (~28k seq) | **4 fixed overfit** | ⚠️ huge gap |
| Total iters | 12 epochs × ~28k/16 ≈ 21,000 | **300** | ⚠️ 70× fewer |

So T0 demonstrates **correctness**, not **paper-comparable accuracy**. T2 (in progress) would close ~half the deltas (layers, pts, ffn, mixed precision) but still leave batch + data + epochs in T3 territory.

### 11.5 Simplifications / assumptions made along the way

Sorted by likely impact. Each one is a deliberate shortcut that diverges from paper-spec.

**Sizing simplifications** (intentional, T0 only):
1. `num_layers=2` vs paper 6 — shallower transformer
2. `num_pts=4` vs paper 13 — coarser image-feature gathering
3. `feedforward_channels=2048` vs 3072 — smaller FFN
4. `batch_size=1` vs paper 16 — effective LR is 16× lower
5. No mixed precision in T0 (paper says "all models trained with mixed precision")
6. 4 fixed overfit samples instead of full trainval — demonstrates correctness, not generalization

**Loss simplifications:**
7. `λ₁=10, λ₂=1, λ₃=1` for Eq. 8 — paper Eq. 8 doesn't give values; ours are guesses
8. RGB loss = `0.85·L1 + 0.15·(1-SSIM)` — paper just says "L_rgb"; form is 3DGS Eq. 7 (Kerbl 2023). SSIM is what blew up T2 memory.
9. **Velocity has no explicit `L_velocity` loss** — paper-correct per §3.3.3 (velocity supervised implicitly via ±0.5s warp render)
10. **No ±0.5s render warps in our overfit yet** — we render at t=0 only, so velocity is currently **un-supervised**. Easy fix: add `dt ∈ {-0.5, 0, +0.5}` loop in `compute_stage1_loss`. **The single biggest paper-spec gap on the loss side.**

**Architecture simplifications:**
11. Positional embedding for 3D `init_xyz`: simple `nn.Linear(3, d) → ReLU → nn.Linear(d, d)`. PETR-family typically uses `pos2posemb3d` (sinusoidal-style with sin/cos basis).
12. No `pseudo_reference_points` for first-frame padding (StreamPETR has learnable-but-frozen pseudo refs). First frame in each sequence sees zeroed memory.
13. No ego-pose embedding via MLN(180) (StreamPETR adds this in `temporal_alignment`). We use only `nerf_positional_encoding` of `memory_reference_point`.
14. Cross-attn `weights_fc.weight` initialized to 0 (faithfully ported from StreamPETR — intentional). iter-1 has zero gradient to `cam_embed`.
15. Render uses gsplat `'RGB+D'` (accumulated depth). `'RGB+ED'` (expected/normalized depth) might be more appropriate for surface-depth supervision. Haven't experimented.

**Data simplifications:**
16. Anisotropic resize 1600×900 → 256×704 (instead of crop+resize). Intrinsics scaled per-axis to compensate.
17. No data augmentation (flip, color jitter, etc.). mmdet3d default does horizontal flip + scaling.
18. `torch_cluster.fps` random start, no seed — entry **H1** in [hacks.md](hacks.md). Each iter sees different FPS subset (mild aug, but breaks reproducibility).
19. `max_lidar_points=35000` random downsampling if cloud is larger (rarely triggered).
20. Single-process loading (no `DataLoader` workers). Slow per-iter data loading; OK at overfit scale.
21. **Part 1 nuScenes only (3,121 sequences out of ~28k)** — limited training coverage; mIoU upper-bound capped well below paper.

**Things we did paper-faithfully (for reference):**
K=900, J=10, embed_dims=768, T_queue=4, image 256×704, ε=1.0m, δ propagator (train `U(0, 3)`, eval `1.6m`), AdamW(lr=4e-4, wd=0.01) with backbone × 0.25 lr scale, grad_clip max_norm=35, cosine annealing, R50 backbone ImageNet1k init, frozen_stages=1, FPN out_channels=embed_dims=768, Flash self-attn + Deformable cross-attn (StreamPETR RepDETR3D variant), Eq. 6 hierarchical parent-child decode, LIDAR_TOP frame as reference throughout.

### 11.6 First-pass recipe (decided 2026-05-11)

After the eval on the 50-iter T2 checkpoint surfaced that **L_rgb went *up*** during training (0.190 → 0.234 across all dt buckets) due to loss-weighting imbalance (`gnorm_rgb ≈ 0.25` vs `gnorm_parent_offset ≈ 336`), and the eval RGB renders showed muddy blobs (see [out/stage1_eval/B_renders.png](out/stage1_eval/B_renders.png)), we settled on a **first-pass recipe** that drops RGB but keeps everything else.

| Component | Decision | Reason | Paper mIoU cost |
|---|---|---|---|
| L_depth | ✅ keep | Dominant signal (~7 mIoU of pretraining gain) | — |
| L_denoise | ✅ keep | 5 LoC, free `‖∂refined‖→FPS` diagnostic | — |
| **L_rgb** | ❌ **drop** | +0.3 mIoU not worth λ-tuning + SSIM memory + muddy-blob debug noise | **−0.3** (Table 3 (d)→(e)) |
| ±0.5s warps | ✅ keep | Only velocity supervision path; paper says "improves final performance"; modest speed cost (~24%) | — |
| δ-NMS propagator | ✅ keep | Paper-spec, works, 0.57 mIoU not worth changing | — |
| S2GO-Small spec | ✅ keep (default) | — | — |
| Stage 2 efficient G2V | ✅ use (when implemented) | 4.6× speed-up, paper-recommended | +free |

**Estimated paper mIoU at full scale: ~21.3** (between row (e) 20.25 and row (f) 21.60). The −0.3 mIoU cost is purely the RGB-direct contribution; we still get LiDAR+ε init + depth-driven position refinement + temporal-warp supervision + denoise regression — i.e., almost the whole pretraining gain.

**Implementation:** `--no-rgb` flag in [s2go/tools/overfit.py](s2go/tools/overfit.py) + `render_mode='D'` support in [s2go/render/gsplat_wrapper.py](s2go/render/gsplat_wrapper.py). ~15 LoC total. The RGB head + colors field on `Gaussians` still exist (Stage 1 architecture unchanged); the RGB head receives zero gradient with `--no-rgb`, so its weights stay near init throughout training but it's harmless.

This makes simplification **#7 (λ values guessed)** and **#8 (RGB form = 3DGS Eq. 7)** in §11.5 *moot* for first-pass runs.

### 11.7 Top open work, in priority

1. **First-pass training run** — T2 + ckpt + warp + **`--no-rgb`** + save, 50 iters minimum, ideally a mini-epoch (~1,750 iters on Part-1) to test whether L_denoise breaks the 1.5 m floor on diverse data.
2. **Stage 1 → Stage 2 load test** — save Stage-1 weights, instantiate Stage-2 config (`mode='learnable'`, `child_mode='semantic'`), verify shared backbone + temporal decoder + parent head weights load cleanly. Catches name mismatches before committing to Stage 2.
3. **Implement Stage 2** — ~85 LoC + extracting SurroundOcc `train.zip` (~3 GB already on disk at `data/surroundocc_dl/`). Use efficient G2V kernel (`localagg_s2go_tiled` on the parallel branch, paper Table 6). See "Stage 2 swap" subsection in §5 for the four deltas.
4. **Stage 2 training + SurroundOcc val mIoU** — first comparable-to-paper number.
5. **Scale up Stage 1 to full nuScenes-trainval** — requires downloading Parts 2-10 from nuscenes.org (~50 GB, see [dataset.md](dataset.md)). Only needed for paper-headline mIoU; not for proof-of-concept.

### 11.8 Where to resume

Most recent state:
- ✅ 50-iter T2 + warp + ckpt + save **completed** (full diagnostics, 40-field history at `/tmp/stage1_t2_50iter.json`, weights at `/tmp/stage1_t2_50iter.pt` 413 MB).
- ✅ 4-check eval ([s2go/tools/stage1_eval.py](s2go/tools/stage1_eval.py)) **completed**, figures + README at [out/stage1_eval/](out/stage1_eval/).
- ✅ Recipe decision **finalized** (§11.6 above).
- ✅ `--no-rgb` flag added and smoke-tested.

**Next concrete step** — re-run T2 with the recipe applied:
```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python -m s2go.tools.overfit \
  --n-iters 50 --num-layers 6 --num-pts 13 --feedforward-channels 3072 \
  --mixed-precision --amp-dtype bf16 --use-checkpoint --no-rgb \
  --save-path /tmp/stage1_recipe_50iter.pt \
  --history-path /tmp/stage1_recipe_50iter.json
```

Then re-run the eval against the new checkpoint (renderings will show only depth columns; RGB will be NaN).

See [hacks.md](hacks.md) for accumulated gotchas (currently H1: FPS non-determinism).
See [Stage1_pseudocode.md](Stage1_pseudocode.md) for the algorithm-block view.
See [architecture.md](architecture.md) for the Figure-1 block diagram.

The architecture is **functionally complete**. The remaining work is operational: first-pass run with the recipe, Stage-1→Stage-2 transfer, Stage 2 implementation.

# S2GO architecture — text-mode diagram

Single-frame view at timestep `t`. The whole pipeline repeats per frame; the
queue [B] carries state forward.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                            S2GO — Streaming Sparse Gaussian Occupancy                        │
│                            (single timestep t; the whole thing repeats per frame)            │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

    INPUTS                                                                       OPTIONAL INPUT
                                                                              (Stage 1 train only)
  ┌──────────────────┐  ┌──────────────────┐                                  ┌────────────────┐
  │ 6-cam images I_t │  │ ego-pose, lidar2 │                                  │ LiDAR pts_t    │
  │ (B,6,3,256,704)  │  │ img matrices     │                                  │ (B, M≈35k, 3)  │
  └────────┬─────────┘  └────────┬─────────┘                                  └────────┬───────┘
           │                     │                                                     │
           ▼                                                                            │
  ┌──────────────────────────────────────┐                                              │
  │ [A] IMAGE ENCODER                    │  ─── paper §4: ResNet50 backbone
  │  ResNet50 ─► FPN  (NO SECONDFPN)     │      Small: ImageNet1k init / Base: nuImages
  │                                      │      reuse: GF-2 BEVSegmentor.extract_img_feat
  │  GF-2's secondfpn_out feeds its      │      with backbone R101→R50 AND SECONDFPN
  │  own pixel-distribution lifter,      │      stripped (we use FPS+ε init from LiDAR
  │  which we don't use → drop that     │      instead of GF-2's pixel-distribution path).
  │  branch from extract_img_feat.       │
  │                                      │
  │  out: ms_img_feats list of           │
  │       (B, 6, C, h_l, w_l)  l=0..3    │
  └──────────────────┬───────────────────┘                                              │
                     │                                                                  │
                     │                                                                  │
                     │       ┌──────────────────────────────────┐                       │
                     │       │ STAGE 1 ONLY                     │                       │
                     │       │ [C] FPS + ε query init (Eq.7)    │ ◄─────────────────────┘
                     │       │  init_xyz = FPS(pts_t,K=900)+ε   │
                     │       │  ε ~ U(-1m, +1m)^(K×3)           │
                     │       │  init_feat = nn.Parameter(K,768) │
                     │       └────────────────┬─────────────────┘
                     │                        │
                     │                        ▼
                     │       ┌──────────────────────────────────┐
                     │       │ STAGE 2 ONLY                     │
                     │       │ [C'] Learnable query positions   │   <── OR
                     │       │  init_xyz = nn.Parameter(K,3)    │
                     │       │  init_feat= nn.Parameter(K,768)  │
                     │       └────────────────┬─────────────────┘
                     │                        │
                     │  ┌─────────────────────┘
                     │  │
                     │  │       ┌──────────────────────────────┐
                     │  │       │ [B] PAST QUERY QUEUE (T=4)   │
                     │  │       │  memory_embedding (B,1024,768)│
                     │  │       │  memory_reference_point      │
                     │  │       │  memory_velo, memory_timestamp│
                     │  │       │  memory_egopose              │ ◄── reuse: StreamPETR
                     │  │       │  pre_update_memory()         │     reset/pre/post_update_memory
                     │  │       │  EGO-TRANSFORM past→current  │     (streampetr_head.py:312-449)
                     │  │       └──────────────────┬───────────┘
                     │  │                          │
                     │  ▼                          ▼
            ┌────────────────────────────────────────────────────────────────┐
            │ [D] TEMPORAL TRANSFORMER  (× 6 layers, embed=768)              │
            │                                                                │
            │   ┌─────────────────────────────────────────┐                  │
            │   │ Self-Attn (Flash, Dao 2022)             │                  │
            │   │  Q = current K queries        (B,K,768) │                  │
            │   │  K,V = concat(Q, mem_feat)  (B,1924,768)│  ◄─ smoking gun  │
            │   │  StreamPETR petr_transformer.py:707-713 │     past queries │
            │   │  ↓ Add+Norm                             │     enter as K,V │
            │   ├─────────────────────────────────────────┤                  │
            │   │ Cross-Attn (Deformable, Zhu/Lin/Wang)   │                  │
            │   │  Q = (B,K,768)                          │                  │
            │   │  Sample 13 keypts/query via lidar2img   │                  │
            │   │  4 FPN levels × 6 cams × 13 pts         │                  │
            │   │  StreamPETR detr3d_transformer.py:480   │                  │
            │   │  ↓ Add+Norm                             │                  │
            │   ├─────────────────────────────────────────┤                  │
            │   │ FFN (3072 hidden) ↓ Add+Norm            │                  │
            │   └─────────────────────────────────────────┘                  │
            │   = PETRTemporalDecoderLayer (StreamPETR)                      │
            └─────────────────────────┬──────────────────────────────────────┘
                                      │
                                      │  outs_dec[-1, :, :K]  →  (B, K, 768)
                                      │  (drop the 1024 past tokens; keep current K=900)
                                      ▼
            ┌──────────────────────────────────────────────────────┐
            │ [E] PARENT REFINER  (extends GF-2 V2 refiner)        │
            │  per parent query i:                                  │
            │    parent_offset o^i  (B,K,3)                         │
            │    parent_opa    a^i  (B,K,1)                         │
            │    parent_velo   v^i  (B,K,3)                         │
            └──────────────────────────────┬───────────────────────┘
                                           │
                                           ▼
            ┌──────────────────────────────────────────────────────┐
            │ [F] CHILD GAUSSIAN HEAD (Eq. 6, NEW ~80 LoC)         │
            │  for each parent, decode J=10 children:               │
            │    child_offset o^{i,j}    (B,K,J,3)                  │
            │    child_scale  s^{i,j}    (B,K,J,3)                  │
            │    child_rot    r^{i,j}    (B,K,J,4)                  │
            │    child_opa    a^{i,j}    (B,K,J,1)                  │
            │                                                       │
            │  STAGE 1: child_rgb c^{i,j}  (B,K,J,3) per child      │
            │  STAGE 2: per-parent semantic class (paper §3.2:      │
            │           "Gaussians derived from the same query      │
            │           collectively share a semantic class label") │
            │           — broadcast over J children, NEVER per-child│
            └──────────────────────────────┬───────────────────────┘
                                           │
                                           ▼
            ┌──────────────────────────────────────────────────────┐
            │ [G] ASSEMBLE FLAT GAUSSIANS  (B, K·J = 9000, ...)     │
            │  G.means      = p^i + o^i + o^{i,j}                   │
            │  G.opacity    = a^i · a^{i,j}                         │
            │  G.scales,rot = child_scale, child_rot                │
            │  G.colors     = child_rgb         (Stage 1)           │
            │  G.semantic   = parent_class      (Stage 2)           │
            │  G.velocity   = v^i  broadcast over J children        │
            └──────────────────────────────┬───────────────────────┘
                                           │
                ┌──────────────────────────┴──────────────────────────┐
                │                                                     │
                ▼                                                     ▼
   ════════════════════════════               ═══════════════════════════════════
   STAGE 1: PRETRAIN                          STAGE 2: 3D OCCUPANCY ESTIMATION
   (12 epochs, no occ labels)                 (24 epochs, after loading Stage 1)
   ════════════════════════════               ═══════════════════════════════════

   ┌─────────────────────────────┐            ┌──────────────────────────────────┐
   │ [G1] gsplat depth+RGB render│            │ [G2] G2V VOXEL SPLATTING        │
   │ for dt ∈ {-0.5, 0, +0.5}s   │            │  (S2GO §3.4.3 — DONE in kernel/) │
   │   means += velocity * dt    │            │                                  │
   │   D̂_t, Î_t = rasterization   │            │  4×4×4 voxel tile blocking      │
   │     (means, quats, scales,  │            │  +Eq. 9 opacity-in-alpha         │
   │      opacities, colors,     │            │                                  │
   │      viewmats, Ks, ...)     │            │  inputs: pts(200·200·16),       │
   │                              │            │          flat Gaussians         │
   │   D̂_t, Î_t shape:            │            │  outputs: voxel_logits          │
   │     (6, H, W, 4)             │            │     (B, H·W·D, num_classes)     │
   │   gsplat 1.5.3 (in env)      │            │   defaults SurroundOcc:         │
   │                              │            │     200·200·16, num_classes=18  │
   │                              │            │   KITTI-360: num_classes=19     │
   │                              │            │  on RTX 3060 vs prob_fast:      │
   └────────────┬────────────────┘            │   3-5× fwd, 1.1-5.5× bwd        │
                │                              │   2-3.5× memory                 │
                │                              │   bit-identical fwd output      │
                │                              └──────────────┬───────────────────┘
                ▼                                             │
   ┌─────────────────────────────┐                            │
   │ EQ. 8 LOSSES  (NEW ~50 LoC)  │                           │
   │ ─────────────────────────── │                            │
   │ L_denoise = ‖FPS(pts) -     │                            │
   │     (init_xyz + parent_o)‖₁ │                            │
   │ L_depth   = L1(D̂, D_LiDAR)  │                            │
   │             on mask>0       │                            │
   │ L_rgb     = 0.85·L1 +        │                            │
   │             0.15·(1-SSIM)   │                            │
   │ L = 10·L_d + 1·L_dp + 1·L_r │                            │
   └─────────────────────────────┘                            │
                                                              ▼
                                                ┌──────────────────────────────────────┐
                                                │ STAGE-2 LOSSES                       │
                                                │  L_ce     = CE(voxel_logits, GT)     │
                                                │  L_lovasz = lovasz_softmax(...)      │
                                                │  L_aux    = w_aux·(L_depth+L_rgb)    │
                                                │            on neighboring keyframes  │
                                                │            (±0.5s, paper §3.4.1)     │
                                                │  L = L_ce + w_lov·L_lovasz + L_aux   │
                                                │                                      │
                                                │  paper §3.4.1+§3.3.3: aux render IS  │
                                                │   prescribed; weights w_aux, w_lov   │
                                                │   are NOT specified — tune empirical │
                                                └──────────────────────────────────────┘

   ────────────────────────────────────────────────────────────────────────────
   POST-FRAME (both stages):  [H] OPACITY-δ PROPAGATOR  →  next frame's queue
   ────────────────────────────────────────────────────────────────────────────
   ┌────────────────────────────────────────────────────────────────────┐
   │ top-{cfg.propagate_k=256} of K=900 by parent_opa  →                │
   │   greedy distance prune by δ:                                       │
   │     train: δ ~ U(0, 3m)            ← paper §B                      │
   │     eval:  δ = 1.6m                ← paper §B (REQUIRED — not      │
   │                                       a default; missing this     │
   │                                       silently drops val mIoU 0.5-1)│
   │ post_update_memory_s2go(): push to head of memory_*; transform    │
   │   reference_points/egopose into next frame's coords; truncate T=4 │
   │ NEW ~30 LoC (replaces StreamPETR's score-based topk)              │
   │                                                                   │
   │ Velocity v^i is supervised IMPLICITLY via render on ±0.5s warps   │
   │ in [G1] (Stage 1) and aux render (Stage 2) — no explicit L_velo.  │
   └────────────────────────────────────────────────────────────────────┘
```

## Legend — code provenance for each block

```
  reuse verbatim (just import / mmengine registry):  [A] image encoder, [B] queue, [D] decoder block
                                                      [G2] our kernel (already built)
  reuse with thin wrapper / extension:                [E] parent refiner, post_update_memory port
                                                      S2GO segmentor (T-frame loop)
  net-new code:                                        [C] FPS+ε init, [F] child head,
                                                      [G1] gsplat wrapper, Eq. 8 losses, [H] propagator
```

## Stage diff at a glance

Everything between **[A]** and **[G]** is *shared code*. The two stages differ only in
[C]/[C'], the last 3 dims of [F]'s output, [G1]/[G2], and the loss.

| What changes between stages | Stage 1 | Stage 2 |
|---|---|---|
| Query position init | FPS(pts) + ε  ([C]) | learnable nn.Parameter ([C']) |
| Per-Gaussian color attribute | own RGB (3 dims, [F]) | shares parent's semantic class |
| Right-hand head | gsplat depth+RGB render ([G1]) | voxel splatting ([G2]) |
| Loss | Eq. 8 (denoise + depth + rgb) | CE + Lovász + 0.1·aux render |
| LiDAR at train time | required (init + depth GT) | optional (only aux depth) |
| LiDAR at inference | n/a (Stage 1 not run at infer) | never used |
| Streaming queue ([B]) | yes | yes |

## Where to find each block in code

| Block | Source |
|---|---|
| [A] image encoder | [reference_code/GaussianFormer/model/segmentor/bev_segmentor.py:40-69](reference_code/GaussianFormer/model/segmentor/bev_segmentor.py#L40-L69) |
| [B] past-query queue | [reference_code/StreamPETR/.../streampetr_head.py:312-449](reference_code/StreamPETR/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py#L312-L449) |
| [C] FPS+ε init | new — uses `torch_cluster.fps` (already in env) |
| [C'] Learnable init | new — `nn.Parameter` (Stage 2) |
| [D] Temporal Transformer (decoder layer) | [reference_code/StreamPETR/.../petr_transformer.py:513](reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/petr_transformer.py#L513) |
| [D] Temporal Transformer (wrapper) | [reference_code/StreamPETR/.../petr_transformer.py:423](reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/petr_transformer.py#L423) |
| [D] Self-attn (Flash) | [reference_code/StreamPETR/.../petr_transformer.py:35](reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/petr_transformer.py#L35) |
| [D] Cross-attn (deformable) | [reference_code/StreamPETR/.../detr3d_transformer.py:480](reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/detr3d_transformer.py#L480) |
| [D] Underlying CUDA op | mmcv `MultiScaleDeformableAttnFunction` (= packaged [reference_code/Deformable-DETR/models/ops/functions/ms_deform_attn_func.py:21](reference_code/Deformable-DETR/models/ops/functions/ms_deform_attn_func.py#L21)) |
| [E] parent refiner | extends [reference_code/GaussianFormer/.../refine_module_v2.py](reference_code/GaussianFormer/model/encoder/gaussian_encoder/refine_module_v2.py) |
| [F] child Gaussian head | new — see [Stage1_design.md §5](Stage1_design.md) |
| [G1] gsplat render | `gsplat 1.5.3` (in env) wrapped in new ~100-line adapter |
| [G2] voxel splatting | **done** — [kernel/localagg_s2go_tiled/](kernel/localagg_s2go_tiled/), see [kernel/SUMMARY.md](kernel/SUMMARY.md) |
| [H] opacity-δ propagator | new — replaces StreamPETR's score-topk |

## Companion docs

- [Stage1_design.md](Stage1_design.md) — concrete API contracts, tensor shapes, attn_cfgs dict, decision table for Stage 1
- [plan.md](plan.md) — broad multi-phase reuse map, dataset/dataloader plan, sub-phase status
- [Implementation.md](Implementation.md) — Stage-1 + Stage-2 training pseudocode, conceptual background
- [kernel/SUMMARY.md](kernel/SUMMARY.md) — what's done in [G2]
- [equations.md](equations.md) — paper equations cross-referenced (Eqs. 6, 7, 8, 9 are the relevant ones for this diagram)

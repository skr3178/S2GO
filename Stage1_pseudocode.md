# Stage 1 training pseudocode — S2GO-Small

> Algorithmic spec for one S2GO-Small Stage-1 training step.
> Style follows the academic algorithm-block convention (numbered lines, `←` for
> assignments, `▷` for inline comments, paired `for/end for` and `if/end if`).
>
> Companion to:
> - [architecture.md](architecture.md) — pipeline diagram with [A]–[H] block labels
> - [Stage1_design.md](Stage1_design.md) — module API contracts, decision table, attn_cfgs
> - [Implementation.md §3](Implementation.md) — Stage-2 counterpart pseudocode

---

## Inputs and notation

```
B            batch size
T = 4        # past+current frames in mini-sequence
K = 900      # parent queries (S2GO-Small)
J = 10       # children per parent (S2GO-Small)
d = 768      # embed_dims
N_cam = 6    # nuScenes cameras
H, W = 256, 704        # image resolution

I_t       ∈ ℝ^(B × N_cam × 3 × H × W)         current-frame images
pts_t     ∈ ℝ^(B × M × 3)                      LiDAR points, M ≈ 35k
ego_t     ∈ SE(3)                              ego pose at frame t
D_t^LiDAR ∈ ℝ^(B × N_cam × H × W)              sparse LiDAR-projected depth GT

p_t^i     ∈ ℝ^3        query position    (i = 1..K)
o_t^i     ∈ ℝ^3        parent offset
v_t^i     ∈ ℝ^3        parent velocity
a_t^i     ∈ ℝ          parent opacity

(o_{t,j}^i, s_{t,j}^i, r_{t,j}^i, a_{t,j}^i, c_{t,j}^i)        child Gaussian j of parent i
                       ∈ ℝ^3 × ℝ^3 × ℝ^4 × ℝ × ℝ^3

ε ~ Uniform(−σ, σ)³,  σ = cfg.lidar_noise = 1.0 m   (paper §B, nuScenes-SurroundOcc)
λ_1, λ_2, λ_3 = cfg.lambda_{denoise, depth, rgb} = 10, 1, 1  (defaults; not paper-pinned)
```

> **First-pass recipe (decided 2026-05-11, see [Stage1_design.md §11.6](Stage1_design.md))**:
> for early training runs we set `λ_3 = 0` (drop L_rgb) and use `gsplat`
> `render_mode='D'` to skip the RGB rasterization path entirely. The RGB head
> and per-child color outputs still exist in the architecture (so Stage-2
> weight transfer is unaffected), but receive no gradient. Estimated cost:
> ~0.3 mIoU at full paper scale (Table 3 (d)→(e)). All L_rgb steps below
> are no-ops when this flag is on. Lines 17–19 and 25 still describe the
> full recipe — read with `λ_3 := 0` for first-pass.

---

## Algorithm 1 — Stage-1 training step (one mini-sequence of T frames)

```
─────────────────────────────────────────────────────────────────────────────────
Algorithm 1   S2GO-Small Stage-1 Training Step
─────────────────────────────────────────────────────────────────────────────────
Input:   batch B of T contiguous frames {(I_t, pts_t, ego_t)}_{t=1..T}
Output:  scalar loss L
─────────────────────────────────────────────────────────────────────────────────
 1:   reset_memory()                                                                                ▷ wipe streaming queue [B]
 2:   L_den, L_dep, L_rgb ← 0, 0, 0
 3:   for t = 1, …, T do
 4:       F_t ← ImgEncoder(I_t)                                                                     ▷ R50 → FPN, 4 levels [A]
 5:       a_t ← FPS_K(pts_t)                                                                        ▷ noise-free anchors (Eq. 7)
 6:       ε ~ Uniform(−σ, σ)^(K × 3),    σ = cfg.lidar_noise                                       ▷ ε scale per dataset
 7:       p_t ← a_t + ε                                                                             ▷ noised init positions
 8:       f_t ← lifter.query_feat (broadcast over B)                                                ▷ ∈ ℝ^(B × K × d)  [C]
                                                                                                    ▷ query_feat is an nn.Parameter(K, d) — learnable,
                                                                                                    ▷ shared across all frames + batch (Stage 2 keeps
                                                                                                    ▷ this exact convention; only p_t changes from FPS+ε
                                                                                                    ▷ to a learnable nn.Parameter(K, 3))
 9:       pre_update_memory(ego_t)                                                                  ▷ ego-compensate past queries [B]
10:       M ← model.memory_embedding                                                                ▷ ∈ ℝ^(B × (T·k_prop) × d), k_prop = 256
11:       f_t' ← TemporalDecoder(f_t, F_t, M)                                                       ▷ 6 layers; self-attn K,V = (current ⊕ M) [D]
12:       (o_t, a_t^par, v_t) ← ParentRefiner(f_t')                                                ▷ offsets, opacity, velocity [E]
13:       {(o_{t,j}, s_{t,j}, r_{t,j}, a_{t,j}, c_{t,j})}_{j=1..J} ← ChildHead(parent_feat, mode='rgb')   ▷ Eq. 6 [F]
14:       𝒢_t ← Assemble(p_t, o_t, {children})                                                      ▷ flat K·J = 9000 Gaussians [G]
15:       for dt ∈ cfg.render_dt_seconds = {−0.5, 0, +0.5} do                                      ▷ paper §3.3.3 (nuScenes 2 Hz)
16:           𝒢_t^(dt) ← 𝒢_t with means ← means + v_t · dt                                          ▷ ego-motion warp
17:           (D̂, Î) ← gsplat_render(𝒢_t^(dt), cams_{t+dt})                                       ▷ depth + RGB [G1]
18:           L_dep ← L_dep + L1_masked(D̂, D_{t+dt}^LiDAR)                                         ▷ mask: D^LiDAR > 0
19:           L_rgb ← L_rgb + 0.85·L1(Î, I_{t+dt}) + 0.15·(1 − SSIM(Î, I_{t+dt}))                  ▷ 3DGS Eq. 7 weighting
20:       end for
21:       L_den ← L_den + Σ_i ‖ a_t^i − (p_t^i + o_t^i) ‖_1                                        ▷ Eq. 8 first term
22:       prop ← TopK_δ_Opacity(parent_t, k = cfg.propagate_k, δ ~ Uniform(0, 3))                  ▷ training δ branch [H]
23:       post_update_memory_s2go(prop, ego_t)                                                     ▷ push to head; transform → next ego frame
24:   end for
25:   L ← (λ_1 · L_den + λ_2 · L_dep + λ_3 · L_rgb) / T
26:   L.backward();   clip_grad_norm_(model, max_norm = 35);   optimizer.step();   scheduler.step()  ▷ paper §B
27:   return  L
─────────────────────────────────────────────────────────────────────────────────
```

---

## Algorithm 2 — Stage-1 full training schedule (wraps Algorithm 1)

```
─────────────────────────────────────────────────────────────────────────────────
Algorithm 2   S2GO-Small Stage-1 Training (12 epochs)
─────────────────────────────────────────────────────────────────────────────────
Input:   model M with R50 backbone init from torchvision://resnet50
         train_loader  (mini-sequences of T = 4 contiguous frames)
         val_loader
         n_epochs = 12                                                                              ▷ paper §B
Output:  checkpoint stage1_final.pth
─────────────────────────────────────────────────────────────────────────────────
 1:   optimizer ← AdamW(M.params, lr = 4·10^−4, weight_decay = 0.01)                              ▷ paper §B
 2:                  with backbone lr scaled × 0.25
 3:   scheduler ← CosineAnnealing(optimizer, T_max = n_epochs · |train_loader|)
 4:   for e = 1, …, n_epochs do
 5:       for batch in train_loader do
 6:           L ← StageOneTrainStep(M, batch, optimizer)                                            ▷ Algorithm 1
 7:           log({epoch: e, loss: L})
 8:       end for
 9:       evaluate_render_quality(M, val_loader)                                                    ▷ qualitative depth + RGB
10:       save_checkpoint(M, "stage1_e" + e + ".pth")
11:   end for
12:   save_checkpoint(M, "stage1_final.pth")
13:   return  M                                                                                     ▷ next: load into Stage-2 config;
                                                                                                    ▷ swap lifter mode='fps_eps' → 'learnable',
                                                                                                    ▷ swap child_head mode='rgb' → 'semantic'.
─────────────────────────────────────────────────────────────────────────────────
```

---

## Algorithm 3 — TopK_δ_Opacity propagator (used by line 22 of Algorithm 1)

```
─────────────────────────────────────────────────────────────────────────────────
Algorithm 3   Opacity-δ Propagator (block [H])
─────────────────────────────────────────────────────────────────────────────────
Input:   parent.{xyz, opa, feat, velo}, k = cfg.propagate_k = 256, training flag
Output:  prop.{xyz, opa, feat, velo}  with |prop| = k
─────────────────────────────────────────────────────────────────────────────────
 1:   if training then  δ ~ Uniform(0, 3)  else  δ ← cfg.delta_eval = 1.6  end if                   ▷ paper §B; eval value REQUIRED
 2:   idx ← argsort(parent.opa, descending)                                                         ▷ candidates by opacity
 3:   keep ← []                                                                                     ▷ greedy mutual-distance prune
 4:   for i in idx do
 5:       if min_{j ∈ keep} ‖ parent.xyz[i] − parent.xyz[j] ‖_2  ≥ δ then
 6:           keep ← keep ∪ {i}
 7:       end if
 8:       if |keep| = k then  break  end if
 9:   end for
10:   return  parent[keep]                                                                          ▷ gather xyz, opa, feat, velo
─────────────────────────────────────────────────────────────────────────────────
```

---

## Loss definitions (math)

The three terms in line 18, 19, 21 of Algorithm 1, with the loss bodies:

```
L_denoise  =  Σ_i  ‖ a_t^i  −  (p_t^i + o_t^i) ‖_1                              (Eq. 8 first term)

L_depth    =  ⟨ |D̂ − D^LiDAR| · 𝟙[D^LiDAR > 0] ⟩                                masked-mean L1

L_rgb      =  0.85 · ⟨|Î − I|⟩  +  0.15 · (1 − SSIM(Î, I))                      3DGS Eq. 7

L_total    =  λ_1 · L_denoise  +  λ_2 · L_depth  +  λ_3 · L_rgb                 Eq. 8
              with λ_1 = 10,  λ_2 = 1,  λ_3 = 1   (defaults — not paper-pinned)
```

**Gradient flow** (which params each term touches):

| Term | Backbone | Decoder | Parent refiner | Child head | init_feat |
|---|:---:|:---:|:---:|:---:|:---:|
| L_denoise | – (indirect via decoder) | ✓ | ✓ (offset only) | – | ✓ |
| L_depth   | ✓ | ✓ | ✓ (offset, opa, **velocity**) | ✓ | ✓ |
| L_rgb     | ✓ | ✓ | ✓ (offset, opa, **velocity**) | ✓ (incl. RGB) | ✓ |

The velocity head receives gradients **only** through the ±0.5s render warps in lines 15–20 of Algorithm 1. There is no explicit `L_velocity` — velocity is supervised implicitly. This is by design (paper §3.3.3 / Table 5).

---

## Line-to-code mapping (where each step lives)

| Algorithm 1 line | Module / file | Status |
|---|---|---|
| 1, 9, 23 | StreamPETR `reset_memory` / `pre_update_memory` / `post_update_memory` ([streampetr_head.py:312-374](reference_code/StreamPETR/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py#L312-L374)) | port (~150 LoC) |
| 4 | `BEVSegmentor.extract_img_feat` ([bev_segmentor.py:40-69](reference_code/GaussianFormer/model/segmentor/bev_segmentor.py#L40-L69)), strip SECONDFPN | reuse |
| 5–8 | `S2GOLifter.fps_eps` (uses `torch_cluster.fps`) | new (~30 LoC) |
| 10–11 | `PETRTemporalTransformer` ([petr_transformer.py:423](reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/petr_transformer.py#L423)) + `PETRTemporalDecoderLayer` (L513) — past-query concat at [L707-723](reference_code/StreamPETR/projects/mmdet3d_plugin/models/utils/petr_transformer.py#L707-L723) | reuse |
| 12 | `ParentRefiner` (extends [refine_module_v2.py](reference_code/GaussianFormer/model/encoder/gaussian_encoder/refine_module_v2.py)) | new (~50 LoC) |
| 13 | `ChildGaussianHead` | new (~80 LoC) |
| 14 | `assemble_gaussians` | new (~30 LoC) |
| 16 | `ego_warp` | new (~20 LoC) |
| 17 | `gsplat_wrapper.render` (wraps `gsplat.rasterization`) | new (~100 LoC) |
| 18 | `DepthRenderLoss` | new (~15 LoC) |
| 19 | `RGBRenderLoss` (cribbed from 3DGS train.py + kornia SSIM) | new (~20 LoC) |
| 21 | `DenoiseLoss` | new (~10 LoC) |
| 22 | `OpacityDeltaPropagator` (Algorithm 3) | new (~30 LoC) |
| 26 | mmengine `MultiLoss` aggregator + AdamW + cosine LR | reuse (GF-2 train.py harness) |

**Total new code for Stage 1: ~535 LoC**, plus configs.

---

## Sanity-check assertions for the first iteration

After running Algorithm 1 once on a single batch (overfit setting) — verify before scaling to a full epoch:

```
  L > 0     ∧ ¬isnan(L)                                                            (1)
  ∀ p ∈ M.parameters() with p.requires_grad:  p.grad ≠ None                         (2)
  child_head.expand.weight.grad ≠ None                                              (3)   ▷ Stage-1 RGB path active
  parent_refiner.velocity_head.weight.grad ≠ None                                   (4)   ▷ render warps reaching velocity
  decoder.layers[-1].attentions[0].weight.grad ≠ None                               (5)   ▷ self-attn alive
  shape(𝒢.means)  = (B, K · J = 9000, 3)                                            (6)
  shape(D̂)        = (B, N_cam = 6, H = 256, W = 704)                                (7)
  shape(Î)        = (B, N_cam = 6, 3, H = 256, W = 704)                             (8)
```

Common failure modes if any of (1)–(8) breaks:

| Symptom | Likely cause |
|---|---|
| (1) NaN | gsplat received non-contiguous tensor, or SSIM input outside [0, 1] |
| (3) zero | ChildGaussianHead is using `mode='semantic'` (Stage-2 path) by accident |
| (4) zero | velocity not flowing into `means += v · dt` — check `ego_warp` |
| (6) wrong | `parent.offset` not broadcasting over J children in `assemble_gaussians` |
| (5) zero with (1)–(4) fine | `prev_exists` not propagated → memory wiped every frame, no temporal signal |

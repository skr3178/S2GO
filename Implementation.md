# S2GO — conceptual background, Stage-2 pseudocode, scratchpad

> Reference companion to:
> - [architecture.md](architecture.md) — full pipeline diagram
> - [Stage1_design.md](Stage1_design.md) — Stage-1 engineering specifics + decision table + attn_cfgs
> - [Stage1_pseudocode.md](Stage1_pseudocode.md) — Stage-1 algorithmic walkthrough
> - [kernel/SUMMARY.md](kernel/SUMMARY.md) — Stage-2 voxel splatter (built, validated)
>
> This file holds paper-to-implementation context and the **Stage-2 training
> pseudocode** (its Stage-1 counterpart now lives in
> [Stage1_pseudocode.md](Stage1_pseudocode.md)). The earlier "kernel
> implementation plan" content has been superseded by `kernel/`.

---

## 1. Online framework

Streaming-perception terminology: at timestep `t` the model uses **only past + current information** with bounded latency, carrying state across frames.

| Mode | Info available | State |
|---|---|---|
| Offline | full sequence past+future | accumulates over whole sequence |
| Single-frame | current frame only | none |
| **Online (S2GO)** | past + current | persistent queue of past queries |

Per frame: take past queries `Q_{t-1..t-4}` + images `I_t` → refine → push subset to queue for `t+1`. The win vs. SurroundOcc/GaussianWorld is that S2GO carries forward **decoded queries** (~1k vectors) instead of re-projecting **image features** (millions of pixels) every frame.

---

## 2. Direct Hungarian matching (DETR) — and why occupancy can't use it

DETR pattern: `N` predictions vs `K << N` ground-truth objects → optimal one-to-one assignment via Hungarian algorithm minimizing `λ_cls·cls + λ_box·L1(box) + …`. "Direct" = no NMS / proposal grid.

**Why occupancy can't use it.** 640k voxel GT vs ~9k Gaussians is a many-to-many mapping with no unique cost matrix. S2GO replaces Hungarian with the denoising-pretraining objective (Eq. 8) — that is what tells each query "where it should be."

---

## 3. Stage-2 training pseudocode

Stage-1 pseudocode lives in [Stage1_pseudocode.md](Stage1_pseudocode.md). This is its Stage-2 counterpart — same architecture, different lifter / per-Gaussian output / loss. Will be fleshed out into a dedicated `Stage2_pseudocode.md` when we get there.

```
# ─── STAGE 2: SEMANTIC OCCUPANCY (24 epochs) ──────────────────────────────
# Prereq: Stage 1 checkpoint loaded; lifter swaps fps_eps → learnable;
#         child_head swaps mode='rgb' → 'semantic'.
load_weights("stage1_epoch_12.pth")

for batch in train_loader:                     # batch = T=4-frame sequence
    memory_queue = []
    for t in range(T):
        # Learnable query positions (NO LiDAR at inference)
        p_init = learnable_query_positions     # nn.Parameter(K, 3)

        feats     = image_encoder(batch.images[t])
        queries   = ego_compensate(memory_queue, batch.ego_pose_inv[t])
        tgt       = temporal_decoder(init_embed(p_init), feats, memory_queue)
        # Stage 2: J children share PARENT query's semantic class (paper §3.2)
        queries_refined, gaussians = parent_child_decode(tgt, p_init,
                                                          mode='semantic')

        # Voxel splatting via our S2GO CUDA kernel ([G2] in architecture.md)
        voxel_logits = g2v_splat(gaussians, grid=(200, 200, 16))   # see kernel/
        L_ce     = CE(voxel_logits, batch.occ_gt[t], ignore=255)
        L_lovasz = lovasz_softmax(voxel_logits, batch.occ_gt[t])

        # paper §3.4.1: "we additionally supervise neighboring frames similar
        #               to Stage 1"
        # paper §3.3.3: rendering done on ±0.5s with velocity warp.
        # NOTE: w_aux=0.1 and w_lovasz=0.25 below are NOT specified in the paper
        # — plausible defaults inherited from GF-2 / 3DGS conventions; tune.
        L_aux = 0
        for dt in [-0.5, +0.5]:
            G_warped = ego_motion_warp(gaussians, tgt.velocity, dt)
            D_hat, I_hat = gsplat_render(G_warped, batch.cams[t+dt])
            L_aux += 0.1 * (L1(D_hat, ...) + L1+SSIM(I_hat, ...))   # 0.1 = guess

        L = L_ce + 0.25 * L_lovasz + L_aux                          # 0.25 = guess
        backprop(L)
        memory_queue.append(topk_by_opacity(queries_refined,
                                            k=cfg.propagate_k,
                                            delta=U(0,3) if training else 1.6))
        memory_queue = memory_queue[-4:]
```

**Key invariant across stages:** identical architecture; only the **lifter mode** (FPS+ε vs learnable), **per-Gaussian output** (RGB vs shared parent class), and **loss** differ. LiDAR is **train-time-only** — never used at inference.

---

## 4. Background papers (lookup)

| Citation | Topic | arXiv | Local file |
|---|---|---|---|
| Carion 2020 | DETR — direct Hungarian matching | [2005.12872](https://arxiv.org/abs/2005.12872) | [papers/1_DETR_Carion2020.pdf](papers/1_DETR_Carion2020.pdf) |
| Wang 2020 | DETR3D — 3D-to-2D query projection | [2110.06922](https://arxiv.org/abs/2110.06922) | [papers/2_DETR3D_Wang2020.pdf](papers/2_DETR3D_Wang2020.pdf) |
| Wang 2023 | StreamPETR — streaming temporal queues *(closest pattern S2GO inherits)* | [2303.11926](https://arxiv.org/abs/2303.11926) | [papers/3_StreamPETR_Wang2023.pdf](papers/3_StreamPETR_Wang2023.pdf) |
| Lin 2022 | Sparse4D — sparse spatio-temporal fusion | [2211.10581](https://arxiv.org/abs/2211.10581) | [papers/4_Sparse4D_Lin2022.pdf](papers/4_Sparse4D_Lin2022.pdf) |
| Yuan 2024 | StreamMapNet — streaming HD-map | [2308.12570](https://arxiv.org/abs/2308.12570) | [papers/5_StreamMapNet_Yuan2024.pdf](papers/5_StreamMapNet_Yuan2024.pdf) |
| Liu 2022 | PETR — basic position-embedding decoder | [2203.05625](https://arxiv.org/abs/2203.05625) | [papers/6_PETR_Liu2022.pdf](papers/6_PETR_Liu2022.pdf) |
| Zhu 2020 | Deformable DETR — `MultiScaleDeformableAttnFunction` op | [2010.04159](https://arxiv.org/abs/2010.04159) | [papers/7_DeformableDETR_Zhu2020.pdf](papers/7_DeformableDETR_Zhu2020.pdf) |

Read order if time-bound: **DETR → StreamPETR → Sparse4D**.

---

## 5. §3.3.2 distilled — pretraining vs occupancy

Same architecture across stages, three differences:

| Element | Stage 2 (§3.2) | Stage 1 (§3.3.2) |
|---|---|---|
| Query positions `p^i` | learnable parameters | `FPS_K(pts) + ε` (Eq. 7) |
| Per-Gaussian color | shared parent semantic class | each Gaussian predicts own RGB |
| Supervision | voxel-CE on dense GT | denoise + render against (D_t, I_t) |

**Why pretraining "unlocks streaming":** random-init queries have no signal telling them to *move* into occupied space → stuck in local minima. Denoising provides explicit per-query target ("land at this anchor"); rendering provides per-Gaussian target ("scale/rotate so depth+RGB match observations"). Stage 2 then inherits this prior and works fine without LiDAR.

---

## 6. Equation 8 — three-term ablation impact

$$\mathcal{L} = \lambda_1 \sum_i \|\mathrm{FPS}_K(\mathbf{pts}_t) - (\mathbf{p}^i + \mathbf{o}_t^i)\|
              + \lambda_2 \mathcal{L}_{depth}(\mathcal{G}, D)
              + \lambda_3 \mathcal{L}_{rgb}(\mathcal{G}, I)$$

| Term | Pulls toward | Removed → drops (Table 3) |
|---|---|---|
| **λ₁ denoise** (L1, FPS anchor vs noised-pos + offset) | queries onto exact LiDAR anchors | ~1.4 mIoU (row e → row f reverse) |
| **λ₂ L_depth** (gsplat depth vs LiDAR-projected sparse depth) | Gaussian scales/positions → 3DGS-style geometric consistency | ~7 mIoU (row a → row d/e) |
| **λ₃ L_rgb** (gsplat RGB vs image, L1+SSIM) | Gaussian appearance → photometric consistency | ~0.3 mIoU (row e → row d) |

**Defaults** (pinned in [Stage1_design.md D6](Stage1_design.md)): `λ₁=10, λ₂=1, λ₃=1` — **not paper-specified**. Surface as `cfg.lambda_*`.

**Empirical takeaway from Table 3:** denoising alone gives ~6 of the ~8 mIoU pretraining gain; depth does most of the rest; RGB is refinement. Order of priority for ablation: depth ≫ denoise ≫ rgb.

---

## 7. SKR Notes (scratchpad)

Original handwritten notes that drove the early scoping of this work — kept as-is for context. Most points are now resolved by the docs above.

- *what is online framework mean?*
  > "This online framework enables efficient propagation and global feature interaction among a sparser set of 3D queries (∼1k) while retaining the high fidelity of Gaussian-based representations."
  → resolved in §1 above.

- *Direct Hungarian matching meaning* → resolved in §2 above.

- *Most important implementation caveat (paper §3.3.1):*
  > "To fully unlock the streaming potential of query-based occupancy estimation, we introduce a pretraining phase that first trains the network to capture 3D scene geometry. During pretraining, query locations are initialized at noised LiDAR points, and the network is trained to recover 3D geometry through a denoising objective. To capture fine-grained local shape, decoded Gaussians are rendered from the current and neighboring views and supervised accordingly. The network also predicts a velocity for each query to model dynamic objects. This pretraining addresses the aforementioned challenges of using sparse queries by 1) supervising queries and their decoded Gaussians to model local scene structure, 2) training queries to self-organize to evenly cover the scene, and 3) supervising queries explicitly to move from empty space to occupied regions. Then, during the following semantic occupancy estimation stage – when LiDAR is no longer used and queries are randomly initialized throughout the 3D scene – the network uses its pretrained knowledge to precisely reposition the queries and decode to Gaussians to capture dense 3D structure."

- *Lets first write the training algorithm in pseudocode* → done: §3 above (Stage 2) and [Stage1_pseudocode.md](Stage1_pseudocode.md) (Stage 1).

- *Download the papers* (Carion 2020, Wang 2020/2023, Lin 2022, Yuan 2024) → done; see §4 above.

- *Denoising and rendering frameworks — build the section 3.3.2* → done; see §5 above.

- *Write loss equation terms — equation 8* → done; see §6 above.

- *Standalone script for §3.4.3 efficient G2V splatting:*
  > "To implement Gaussian-to-voxel splatting, GaussianFormer first determines pairs of interacting Gaussians and voxels, then parallelizes over voxels in the forward pass and over Gaussians in the backward pass. However, this formulation does not account for the inherent locality of the splatting operation: neighboring voxels process a similar set of Gaussians and vice-versa. Such voxels and Gaussians should be processed together in a CUDA block for optimized L1 cache usage. This is especially a problem for the backward pass since naively parallelizing over Gaussians incurs random access costs on a large number of voxels (640k). To address this problem in the forward pass, we block voxels into 4×4×4 grids and have threads tied to each voxel collaboratively load nearby Gaussians onto memory before splatting them, similar to 3DGS. In the backward pass, we adopt a similar approach but additionally take care to tie threads to individual Gaussians to avoid atomic operations on the gradients."

  → **DONE.** Implemented at [kernel/localagg_s2go_tiled/](kernel/localagg_s2go_tiled/), validated, benchmarked. Speedups + ncu profile in [kernel/SUMMARY.md](kernel/SUMMARY.md).

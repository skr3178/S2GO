# Stage 1 vs Stage 2 — change summary

**Purpose:** a compact delta document. Stage 1 is depth/RGB rendering pretraining; Stage 2 is semantic occupancy training. Stage 2 reuses the Stage 1 architecture wholesale and changes only the *output side*, the *query-init side* (paper §3.4.1), and the *loss/data side*. Nothing in `s2go/models/`, `s2go/datasets/`, `s2go/losses/`, `s2go/render/`, or Stage 1's training/eval scripts is edited.

**Update 2026-05-13 — Gap 1 + Gap 8 fixes applied:**
- **Gap 1 (paper §3.4.1):** Stage 2 no longer uses LiDAR FPS+ε query init. A new `Stage2Lifter` ([stage2_lifter.py](stage2_lifter.py)) provides learnable query positions (`nn.Parameter(K, 3)`) and is hot-swapped onto `self.segmentor.lifter` by `S2GOStage2.__init__` when `query_init='learned'` (new default). The lidar input is still passed through `S2GOSegmentor.forward_one_frame` for signature parity but is ignored by the Stage 2 lifter. Selectable via `--query-init {learned, fps_lidar}` in the training CLI.
- **Gap 8 (bridge bug):** `load_stage1_state` now **partial-copies the first 11 rows** of `child_head.head.weight/bias` from Stage 1 (offset/scale/rot/opa projections) into Stage 2 instead of dropping the whole head. Only the trailing 3 RGB rows are discarded. Stage 1's 5000 iters of learned child-geometry projection now survive the bridge.

---

## TL;DR (one paragraph)

Stage 1 trains a backbone + segmentor + per-child *RGB* head to **render** RGB+depth via gsplat against LiDAR-depth and image GT. Stage 2 reuses the same backbone + segmentor (same params, same shapes), flips one constructor flag on `ChildGaussianHead` to drop the RGB output, adds a small `SemanticHead` on top of `parent.feat` to emit per-parent class logits, and runs a custom CUDA **Gaussian-to-Voxel** splatter to produce a dense voxel grid that's supervised against Occ3D-style semantic labels with an occupancy-BCE + (KL + CE + Lovász) loss. The two stages share the architecture and run sequentially; Stage 2 initialises from the Stage 1 `.pt` checkpoint (with a single-key shape mismatch on `child_head.head.*` that is dropped at load).

---

## 1. Architectural diff (only two things change)

### 1.1 `ChildGaussianHead` — mode flip

Same class in [s2go/models/encoder/heads.py:99-163](../models/encoder/heads.py#L99). The constructor arg `mode` swaps the last 3 output dims:

| Arg | Stage 1 | Stage 2 |
|---|---|---|
| `mode` | `'rgb'` | `'semantic'` |
| `head` Linear out dim | **14** = offset(3) + scale(3) + rot(4) + opa(1) + **rgb(3)** | **11** = offset(3) + scale(3) + rot(4) + opa(1) |
| `head` params | 10,766 | 8,459 |
| Δ | — | **−2,307 params** (the RGB rows) |

All other ChildGaussianHead config — `embed_dims=768`, `J=10`, `child_unit_xyz=(1, 1, 0.5)`, `scale_range=(0.01, 2.5)`, the `expand = Linear(768, 7680)` trunk — is identical.

**Exact `__init__` — the mode-conditioned branch:**

```python
# s2go/models/encoder/heads.py:114-140
def __init__(self, embed_dims: int = 768, J: int = 10,
             mode: str = 'rgb',
             num_classes: int = 18,
             child_unit_xyz=(1.0, 1.0, 0.5),
             scale_range=(0.01, 2.5)):
    super().__init__()
    assert mode in ('rgb', 'semantic')
    self.embed_dims = embed_dims
    self.J = J
    self.mode = mode
    self.num_classes = num_classes
    self.register_buffer("child_unit_xyz",
                          torch.tensor(child_unit_xyz, dtype=torch.float32))
    self.scale_min, self.scale_max = float(scale_range[0]), float(scale_range[1])

    # Expand parent feature into J child slots (one nn.Linear per parent
    # producing embed_dims*J outputs, matching Stage1_design.md §5).
    self.expand = nn.Linear(embed_dims, embed_dims * J)
    # Per-child final projection. RGB is Stage-1-only; Stage-2 emits a parent-
    # shared semantic class instead (handled separately at the parent level).
    # The child head only differs in its last 3 dims.
    if mode == 'rgb':
        self.head = nn.Linear(embed_dims, 3 + 3 + 4 + 1 + 3)        # 14
    else:  # 'semantic'
        self.head = nn.Linear(embed_dims, 3 + 3 + 4 + 1)             # 11
```

**Exact `forward` — last 3 dims diverge:**

```python
# s2go/models/encoder/heads.py:142-163
def forward(self, parent_feat: torch.Tensor) -> ChildPred:
    B, K, d = parent_feat.shape
    # (B, K, d) -> (B, K, J*d) -> (B, K, J, d)
    x = self.expand(parent_feat).view(B, K, self.J, d)
    out = self.head(x)                                              # (B, K, J, 14|11)

    offset = (2.0 * torch.sigmoid(out[..., :3]) - 1.0) * self.child_unit_xyz
    scale = self.scale_min + (self.scale_max - self.scale_min) * torch.sigmoid(out[..., 3:6])
    rot = F.normalize(out[..., 6:10], dim=-1)
    opa = torch.sigmoid(out[..., 10:11])
    if self.mode == 'rgb':
        rgb = torch.sigmoid(out[..., 11:14])
    else:
        rgb = None                       # Stage 2 — per-parent class is set elsewhere
    return ChildPred(offset=offset, scale=scale, rot=rot, opa=opa, rgb=rgb)
```

### 1.2 `SemanticHead` — new in Stage 2

Defined in [s2go/stage2/semantic_head.py:20-51](semantic_head.py#L20). A 2-layer MLP on top of `parent.feat`:

```
parent_feat (B, K=900, 768)
       │
       ▼  Linear(768, 256)         196,864 params
       │
       ▼  GELU
       │
       ▼  Linear(256, 18)            4,626 params       bias[empty_id=17] = log(2.0)
       │
sem_per_parent (B, K=900, 18)      ← class logits
       │
       ▼  unsqueeze(2).expand(-1, -1, J=10, -1).reshape  (broadcast in S2GOStage2)
       │
sem_logits_per_g (B, K·J=9000, 18) ← per-Gaussian class logits
```

Total: **201,490 params** (~0.2 M). No Stage 1 counterpart.

**Exact code — full `SemanticHead` class:**

```python
# s2go/stage2/semantic_head.py:20-51
class SemanticHead(nn.Module):
    """Two-layer MLP: parent_feat (B, K, D) -> class logits (B, K, C)."""

    def __init__(self,
                 feat_dim: int = 768,
                 num_classes: int = NUM_CLASSES,      # 18
                 hidden: int = 256,
                 empty_id: int = EMPTY_CLASS_ID):     # 17
        super().__init__()
        self.num_classes = num_classes
        self.empty_id = empty_id
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, num_classes)

        # Output bias prior: small positive bump on the empty class.
        # log(2.0) for empty, 0 elsewhere → softmax ≈ ~10% empty, ~5% per other
        # class at init (for C=18). Keeps initial loss in a sane range and
        # avoids saturating any one logit before training begins.
        nn.init.zeros_(self.fc2.bias)
        with torch.no_grad():
            self.fc2.bias[empty_id] = float(torch.log(torch.tensor(2.0)))

    def forward(self, parent_feat: torch.Tensor) -> torch.Tensor:
        """parent_feat: (B, K, D)  ->  logits: (B, K, C)."""
        return self.fc2(self.act(self.fc1(parent_feat)))
```

**Where it sits in `S2GOStage2.forward_one_frame` — broadcast to per-Gaussian:**

```python
# s2go/stage2/stage2_segmentor.py:62-73
def forward_one_frame(self, frame: Dict[str, torch.Tensor]) -> Stage2FrameOutput:
    out: FrameOutput = self.segmentor.forward_one_frame(frame)
    # parent.feat: (B, K, D); per-parent logits (B, K, C)
    sem_parent = self.semantic_head(out.parent.feat)
    B, K, C = sem_parent.shape
    # Broadcast over J=10 children → (B, K*J, C). Matches how
    # `assemble_gaussians` broadcasts parent.opa and parent.velocity.
    sem_per_g = (sem_parent.unsqueeze(2)
                 .expand(-1, -1, self.J, -1)
                 .reshape(B, K * self.J, C)
                 .contiguous())
    return Stage2FrameOutput(raw=out, sem_logits_per_g=sem_per_g)
```

### 1.2.5 `Stage2Lifter` — query init swap (Gap 1 fix, paper §3.4.1)

Defined in [s2go/stage2/stage2_lifter.py](stage2_lifter.py). Replaces the inherited `S2GOLifter` (FPS+ε on LiDAR) at runtime so Stage 1 segmentor code remains untouched. Constructor in `S2GOStage2.__init__`:

```python
# s2go/stage2/stage2_segmentor.py
if query_init == 'learned':
    self.segmentor.lifter = Stage2Lifter(
        K=self.segmentor.K,
        embed_dims=embed_dims,
        init_range=stage2_init_range,   # default = Occ3D AABB
    )
```

**Architecture (signature-compatible drop-in for `S2GOLifter`):**

```python
# s2go/stage2/stage2_lifter.py — key fields
self.query_xyz  = nn.Parameter(torch.empty(K, 3))     # uniform-init in init_range
self.query_feat = nn.Parameter(torch.empty(K, embed_dims))  # σ=0.02 normal init

def forward(self, pts, add_noise=True):
    # pts is ACCEPTED AND IGNORED — kept for signature parity.
    B = pts.shape[0]
    init_xyz  = self.query_xyz.unsqueeze(0).expand(B, -1, -1).contiguous()
    init_feat = self.query_feat.unsqueeze(0).expand(B, -1, -1).contiguous()
    return init_xyz, init_xyz, init_feat   # anchors_xyz == init_xyz (no denoise loss)
```

**Key properties:**
- Same name + shape `(K, embed_dims)` for `query_feat` as `S2GOLifter.query_feat` → Stage 1's `lifter.query_feat` weight transfers cleanly via `load_state_dict(strict=False)`.
- New `query_xyz` parameter unique to Stage 2 → shows up as `missing` on bridge load (intended).
- Total: **K·3 + K·768 ≈ 693K params** (vs S2GOLifter's K·768 ≈ 691K — just K extra for the xyz).

### 1.3 What does NOT change

The full inventory of reused Stage 1 modules:

| Module | File | Change |
|---|---|---|
| R50 + FPN backbone | [s2go/models/backbone/r50_fpn.py](../models/backbone/r50_fpn.py) | None |
| `S2GOLifter` (FPS + ε on LiDAR) | [s2go/models/lifter/s2go_lifter.py](../models/lifter/s2go_lifter.py) | **Class unchanged**, but at runtime Stage 2 swaps `self.segmentor.lifter` for `Stage2Lifter` when `query_init='learned'` (paper §3.4.1, Gap 1 fix) |
| `TemporalDecoder` (deformable cross-attn ×L) | [s2go/models/encoder/temporal_decoder.py](../models/encoder/temporal_decoder.py) | None |
| `ParentRefiner` | [s2go/models/encoder/heads.py](../models/encoder/heads.py) | None |
| `ChildGaussianHead` | [s2go/models/encoder/heads.py](../models/encoder/heads.py) | **constructor arg only** (`mode`) |
| `assemble_gaussians` | [s2go/models/encoder/assembly.py](../models/encoder/assembly.py) | None — `Gaussians.colors` is `None` in Stage 2 |
| `MemoryQueue` | [s2go/models/queue/memory.py](../models/queue/memory.py) | None |
| `OpacityDeltaPropagator` | [s2go/models/queue/propagator.py](../models/queue/propagator.py) | None |
| `S2GOSegmentor` | [s2go/models/segmentor.py](../models/segmentor.py) | None — composed by `S2GOStage2`, passed `child_mode='semantic'` |
| `pos_mlp` (3→d embedding) | inside `S2GOSegmentor` | None |
| `NuScenesLoader` | [s2go/datasets/nusc_loader.py](../datasets/nusc_loader.py) | None — wrapped by `Stage2OccLoader` |

---

## 2. Loss diff

### Stage 1 (S2GO Eq. 8, depth-only barebones recipe)

| Term | Description |
|---|---|
| **L_denoise** | L1 between `init_xyz` (FPS+ε noised) and `init_xyz + parent.offset` (refined). Trains parents to denoise their own positions back toward LiDAR. *Skipped entirely in the barebones recipe.* |
| **L_depth** | L1 between rendered depth (via gsplat with `render_mode='D'`) and sparse LiDAR-projected depth maps, at dt=0 (current frame). Always on. |
| **L_rgb** | L1 + (1−SSIM) on rendered RGB vs. camera image. *Off in barebones.* |
| **±0.5 s warp renders** | Optional renders at neighbour timesteps via velocity-warped Gaussians. Trains the velocity head. *Off in barebones.* |

v2 ckpt (`out/barebones_full_part1_5000iter_v2/ckpt.pt`) used: **L_depth only**, at dt=0, no warps, no RGB.

### Stage 2 (paper §3.4, voxel-space)

| Term | Default weight | Description |
|---|---|---|
| **`occupancy_bce`** | `w_occ=1.0`, **`pos_weight=16`** | Weighted BCE between predicted occupancy probability (G2V output Eq. 1) and binary GT occupancy mask. `pos_weight=16` ≈ class-frequency ratio. |
| **`kl_semantic`** | `w_kl=1.0` | KL(softmax(logits) ‖ one-hot(GT)). For one-hot GT collapses to NLL. `ignore_index=EMPTY_CLASS_ID=17`. |
| **`ce_semantic`** | `w_ce=10.0` | Weighted cross-entropy (optional class weights). `ignore_index=EMPTY`. |
| **`lovasz_semantic`** | `w_lovasz=1.0` | Lovász-softmax — IoU surrogate over present classes. `ignore_index=EMPTY`. |

Stage 1's depth-rendering loss is **not** computed in Stage 2. Stage 2's voxel loss is **not** computed in Stage 1.

**Exact code — Stage 2 occupancy BCE with positive-class weighting:**

```python
# s2go/stage2/losses.py:34-55
def occupancy_bce(occ_pred: torch.Tensor,
                  occ_gt: torch.Tensor,
                  pos_weight: float = 1.0) -> torch.Tensor:
    """Per-voxel BCE between predicted occ_prob ∈ (0, 1) and GT binary mask.

    pos_weight: scalar weight on the positive-class loss term. Recommended
                ≈ N_empty / N_occ ratio (~16 for typical Occ3D frames).
                Combats imbalance-driven "predict empty everywhere"
                collapse seen on the dense-supervision path.
    """
    eps = 1e-6
    p = occ_pred.clamp(eps, 1 - eps)
    y = occ_gt.float()
    # standard form: -[ pos_weight * y * log p + (1-y) * log(1-p) ]
    loss = -(pos_weight * y * torch.log(p) + (1 - y) * torch.log(1 - p))
    return loss.mean()
```

**Exact code — Stage 2 loss combiner (`compute_stage2_loss`):**

```python
# s2go/stage2/losses.py:149-205
def compute_stage2_loss(occ_pred, sem_logits, occ_gt, sem_gt,
                          *,
                          sem_loss: str = 'both',
                          w_occ: float = 1.0,
                          w_kl: float = 1.0,
                          w_ce: float = 10.0,
                          w_lovasz: float = 1.0,
                          class_weights=None,
                          ignore_index: int = -1,
                          occ_pos_weight: float = 1.0):
    assert sem_loss in ('kl', 'ce_lovasz', 'both')
    diag = {}

    L_occ = occupancy_bce(occ_pred, occ_gt, pos_weight=occ_pos_weight)
    diag['occ_bce'] = float(L_occ.item())
    total = w_occ * L_occ

    L_kl = torch.zeros((), device=occ_pred.device, dtype=occ_pred.dtype)
    L_ce = torch.zeros_like(L_kl)
    L_lz = torch.zeros_like(L_kl)

    if sem_loss in ('kl', 'both'):
        L_kl = kl_semantic(sem_logits, sem_gt, ignore_index=ignore_index)
        total = total + w_kl * L_kl
    if sem_loss in ('ce_lovasz', 'both'):
        L_ce = ce_semantic(sem_logits, sem_gt, class_weights=class_weights,
                            ignore_index=ignore_index)
        L_lz = lovasz_semantic(sem_logits, sem_gt, ignore_index=ignore_index,
                                classes='present')
        total = total + w_ce * L_ce + w_lovasz * L_lz

    diag['sem_kl']     = float(L_kl.item())
    diag['sem_ce']     = float(L_ce.item())
    diag['sem_lovasz'] = float(L_lz.item())
    diag['total']      = float(total.item())
    return total, diag
```

Stage 2 smoke runs use `sem_loss='both'`, `occ_pos_weight=16`, `ignore_index=EMPTY_CLASS_ID=17`.

---

## 3. Data diff

| | Stage 1 | Stage 2 |
|---|---|---|
| **Loader** | `NuScenesLoader` | `Stage2OccLoader` wraps `NuScenesLoader` (no edits) |
| **Per-frame GT consumed** | `lidar_depth` (6×H×W sparse depth maps from LiDAR projection); `imgs` for RGB loss | `sem_voxel_gt` (Vx, Vy, Vz) int64 + `occ_voxel_gt` bool — derived from `.npy` voxel labels |
| **GT source on disk** | nuScenes `samples/LIDAR_TOP/*.pcd.bin` + `samples/CAM_*/*.jpg` | `data/nuscenes_occ/nuscenes_occ/samples/*.npy` (SurroundOcc-generated Occ3D voxels) |
| **Usable seqs (Part-1)** | 3,376 T=1 sequences | 914 with matching `.npy` GT |
| **Voxel grid** | not used | 200 × 200 × 16, voxel 0.5 m, pc_range [-50,-50,-5, 50,50,3] |
| **Class scheme** | n/a | 18 classes (17 semantic 0..16 + 1 empty=17) |

---

## 4. Eval diff

| | Stage 1 ([stage1_eval.py](../tools/stage1_eval.py)) | Stage 2 ([stage2_eval.py](../tools/stage2_eval.py)) |
|---|---|---|
| **Forward output checked** | Rendered RGB + depth (via gsplat) | Voxel occupancy + argmax class (via G2V) |
| **Quantitative metric** | Nearest-LiDAR distance per query (mean/median/min/max), opacity histograms, scale histograms | **mIoU** + **occupancy IoU** (per-class + aggregate) |
| **Qualitative artefacts** | `eval_A_bev.png` (BEV scatter), `eval_B_renders.png` (rendered vs GT depth/RGB), `eval_C_histograms.png` | `eval_A_bev.png` (top-down argmax), `eval_B_voxel_slices.png` (z-slices GT vs pred), `eval_C_confusion.png` |
| **Reference for metric math** | gsplat outputs, paper §3.3 | MonoScene's `SSCMetrics` (verified in §7 of `Stage2_summary.md`) |

---

## 5. Training recipe diff (current settings)

| | Stage 1 v2 (5000-iter streaming) | Stage 2 smoke (50-iter overfit) |
|---|---|---|
| Optimizer | AdamW (lr_seg=2e-4, lr_bb=5e-5, wd=0.01) | AdamW (lr_seg=2e-4, lr_bb=5e-5, wd=0.01) — same |
| LR schedule | Constant | Constant |
| Grad clip | 10 (NaN/inf guarded) | 10 (NaN/inf guarded) |
| Precision | bf16 autocast (segmentor); fp32 (backbone, gsplat) | bf16 autocast (segmentor); fp32 (backbone, G2V) |
| Mixed-precision dtype | bf16 | bf16 |
| Iters | 5,000 (≈ 1.5 epochs over 3,376 seqs) | 50 (cycling 4 seqs × 12.5) |
| `T_seq`, `T_queue` | 1, 1 (barebones) | 1, 1 |
| Backbone freeze | No | No |
| Architecture | **paper-spec T2**: num_layers=6, num_pts=13, ffn=3072 | **T0-tier**: num_layers=2, num_pts=4, ffn=2048 (default when `--from-scratch`) |
| Trainable params | ~108 M total | ~70 M total |
| Peak GPU memory | 4.24 GB | 3.75 GB (CUDA G2V dense) / 5.22 GB (torch sparse) |

The Stage 2 smoke deliberately uses T0-tier *only because* `--from-scratch` skips Stage 1 ckpt loading. The next-step bridge run will use paper-spec auto-detected from the v2 ckpt's config.

---

## 6. The "bridge" — how Stage 1 ckpt becomes Stage 2 init

Implemented in [stage2_segmentor.py:78-103](stage2_segmentor.py#L78) (`S2GOStage2.load_stage1_state`) and used by [stage2_overfit.py:122-160](../tools/stage2_overfit.py#L122). Three steps:

**Step 1 — arch detection in the training script** ([stage2_overfit.py:122-148](../tools/stage2_overfit.py#L122)):

```python
# s2go/tools/stage2_overfit.py:122-148
if from_scratch:
    # T0-tier sizing: fits on 12 GB with full G2V graph (vs paper-spec
    # which holds ~11 GB of activations in autograd alone). Matches the
    # defaults of `s2go.tools.overfit`'s smoke runs.
    print("[2/5] From-scratch T0-tier arch (no Stage-1 ckpt)")
    arch = dict(K=900, J=10, embed_dims=768,
                 num_layers=2, num_pts=4, feedforward_channels=2048)
    ckpt = None
else:
    print(f"[2/5] Reading Stage-1 ckpt arch: {stage1_ckpt}")
    ckpt = torch.load(stage1_ckpt, map_location=device)
    s1_cfg = ckpt.get('config', {})
    arch = dict(
        K=int(s1_cfg.get('K', 900)),
        J=int(s1_cfg.get('J', 10)),
        embed_dims=int(s1_cfg.get('embed_dims', 768)),
        num_layers=int(s1_cfg.get('num_layers', 6)),
        num_pts=int(s1_cfg.get('num_pts', 13)),
        feedforward_channels=int(s1_cfg.get('feedforward_channels', 3072)),
    )
```

**Step 2 — backbone direct load** ([stage2_overfit.py:151-153](../tools/stage2_overfit.py#L151)):

```python
# s2go/tools/stage2_overfit.py:151-153
if ckpt is not None:
    # Backbone weights load directly (R50+FPN shape didn't change)
    missing_b, unexpected_b = backbone.load_state_dict(ckpt['backbone'],
                                                        strict=False)
```

**Step 3 — segmentor bridge with partial child-head transfer (Gap 8 fix):**

```python
# s2go/stage2/stage2_segmentor.py — current
def load_stage1_state(self, ckpt: dict, strict: bool = False):
    """Notes (post-fix for Gap 8):
        - Stage 1's child_head.head is Linear(d, 14); rows correspond to:
            rows 0..2  : child offset
            rows 3..5  : child scale
            rows 6..9  : quaternion rotation
            row  10    : child opacity
            rows 11..13: RGB (Stage 1 only)
          The first 11 rows have identical meaning between stages →
          we COPY them so 5000 iters of Stage 1's learned child-geometry
          projection survive the bridge. RGB rows (11..13) are dropped.
        - SemanticHead has no Stage 1 counterpart → fresh init.
        - When query_init='learned', Stage 1's `lifter.query_feat` loads
          cleanly into Stage2Lifter.query_feat (same name + shape (K, d));
          Stage2Lifter.query_xyz has no Stage 1 counterpart and stays at
          its uniform-AABB init.

    Returns:
        (missing, unexpected, head_partial_transferred: bool)
    """
    seg_state = dict(ckpt['segmentor'])

    # Pop the mismatched child_head.head keys before load_state_dict so
    # strict=False doesn't simply ignore them — we want to actively
    # partial-copy them afterwards.
    s1_head_w = seg_state.pop('child_head.head.weight', None)
    s1_head_b = seg_state.pop('child_head.head.bias',   None)

    missing, unexpected = self.segmentor.load_state_dict(seg_state, strict=strict)

    head_partial_transferred = False
    if s1_head_w is not None and s1_head_b is not None:
        s2_w = self.segmentor.child_head.head.weight   # (11, d)
        s2_b = self.segmentor.child_head.head.bias     # (11,)
        with torch.no_grad():
            s2_w.copy_(s1_head_w[:11].to(s2_w.device, s2_w.dtype))
            s2_b.copy_(s1_head_b[:11].to(s2_b.device, s2_b.dtype))
        head_partial_transferred = True

    return missing, unexpected, head_partial_transferred
```

**Verified against the real v2 ckpt** (`out/barebones_full_part1_5000iter_v2/ckpt.pt`):
- `s2_w[:11]` exactly matches `s1_w[:11]` after the call (allclose=True).
- `s2_b[:11]` exactly matches `s1_b[:11]` (allclose=True).
- Both differ from their random init (changed_from_init=True).
- With `query_init='learned'`: `unexpected=0` keys (Stage 1's `lifter.query_feat` loaded cleanly into `Stage2Lifter.query_feat`), `missing=[child_head.head.weight, child_head.head.bias, lifter.query_xyz]` — first two are populated by the partial copy, the third is intentionally fresh.

**Before the Gap 8 fix:** the entire `child_head.head` was dropped, so Stage 2 started with a random-init `Linear(d, 11)` head. 5000 iters of Stage 1's learned offset/scale/rot/opa projection were thrown away on the bridge.

After load:
- **`SemanticHead`** has no Stage 1 counterpart → stays at fresh init (empty-class-biased bias).
- **`ChildGaussianHead.head`** is freshly initialised (only the small 11-dim Linear).
- All other segmentor params are warm-started from Stage 1's 5000 iters of depth-rendering training.

Optimization: single AdamW over all params (backbone + segmentor + semantic head). **No freezing**. lr_bb = lr_seg × 0.25 = 5e-5 (same as Stage 1).

→ **Smoke runs have not exercised this path yet.** Both ran with `--from-scratch` because of an earlier memory OOM at paper-spec; the CUDA G2V backend now frees enough memory to retry.

---

## 7. Code-path diff (new files only)

| Path | Status |
|---|---|
| `s2go/stage2/__init__.py` | new |
| `s2go/stage2/semantic_head.py` | new |
| `s2go/stage2/g2v.py` (torch + CUDA backends + factory) | new |
| `s2go/stage2/stage2_segmentor.py` | new (wraps Stage 1 segmentor) |
| `s2go/stage2/losses.py` | new |
| `s2go/stage2/occ_dataset.py` | new (wraps `NuScenesLoader`) |
| `s2go/stage2/miou.py` | new |
| `s2go/tools/stage2_overfit.py` | new |
| `s2go/tools/stage2_eval.py` | new |
| **Stage 1 code under `s2go/models/`, `s2go/datasets/`, `s2go/losses/`, `s2go/render/`, `s2go/tools/{overfit.py, stage1_eval.py}`** | **unchanged** |

---

## 8. Side-by-side cheat sheet

| Dimension | Stage 1 (v2 ckpt) | Stage 2 (smoke v2) |
|---|---|---|
| Goal | Geometric Gaussian pretraining (depth render fits LiDAR) | Semantic occupancy prediction (voxel argmax = correct class) |
| Output | `Gaussians(B, 9000, ...)` rendered to depth+RGB images | Voxel grid `(200, 200, 16)` semantic argmax |
| ChildHead mode | `'rgb'` (Linear → 14) | `'semantic'` (Linear → 11) |
| Extra heads | none | `SemanticHead` (MLP, 0.2 M params) |
| Loss | L_depth (+ optional L_denoise, L_rgb, ±0.5s warps) | occupancy BCE + KL + CE + Lovász |
| GT data | LiDAR depth, camera RGB | Occ3D `.npy` voxel labels |
| Renderer / splatter | **gsplat** (3DGS, image-space) | **`local_aggregate_s2go` CUDA kernel** (G2V, voxel-space) |
| Metric reported | nearest-LiDAR distance, opacity histograms | **mIoU**, **occupancy IoU** |
| Iters trained | 5,000 over 3,376 seqs | 50 over 4 seqs |
| Arch used | paper-spec (6 / 13 / 3072) | T0-tier (2 / 4 / 2048) — bridge run will be paper-spec |
| Stage init | ImageNet pretrained R50 only | Either ImageNet R50 only (`--from-scratch`) OR Stage 1 v2 ckpt warm-start (bridge, deferred) |
| Final result | L_depth 14.4 → 2.45 m (best @ iter 2300) | total loss 35.5 → 11.4; mIoU 5.88%, occ-IoU 16.12% (smoke-grade only) |

---

## Related documents

- [Stage2_plan.md](Stage2_plan.md) — the original design plan
- [Stage2_summary.md](Stage2_summary.md) — full implementation summary with run trajectories, eval results, MonoScene metric verification, and deferred items

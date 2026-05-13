# S2GO Stage 2 implementation plan

## Context

Stage 1 (depth-only Gaussian pretraining) is now stable: the v2 streaming
checkpoint at [out/barebones_full_part1_5000iter_v2/ckpt.pt](out/barebones_full_part1_5000iter_v2/ckpt.pt)
completed 5000 iters cleanly (0 NaN skips, L_depth 14.4→2.45m), and the
synthetic 3-Gaussian opacity-gradient test at
[s2go/tools/synthetic_opacity_grad_test.py](s2go/tools/synthetic_opacity_grad_test.py)
confirmed gsplat's backward pathway is correct. The natural next move is
**Stage 2 — semantic occupancy training** (S2GO paper §3.4): per-Gaussian
class head + Gaussian-to-voxel (G2V) splatting + voxel-space loss + mIoU
eval.

**Constraint from user:** Stage 2 lives in its own files/folders so it
can be reviewed/landed independently. No Stage 1 file is touched.
A future "bridge" task will integrate the two pipelines more tightly;
for now, Stage 2 *consumes* the Stage 1 checkpoint as init but doesn't
modify Stage 1 code.

**Decisions (locked in via earlier Q&A):**
- Loss: KL **and** CE+Lovász behind a `--sem-loss {kl,ce_lovasz,both}` flag.
- G2V backend: pure-torch reference from [tests/g2v_reference.py](tests/g2v_reference.py).
  CUDA kernel ([localagg_s2go/](localagg_s2go/local_aggregate_s2go/__init__.py))
  deferred to a follow-up.
- First milestone: smoke test (50 iters × 4 sequences), to validate the
  pipeline wires up and backprops. No mIoU claim yet.
- Init: load Stage 1 ckpt's backbone + segmentor weights, randomly init
  the new semantic head, train all params jointly.

## What's already in place (reusable, no edits)

| Piece | Path | Why reusable |
|---|---|---|
| Pure-torch G2V reference | [tests/g2v_reference.py](tests/g2v_reference.py) | Computes per-voxel occupancy + class mixture from (means, rots, scales, opas, colors). Autograd-friendly. |
| G2V test scaffold | [tests/test_g2v_scaffold.py](tests/test_g2v_scaffold.py) | Shape/grad sanity contract; we can extend rather than rewrite. |
| Semantic mode in child head | [s2go/models/encoder/heads.py](s2go/models/encoder/heads.py) | `ChildGaussianHead(mode='semantic', num_classes=18)` already supported. |
| Stage 1 segmentor with semantic switch | [s2go/models/segmentor.py:79](s2go/models/segmentor.py#L79) | `child_mode='semantic'` is a constructor arg; we just pass it. |
| NuScenesLoader | [s2go/datasets/nusc_loader.py](s2go/datasets/nusc_loader.py) | Yields T-frame sequences keyed by sample_token + lidar_token. We *wrap* (not edit) it. |
| Local Occ3D-style GT | `data/nuscenes_occ/nuscenes_occ/samples/*.npy` | Sparse `(M, 4)` int arrays: (x, y, z, class_id). ~680 files locally; enough for the 4-sequence smoke. |
| Stage 1 ckpt convention | [s2go/tools/overfit.py](s2go/tools/overfit.py) `_build_ckpt()` | Same dict layout: `backbone`, `segmentor`, `config`. We'll mirror it. |
| Eval bundle convention | [s2go/tools/stage1_eval.py](s2go/tools/stage1_eval.py) + [out/barebones_full_part1_5000iter_v2_results/](out/barebones_full_part1_5000iter_v2_results/) | PNG figures + `eval_stats.json` + `SUMMARY.md`. We mirror this for Stage 2. |
| MeanIoU reference | [reference_code/GaussianFormer/misc/metric_util.py:9](reference_code/GaussianFormer/misc/metric_util.py#L9) | Port (don't import — vendored ref).

## What's missing (Stage 2 scope)

| Piece | New file |
|---|---|
| Per-parent semantic classifier | [s2go/stage2/semantic_head.py](s2go/stage2/semantic_head.py) |
| Stage 2 segmentor wrapper | [s2go/stage2/stage2_segmentor.py](s2go/stage2/stage2_segmentor.py) |
| G2V layer (thin nn.Module around the torch reference) | [s2go/stage2/g2v.py](s2go/stage2/g2v.py) |
| Voxel-space losses (occupancy BCE + KL sem + CE/Lovász sem) | [s2go/stage2/losses.py](s2go/stage2/losses.py) |
| Occ3D GT loader / sparse-to-dense scatter | [s2go/stage2/occ_dataset.py](s2go/stage2/occ_dataset.py) |
| mIoU metric | [s2go/stage2/miou.py](s2go/stage2/miou.py) |
| Smoke-test training entry | [s2go/tools/stage2_overfit.py](s2go/tools/stage2_overfit.py) |
| Eval entry | [s2go/tools/stage2_eval.py](s2go/tools/stage2_eval.py) |
| Package marker | [s2go/stage2/__init__.py](s2go/stage2/__init__.py) |

## Pre-flight verification (do FIRST, before writing module code)

Two hard prerequisites that block the plan if wrong:

**P1. Voxel grid extent must match the local GT data.**
Two candidate conventions are in the repo:
- Paper/GaussianFormer: `pc_range=[-50,-50,-5, 50,50,3]`, 200×200×16, 0.5 m
- Existing CUDA wrapper [localagg_s2go/local_aggregate_s2go/__init__.py:118](localagg_s2go/local_aggregate_s2go/__init__.py#L118): `pc_min=[-40,-40,-1]`, grid 200×200×16, 0.4 m → `[-40,40,-40,40,-1,5.4]`

Load 2-3 `.npy` files from `data/nuscenes_occ/nuscenes_occ/samples/` and
inspect `coords[:,0:3].min()/max()` to pick the correct one. Pin the
result as constants in `s2go/stage2/__init__.py` (`PC_RANGE`, `VOXEL_SIZE`,
`GRID_SHAPE`, `NUM_CLASSES`, `EMPTY_CLASS_ID`).

**P2. GT coverage for our 4 smoke-test sequences.**
Walk the 4 sample_tokens used in the Stage 1 smoke test, resolve their
`lidar_token`, and confirm a matching `.npy` exists. If not, swap to 4
sequences where GT is present. Record the chosen tokens in the script.

## Module-by-module design

### `s2go/stage2/__init__.py`
Re-exports + pinned constants from P1:
```python
PC_RANGE      = (..., ..., ...)        # from P1
VOXEL_SIZE    = (..., ..., ...)
GRID_SHAPE    = (Vx, Vy, Vz)           # e.g. (200, 200, 16)
NUM_CLASSES   = 18                      # 17 sem + 1 empty (cross-check P1)
EMPTY_CLASS_ID = 17
```

### `s2go/stage2/semantic_head.py`
```python
class SemanticHead(nn.Module):
    """parent feat (B, K, D) -> per-parent class logits (B, K, C)."""
    def __init__(self, feat_dim, num_classes, hidden=256): ...
    def forward(self, parent_feat): ...
```
Two-layer MLP (Linear → GELU → Linear). Init: bias toward `EMPTY_CLASS_ID`
at start so early loss is well-conditioned (logit prior).

### `s2go/stage2/stage2_segmentor.py`
```python
class S2GOStage2(nn.Module):
    def __init__(self, segmentor_kwargs, num_classes):
        self.segmentor = S2GOSegmentor(**segmentor_kwargs, child_mode='semantic')
        self.semantic_head = SemanticHead(segmentor.parent_feat_dim, num_classes)

    def forward_one_frame(self, feats, lidar_pts, ego_pose, lidar2img, prev_exists):
        out = self.segmentor.forward_one_frame(...)
        # parent.feat shape (B, K, D); semantic logits (B, K, C)
        sem_parent = self.semantic_head(out.parent.feat)
        # Broadcast parent semantics to children (matches assembly's parent->child pattern)
        sem_per_g = sem_parent[:, :, None, :].expand(-1, -1, J, -1).reshape(B, K*J, C)
        return Stage2FrameOutput(gaussians=out.gaussians, sem_per_g=sem_per_g, raw=out)
```
**Important:** No edits to [s2go/models/segmentor.py](s2go/models/segmentor.py) or
[s2go/models/encoder/assembly.py](s2go/models/encoder/assembly.py). The Gaussians dataclass
stays unchanged; semantic logits travel as a sibling tensor.

### `s2go/stage2/g2v.py`
Thin nn.Module wrapper around the existing `g2v_forward_reference` in
[tests/g2v_reference.py](tests/g2v_reference.py). If `tests/` isn't importable as a
package, copy the pure function into `s2go/stage2/g2v.py` (it's a single
short function) and add a unit test that asserts equality with the
original to keep them in sync.

Returns `(occ_logits (Vx*Vy*Vz,), sem_logits (Vx*Vy*Vz, C))`.

### `s2go/stage2/losses.py`
```python
def occupancy_bce(occ_pred, occ_gt):
    return F.binary_cross_entropy_with_logits(occ_pred, occ_gt.float())

def kl_semantic(sem_logits, sem_gt_onehot, occupied_mask):
    # KL(softmax(pred) || gt_onehot), masked to occupied voxels
    ...

def ce_semantic(sem_logits, sem_gt_idx, occupied_mask, class_weights=None):
    ...

def lovasz_semantic(sem_logits, sem_gt_idx, occupied_mask):
    # Port from reference_code/GaussianFormer/loss/lovasz_loss.py
    ...

def compute_stage2_loss(occ_pred, sem_logits, occ_gt, sem_gt_idx, *,
                        sem_loss: str,            # 'kl' | 'ce_lovasz' | 'both'
                        w_occ=1.0, w_kl=1.0, w_ce=10.0, w_lovasz=1.0):
    ...
    return total, dict_of_terms
```

### `s2go/stage2/occ_dataset.py`
```python
class Stage2OccLoader:
    """Wraps NuScenesLoader, adds dense voxel GT per frame.

    For each frame in a T-sequence, look up the lidar_token, load
    {occ_root}/{lidar_token}.npy (sparse (M,4) int64), and scatter into
    a dense (Vx, Vy, Vz) int64 with EMPTY_CLASS_ID fill.
    Also yields binary occupancy mask = (dense != EMPTY_CLASS_ID).
    """
    def __init__(self, nusc_loader, occ_root, grid_shape, empty_id): ...
    def __getitem__(self, idx):
        frames = self.nusc_loader[idx]
        for f in frames:
            sem_dense, occ_mask = self._load_voxel_gt(f['lidar_token'])
            f['sem_voxel_gt']  = sem_dense   # (Vx, Vy, Vz) int64
            f['occ_voxel_gt']  = occ_mask    # (Vx, Vy, Vz) bool
        return frames
```
No edits to [s2go/datasets/nusc_loader.py](s2go/datasets/nusc_loader.py).

### `s2go/stage2/miou.py`
Port the per-class IoU + occupancy IoU calculation from
[reference_code/GaussianFormer/misc/metric_util.py:9](reference_code/GaussianFormer/misc/metric_util.py#L9).
Class names list from the explore output (16 fg + 1 empty; or 17 + 1 if
P1 confirms 18 total).

### `s2go/tools/stage2_overfit.py`
CLI mirror of [s2go/tools/overfit.py](s2go/tools/overfit.py):
```
python -m s2go.tools.stage2_overfit \
    --stage1-ckpt out/barebones_full_part1_5000iter_v2/ckpt.pt \
    --iters 50 \
    --num-sequences 4 \
    --t-seq 1 \
    --lr 2e-4 \
    --grad-clip 10 \
    --sem-loss both \
    --w-occ 1.0 --w-kl 1.0 --w-ce 10.0 --w-lovasz 1.0 \
    --save-path out/stage2_smoke_50iter/ckpt.pt
```
- Loads Stage 1 ckpt → `model.backbone.load_state_dict(ckpt['backbone'])`
  and `model.segmentor.load_state_dict(ckpt['segmentor'])`.
- New semantic head random-initialized (Kaiming for hidden, bias toward
  empty class).
- Single AdamW over all params, same NaN/inf guard and `_build_ckpt()`
  helper pattern as Stage 1.
- Per-iter log: `total`, `occ_bce`, `sem_kl` (if used), `sem_ce`,
  `sem_lovasz`, `gnorm`, `peak_mem`, `skipped`.
- Saves `training_history.json` like Stage 1.

### `s2go/tools/stage2_eval.py`
Mirror of [s2go/tools/stage1_eval.py](s2go/tools/stage1_eval.py):
- Forward one held sequence
- Compute per-class IoU (mIoU) + occupancy IoU
- Save:
  - `eval_A_bev.png` — top-down argmax voxel slice colored by class
  - `eval_B_voxel_slices.png` — 4 z-slices (e.g. z = 0.6, 1.4, 2.2, 3.0 m)
  - `eval_C_confusion.png` — 18×18 confusion matrix
  - `eval_stats.json` — per-class IoU + mean

## Output bundle

`out/stage2_smoke_50iter/`
- `ckpt.pt`
- `train.log`
- `training_history.json`
- (after eval) `eval_A_bev.png`, `eval_B_voxel_slices.png`,
  `eval_C_confusion.png`, `eval_stats.json`, `SUMMARY.md`

Bundle layout matches [out/barebones_full_part1_5000iter_v2_results/](out/barebones_full_part1_5000iter_v2_results/).

## Execution order

1. **P1 + P2 verification** (read-only Bash + small script). ~15 min.
   Pin grid constants; confirm 4 sequences have GT.
2. **Module scaffolding** — write the 7 new files in `s2go/stage2/`.
   Each module gets a short docstring + the structure above. No training
   code yet.
3. **Sanity tests** — extend [tests/test_g2v_scaffold.py](tests/test_g2v_scaffold.py)
   with:
   - `test_semantic_head_shape` — `SemanticHead` produces (B, K, C).
   - `test_g2v_layer_grad` — `G2VLayer.backward()` gives non-zero grads to
     opacities, scales, and sem_logits.
   - `test_occ_loader_scatter` — sparse `(M, 4)` → dense fills the right
     voxels and leaves the rest = EMPTY_CLASS_ID.
4. **Training entry** — write `stage2_overfit.py`; smoke-run for **5
   iters first** (not the full 50) just to verify forward/backward/save.
5. **First real smoke run** — 50 iters × 4 sequences. Watch for: total
   loss decreasing, gnorm bounded, no NaNs.
6. **Eval entry** — write `stage2_eval.py`; run on a training sequence.
   Goal at this stage is *not* a strong mIoU number; it's "the pipeline
   produces a plausible voxel argmax that overlaps the GT."
7. **Bundle + SUMMARY.md** following the v2 results pattern.

## Verification (how we know it works end-to-end)

- Each test in step 3 must pass.
- 50-iter smoke run completes with no NaN skips.
- `total` loss at iter 49 < `total` at iter 0.
- `sem_kl` (or `sem_ce`) > 0 at iter 0 and trending down.
- `occ_bce` > 0 and trending down.
- `gnorm` bounded < ~50 throughout (will be larger than Stage 1 because
  there are more loss terms; tighten only if it explodes).
- Eval produces a non-degenerate argmax (>1 class predicted).

## Non-goals (explicitly deferred)

- Stage 1 ↔ Stage 2 "bridge" (joint training, freezing schedules,
  warmup). The smoke run trains all params from the Stage 1 init in one
  go; richer integration comes later.
- CUDA G2V backend. The pure-torch path is correct and gradient-checked;
  performance gain from the existing CUDA kernel can be claimed in a
  follow-up by adding a `--g2v cuda` flag and a gradcheck against torch.
- Full 5000-iter streaming Stage 2 training. We commit to that only
  after the 50-iter smoke + 500-iter overfit pass.
- Class balancing tuning, Lovász weight sweeps, occupancy-vs-semantic
  weight sweeps. All gated by `--w-*` flags from day one but no sweep
  in this milestone.
- Validation split. Stage 1 didn't have one yet; Stage 2 will mirror
  that gap and we'll close both at once in a later task.

## Critical files to read before writing each module

- Semantic head signature: [s2go/models/segmentor.py:130-160](s2go/models/segmentor.py#L130-L160)
  (need parent_feat_dim and the parent feat slot in FrameOutput).
- ChildGaussianHead semantic mode: [s2go/models/encoder/heads.py:99-163](s2go/models/encoder/heads.py#L99-L163)
  (confirm what changes when `mode='semantic'`).
- G2V reference math: [tests/g2v_reference.py:56-112](tests/g2v_reference.py#L56-L112).
- Stage 1 train loop + NaN guard + `_build_ckpt`: [s2go/tools/overfit.py](s2go/tools/overfit.py)
  (mirror its structure).
- Stage 1 eval template: [s2go/tools/stage1_eval.py](s2go/tools/stage1_eval.py).
- MeanIoU reference: [reference_code/GaussianFormer/misc/metric_util.py:9-111](reference_code/GaussianFormer/misc/metric_util.py#L9-L111).
- Lovász reference: [reference_code/GaussianFormer/loss/lovasz_loss.py](reference_code/GaussianFormer/loss/lovasz_loss.py).
- Stage 1 ckpt format (the loading contract): [s2go/tools/overfit.py](s2go/tools/overfit.py) `_build_ckpt()`.

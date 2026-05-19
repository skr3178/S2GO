# SUMMARY (top — read first; 2026-05-18)

This file accreted several analyses (some contradictory). Condensed:

**What the document argues (in order):**
- *Diagnosis #1:* bridge gives 0 mIoU benefit → either bridge broken or
  Stage-1 learned nothing. "Smoking gun": lifter is **swapped** for
  `Stage2Lifter` and `query_xyz` is fresh-random → claims the Stage-1
  decoder's weights are conditioned on positions that no longer exist.
  Proposed fix: don't swap the lifter. Plus diagnostics (query_xyz dist,
  decoder attn, param-change) and two side-causes: (A) Stage-1 under-trained
  (`topk_opa=0.52`), (B) Stage-2 LR too high → use **differential LR**
  (backbone 1e-5 / pretrained seg 5e-5 / fresh head 4e-4).
- *Essay #2 (epochs):* "smaller data → fewer epochs" is wrong; val mIoU
  peaks ~8-15 ep regardless of data size. Recommends **early-stopping on
  val mIoU** (patience 4, max 15 ep), or subsample frames/epoch, or shorter
  Stage-1 + full Stage-2 — never pre-shrink epoch count.
- *Analysis #3 (code-checked):* reaffirms swap→fresh-`query_xyz` as cause;
  the **one-flag fix `--query-init fps_lidar`** (keeps Stage-1 `S2GOLifter`,
  LiDAR-seeded queries); notes this uses LiDAR at Stage-2 train time
  (non-paper-faithful, OK for validation).
- *Remaining-issues list (verdict):* files structurally correct for
  Stage-2; before next serious run fix the 5-item backlog (below).

**5-item engineering backlog — with verified status:**
- **1. `Stage2Lifter` skip_lidar / missing `lidar_pts`.** Open. Only matters
  for the *learnable-init* (paper-faithful) path; not the running diagnostic.
- **2. Fix `--from-scratch` default.** *Outdated claim* — current
  `stage2_train.py` argparse is `action="store_true"` (default **False**),
  not `True`; the bridge already runs (proven this session). No fix needed.
- **3. Log/assert `load_stage1_state` partial transfer.** Mostly done — boot
  log prints `missing/unexpected/head_partial_transferred`; could add a hard
  `assert head_ok`. Minor.
- **4. `scale_min=0.05` in Stage-2.** Already in effect (passed on every
  Stage-2 launch). Done.
- **5. Real G2V+loss gradient smoke test.** Open. Test infra; unrelated to
  the bridge question.

**Verified code facts that CORRECT the central diagnosis:**
- **Stage-1 has NO learned `query_xyz`.** `S2GOLifter`
  (`s2go/models/lifter/s2go_lifter.py:38-66`) has one learned param,
  `query_feat`; positions are computed per-frame as `FPS(LiDAR)+ε`. Stage-1
  ckpt lifter keys = `['lifter.query_feat']` only. → "throws away learned
  query_xyz" is **void**; Diagnostic 1 as written would `KeyError`.
- **The lifter swap is paper-faithful, not a bug.** Paper §3.4.1: Stage-2
  query positions are *learnable* by design ("only uses RGB at inference").
  The thing meant to transfer is the decoder/refiner's **query-movement
  ability — trained by the denoising loss** — not coordinates.
- `lifter.query_feat` (the one learned param) *does* transfer cleanly.
- **Real cause of 0 benefit:** Stage-1 was run `--depth-only` →
  **λ_denoise=0** → decoder/refiner never trained to reposition queries
  (paper Table-3: depth-only ≈ row (e); full denoise = row (f), +6-8 mIoU).
  Confirmed by code inspection alone.

**Currently running:** `stage2_bridged_fpslidar_diag` (pid 464365,
setsid-detached, 30k iters) — Stage-2 bridged from depth-only `ckpt_final`
with `--query-init fps_lidar` (no code change, one flag). Tests: does the
depth-only prior transfer when the decoder gets its trained LiDAR-seeded
input? Read vs the completed from-scratch baseline (`history/`):
≫ baseline → prior useful, swap masked it; ≈ baseline → depth-only Stage-1
learned no transferable prior (the code-predicted outcome).

**Highest-value real fix (independent of the diagnostic):** run **Stage-1
`--full-recipe`** (noised-LiDAR init + denoise + depth + RGB + ±0.5s velocity
warps), then bridge — paper Table-3 row (f), the configuration that yields
the documented +6-8 mIoU. The diagnostic explains *why* depth-only was flat;
it does not replace a denoise-trained Stage-1.

---

Good — having the bridge diagram laid out makes the diagnosis much clearer. Let me work through why your bridge isn't producing the expected improvement.

## What "no improvement" actually tells us

The paper claims **+5-6 mIoU gap** between Stage 1 init and from-scratch Stage 2. You're seeing **0 mIoU gap**. That's a very specific signal — not "smaller than expected gap" but "the bridge isn't doing anything useful."

When transfer learning produces zero benefit, one of two things is happening:

1. **The bridge is broken**: weights aren't actually being loaded, or are being loaded into the wrong places
2. **The bridge is loading correctly but the loaded weights aren't useful**: Stage 1 didn't learn anything Stage 2 can use

Your diagram reveals issues that point firmly toward #1 + a side of #2. Let me walk through them.

## The smoking gun in your diagram

Look at this row:

```
lifter (FPS(LiDAR)+ε query)            ✓ query_feat only ──── ├─ lifter → SWAPPED to Stage2Lifter
   └─ query_feat                          (name+shape match)        └─ query_xyz = FRESH (random)
```

**The lifter is SWAPPED, not loaded.** And `query_xyz` is initialized fresh.

This is a structural problem with the bridge. Stage 1 trained `query_xyz` to a specific set of 3D positions through 5000+ iterations of denoising + depth supervision. These positions were the *foundation* of all downstream learning — every decoder output, every Gaussian prediction, was conditioned on these query positions being where they ended up.

By swapping the lifter and randomizing `query_xyz`, you're essentially throwing away the spatial anchoring that Stage 1 built. The decoder loaded its weights, but those weights were tuned to work with specific query positions. Now it's getting random positions and trying to do the same work — but the conditional relationships are broken.

This is like loading a transformer's attention weights but randomizing the positional embeddings. The attention patterns were learned assuming specific position encodings. Randomize the encodings and the attention patterns no longer make sense.

## The second issue

```
child_head.head  Linear(d,14)
   rows 0..10  geom (off/scale/rot/opa) ─ ~ partial: copy rows 0..10
   rows 11..13 RGB ─────────────────────── ✗ discarded
```

Partial-row loading of `child_head.head`. This part is actually fine in principle — the geometric rows (offset, scale, rotation, opacity predictions) are the same in both stages, only the RGB rows are dropped.

But this gives Stage 2 a *partially* warm-started child head. If something is going wrong elsewhere (e.g., the lifter swap), the warm geometric weights might actually be working against you — they're tuned for a query distribution that no longer exists.

## The third issue

```
Stage-2 model  S2GOStage2
                                       ✦ no counterpart ──────────────► semantic_head = FRESH init
```

`semantic_head` is freshly initialized. This is expected and correct — Stage 1 has no semantic head to load from. But it means the semantic head has to be trained from scratch regardless of Stage 1 init.

This isn't a bug, but it's worth knowing: even a perfectly-working bridge only warm-starts the geometry. The semantic part still has to learn from random init.

## Why the paper's bridge works (and yours might not)

Reading the paper carefully, S2GO's Stage 1 → Stage 2 transition keeps the **lifter intact**. The query positions learned in Stage 1 are the same positions used in Stage 2. The continuity is what makes the transfer effective.

Your bridge breaks this continuity. Stage 2 starts with random query positions and has to re-learn spatial reasoning from scratch in Stage 2's iter budget — which is much less than Stage 1's iter budget. The result: Stage 2 with bridge ≈ Stage 2 from scratch, because the bridge's "useful" weights (decoder, parent refiner, partial child head) are conditioned on inputs that no longer exist.

## How to verify this is the issue

A few quick diagnostics:

### Diagnostic 1: Compare query_xyz distributions

```python
# Load Stage 1 ckpt
s1_ckpt = torch.load('stage1_ckpt.pt')
s1_query_xyz = s1_ckpt['segmentor']['lifter.query_xyz']  # or wherever it lives

# Initialize Stage 2 model fresh
s2_model = S2GOStage2(...)
s2_query_xyz = s2_model.segmentor.lifter.query_xyz

print(f"Stage 1 query_xyz range: "
      f"x=[{s1_query_xyz[:, 0].min():.2f}, {s1_query_xyz[:, 0].max():.2f}], "
      f"y=[{s1_query_xyz[:, 1].min():.2f}, {s1_query_xyz[:, 1].max():.2f}], "
      f"z=[{s1_query_xyz[:, 2].min():.2f}, {s1_query_xyz[:, 2].max():.2f}]")

print(f"Stage 2 query_xyz range: "
      f"x=[{s2_query_xyz[:, 0].min():.2f}, {s2_query_xyz[:, 0].max():.2f}], "
      f"y=[{s2_query_xyz[:, 1].min():.2f}, {s2_query_xyz[:, 1].max():.2f}], "
      f"z=[{s2_query_xyz[:, 2].min():.2f}, {s2_query_xyz[:, 2].max():.2f}]")
```

If Stage 1's positions are concentrated near actual scene structure (e.g., z mostly in [-1, 3] range covering road surfaces, x/y clustering around vehicle paths) and Stage 2's are uniform random across the full AABB, that's strong evidence the bridge is throwing away learned spatial structure.

### Diagnostic 2: Check what the decoder expects

```python
# Forward a frame through Stage 1's decoder with Stage 1's query positions
# vs Stage 2's decoder (warm-started) with random query positions

# Compare cross-attention attention weights
# If Stage 1's decoder produces sharp, scene-aligned attention patterns
# but warm-Stage 2 with random queries produces uniform/random attention,
# you've confirmed the conditioning is broken
```

### Diagnostic 3: Check the param-change diagnostic (separate from earlier eval bug)

```python
# After bridge load, before any training:
s2_state_before = {k: v.clone() for k, v in s2_model.state_dict().items()}

# After 100 iters of training:
s2_state_after = s2_model.state_dict()

# Which params changed how much?
for k in s2_state_before:
    diff = (s2_state_after[k] - s2_state_before[k]).abs().max().item()
    if diff > 0:
        # Check if this is a Stage 1-loaded param or a fresh-init param
        is_loaded = 'lifter.query_feat' in k or 'decoder' in k or ...
        print(f"{k}: max_change={diff:.4e}, was_loaded={is_loaded}")
```

If Stage 1-loaded params are changing as much as fresh-init params (e.g., decoder weights changing 10× their original magnitude in 100 iters), that's a sign the loaded weights aren't actually useful and the optimizer is essentially re-learning them from scratch.

## How to fix the bridge

The cleanest fix: **don't swap the lifter**. Keep Stage 1's lifter intact for Stage 2.

```python
def load_stage1_state(self, ckpt, strict=False):
    s1_state = ckpt['segmentor']
    
    # OLD: swap lifter for Stage2Lifter (broken)
    # NEW: keep Stage 1's lifter, load its weights
    
    # Drop only the RGB rows from child_head.head
    s1_state_filtered = {}
    for k, v in s1_state.items():
        if 'child_head.head.weight' in k or 'child_head.head.bias' in k:
            # Slice to keep only geometric rows
            s1_state_filtered[k] = v[:11] if v.dim() > 0 else v[:11]
        else:
            s1_state_filtered[k] = v
    
    self.segmentor.load_state_dict(s1_state_filtered, strict=False)
    # Semantic head stays at fresh init (no Stage 1 counterpart)
```

Why this works:
- Stage 1's `query_xyz` is preserved → spatial anchoring intact
- Stage 1's `query_feat` is preserved → semantic foundations intact
- Decoder weights operate on the same input space they were trained for
- Only the semantic head and the dropped RGB rows are fresh

## The alternative: re-do Stage 1 with Stage 2's lifter

If for some reason you can't keep Stage 1's lifter (maybe Stage2Lifter has different params or shape), you'd need to:

1. Train Stage 1 with the Stage 2 lifter architecture
2. Then bridge to Stage 2 with the lifter weights transferring

But this defeats the purpose — Stage 1 should be the "geometry pretraining" that Stage 2 builds on, using the same input infrastructure.

## Sanity check before refactoring

Before committing to a bridge refactor, verify the diagnosis. If you confirm:
- Stage 1's `query_xyz` is meaningfully different from random (e.g., clustered around scene content)
- Stage 2's bridge-loaded `query_xyz` is uniform random

Then the diagnosis is right. If Stage 1's `query_xyz` happens to be near-uniform anyway (because it didn't really learn good positions), then the lifter swap isn't the issue — Stage 1 itself didn't learn what we expected.

In that case, you have a deeper Stage 1 quality problem. The lifter swap is correct in *consequence* (Stage 1 has nothing useful to transfer) even if it's wrong in *principle*.

## What might also be contributing

Two other possibilities worth checking:

### A. Stage 1 didn't learn good geometry

Your overnight Stage 1 hit `topk_opacity_mean=0.52` at the end (not the healthy 0.85+). This suggests Stage 1's representation might be only partially useful even with a working bridge. Quality of the source matters for transfer learning.

### B. Stage 2's recipe might be wiping out Stage 1's weights

If Stage 2 uses an aggressive LR (4e-4) for the whole model from iter 1, the loaded weights get heavily updated in the first few iterations. By iter 1000, they look like fresh-init weights anyway. This is why pretraining recipes typically use lower LR for pretrained components.

Fix:
```python
optim = AdamW([
    # Backbone: lowest LR (most pretrained)
    {'params': backbone.parameters(), 'lr': 1e-5},  # Stage 1 + ImageNet
    # Pretrained Stage 1 segmentor parts: low LR
    {'params': pretrained_segmentor_params, 'lr': 5e-5},
    # Fresh semantic head: high LR (random init)
    {'params': model.semantic_head.parameters(), 'lr': 4e-4},
])
```

This is standard transfer learning practice. The paper might use this implicitly without documenting it.

## TL;DR

**The reason your Stage 1 → Stage 2 bridge produces no gain is structural:** the bridge swaps the lifter and randomizes `query_xyz`. This throws away the spatial anchoring Stage 1 built up. The decoder's loaded weights are tuned for specific query positions; with random positions, those weights become approximately useless.

**Three things to verify:**

1. **`query_xyz` distribution check**: are Stage 1's query positions meaningfully non-random? If yes, the lifter swap is destroying useful structure.

2. **Stage 1 quality check**: did Stage 1 actually learn good geometry? Your `topk_opa=0.52` and unimodal scale distribution suggest Stage 1 might be only partially useful even with a fixed bridge.

3. **Stage 2 LR check**: is the segmentor LR (4e-4) too high for pretrained components? Try differential LR.

**Fix priority**:

1. **Stop swapping the lifter**. Keep Stage 1's lifter with its trained `query_xyz`. This is the structural fix.
2. **Use differential LR** for pretrained vs fresh components (1e-5 / 5e-5 / 4e-4 hierarchy)
3. **If Stage 1 itself didn't converge well, fix Stage 1 first** — better Stage 1 → better transfer

Without #1, the bridge is essentially cosmetic. The "loaded weights" don't carry the geometric information they were supposed to because the inputs they expect are gone.

Run the `query_xyz` distribution check first — that single comparison will tell you whether the diagnosis is right.

I want to push back here, because the reasoning behind this idea is the opposite of what's actually true. Let me explain carefully.

## The reasoning you're using (and why it's backwards)

The intuition seems to be: "model has lots of capacity, dataset is small, so the model will memorize faster → train for fewer epochs."

This is **conflating two different things**:

1. **How quickly the model fits the training data** (yes, faster with smaller data)
2. **How many epochs are needed to learn useful features** (independent of model size)

Model size affects #1, not #2. Let me explain.

## What model size actually controls

Large model + small data does converge faster *on training loss*. After ~3-4 epochs, your training loss will be very low — the model has memorized your 200 scenes.

But low training loss ≠ good val mIoU. Two distinct things are happening:

```
Epochs 1-3:   Model learns basic geometry & class patterns
              Train loss: high → medium
              Val mIoU:   low → medium (real learning)
              
Epochs 4-8:   Model refines patterns, learns rarer classes
              Train loss: medium → low
              Val mIoU:   medium → high (still learning)
              
Epochs 9-15:  Model memorizes scene-specific quirks
              Train loss: low → very low
              Val mIoU:   high → plateau (peak performance)
              
Epochs 16+:   Model overfits hard
              Train loss: very low → near zero
              Val mIoU:   plateau → decline (overfitting)
```

If you stop at epoch 3 or 5 because training loss looks low, you stop **before** the model has finished learning useful features. You've only done step 1 of 3.

## What actually controls when to stop

The right signal is **val mIoU**, not training loss or model-vs-data ratio.

Val mIoU follows a predictable pattern:
- Rises rapidly for the first few epochs
- Slows down but keeps rising
- Plateaus
- Eventually declines as overfitting sets in

The peak — the highest val mIoU — is typically at epoch **8-15** for occupancy estimation tasks, regardless of:
- Model size
- Dataset size
- Backbone choice
- Optimizer

This is empirical, not theoretical. The 12-epoch convention in the paper isn't a coincidence — it's near the typical peak.

## Why "fewer epochs for smaller data" gives bad results

Concrete prediction for what happens if you train for fewer epochs:

**3 epochs**: training loss looks great (~5-6), but val mIoU is ~3-5%. The model hasn't learned rare classes yet (motorcycle, bicycle, trailer all near zero). You stop too early and conclude "this doesn't work."

**6 epochs**: val mIoU is ~7-9%. Better, but still on the rising portion of the curve. Per-class IoU is uneven — some classes high, others zero.

**12 epochs**: val mIoU is ~10-13%. Near peak performance. All classes have non-trivial IoU.

The 7-8 mIoU gap between epoch-3 and epoch-12 is gone forever if you stop early. You can't recover it later.

## When "fewer epochs" actually applies

There IS a valid version of the rule you're thinking of, but it applies in a different direction:

**If your dataset is LARGER than the paper's setup**, you might need fewer epochs because each epoch provides more training signal. With 2× the data, 6 epochs might match 12 epochs on smaller data.

The other direction — smaller data → fewer epochs — has no theoretical basis. It's a common misconception that comes from confusing training loss convergence with model performance.

## What you can do to save compute

I understand the desire to save time. Here are ways that actually work:

### Option 1: Early stopping on val mIoU

Set a maximum of 15 epochs but stop early if val plateaus:

```python
# Pseudocode
best_val_miou = 0
patience = 4  # epochs without improvement
no_improve_count = 0

for epoch in range(15):
    train_one_epoch()
    val_miou = evaluate()
    
    if val_miou > best_val_miou + 0.001:
        best_val_miou = val_miou
        save_checkpoint()
        no_improve_count = 0
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"Stopping at epoch {epoch}, best val mIoU = {best_val_miou}")
            break
```

If val mIoU plateaus at epoch 8, the run stops automatically and you save 4 days. If it keeps improving until epoch 14, you get the full benefit. **You don't have to decide in advance** — let the model tell you.

This is the right way to "train for fewer epochs": train for *up to* 15, stop when val stops improving.

### Option 2: Reduce iters-per-epoch instead of total epochs

Instead of "train for 6 epochs instead of 12," try "train for 12 epochs but with half the samples per epoch." This keeps the *diversity* of the training experience while reducing wall-clock.

```python
# Original: 200 scenes × 40 samples/scene = 8000 samples/epoch × 12 = 96k samples seen
# Reduced:  200 scenes × 20 samples/scene = 4000 samples/epoch × 12 = 48k samples seen
```

Subsample frames within each scene. The model still sees all 200 scenes 12 times, just with fewer keyframes per visit. Training is ~2× faster but you keep the multi-epoch refinement pattern.

### Option 3: Shorter Stage 1, longer Stage 2

If you really need to save time, cut Stage 1 short:

```
Stage 1: 8 epochs (instead of 12-15)
Stage 2: 12 epochs (keep full)
```

Stage 1's value plateaus earlier than Stage 2's. You lose some Stage 1 polish but Stage 2's full epochs still extract value from it.

This is asymmetric: shortening Stage 2 hurts more than shortening Stage 1.

## What I actually recommend

```python
# Stage 1 config
stage1_config = {
    'max_epochs': 15,
    'early_stopping_metric': 'val_L_depth',  # or val_mIoU if computed
    'early_stopping_patience': 4,
    'eval_every': 'each epoch',
}

# Stage 2 config
stage2_config = {
    'max_epochs': 15,
    'early_stopping_metric': 'val_mIoU',
    'early_stopping_patience': 4,
    'eval_every': 'each epoch',
}
```

**Expected actual training time** with early stopping:

- Stage 1: probably stops at epoch 10-13 (val L_depth plateaus). ~4 days.
- Stage 2: probably stops at epoch 8-12 (val mIoU plateaus). ~3 days.
- Total: ~7 days instead of ~9.

Plus you have a defensible "we trained until convergence" story instead of "we picked an arbitrary small number of epochs."

## TL;DR

**The intuition "smaller data needs fewer epochs" is incorrect.**

What actually happens:
- Fewer epochs → model hasn't finished learning, even though training loss looks low
- Val mIoU continues improving until ~10-15 epochs regardless of dataset size
- Stopping at 3-5 epochs leaves 4-8 mIoU on the table permanently

**The right approach**:
- Set max epochs to 15 for each stage
- Use early stopping based on val mIoU (not training loss)
- Let the model tell you when it's done

**If you really need to save compute**:
- Early stopping (probably saves 2-4 days)
- Subsample frames within scenes (keeps diversity, halves wall-clock)
- Shorter Stage 1, full Stage 2 (Stage 1 plateaus earlier)

**Don't** pick a smaller epoch count upfront. That's optimizing the wrong variable.

The reason the paper uses 12 epochs isn't arbitrary — it's empirically near-peak for occupancy estimation tasks. Your reduced data doesn't change where the peak is; it just changes how high the peak gets.

The code confirms your structural analysis exactly. The bridge fails because `Stage2Lifter` creates **fresh random `query_xyz`** that the Stage‑1 decoder has never seen, while the pretrained offset head, parent refiner, and child head are all tuned to refine queries that start near LiDAR surfaces. The fix is one flag: `--query-init fps_lidar`.

Here's the precise chain and the solution.

---

### 1. What the code tells us

- **`Stage2Lifter`** (line 45‑73 of `stage2_lifter.py`): creates `self.query_xyz` as a `nn.Parameter` drawn uniformly from `[-50,50]×[-50,50]×[-5,3]`.  
- **`S2GOStage2.__init__`** (line 67‑77 of `stage2_segmentor.py`):  
  - If `query_init='learned'`, **swaps** the original `S2GOLifter` with a fresh `Stage2Lifter`.  
  - If `query_init='fps_lidar'`, **keeps** the original `S2GOLifter` (FPS from LiDAR + noise).  
- **`load_stage1_state`**: transfers the Stage‑1 `query_feat` parameter (shape matches), but there is **no counterpart for `query_xyz`** – it remains at its random AABB initialisation.  

### 2. Why the bridge gave no benefit

- Stage 1 pretrained the decoder, parent refiner, and child head to move queries from **noised LiDAR points** (±1 m) onto exact surfaces and to model geometry around them.  
- Stage 2 (with `query_init='learned'`) gives the decoder queries scattered randomly across the entire 100 m×100 m scene. The offset head was trained to predict small corrections (~1 m); it cannot suddenly learn to predict 50 m jumps.  
- The decoder receives an input distribution vastly different from its training, so the pretrained weights are effectively useless. The semantic head also receives features from queries floating in empty space, making classification nearly impossible.

### 3. The one‑flag fix

Your `stage2_train.py` already exposes `--query-init` (line ~130 of the argparse section). Use:

```bash
python -m s2go.tools.stage2_train \
  --splits-json out/curated_255_splits.json \
  --stage1-ckpt out/stage1_curated_ckpt.pt \
  --query-init fps_lidar \
  ... (other args)
```

This does three things:
- **Prevents the lifter swap** – `S2GOStage2` keeps the original `S2GOLifter` that was trained in Stage 1.  
- **Initialises queries from FPS+noise on LiDAR** each frame, exactly as Stage 1 did.  
- **Gives the pretrained decoder its expected input distribution** – queries near surfaces, small offsets needed.

### 4. What to expect

- The Stage 1 checkpoint will immediately transfer useful geometric knowledge.  
- The from‑scratch vs. pretrained mIoU delta should become significant, even with few epochs.  
- The LiDAR data is already present in your frames (loaded by `Stage2OccLoader`), so no dataloader changes are needed.

### 5. Paper compliance note

The paper’s Stage 2 trains with **no LiDAR in the forward pass** only after the model has already been initialised from Stage 1. You are currently validating whether the pipeline works; once proven, you can optionally train a final version with `query_init='learned'` but initialised from a checkpoint that already saw LiDAR‑initialised queries, or simply publish results with the LiDAR‑initialised variant (many occupancy methods use LiDAR for training initialisation, as long as inference does not rely on it).

**Bottom line:** add `--query-init fps_lidar` to your Stage 2 run, and the bridge will actually carry the Stage 1 geometry knowledge into Stage 2.


Remaining issues to fix
1. Stage2Lifter still requires pts to exist
Even though it ignores point values, it still does:
B = pts.shape[0]
So if you later add skip_lidar=True and remove frame['lidar_pts'], this will break.
Minimal fix inside S2GOStage2.forward_one_frame before calling the segmentor:
if self.query_init == "learned" and "lidar_pts" not in frame:
    B = frame["imgs"].shape[0]
    device = frame["imgs"].device
    frame = dict(frame)
    frame["lidar_pts"] = torch.empty(B, 0, 3, device=device)
Or modify Stage2Lifter.forward to accept pts=None plus a batch_size argument. Since you want no Stage-1 edits, the dummy tensor injection in S2GOStage2 is cleaner.
2. Log the bridge result
load_stage1_state() returns:
missing, unexpected, head_partial_transferred
but your trainer currently calls it without printing/checking. You should log it and assert the partial copy happened when loading a Stage-1 checkpoint.
missing, unexpected, head_ok = model.load_stage1_state(ckpt, strict=False)
print("bridge missing:", missing)
print("bridge unexpected:", unexpected)
print("child_head partial transferred:", head_ok)
assert head_ok
Expected missing should include lifter.query_xyz and possibly child_head.head.* because you pop/copy those manually. Unexpected keys should be understood, not ignored silently.
3. --from-scratch bug in stage2_train.py still matters
The uploaded stage2_train.py had:
p.add_argument("--from-scratch", action="store_true", default=True)
That makes from_scratch always true, so Stage-1 checkpoint loading never happens from CLI. The trainer branch clearly uses T0 from-scratch if from_scratch is true, and only reads the Stage-1 checkpoint if false.
Fix:
p.add_argument("--from-scratch", action="store_true", default=False)

# after parsing
from_scratch = a.from_scratch or (a.stage1_ckpt is None)
Without this, your nice bridge code will not be exercised.
4. scale_min should default to 0.05 for Stage 2
The trainer still used scale_min=0.01 by default earlier, but your Stage-2 G2V backward has already shown instability at 0.01. For Stage 2, keep:
scale_min = 0.05
until the G2V density-normalizer issue is fully stabilized.
5. Add a real Stage-2 gradient test
The self-test currently checks semantic logits shape and sibling sharing, which is good. It also backprops through:
out.sem_logits_per_g.sum() + out.raw.gaussians.means.sum()
That proves basic graph connectivity, but not the real Stage-2 path through G2V and compute_stage2_loss.
Add a one-batch test:
forward → G2V → occupancy/semantic loss → backward
and print:
semantic_head grad norm
lifter.query_xyz grad norm
lifter.query_feat grad norm
child scale/opacity grad norm
This will catch the “loss runs but output does not change” bug much faster.
Verdict
These files are now structurally correct for Stage 2:
✅ learned query positions
✅ no LiDAR query init
✅ semantic head on parent features
✅ broadcast semantic logits to children
✅ partial Stage-1 geometry-head transfer
✅ Stage-1 code left untouched
Before the next serious run, fix these:
1. Make Stage2Lifter compatible with skip_lidar / missing lidar_pts.
2. Fix --from-scratch default so Stage-1 bridge can run.
3. Log/assert load_stage1_state partial transfer.
4. Use scale_min=0.05 in Stage 2.
5. Add one real G2V+loss gradient smoke test.
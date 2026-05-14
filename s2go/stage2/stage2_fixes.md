# Stage-2 fixes — ranked by impact

Consolidated from the three diagnostic passes on `stage2_train.py`. Items are
ordered by impact: known bugs first, then unsafe defaults, then diagnostic probes
that pinpoint the silent semantic-head failure.

Two symptoms are being addressed simultaneously:

- **Backward NaN under some configs** — Tier A items 3 and Tier B item 5 are the
  direct fixes.
- **Identical eval mIoU at iter 500 vs iter 1000** (the silent non-convergence
  hiding behind the NaN guard) — Tier A items 1, 2, 4 and Tier B item 6 are the
  most likely root causes; Tier C probes localise which one.

Apply Tier A and Tier B before the next training run. Use Tier C to localise
the remaining hypothesis if the loss plateau persists.

---

## Tier A — Definite bugs

### 1. `--from-scratch` is permanently True

```python
p.add_argument("--from-scratch", action="store_true", default=True)
```

`action="store_true"` makes the flag True whenever passed, but `default=True`
makes it True when *not* passed. Net effect: always True. Every Stage-2 run to
date has been row (a) from scratch, regardless of `--stage1-ckpt`.

Fix:

```python
p.add_argument("--from-scratch", action="store_true", default=False)
# after parse:
from_scratch = a.from_scratch or (a.stage1_ckpt is None)
```

### 2. Eval / periodic-save never fire under grad accumulation

Current gate:

```python
if is_accum_end and eval_every and i > 0 and i % eval_every == 0:
```

For `grad_accum_steps=16`, `eval_every=500` this requires `i % 500 == 0` AND
`(i + 1) % 16 == 0` — jointly satisfied at essentially no `i`. Explains the v4
postmortem "no periodic eval recorded" symptom.

Fix: count optimizer steps, not micro-iters.

```python
if did_optimizer_step:
    optim_step += 1
    if eval_every and optim_step >= next_eval_step:
        run_eval(); next_eval_step += eval_every
    if save_periodic_every and optim_step >= next_periodic_step:
        save_periodic(); next_periodic_step += save_periodic_every
```

### 3. `scheduler.step()` runs even when the optimizer step is skipped

```python
if not finite:
    optim.zero_grad(...); skipped = True
else:
    optim.step(); skipped = False
scheduler.step()         # ← runs in both branches
```

A 50-iter NaN storm burns 50 LR ticks while weights are frozen. The schedule
should advance only on real optimizer steps.

Fix:

```python
if finite:
    optim.step(); scheduler.step()
    did_optimizer_step = True
else:
    optim.zero_grad(set_to_none=True)
    did_optimizer_step = False
```

### 4. `occ_pos_weight=16` paired with balanced sparse sampling

Training G2V samples `n_pos = voxel_sample_size // 2` and
`n_neg = voxel_sample_size - n_pos`, so positives are already ~50 % of every
batch. `compute_stage2_loss` then applies `occ_pos_weight=16` on top —
double-correcting the imbalance and biasing the model toward "occupied
everywhere".

`pos_weight=16` is correct for **dense full-grid** training where >95 % of
voxels are empty. For **balanced sparse** training (the default Stage-2 path)
it must be `1.0`.

Fix:

```python
train_occ_pos_weight = 1.0 if voxel_sample_size > 0 else occ_pos_weight
# pass train_occ_pos_weight into compute_stage2_loss
```

---

## Tier B — Unsafe defaults

### 5. `scale_min` default should be 0.05 for Stage 2

Current default is `0.01`, with help text still asserting it's safe. The v4
postmortem disproved that: Stage-2 G2V backward can still go NaN after scale
collapse below 0.05. The `0.01 / grad_clip=35` combination is appropriate only
for Stage-1 denoise-only training.

Fix:

```python
p.add_argument("--scale-min", type=float, default=0.05)
```

Stage-2 should run with `scale_min=0.05, grad_clip=10`.

### 6. SurroundOcc class-ID mapping (most plausible root cause)

The SurroundOcc `.npy` files store raw class IDs in `{1..17}`. The model
convention is `{0..16}` for occupied semantic classes with `empty=17`. If raw
labels were ingested verbatim into the dense tensor, class `17` is being
treated as empty everywhere and class `0` never receives supervision —
precisely the per-class IoU pattern observed (every class ≈ 0 % except the
three dominant background classes).

One-line verification:

```python
raw = np.load(some_npy); print(np.unique(raw[:, 3], return_counts=True))
```

If raw labels are `1..17`, remap inside `Stage2OccLoader`:

```python
dense_sem = torch.full((200, 200, 16), EMPTY_CLASS_ID, dtype=torch.long)
dense_sem[vx, vy, vz] = raw_class_id - 1   # 1..17 → 0..16
```

If raw labels are already `0..16`, leave as-is. **Verify before assuming the
rest of the stack is broken** — this single check eliminates ~50 % of the
hypothesis space.

---

## Tier C — Diagnostic probes

Each is a single-iteration smoke test that takes ≤ 5 min and pins down which
remaining hypothesis is real.

### 7. Semantic-head gradient + update probe

Run one forward + backward + optimizer.step():

```python
opt_ids = {id(p) for g in optim.param_groups for p in g["params"]}
before  = model.semantic_head.fc2.weight.detach().clone()

# forward / loss / backward / step here

for n, p in model.named_parameters():
    if "semantic_head" in n:
        gn = None if p.grad is None else p.grad.norm().item()
        print(n, "in_opt", id(p) in opt_ids,
                "rg",     p.requires_grad,
                "grad",   gn)

delta = (model.semantic_head.fc2.weight.detach() - before).abs().max().item()
print("fc2 max update:", delta)
```

Expected: `in_opt=True`, `requires_grad=True`, `grad_norm > 0`, `delta > 0`.

- `delta == 0` → head is frozen (Tier C #8 likely)
- `grad_norm == 0` but `in_opt=True` → gradient flow into the semantic path is
  broken upstream (G2V kernel, see #8)

### 8. G2V CUDA kernel may not back-prop into `class_logits`

If `G2VLayerCUDA.forward_sparse` calls `.detach()` on `class_logits`, or the
custom backward doesn't compute `dL / dclass_logits`, the semantic head is in
the optimizer but never receives gradient.

Isolation test:

```python
sem = out.sem_logits_per_g.float().detach().requires_grad_(True)
occ, sem_out = g2v_train.forward_sparse(
    means=..., rotations=..., scales=..., opacities=...,
    class_logits=sem, voxel_flat_idx=...)
sem_out.sum().backward()
print("class_logits.grad norm:", sem.grad.norm().item())   # must be > 0
```

If this prints `None` or `0.0`, audit `s2go/stage2/g2v.py:G2VLayerCUDA` for a
`.detach()` on `class_logits`, and check whether the custom CUDA backward
(`local_aggregate_s2go`) actually writes the `class_logits` gradient slot. The
workaround until the kernel is fixed: route semantic logits through the torch
reference G2V while keeping CUDA for occupancy.

### 9. Eval-side: log argmax distribution

If logits change but argmax doesn't, mIoU stays bit-identical. Inside the eval
loop:

```python
print("occ_pred min/mean/max",
      occ_pred.min().item(), occ_pred.mean().item(), occ_pred.max().item())
print("frac >= thresh:", (occ_pred >= occ_thresh).float().mean().item())

u, c = torch.unique(pred_argmax.reshape(-1), return_counts=True)
print("pred hist:", dict(zip(u.tolist(), c.tolist())))
```

- Histograms identical at iter 500 and iter 1000 → prediction truly unchanged
  → head frozen (back to #7)
- Histograms differ but mIoU coincides exactly → numerical coincidence on the
  eval slice, re-run with more sequences

### 10. Eval-side: confirm checkpoint identity

Identical eval also fits "the same stale ckpt is being loaded twice":

```python
print("EVAL_CKPT:", ckpt_path)
print("EVAL_ITER:", ckpt.get("iter"))
print("SEM_HEAD_NORM:", model.semantic_head.fc2.weight.norm().item())
```

`SEM_HEAD_NORM` must differ between iter-500 and iter-1000 checkpoints. If
identical, `save_best` never updated since iter 0, or eval is loading the
wrong file.

---

## Quick-start order

Each step is ≤ 30 min.

1. **Apply Tier A (1–4).** All are definite bugs; not optional.
2. **Verify the label mapping (6).** One-line numpy print; remap if `1..17`.
3. **Bump `scale_min` to 0.05 (5).** Direct NaN-prevention.
4. **Run probes #7 + #8 as a single-iter smoke test.** Tells you whether the
   semantic head is actually receiving gradient.
5. **Only after #7/#8 pass:** launch a 5k-iter scaled run.

If Tier A + label remap alone resolve convergence, you'll see it inside the
first 200 iters: loss should drop past ~17 (vs the current ~22 floor) and
per-class IoU should be non-zero on at least 2–3 frequent classes.

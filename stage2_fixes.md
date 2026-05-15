Yes — your mapping is **mostly correct**, but I would make **one important distinction**:

```text
Eq. 1–2: binary occupancy probability
Eq. 4–5: semantic class mixture / density weighting
```

The dangerous `1 / sqrt(det(Σ))` factor belongs to the **density / mixture weighting path**, not the plain occupancy alpha term, unless your implementation deliberately uses it in the occupancy path too.

## Correct interpretation

Your chain is right at the implementation level if your `g2v.py` does:

```python
Sigma = R @ diag(s²) @ R.T
inv_sqrt_det = 1.0 / sqrt(det(Sigma))
w = opacity * inv_sqrt_det * exp(q)
```

The postmortem also identifies exactly this implementation: `Sigma`, `inv_sqrt_det`, and `w_full = (opacities * inv_sqrt_det) * exp(q)` inside the G2V layer. It further notes that forward values stayed finite, while backward through this scale/determinant path produced non-finite gradients. 

So the practical failure explanation is valid:

```text
scale → Σ → det(Σ) → inv_sqrt_det → semantic/mixture weights → backward explosion
```

And `scale_min` is indeed enforced upstream in the child head:

```python
scale = scale_min + (scale_max - scale_min) * sigmoid(out[..., 3:6])
```

Your Stage1-vs-Stage2 doc shows this exact scale formula.  The current trainer also overrides `model.segmentor.child_head.scale_min` after model creation, specifically to stabilize Stage 2. 

## Small correction: do not say Eq. 2 and Eq. 5 are the same role

I would phrase the equation roles like this:

| Equation                  | Role                                                | Should contain `1/sqrt(det)`? |
| ------------------------- | --------------------------------------------------- | ----------------------------: |
| Eq. 1                     | Combine per-Gaussian occupancy into voxel occupancy |                            No |
| Eq. 2 / Eq. 9-style alpha | Per-Gaussian occupancy influence, `a·exp(-½d²)`     |      No, usually unnormalized |
| Eq. 3                     | Build covariance `Σ = R diag(s²) Rᵀ`                |                           n/a |
| Eq. 4                     | Semantic class mixture across Gaussians             |                  Uses weights |
| Eq. 5                     | Gaussian density used as a mixture weight           |     Yes, has `1/sqrt(det(Σ))` |

So the safer statement is:

```text
The determinant normalizer should affect the semantic mixture density path.
It should not be necessary for the binary occupancy alpha path.
```

If your kernel uses `inv_sqrt_det` for both occupancy and semantic class mixture, that may be less paper-faithful and much more unstable.

## About “same gradients”

I would soften this line:

> “Same forward expressions, same gradients in the backward.”

Better:

```text
The CUDA kernel is intended to implement the same forward math and corresponding backward gradients, but because it is fused/binned/bf16-sensitive, it needs explicit parity tests against the torch reference.
```

Your Stage2 summary says the CUDA backend exists specifically to avoid the huge Python autograd `(V,N,C)` intermediate and wraps the compiled `local_aggregate_s2go` kernel.  That makes it necessary, but not automatically numerically identical under all edge cases.

Do this parity test on a tiny case:

```text
same Gaussians
same voxel subset
torch G2V vs CUDA G2V

compare:
  occ_prob
  semantic logits
  grad wrt means
  grad wrt scales
  grad wrt opacities
  grad wrt class logits
```

## About the scale-gradient explosion

Your qualitative chain is correct. I would adjust the math wording slightly:

```text
∂w/∂s has multiple terms:
  normalizer term from 1/(s_x s_y s_z)
  exponent term from dᵀΣ⁻¹d
```

Near the Gaussian center, the normalizer derivative behaves roughly like:

```text
p ∝ 1 / (s_x s_y s_z)
∂p/∂s_i ∝ -1 / (s_i · s_x · s_y · s_z)
```

For isotropic `s`, that is roughly:

```text
∂p/∂s ∝ 1 / s⁴
```

So your `1/s⁴` statement is a good rule-of-thumb for the local density derivative. The postmortem’s `det(Σ)^-1.5` wording is another way of describing how determinant-related terms can become enormous near the scale floor. 

The important practical fact is unchanged:

```text
s = 0.01 creates huge inverse-scale factors.
s = 0.05 reduces those factors by orders of magnitude.
```

The postmortem already shows that `scale_min=0.01` repeatedly cascaded in longer runs, while `scale_min=0.05` gave a clean 1500-iter run. 

## What I would do next

### 1. Keep `scale_min=0.05` for Stage 2

This is the direct fix.

```bash
--scale-min 0.05
```

Do not go back to `0.01` until you have either fp32-stable G2V backward or an unnormalized occupancy/mixture ablation that proves it is safe.

### 2. Separate occupancy weighting from semantic density weighting

Add explicit flags:

```text
--occ-use-density-normalizer false
--sem-use-density-normalizer true
```

Then test:

```text
A. occupancy alpha = opacity * exp(q)
B. semantic mixture weight = opacity * inv_sqrt_det * exp(q)
```

If NaNs persist, test:

```text
C. semantic mixture weight = opacity * exp(q)
```

That tells you whether the determinant factor is actually needed for semantic mIoU.

### 3. Run an “occ-only” ablation

Set:

```text
w_kl = 0
w_ce = 0
w_lovasz = 0
```

If NaNs disappear, the unstable path is almost certainly Eq. 4–5 semantic density weighting, not Eq. 1 occupancy BCE.

### 4. Keep scale diagnostics

Log:

```text
scale_min_actual
scale_p01
scale_p05
num_scale_at_floor
first_bad_grad_param
```

The postmortem’s causal chain depends on scales hitting the floor; make that visible every window.

## Bottom line

Your explanation is correct in spirit:

```text
Stage 2 Eq. 1–5 define the G2V computation.
scale_min enters through the child head.
small scale makes Σ nearly singular.
the density normalizer creates huge backward factors.
CUDA/bf16 makes that instability visible in production.
```

The one correction is:

```text
Do not conflate occupancy alpha with normalized Gaussian density.
The determinant normalizer is mainly the Eq. 5 semantic mixture-density path, not the plain Eq. 1–2 occupancy path.
```

So the best next implementation change is: **keep `scale_min=0.05`, and explicitly separate unnormalized occupancy alpha from normalized semantic mixture density.**



Yes — for your training, the two available floating‑point formats are **bf16** (bfloat16) and **fp32** (float32).  Your current setup uses bf16 autocast, which speeds up training and saves memory, but leads to the NaN gradients when Gaussian scales become too small.

The practical fix is **not** to switch the entire model to fp32 (which would roughly double memory and slow down the run), but to handle the one problematic layer (G2V) appropriately.  You have two good options:

1. **Keep bf16 everywhere, raise `scale_min` to 0.05** – This is the simplest, paper‑aligned fix.  It keeps training fast and avoids any precision‑sensitive gradient magnitudes entirely.  (This is the recommended path.)

2. **Keep `scale_min = 0.01` but force the G2V layer to fp32** – If you need to experiment with very small scales, you can wrap the G2V forward in `autocast(enabled=False)`.  That will compute the splatting and its backward pass in full fp32, removing the numerical instability while the rest of the network still uses bf16 for speed.  This adds a little memory and time only for the G2V part, which is a small fraction of the total.

In both cases you remain with bf16 as the main training precision.  The choice is between changing a hyperparameter or making a small code adjustment; the first option is cleaner for most purposes.


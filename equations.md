# Equations from S2GO Paper

## Section 3.1: Preliminary — Gaussian Occupancy Estimation

### Equation 1 — Occupancy probability as union over P nearby Gaussians
$$\alpha(\mathbf{x}) = 1 - \prod_{i=1}^{P}\bigl(1 - \alpha(\mathbf{x};\mathbf{G}_i)\bigr)$$

where $\alpha(\mathbf{x};\mathbf{G}_i)$ is the probability that $\mathbf{x}$ is occupied by Gaussian $\mathbf{G}_i$.

### Equation 2 — Single-Gaussian occupancy probability
$$\alpha(\mathbf{x};\mathbf{G}) = \exp\!\Bigl(-\tfrac{1}{2}(\mathbf{x}-\mathbf{m})^{\mathrm{T}}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\mathbf{m})\Bigr)$$

### Equation 3 — Covariance decomposition
$$\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^{\mathrm{T}}\mathbf{R}^{\mathrm{T}},\qquad \mathbf{S} = \mathrm{diag}(\mathbf{s}),\qquad \mathbf{R} = \mathrm{q2r}(\mathbf{r})$$

### Equation 4 — Foreground class distribution as opacity-weighted Gaussian mixture
$$\mathbf{e}(\mathbf{x};\mathcal{G}) = \sum_{i=1}^{P} p(\mathbf{G}_i \mid \mathbf{x})\, \tilde{\mathbf{c}}_i = \frac{\sum_{i=1}^{P} p(\mathbf{x} \mid \mathbf{G}_i)\, a_i\, \tilde{\mathbf{c}}_i}{\sum_{j=1}^{P} p(\mathbf{x} \mid \mathbf{G}_j)\, a_j}$$

### Equation 5 — Gaussian density
$$p(\mathbf{x} \mid \mathbf{G}_i) = \frac{1}{(2\pi)^{\frac{3}{2}}\,|\boldsymbol{\Sigma}|^{\frac{1}{2}}} \exp\!\Bigl(-\tfrac{1}{2}(\mathbf{x}-\mathbf{m})^{\mathrm{T}}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\mathbf{m})\Bigr)$$

The joint semantic occupancy distribution over foreground classes and the empty background is written as
$\bigl[\alpha(\mathbf{x}) \cdot \mathbf{e}(\mathbf{x};\mathcal{G});\; 1-\alpha(\mathbf{x})\bigr] \in \mathbb{R}^{(C+1)}$.

---

## Section 3.2: Architecture

### Equation 6 — Hierarchical query → Gaussian decomposition at timestep $t$
$$\mathcal{G}_t = \Bigl\{\bigl\{\bigl(\mathbf{p}^i + \mathbf{o}^i + \mathbf{o}_j^i,\; \mathbf{v}^i,\; \mathbf{r}_j^i,\; \mathbf{s}_j^i,\; a^i \cdot a_j^i\bigr)\bigr\}_{j=1}^{J}\Bigr\}_{i=1}^{K}$$

where $J$ is the number of Gaussians per query and $K$ the number of queries. Each Gaussian has 3D position $\mathbf{p}^i + \mathbf{o}^i + \mathbf{o}_j^i$ (query position + query offset + its own offset), velocity $\mathbf{v}^i$ inherited from its parent query, rotation $\mathbf{r}_j^i$, scale $\mathbf{s}_j^i$, and opacity $a^i \cdot a_j^i$.

---

## Section 3.3: Stage 1 — 3D Geometry Denoising

### Equation 7 — Noised LiDAR-FPS query initialization
$$\{\mathbf{p}^i\}_{i=0}^{K} = \mathrm{FPS}_K(\mathbf{pts}) + \epsilon$$

where $m$ is the number of LiDAR points, $\epsilon \sim \mathcal{U}(-e, e)^{K\times 3}$, $\mathrm{FPS}_K$ applies Furthest-Point-Sampling to yield $K$ points, and $\mathcal{U}(-e, e)$ is the uniform distribution with hyperparameter $e$.

### Equation 8 — Pretraining loss (denoising + depth + RGB rendering)
$$\mathcal{L} = \lambda_1 \sum_{i=1}^{K} \bigl\|\mathrm{FPS}_K(\mathbf{pts}_t) - (\mathbf{p}^i + \mathbf{o}_t^i)\bigr\| + \lambda_2 \mathcal{L}_{depth}(\mathcal{G}, D) + \lambda_3 \mathcal{L}_{rgb}(\mathcal{G}, I)$$

The first term is the denoising objective; $\mathcal{L}_{depth}$ and $\mathcal{L}_{rgb}$ render depth maps and RGB images from the Gaussians and supervise them with LiDAR-projected depth maps $D_t$ and image observations.

---

## Section 3.4.2: Opacity-Weighted Geometry Estimation

### Equation 9 — Opacity-weighted single-Gaussian occupancy probability
$$\alpha(\mathbf{x};\mathbf{G}) = a \cdot \exp\!\Bigl(-\tfrac{1}{2}(\mathbf{x}-\mathbf{m})^{\mathrm{T}}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\mathbf{m})\Bigr)$$

Adds the opacity term $a$ missing in Eq. 2, so opacity directly influences binary occupancy rather than only mixture weighting.

---

## Appendix A: Evaluation Metrics

### Equation 10 — mIoU / RayIoU
$$\mathrm{mIoU}/\mathrm{RayIoU} = \frac{1}{|C|}\sum_{i \in C} \frac{TP_i}{TP_i + FP_i + FN_i}$$

### Equation 11 — IoU (binary occupied vs. empty)
$$\mathrm{IoU} = \frac{TP_{\neq c_0}}{TP_{\neq c_0} + FP_{\neq c_0} + FN_{\neq c_0}}$$

where $TP_i$, $FP_i$, $FN_i$ are the true positive, false positive, and false negative predictions for class $i$; $C$ is the set of semantic classes; $c_0$ is the empty class. For RayIoU, a query ray is a true positive if the predicted class matches the ground truth and the L1 error between predicted and ground-truth depth is within thresholds 1 m, 2 m, 4 m.

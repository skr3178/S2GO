"""Stage-2 voxel-space losses.

Three terms; selectable via `sem_loss ∈ {'kl', 'ce_lovasz', 'both'}`:

  1. occupancy_bce — per-voxel BCE between predicted occupancy probability
     (from G2V Eq. 1) and binary GT occupancy mask. Always computed.

  2. kl_semantic — KL(softmax(pred_logits) || GT one-hot), per occupied
     voxel. Equivalent to negative log-likelihood when GT is one-hot.

  3. ce_semantic + lovasz_semantic — GaussianFormer-2's recipe. CE on
     argmaxed labels (weighted) + Lovász-softmax for IoU surrogate.

All three masked to non-ignore voxels (GT label != ignore_index, default
none — empty class is treated as a normal class). Loss weights are
exposed as kwargs.

Lovász port:
  Distilled from `reference_code/GaussianFormer/loss/utils/lovasz_softmax.py`
  (Berman 2018, MIT). Cleaned up: removed Variable boilerplate, kept the
  multi-class flat version, default 'present' classes (only sums those
  appearing in GT to avoid empty-mass false-positive credit).
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import NUM_CLASSES, EMPTY_CLASS_ID


# ── Occupancy BCE ──────────────────────────────────────────────────────────
def occupancy_bce(occ_pred: torch.Tensor,
                  occ_gt: torch.Tensor,
                  pos_weight: float = 1.0) -> torch.Tensor:
    """Per-voxel BCE between predicted occ_prob ∈ (0, 1) and GT binary mask.

    Args:
        occ_pred:   (B, Vx, Vy, Vz) probability of occupancy ∈ [0, 1).
        occ_gt:     (B, Vx, Vy, Vz) bool / float — 1 if voxel is occupied.
        pos_weight: scalar weight on the positive-class loss term. Recommended
                    ≈ N_empty / N_occ ratio (~16 for typical Occ3D frames).
                    Combats imbalance-driven "predict empty everywhere"
                    collapse seen on the dense-supervision path.

    G2V's occ_pred is already a probability (Eq. 1 directly), so we use a
    manual weighted BCE.
    """
    eps = 1e-6
    p = occ_pred.clamp(eps, 1 - eps)
    y = occ_gt.float()
    # standard form: -[ pos_weight * y * log p + (1-y) * log(1-p) ]
    loss = -(pos_weight * y * torch.log(p) + (1 - y) * torch.log(1 - p))
    return loss.mean()


# ── KL semantic ────────────────────────────────────────────────────────────
def kl_semantic(sem_logits: torch.Tensor,
                sem_gt: torch.Tensor,
                ignore_index: int = -1) -> torch.Tensor:
    """KL(softmax(sem_logits) || onehot(sem_gt)), masked.

    For one-hot GT this is equivalent to NLL: -log softmax(pred)[gt].
    Implemented via `cross_entropy` which is numerically stable.

    Args:
        sem_logits: (B, Vx, Vy, Vz, C) — class logits.
        sem_gt:     (B, Vx, Vy, Vz)    — long, GT class index.
        ignore_index: voxel labels equal to this are excluded.
    """
    B, Vx, Vy, Vz, C = sem_logits.shape
    logits_flat = sem_logits.reshape(-1, C)
    gt_flat     = sem_gt.reshape(-1)
    return F.cross_entropy(logits_flat, gt_flat,
                            ignore_index=ignore_index,
                            reduction='mean')


# ── CE semantic ────────────────────────────────────────────────────────────
def ce_semantic(sem_logits: torch.Tensor,
                sem_gt: torch.Tensor,
                class_weights: Optional[torch.Tensor] = None,
                ignore_index: int = -1) -> torch.Tensor:
    """Class-weighted cross-entropy over voxels."""
    B, Vx, Vy, Vz, C = sem_logits.shape
    return F.cross_entropy(sem_logits.reshape(-1, C),
                             sem_gt.reshape(-1),
                             weight=class_weights,
                             ignore_index=ignore_index,
                             reduction='mean')


# ── Lovász softmax (multi-class, flat, present-only) ──────────────────────
def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Lovász extension gradient w.r.t sorted errors (Berman et al. 2018)."""
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p].clone() - jaccard[0:-1].clone()
    return jaccard


def lovasz_semantic(sem_logits: torch.Tensor,
                    sem_gt: torch.Tensor,
                    ignore_index: int = -1,
                    classes: str = 'present') -> torch.Tensor:
    """Lovász-softmax loss over voxels, multi-class.

    Args:
        sem_logits: (B, Vx, Vy, Vz, C)
        sem_gt:     (B, Vx, Vy, Vz) long
        ignore_index: skipped voxels
        classes: 'present' (default — only classes that appear in GT) or 'all'
    """
    C = sem_logits.shape[-1]
    probs = sem_logits.softmax(dim=-1).reshape(-1, C)
    labels = sem_gt.reshape(-1)
    if ignore_index is not None and ignore_index >= 0:
        valid = labels != ignore_index
        probs  = probs[valid]
        labels = labels[valid]
    if probs.numel() == 0:
        return sem_logits.sum() * 0.0

    losses = []
    class_iter = range(C) if classes == 'all' else None
    if class_iter is None:
        # 'present' — only classes that appear in this batch's GT
        class_iter = labels.unique().tolist()
    for c in class_iter:
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    if not losses:
        return sem_logits.sum() * 0.0
    return torch.stack(losses).mean()


# ── Combiner ───────────────────────────────────────────────────────────────
def compute_stage2_loss(occ_pred: torch.Tensor,
                          sem_logits: torch.Tensor,
                          occ_gt: torch.Tensor,
                          sem_gt: torch.Tensor,
                          *,
                          sem_loss: str = 'both',
                          w_occ: float = 1.0,
                          w_kl: float = 1.0,
                          w_ce: float = 10.0,
                          w_lovasz: float = 1.0,
                          class_weights: Optional[torch.Tensor] = None,
                          ignore_index: int = -1,
                          occ_pos_weight: float = 1.0,
                          ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Total Stage-2 loss + diagnostic breakdown.

    Args:
        occ_pred:   (B, Vx, Vy, Vz) occupancy probability from G2V.
        sem_logits: (B, Vx, Vy, Vz, C) class logits from G2V.
        occ_gt:     (B, Vx, Vy, Vz) bool / float binary occupancy GT.
        sem_gt:     (B, Vx, Vy, Vz) long class-id GT.
        sem_loss:   'kl' | 'ce_lovasz' | 'both'.
        w_*:        per-term weights.
        class_weights: optional (C,) class balance for ce_semantic.
        ignore_index: -1 to use all voxels; set to EMPTY_CLASS_ID to mask empty.

    Returns:
        (L_total, diag) where diag has per-term .item() floats.
    """
    assert sem_loss in ('kl', 'ce_lovasz', 'both'), sem_loss
    diag: Dict[str, float] = {}

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
        L_ce = ce_semantic(sem_logits, sem_gt,
                            class_weights=class_weights,
                            ignore_index=ignore_index)
        L_lz = lovasz_semantic(sem_logits, sem_gt,
                                ignore_index=ignore_index,
                                classes='present')
        total = total + w_ce * L_ce + w_lovasz * L_lz

    diag['sem_kl']     = float(L_kl.item())
    diag['sem_ce']     = float(L_ce.item())
    diag['sem_lovasz'] = float(L_lz.item())
    diag['total']      = float(total.item())
    return total, diag


# ────────────────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────────────────
def _self_test():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, Vx, Vy, Vz, C = 1, 10, 10, 4, NUM_CLASSES

    # Random "predictions"
    sem_logits = torch.randn(B, Vx, Vy, Vz, C, device=device, requires_grad=True)
    occ_pred = torch.rand(B, Vx, Vy, Vz, device=device, requires_grad=True)
    # GT: ~30% occupied; occupied voxels get random class 0..16, empty get class 17
    occ_gt = (torch.rand(B, Vx, Vy, Vz, device=device) < 0.3)
    sem_gt = torch.full((B, Vx, Vy, Vz), EMPTY_CLASS_ID,
                          device=device, dtype=torch.long)
    sem_gt[occ_gt] = torch.randint(0, NUM_CLASSES - 1,
                                      sem_gt[occ_gt].shape, device=device)

    for mode in ('kl', 'ce_lovasz', 'both'):
        # Re-create grad-enabled tensors per mode (backward consumes them)
        sl = sem_logits.detach().clone().requires_grad_(True)
        op = occ_pred.detach().clone().requires_grad_(True)
        L, diag = compute_stage2_loss(op, sl, occ_gt, sem_gt, sem_loss=mode)
        L.backward()
        print(f"sem_loss={mode}: total={diag['total']:.3f}  "
              f"occ_bce={diag['occ_bce']:.3f}  kl={diag['sem_kl']:.3f}  "
              f"ce={diag['sem_ce']:.3f}  lovasz={diag['sem_lovasz']:.3f}")
        assert torch.isfinite(L), f"{mode}: loss is non-finite"
        assert sl.grad is not None and sl.grad.norm().item() > 0, "no grad on sem_logits"
        assert op.grad is not None and op.grad.norm().item() > 0, "no grad on occ_pred"
    print("losses self-test PASSED.")


if __name__ == "__main__":
    _self_test()

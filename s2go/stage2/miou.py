"""MeanIoU — Stage-2 voxel-level evaluation metric.

Port of `reference_code/GaussianFormer/misc/metric_util.py:MeanIoU`,
cleaned up for our use case:
  - No distributed all_reduce (single-GPU eval).
  - No mmengine logger dependency (caller decides logging format).
  - Per-call accumulation: `update(pred, gt)` for each batch / frame, then
    `compute()` at the end returns dict of per-class IoU + mIoU + occ-IoU.

Two IoUs are computed:
  - Per-class IoU over voxels (standard semantic segmentation metric).
    Empty class is *included* in the class list by default — drop it from
    the mean if you want fg-only mIoU (mask via `ignore_classes`).
  - Occupancy IoU: binary {empty vs not-empty} regardless of class.
"""
from typing import Dict, List, Optional, Sequence

import torch

from . import NUM_CLASSES, EMPTY_CLASS_ID, CLASS_NAMES


class MeanIoU:
    """Streaming per-class voxel IoU.

    Args:
        num_classes:    total class count (default 18; empty included).
        class_names:    optional list of human-readable names (length num_classes).
        empty_label:    class id treated as "empty" for occupancy IoU.
        ignore_classes: per-class IoU mean excludes these (e.g. [empty_label]
                         for fg-only mIoU).
    """
    def __init__(self,
                 num_classes: int = NUM_CLASSES,
                 class_names: Sequence[str] = CLASS_NAMES,
                 empty_label: int = EMPTY_CLASS_ID,
                 ignore_classes: Optional[Sequence[int]] = None):
        assert len(class_names) == num_classes
        self.num_classes  = int(num_classes)
        self.class_names  = list(class_names)
        self.empty_label  = int(empty_label)
        self.ignore_classes = set(ignore_classes or [])
        self.reset()

    def reset(self) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Per-class buckets + a final slot for occupancy.
        self.total_seen     = torch.zeros(self.num_classes + 1, device=device)
        self.total_correct  = torch.zeros(self.num_classes + 1, device=device)
        self.total_positive = torch.zeros(self.num_classes + 1, device=device)

    @torch.no_grad()
    def update(self,
               pred: torch.Tensor,   # (...) long, predicted class ids
               gt: torch.Tensor,     # (...) long, GT class ids
               mask: Optional[torch.Tensor] = None,
               ) -> None:
        """Accumulate counts. pred/gt must share shape (any rank)."""
        assert pred.shape == gt.shape, f"shape mismatch: {pred.shape} vs {gt.shape}"
        if mask is not None:
            pred = pred[mask]
            gt   = gt[mask]
        pred = pred.flatten()
        gt   = gt.flatten()

        for c in range(self.num_classes):
            gt_c = (gt == c)
            pr_c = (pred == c)
            self.total_seen[c]     += gt_c.sum()
            self.total_correct[c]  += (gt_c & pr_c).sum()
            self.total_positive[c] += pr_c.sum()

        # Occupancy bucket
        gt_occ = (gt != self.empty_label)
        pr_occ = (pred != self.empty_label)
        self.total_seen[-1]     += gt_occ.sum()
        self.total_correct[-1]  += (gt_occ & pr_occ).sum()
        self.total_positive[-1] += pr_occ.sum()

    def compute(self) -> Dict:
        """Return per-class IoU, precision, recall + mIoU + occupancy IoU."""
        per_class = {}
        ious_for_mean = []
        for c in range(self.num_classes):
            seen   = float(self.total_seen[c].item())
            correct = float(self.total_correct[c].item())
            pos    = float(self.total_positive[c].item())
            iou = correct / (seen + pos - correct) if (seen + pos - correct) > 0 else float('nan')
            prec = correct / pos if pos > 0 else float('nan')
            reca = correct / seen if seen > 0 else float('nan')
            per_class[self.class_names[c]] = {
                'iou':    iou,
                'prec':   prec,
                'recall': reca,
                'seen':   int(seen),
                'pred':   int(pos),
            }
            if c not in self.ignore_classes and seen > 0:
                ious_for_mean.append(iou)
        miou = (sum(ious_for_mean) / len(ious_for_mean)) if ious_for_mean else float('nan')

        occ_seen    = float(self.total_seen[-1].item())
        occ_correct = float(self.total_correct[-1].item())
        occ_pos     = float(self.total_positive[-1].item())
        occ_iou = occ_correct / (occ_seen + occ_pos - occ_correct) \
            if (occ_seen + occ_pos - occ_correct) > 0 else float('nan')

        return {
            'per_class':       per_class,
            'mIoU':            miou,
            'occupancy_IoU':   occ_iou,
            'n_voxels_seen':   int(self.total_seen.sum().item()),
        }


# ────────────────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────────────────
def _self_test():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = MeanIoU(ignore_classes=[EMPTY_CLASS_ID])

    # Synthetic perfect prediction over 4 frames → mIoU = 1.0, occ_iou = 1.0
    for _ in range(4):
        gt = torch.randint(0, NUM_CLASSES, (200, 200, 16), device=device)
        metric.update(gt, gt)
    stats = metric.compute()
    print(f"Perfect prediction: mIoU = {stats['mIoU']:.3f}  "
          f"occ_IoU = {stats['occupancy_IoU']:.3f}")
    assert stats['mIoU'] > 0.999
    assert stats['occupancy_IoU'] > 0.999

    # Slightly noisy prediction → mIoU < 1
    metric.reset()
    for _ in range(4):
        gt = torch.randint(0, NUM_CLASSES, (200, 200, 16), device=device)
        pr = gt.clone()
        noise = torch.rand_like(gt, dtype=torch.float32) < 0.1
        pr[noise] = torch.randint(0, NUM_CLASSES, (int(noise.sum().item()),),
                                    device=device)
        metric.update(pr, gt)
    stats = metric.compute()
    print(f"10% noise:        mIoU = {stats['mIoU']:.3f}  "
          f"occ_IoU = {stats['occupancy_IoU']:.3f}")
    assert 0.5 < stats['mIoU'] < 1.0
    print("MeanIoU self-test PASSED.")


if __name__ == "__main__":
    _self_test()

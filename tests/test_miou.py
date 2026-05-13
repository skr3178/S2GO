"""MeanIoU correctness benchmark.

Validates `s2go.stage2.miou.MeanIoU` against hand-computable and reference
(numpy confusion-matrix) implementations. Run as:

    python tests/test_miou.py

Exits non-zero on any failure. No pytest dependency.

Cases:
  1. Perfect prediction         → mIoU == 1, occ_IoU == 1
  2. All-empty prediction       → occ_IoU == 0; per-fg-class IoU == 0
  3. Hand-rolled tiny example   → matches paper formula TP / (TP+FP+FN)
  4. Numpy confusion-matrix     → per-class IoU agrees on random data
  5. Streaming equivalence      → 1×N update == K×(N/K) updates
  6. Mask                       → only masked voxels are counted
  7. ignore_classes             → empty class excluded from mIoU mean
  8. seen==0 class dropped      → absent class does not pollute mIoU
  9. Occupancy decoupled        → high occ_IoU with low per-class mIoU
"""
from __future__ import annotations
import math
import os
import sys
import numpy as np
import torch

# Make `s2go` importable when running this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from s2go.stage2 import NUM_CLASSES, EMPTY_CLASS_ID
from s2go.stage2.miou import MeanIoU


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATOL = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Reference: per-class IoU from a confusion matrix (independent impl).
# ─────────────────────────────────────────────────────────────────────────────
def reference_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int):
    """Returns array of per-class IoU (NaN for classes with seen+pred==0)."""
    pred = pred.ravel()
    gt   = gt.ravel()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    # cm[i, j] = # voxels with gt==i and pred==j
    idx = gt * num_classes + pred
    flat = np.bincount(idx, minlength=num_classes * num_classes)
    cm = flat.reshape(num_classes, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(axis=1) - tp                    # GT but not predicted
    fp = cm.sum(axis=0) - tp                    # predicted but not GT
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    return iou, tp, fp, fn


# ─────────────────────────────────────────────────────────────────────────────
# Cases
# ─────────────────────────────────────────────────────────────────────────────
def case_perfect():
    m = MeanIoU(ignore_classes=[EMPTY_CLASS_ID])
    g = torch.randint(0, NUM_CLASSES, (50, 50, 8), device=DEVICE)
    m.update(g.clone(), g)
    s = m.compute()
    assert math.isclose(s['mIoU'], 1.0, abs_tol=ATOL), f"perfect mIoU = {s['mIoU']}"
    assert math.isclose(s['occupancy_IoU'], 1.0, abs_tol=ATOL), \
        f"perfect occ_IoU = {s['occupancy_IoU']}"


def case_all_empty_pred():
    """Pred is entirely empty class → every fg class IoU = 0, occ_IoU = 0."""
    m = MeanIoU(ignore_classes=[EMPTY_CLASS_ID])
    gt = torch.randint(0, NUM_CLASSES - 1, (20, 20, 4), device=DEVICE)  # never empty
    pred = torch.full_like(gt, EMPTY_CLASS_ID)
    m.update(pred, gt)
    s = m.compute()
    # No GT or pred is empty=False on the GT side, so occ_seen > 0 and occ_correct = 0
    assert s['occupancy_IoU'] == 0.0, f"occ_IoU = {s['occupancy_IoU']}"
    # Every fg class with seen>0: correct=0, positive=0, seen>0 → iou = 0/(seen) = 0
    for cname, st in s['per_class'].items():
        if st['seen'] > 0 and cname != 'free':  # 'free' is the empty class label
            assert st['iou'] == 0.0, f"{cname}: iou should be 0, got {st['iou']}"
    # mIoU excludes empty class; all remaining IoUs are 0 → mIoU == 0
    assert s['mIoU'] == 0.0, f"mIoU = {s['mIoU']}"


def case_hand_rolled():
    """Tiny worked example:

        gt   = [0, 0, 0, 1, 1, 2, 2, 17, 17, 17]
        pred = [0, 0, 1, 1, 2, 2, 2, 17, 17,  0]
        cls 0: TP=2  FP=2(idx 2 from gt=0,pred=1?? no — FP counts pred==0 with gt!=0)
               pred==0: idx 0,1,9 → 3 entries; gt at those: 0,0,17 → TP=2 FP=1
               gt==0:   idx 0,1,2 → 3 entries → FN = 3 - 2 = 1
               IoU = 2 / (2 + 1 + 1) = 0.5
        cls 1: pred==1 → idx 2,3 (gt=0,1) → TP=1 FP=1
               gt==1   → idx 3,4 (pred=1,2) → FN=1
               IoU = 1 / (1+1+1) = 1/3
        cls 2: pred==2 → idx 4,5,6 (gt=1,2,2) → TP=2 FP=1
               gt==2   → idx 5,6 → FN=0
               IoU = 2 / (2+1+0) = 2/3
        cls 17 (empty): pred==17 → idx 7,8 (gt=17,17) → TP=2 FP=0
                        gt==17   → idx 7,8,9 → FN=1
                        IoU = 2/3
        Occupancy: not-empty gt = {0,1,2,...,16}
                   gt_occ count = 7 ; pred_occ count = 8 ; correct = 7-1 = 6? recompute:
                   gt_occ: idx 0-6 (7 entries)
                   pr_occ: idx 0,1,2,3,4,5,6,9 (8 entries — idx 9 is pred=0)
                   correct = gt_occ & pr_occ → idx 0-6 ∩ 0,1,2,3,4,5,6,9 = 0-6 (7)
                   occ_IoU = 7 / (7 + 8 - 7) = 7/8
        mIoU excluding empty: mean(0.5, 1/3, 2/3, …all other classes with seen=0 dropped)
                              = (0.5 + 1/3 + 2/3) / 3 = 1.5 / 3 = 0.5
    """
    gt   = torch.tensor([0, 0, 0, 1, 1, 2, 2, 17, 17, 17], device=DEVICE)
    pred = torch.tensor([0, 0, 1, 1, 2, 2, 2, 17, 17,  0], device=DEVICE)
    m = MeanIoU(ignore_classes=[EMPTY_CLASS_ID])
    m.update(pred, gt)
    s = m.compute()

    expected = {
        0:     0.5,
        1:     1.0 / 3.0,
        2:     2.0 / 3.0,
        17:    2.0 / 3.0,
    }
    name_by_id = {i: list(s['per_class'].keys())[i] for i in expected}
    for cid, want in expected.items():
        got = s['per_class'][name_by_id[cid]]['iou']
        assert math.isclose(got, want, abs_tol=ATOL), \
            f"class {cid}: got {got}, want {want}"
    assert math.isclose(s['mIoU'], 0.5, abs_tol=ATOL), f"mIoU = {s['mIoU']}"
    assert math.isclose(s['occupancy_IoU'], 7.0 / 8.0, abs_tol=ATOL), \
        f"occ_IoU = {s['occupancy_IoU']}"


def case_numpy_reference():
    """Random data: per-class IoU must match independent confusion-matrix impl."""
    torch.manual_seed(42)
    np.random.seed(42)
    gt   = torch.randint(0, NUM_CLASSES, (40, 40, 6), device=DEVICE)
    # noisy pred: 30% randomly replaced
    pred = gt.clone()
    flip = torch.rand_like(gt, dtype=torch.float32) < 0.3
    pred[flip] = torch.randint(0, NUM_CLASSES, (int(flip.sum().item()),),
                                 device=DEVICE)

    m = MeanIoU(ignore_classes=[EMPTY_CLASS_ID])
    m.update(pred, gt)
    s = m.compute()

    ref_iou, *_ = reference_iou(pred.cpu().numpy(), gt.cpu().numpy(), NUM_CLASSES)
    names = list(s['per_class'].keys())
    for c in range(NUM_CLASSES):
        got = s['per_class'][names[c]]['iou']
        want = ref_iou[c]
        if math.isnan(want):
            assert math.isnan(got), f"class {c}: ref NaN but metric {got}"
        else:
            assert math.isclose(got, want, abs_tol=1e-6), \
                f"class {c}: got {got}, ref {want}"


def case_streaming_equivalence():
    """1 big update == K smaller updates of the same data."""
    torch.manual_seed(0)
    gt   = torch.randint(0, NUM_CLASSES, (10000,), device=DEVICE)
    pred = torch.randint(0, NUM_CLASSES, (10000,), device=DEVICE)

    m1 = MeanIoU(ignore_classes=[EMPTY_CLASS_ID])
    m1.update(pred, gt)
    s1 = m1.compute()

    m2 = MeanIoU(ignore_classes=[EMPTY_CLASS_ID])
    chunks = 7
    for i in range(chunks):
        sl = slice(i * (10000 // chunks),
                    (i + 1) * (10000 // chunks) if i < chunks - 1 else 10000)
        m2.update(pred[sl], gt[sl])
    s2 = m2.compute()

    assert math.isclose(s1['mIoU'], s2['mIoU'], abs_tol=ATOL), \
        f"mIoU: 1-shot {s1['mIoU']} vs streamed {s2['mIoU']}"
    assert math.isclose(s1['occupancy_IoU'], s2['occupancy_IoU'], abs_tol=ATOL), \
        f"occ_IoU: 1-shot {s1['occupancy_IoU']} vs streamed {s2['occupancy_IoU']}"


def case_mask():
    """Mask should drop voxels from accounting entirely."""
    gt   = torch.tensor([0, 0, 1, 1, 2, 2], device=DEVICE)
    pred = torch.tensor([0, 9, 1, 9, 2, 9], device=DEVICE)
    # Without mask: cls 0 IoU = 1/2, cls 1 = 1/2, cls 2 = 1/2, cls 9 = 0/3 (false)
    # With mask keeping only even indices [0,2,4]: pred & gt agree everywhere → all IoUs 1.0
    mask = torch.tensor([True, False, True, False, True, False], device=DEVICE)
    m = MeanIoU(ignore_classes=[EMPTY_CLASS_ID])
    m.update(pred, gt, mask=mask)
    s = m.compute()
    # Only classes 0,1,2 have seen>0
    names = list(s['per_class'].keys())
    for cid in (0, 1, 2):
        got = s['per_class'][names[cid]]['iou']
        assert math.isclose(got, 1.0, abs_tol=ATOL), f"masked class {cid} IoU={got}"
    assert math.isclose(s['mIoU'], 1.0, abs_tol=ATOL), f"masked mIoU = {s['mIoU']}"


def case_ignore_classes():
    """Empty class included in mean changes the number, exclusion matches hand calc."""
    # Use same gt/pred as case_hand_rolled
    gt   = torch.tensor([0, 0, 0, 1, 1, 2, 2, 17, 17, 17], device=DEVICE)
    pred = torch.tensor([0, 0, 1, 1, 2, 2, 2, 17, 17,  0], device=DEVICE)

    m_inc = MeanIoU(ignore_classes=[])              # include empty
    m_exc = MeanIoU(ignore_classes=[EMPTY_CLASS_ID])  # exclude empty
    m_inc.update(pred, gt); m_exc.update(pred, gt)
    s_inc = m_inc.compute(); s_exc = m_exc.compute()

    # Including empty: (0.5 + 1/3 + 2/3 + 2/3) / 4 = 2.1666… / 4 = 0.5416…
    want_inc = (0.5 + 1/3 + 2/3 + 2/3) / 4
    assert math.isclose(s_inc['mIoU'], want_inc, abs_tol=ATOL), \
        f"include-empty mIoU: got {s_inc['mIoU']}, want {want_inc}"
    assert math.isclose(s_exc['mIoU'], 0.5, abs_tol=ATOL), \
        f"exclude-empty mIoU: got {s_exc['mIoU']}"


def case_absent_class():
    """A class that never appears in GT (seen==0) must be dropped from mean."""
    # GT contains only classes {0, 1}; preds also only {0, 1}; everything else absent.
    gt   = torch.tensor([0, 0, 1, 1, 1, 0], device=DEVICE)
    pred = torch.tensor([0, 1, 1, 1, 0, 0], device=DEVICE)
    # cls 0: pred==0 idx 0,4,5 → TP=2 FP=1 ; gt==0 idx 0,1,5 → FN=1 ; IoU = 2/4 = 0.5
    # cls 1: pred==1 idx 1,2,3 → TP=2 FP=1 ; gt==1 idx 2,3,4 → FN=1 ; IoU = 2/4 = 0.5
    m = MeanIoU(ignore_classes=[EMPTY_CLASS_ID])
    m.update(pred, gt)
    s = m.compute()
    # mIoU = mean(0.5, 0.5) = 0.5 (the 15 absent fg classes do NOT contribute)
    assert math.isclose(s['mIoU'], 0.5, abs_tol=ATOL), \
        f"absent-class mIoU = {s['mIoU']} (should ignore unseen classes)"


def case_occupancy_decoupled():
    """High occ_IoU with low per-class mIoU — semantics should be independent.

    Build data where every non-empty voxel is correctly classified as non-empty
    but the class label is wrong. Occupancy IoU = 1.0; per-class mIoU ~ 0.
    """
    N = 1000
    torch.manual_seed(1)
    # Non-empty GT classes uniformly in [0, NUM_CLASSES-2]
    gt = torch.randint(0, NUM_CLASSES - 1, (N,), device=DEVICE)
    # Pred: shift class by 1 mod (NUM_CLASSES - 1) → always non-empty, always wrong
    pred = (gt + 1) % (NUM_CLASSES - 1)
    m = MeanIoU(ignore_classes=[EMPTY_CLASS_ID])
    m.update(pred, gt)
    s = m.compute()
    assert math.isclose(s['occupancy_IoU'], 1.0, abs_tol=ATOL), \
        f"occ_IoU should be 1.0 (both never empty), got {s['occupancy_IoU']}"
    # Per-class: every voxel is misclassified → TP=0 for every class → IoU=0
    assert math.isclose(s['mIoU'], 0.0, abs_tol=ATOL), \
        f"mIoU should be 0 with all-wrong class, got {s['mIoU']}"


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations of the test cases (saved alongside this file).
# ─────────────────────────────────────────────────────────────────────────────
def _viz_hand_rolled(ax_top, ax_mid, ax_bot):
    """Top: GT vs pred per voxel. Middle: TP/FP/FN per class. Bottom: IoU bars."""
    gt   = np.array([0, 0, 0, 1, 1, 2, 2, 17, 17, 17])
    pred = np.array([0, 0, 1, 1, 2, 2, 2, 17, 17,  0])
    N = len(gt)
    # use distinct hues for cls 0/1/2/17
    color = {0: '#4C9BE8', 1: '#E89B4C', 2: '#5BC57E', 17: '#BBBBBB'}
    x = np.arange(N)
    ax_top.bar(x - 0.18, np.ones(N), width=0.36,
                color=[color[g] for g in gt], edgecolor='black', linewidth=0.4)
    ax_top.bar(x + 0.18, np.ones(N), width=0.36,
                color=[color[p] for p in pred], edgecolor='black', linewidth=0.4)
    for i in range(N):
        ax_top.text(i - 0.18, 1.05, str(gt[i]),   ha='center', fontsize=8)
        ax_top.text(i + 0.18, 1.05, str(pred[i]), ha='center', fontsize=8,
                     color='red' if gt[i] != pred[i] else 'black')
    ax_top.set_xticks(x); ax_top.set_yticks([])
    ax_top.set_xlabel("voxel index"); ax_top.set_ylim(0, 1.25)
    ax_top.set_title("Hand-rolled 10-voxel toy: GT (left bar) vs Pred (right bar). "
                      "Red text = mismatch.", fontsize=10)
    # legend
    from matplotlib.patches import Patch
    ax_top.legend(handles=[
        Patch(facecolor=color[0],  edgecolor='k', label='cls 0'),
        Patch(facecolor=color[1],  edgecolor='k', label='cls 1'),
        Patch(facecolor=color[2],  edgecolor='k', label='cls 2'),
        Patch(facecolor=color[17], edgecolor='k', label='cls 17 (empty)'),
    ], loc='upper right', ncol=4, fontsize=8, frameon=False)

    # TP/FP/FN per class
    classes = [0, 1, 2, 17]
    tp = [int(((gt == c) & (pred == c)).sum()) for c in classes]
    fp = [int(((gt != c) & (pred == c)).sum()) for c in classes]
    fn = [int(((gt == c) & (pred != c)).sum()) for c in classes]
    xc = np.arange(len(classes))
    ax_mid.bar(xc - 0.25, tp, width=0.25, color='#5BC57E', label='TP', edgecolor='k', lw=0.4)
    ax_mid.bar(xc,         fp, width=0.25, color='#E89B4C', label='FP', edgecolor='k', lw=0.4)
    ax_mid.bar(xc + 0.25, fn, width=0.25, color='#E84C5B', label='FN', edgecolor='k', lw=0.4)
    ax_mid.set_xticks(xc); ax_mid.set_xticklabels([f'cls {c}' for c in classes])
    ax_mid.set_ylabel("count"); ax_mid.legend(loc='upper right', fontsize=8, frameon=False)
    ax_mid.set_title("TP / FP / FN per class (the three buckets MeanIoU accumulates)",
                      fontsize=10)
    for i, (a, b, c) in enumerate(zip(tp, fp, fn)):
        ax_mid.text(i - 0.25, a + 0.05, str(a), ha='center', fontsize=7)
        ax_mid.text(i,         b + 0.05, str(b), ha='center', fontsize=7)
        ax_mid.text(i + 0.25, c + 0.05, str(c), ha='center', fontsize=7)

    # measured vs expected IoU
    m = MeanIoU(ignore_classes=[EMPTY_CLASS_ID])
    m.update(torch.tensor(pred, device=DEVICE), torch.tensor(gt, device=DEVICE))
    s = m.compute()
    names = list(s['per_class'].keys())
    measured = [s['per_class'][names[c]]['iou'] for c in classes]
    expected = [0.5, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]
    ax_bot.bar(xc - 0.18, expected, width=0.36, color='#9BB7D4',
                label='hand-computed', edgecolor='k', lw=0.4)
    ax_bot.bar(xc + 0.18, measured, width=0.36, color='#2E5C8A',
                label='MeanIoU.compute()', edgecolor='k', lw=0.4)
    for i, (e, mv) in enumerate(zip(expected, measured)):
        ax_bot.text(i - 0.18, e + 0.02, f"{e:.3f}", ha='center', fontsize=7)
        ax_bot.text(i + 0.18, mv + 0.02, f"{mv:.3f}", ha='center', fontsize=7)
    ax_bot.set_xticks(xc); ax_bot.set_xticklabels([f'cls {c}' for c in classes])
    ax_bot.set_ylim(0, 1.0); ax_bot.set_ylabel("IoU")
    ax_bot.set_title(f"Per-class IoU: hand-computed vs MeanIoU.compute()  "
                      f"|  mIoU(excl. empty) = {s['mIoU']:.3f}, "
                      f"occ_IoU = {s['occupancy_IoU']:.3f}",
                      fontsize=10)
    ax_bot.legend(loc='upper right', fontsize=8, frameon=False)


def _viz_numpy_parity(ax):
    """Scatter of MeanIoU value vs numpy-confusion-matrix reference, per class."""
    torch.manual_seed(42); np.random.seed(42)
    gt   = torch.randint(0, NUM_CLASSES, (40, 40, 6), device=DEVICE)
    pred = gt.clone()
    flip = torch.rand_like(gt, dtype=torch.float32) < 0.3
    pred[flip] = torch.randint(0, NUM_CLASSES, (int(flip.sum().item()),),
                                 device=DEVICE)
    m = MeanIoU(ignore_classes=[EMPTY_CLASS_ID])
    m.update(pred, gt); s = m.compute()
    names = list(s['per_class'].keys())
    metric_vals = np.array([s['per_class'][names[c]]['iou'] for c in range(NUM_CLASSES)])
    ref_vals, *_ = reference_iou(pred.cpu().numpy(), gt.cpu().numpy(), NUM_CLASSES)
    valid = ~np.isnan(metric_vals) & ~np.isnan(ref_vals)
    ax.plot([0, 1], [0, 1], ls='--', color='gray', lw=0.8, label='y = x')
    ax.scatter(ref_vals[valid], metric_vals[valid], c='#2E5C8A', s=40,
                edgecolor='k', lw=0.5, label='per-class IoU')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("numpy confusion-matrix reference IoU")
    ax.set_ylabel("MeanIoU.compute() IoU")
    ax.set_title(f"Reference parity: 30% label noise on 40×40×6 volume\n"
                  f"mIoU = {s['mIoU']:.3f}  |  max |Δ| = "
                  f"{np.nanmax(np.abs(metric_vals - ref_vals)):.2e}",
                  fontsize=10)
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    ax.set_aspect('equal')


def _viz_occupancy_decoupled(ax):
    """Bar chart contrasting occ_IoU=1.0 with mIoU=0.0."""
    N = 1000
    torch.manual_seed(1)
    gt   = torch.randint(0, NUM_CLASSES - 1, (N,), device=DEVICE)
    pred = (gt + 1) % (NUM_CLASSES - 1)
    m = MeanIoU(ignore_classes=[EMPTY_CLASS_ID]); m.update(pred, gt); s = m.compute()
    bars = ['mIoU\n(per-class)', 'occupancy IoU\n(empty vs not)']
    vals = [s['mIoU'], s['occupancy_IoU']]
    colors = ['#E84C5B', '#5BC57E']
    ax.bar(bars, vals, color=colors, edgecolor='k', lw=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.15); ax.set_ylabel("IoU")
    ax.set_title("Pred = (gt + 1) mod 17 — every voxel is non-empty\n"
                  "but assigned the wrong class. Metrics decouple cleanly.",
                  fontsize=10)


def _viz_streaming_equivalence(ax):
    """1-shot mIoU vs streamed mIoU across chunk counts."""
    torch.manual_seed(0)
    N = 10000
    gt   = torch.randint(0, NUM_CLASSES, (N,), device=DEVICE)
    pred = torch.randint(0, NUM_CLASSES, (N,), device=DEVICE)
    chunks_list = [1, 2, 4, 7, 16, 64]
    mious = []
    for k in chunks_list:
        m = MeanIoU(ignore_classes=[EMPTY_CLASS_ID])
        step = (N + k - 1) // k
        for i in range(k):
            sl = slice(i * step, min((i + 1) * step, N))
            m.update(pred[sl], gt[sl])
        mious.append(m.compute()['mIoU'])
    ax.plot(chunks_list, mious, 'o-', color='#2E5C8A', lw=1.4, markersize=7)
    ax.axhline(mious[0], ls='--', color='gray', lw=0.7, label=f'1-shot = {mious[0]:.6f}')
    ax.set_xscale('log')
    ax.set_xticks(chunks_list); ax.set_xticklabels([str(k) for k in chunks_list])
    ax.set_xlabel("# of update() calls (same data, split into N chunks)")
    ax.set_ylabel("mIoU")
    spread = max(mious) - min(mious)
    ax.set_title(f"Streaming equivalence:  spread across chunkings = "
                  f"{spread:.2e}", fontsize=10)
    ax.legend(loc='upper right', fontsize=8, frameon=False)


def save_visualizations(out_dir: str):
    """Generate the figures and write PNGs into out_dir."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # Figure 1 — hand-rolled didactic figure (3 stacked panels)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9),
                              gridspec_kw={'height_ratios': [1.0, 1.0, 1.2]})
    _viz_hand_rolled(*axes)
    fig.suptitle("MeanIoU — hand-rolled toy example  "
                  "(GT/Pred → TP/FP/FN → per-class IoU)", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p1 = os.path.join(out_dir, "miou_case_hand_rolled.png")
    fig.savefig(p1, dpi=110); plt.close(fig)
    print(f"  wrote {p1}")

    # Figure 2 — parity + occupancy decoupled + streaming (1 row, 3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    _viz_numpy_parity(axes[0])
    _viz_occupancy_decoupled(axes[1])
    _viz_streaming_equivalence(axes[2])
    fig.suptitle("MeanIoU — invariants checked by the test suite",
                  fontsize=11, y=1.02)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "miou_invariants.png")
    fig.savefig(p2, dpi=110, bbox_inches='tight'); plt.close(fig)
    print(f"  wrote {p2}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    cases = [
        ('perfect prediction',           case_perfect),
        ('all-empty prediction',         case_all_empty_pred),
        ('hand-rolled tiny example',     case_hand_rolled),
        ('vs numpy confusion-matrix',    case_numpy_reference),
        ('streaming equivalence',        case_streaming_equivalence),
        ('mask drops voxels',            case_mask),
        ('ignore_classes semantics',     case_ignore_classes),
        ('absent class dropped',         case_absent_class),
        ('occupancy decoupled from cls', case_occupancy_decoupled),
    ]
    print(f"MeanIoU benchmark on device={DEVICE}\n")
    failed = 0
    for name, fn in cases:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print()
    if failed:
        print(f"{failed}/{len(cases)} cases failed")
        sys.exit(1)
    print(f"{len(cases)}/{len(cases)} cases passed")

    if "--no-viz" not in sys.argv:
        print("\nWriting visualizations…")
        save_visualizations(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()

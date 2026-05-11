"""Stage-1 pretraining losses (Eq. 8) — denoise + depth + rgb.

L_total = λ_1 · L_denoise  +  λ_2 · L_depth  +  λ_3 · L_rgb           (Eq. 8)
        = 10·L_denoise + 1·L_depth + 1·L_rgb       (defaults — NOT paper-pinned)

Each loss is a small nn.Module returning a scalar. The MultiLoss aggregator
(GF-2 reuse) combines them with weights at training time.

Where each term is computed in the training step:
    Stage1_pseudocode.md Algorithm 1
        L_denoise  ← line 21
        L_depth    ← line 18 (per-dt accumulation)
        L_rgb      ← line 19 (per-dt accumulation)

Run self-test:
    cd /media/skr/storage/self_driving/S2GO
    python -m s2go.losses.pretrain_loss
"""
import torch
import torch.nn as nn

from .ssim import ssim


class DenoiseLoss(nn.Module):
    """L1 between noise-free FPS anchors and refined query positions.

    Eq. 8 first term:  L_denoise = Σ_i  || a_t^i  −  (p_t^i + o_t^i) ||_1

    Paper writes a sum over queries; we average for stable batch-size scaling.
    Caller multiplies by λ_1 (default 10).
    """

    def forward(self, fps_anchors: torch.Tensor, refined_xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fps_anchors: (B, K, 3) noise-free FPS anchors  (target)
            refined_xyz: (B, K, 3) init_xyz + parent.offset  (prediction)

        Returns:
            scalar — mean-over-(B,K) of per-query L1 (summed across xyz).
        """
        if fps_anchors.shape != refined_xyz.shape:
            raise ValueError(f"shape mismatch {fps_anchors.shape} vs {refined_xyz.shape}")
        return (fps_anchors - refined_xyz).abs().sum(dim=-1).mean()


class DepthRenderLoss(nn.Module):
    """L1 between rendered depth and LiDAR-projected sparse depth, masked.

    Eq. 8 second term:  L_depth = ⟨ |D̂ − D^LiDAR| · 1[D^LiDAR > 0] ⟩

    Rendered depth from gsplat (mode='RGB+D' returns alpha-weighted accumulated
    depth; for tighter supervision later, switch to 'RGB+ED' for normalized
    expected depth — the math here works for either).
    """

    def forward(self, rendered_depth: torch.Tensor,
                lidar_depth: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            rendered_depth: (..., H, W)  per-camera rendered depth
            lidar_depth:    (..., H, W)  sparse, 0 where no LiDAR return
            mask: bool, same shape as depth. Default = (lidar_depth > 0).

        Returns:
            scalar — mean L1 over valid pixels. Returns 0 if no valid pixels
            (caller may want to skip the loss step in that degenerate case).
        """
        if rendered_depth.shape != lidar_depth.shape:
            raise ValueError(f"shape mismatch {rendered_depth.shape} vs {lidar_depth.shape}")
        if mask is None:
            mask = lidar_depth > 0
        valid = mask.float()
        diff = (rendered_depth - lidar_depth).abs() * valid
        denom = valid.sum().clamp(min=1.0)
        return diff.sum() / denom


class RGBRenderLoss(nn.Module):
    """0.85 · L1 + 0.15 · (1 − SSIM)  — 3DGS Eq. 7 weighting.

    Form is OUR CHOICE inherited from 3DGS conventions; the S2GO paper just
    writes 'L_rgb' without specifying. Plain L1 is a valid fallback
    (set ssim_weight=0).

    Inputs may be channel-first (..., 3, H, W) or channel-last (..., H, W, 3);
    we normalize internally to channel-first for SSIM.
    """

    def __init__(self, l1_weight: float = 0.85, ssim_weight: float = 0.15,
                 window_size: int = 11):
        super().__init__()
        self.l1_w = float(l1_weight)
        self.ssim_w = float(ssim_weight)
        self.window_size = window_size

    @staticmethod
    def _to_chw(x: torch.Tensor) -> torch.Tensor:
        """Normalize a (..., H, W, 3) or (..., 3, H, W) image to channel-first."""
        if x.shape[-1] == 3 and x.shape[-3] != 3:
            # channel-last → channel-first
            return x.movedim(-1, -3)
        return x

    def forward(self, rendered_rgb: torch.Tensor, image_gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rendered_rgb: (C, H, W, 3) (gsplat output) or (C, 3, H, W)
            image_gt:     (C, H, W, 3) or (C, 3, H, W), same convention

        Returns:
            scalar = l1_w · L1 + ssim_w · (1 − SSIM)
        """
        a = self._to_chw(rendered_rgb)
        b = self._to_chw(image_gt)
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
        l1 = (a - b).abs().mean()
        if self.ssim_w == 0.0:
            return self.l1_w * l1
        ssim_val = ssim(a, b, window_size=self.window_size, size_average=True)
        return self.l1_w * l1 + self.ssim_w * (1.0 - ssim_val)


# ────────────────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────────────────
def _self_test():
    torch.manual_seed(0)
    device = "cuda"
    B, K = 2, 900
    C_cam, H, W = 6, 256, 704

    print("S1.4 Stage-1 losses self-test")

    # ── DenoiseLoss ───────────────────────────────────────────────────────
    den = DenoiseLoss()
    anchors = torch.rand(B, K, 3, device=device)
    # case A: refined == anchors → zero loss
    refined_zero = anchors.detach().clone().requires_grad_(True)
    L0 = den(anchors, refined_zero)
    assert L0.item() == 0.0, f"DenoiseLoss(anchors, anchors) = {L0.item()}, expected 0"
    # case B: refined = anchors + 1m noise → loss ≈ 3.0  (sum over xyz of L1=1 per axis)
    noise = torch.empty_like(anchors).uniform_(-1.0, 1.0)
    refined_noisy = (anchors + noise).requires_grad_(True)
    L1 = den(anchors, refined_noisy)
    expected = noise.abs().sum(dim=-1).mean().item()
    assert abs(L1.item() - expected) < 1e-5, \
        f"DenoiseLoss math: got {L1.item():.4f}, expected {expected:.4f}"
    L1.backward()
    assert refined_noisy.grad is not None and refined_noisy.grad.norm().item() > 0, \
        "DenoiseLoss grad missing"
    print(f"  DenoiseLoss      OK: identity→0, noise→{L1.item():.4f} (expected {expected:.4f}), "
          f"grad ‖·‖ = {refined_noisy.grad.norm().item():.4e}")

    # ── DepthRenderLoss ───────────────────────────────────────────────────
    dep = DepthRenderLoss()
    lidar_d = torch.zeros(C_cam, H, W, device=device)
    # Sparse: ~5% of pixels have valid LiDAR returns at random depths in [3, 10] m
    n_valid = int(0.05 * C_cam * H * W)
    flat_idx = torch.randperm(C_cam * H * W, device=device)[:n_valid]
    lidar_d.view(-1)[flat_idx] = torch.empty(n_valid, device=device).uniform_(3.0, 10.0)
    # case A: rendered == lidar_d on valid pixels → zero loss (after masking)
    rendered_zero = lidar_d.detach().clone().requires_grad_(True)
    LA = dep(rendered_zero, lidar_d)
    assert LA.item() < 1e-6, f"DepthRenderLoss(rendered=lidar, mask) = {LA.item()}, expected ~0"
    # case B: rendered = lidar_d + 1m → loss = 1.0 over valid pixels
    rendered_off = (lidar_d + 1.0).detach().clone().requires_grad_(True)
    LB = dep(rendered_off, lidar_d)
    assert abs(LB.item() - 1.0) < 1e-5, f"DepthRenderLoss(+1m): got {LB.item()}, expected 1.0"
    LB.backward()
    assert rendered_off.grad is not None and rendered_off.grad.norm().item() > 0, \
        "DepthRenderLoss grad missing"
    print(f"  DepthRenderLoss  OK: identity→0, +1m→{LB.item():.4f} (expected 1.0), "
          f"grad ‖·‖ = {rendered_off.grad.norm().item():.4e}")

    # ── RGBRenderLoss ─────────────────────────────────────────────────────
    rgb_loss = RGBRenderLoss(l1_weight=0.85, ssim_weight=0.15)
    img_gt = torch.rand(C_cam, H, W, 3, device=device)
    # case A: rendered == GT → loss ≈ 0 (SSIM=1 → 1-SSIM=0; L1=0)
    rendered_eq = img_gt.detach().clone().requires_grad_(True)
    LA = rgb_loss(rendered_eq, img_gt)
    assert LA.item() < 1e-4, f"RGBRenderLoss(identity) = {LA.item()}, expected ~0"
    # case B: rendered = GT + 0.5 (clamped) → loss > 0 with non-zero grad
    rendered_diff = (img_gt + 0.5).clamp(0, 1).detach().clone().requires_grad_(True)
    LB = rgb_loss(rendered_diff, img_gt)
    assert LB.item() > 0.05, f"RGBRenderLoss(+0.5): {LB.item()} too small"
    LB.backward()
    assert rendered_diff.grad is not None and rendered_diff.grad.norm().item() > 0, \
        "RGBRenderLoss grad missing"
    # case C: pure L1 (ssim_w=0) sanity
    pure_l1 = RGBRenderLoss(l1_weight=1.0, ssim_weight=0.0)
    LC = pure_l1(rendered_diff.detach(), img_gt)
    expected_l1 = (rendered_diff.detach() - img_gt).abs().mean().item()
    assert abs(LC.item() - expected_l1) < 1e-5, \
        f"pure-L1 mode: got {LC.item()}, expected {expected_l1}"
    # case D: channel-first input also works
    rendered_chw = img_gt.movedim(-1, -3)            # (C, 3, H, W)
    img_chw = img_gt.movedim(-1, -3)
    LD = rgb_loss(rendered_chw, img_chw)
    assert LD.item() < 1e-4, f"channel-first identity: {LD.item()}"
    print(f"  RGBRenderLoss    OK: identity→{LA.item():.2e}, +0.5→{LB.item():.4f} "
          f"(grad ‖·‖ = {rendered_diff.grad.norm().item():.4e})")
    print(f"                   pure-L1 matches mean|·|, channel-first input works")

    # ── Eq. 8 weighted combo sanity ───────────────────────────────────────
    L_total = 10.0 * L1 + 1.0 * LB + 1.0 * LB
    assert L_total.item() > 0
    print(f"  Eq. 8 weighted combo OK: λ₁·L_d + λ₂·L_dep + λ₃·L_rgb ≈ {L_total.item():.3f}")

    print("\nS1.4 Stage-1 losses self-test PASSED.")


if __name__ == "__main__":
    _self_test()

"""R50+FPN image backbone for Stage 1 — block [A] in architecture.md.

Wraps mmdet's ResNet (depth=50, ImageNet1k pretrained via torchvision) + FPN
(out_channels = embed_dims = 768). Produces the `feat_flatten` tensor that
S2GOSegmentor.forward expects.

Design choices (Stage1_design.md §2 + this S1.7a step):
  - **R50, ImageNet1k init** — paper §4 line 404: *"S2GO uses the ResNet50
    backbone, S2GO-Small uses 900 queries..."*; paper §B line 1045:
    *"S2GO-Small uses an ImageNet1k backbone."*
  - **FPN out_channels = 768** to match temporal-decoder embed_dims, avoiding a
    separate post-FPN projection layer.
  - **No SECONDFPN** — that's GF-2's pixel-distribution lifter path, which we
    replace with FPS+ε init from LiDAR (see Stage1_design.md §2 reuse map).
  - **Backbone lr scaled ×0.25** — applied at optimizer-config time
    (mmengine `paramwise_cfg`), not here. Paper §B verbatim.

Run self-test:
    cd /media/skr/storage/self_driving/S2GO
    python -m s2go.models.backbone.r50_fpn
"""
from typing import Tuple, List
import torch
import torch.nn as nn

from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN


class R50FPNBackbone(nn.Module):
    """ResNet-50 + FPN, expects (B, N_cam, 3, H, W) → multi-scale features.

    Args:
        embed_dims:        FPN output channels (paper §B: 768)
        num_outs:          # FPN levels (default 4 → strides 4/8/16/32 typical)
        frozen_stages:     # ResNet stages frozen (default 1, matches GF-2)
        pretrained:        load ImageNet1k weights via torchvision (default True)
    """

    def __init__(self,
                 embed_dims: int = 768,
                 num_outs: int = 4,
                 frozen_stages: int = 1,
                 pretrained: bool = True):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_outs = num_outs

        backbone_init_cfg = (
            dict(type='Pretrained', checkpoint='torchvision://resnet50')
            if pretrained else None
        )

        self.backbone = ResNet(
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),                 # all 4 stages → 4 levels for FPN
            frozen_stages=frozen_stages,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=backbone_init_cfg,
        )
        self.neck = FPN(
            in_channels=[256, 512, 1024, 2048],         # ResNet-50 stage outputs
            out_channels=embed_dims,                     # = 768 to match decoder
            num_outs=num_outs,
            start_level=0,
            add_extra_convs='on_output',                  # for num_outs > 4
            relu_before_extra_convs=False,
        )
        if pretrained:
            self.backbone.init_weights()
        self.neck.init_weights()

    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        """
        Args:
            imgs: (B, N_cam, 3, H, W)  RGB in [0, 1] or normalized — caller's choice

        Returns:
            feat_flatten:      (B*N_cam, sum_HW, embed_dims)
            spatial_shapes:    (num_outs, 2) long tensor of (h_l, w_l) per level
            level_start_index: (num_outs,) long tensor of flat-index offsets per level
            levels_hw:         python list of (h_l, w_l) per level (for debugging)
        """
        B, N_cam, C, H, W = imgs.shape
        x = imgs.view(B * N_cam, C, H, W)
        # ResNet → tuple of 4 tensors
        backbone_feats = self.backbone(x)
        # FPN → tuple of `num_outs` tensors of shape (BN, embed_dims, h_l, w_l)
        fpn_feats = self.neck(backbone_feats)

        flats = []
        levels_hw = []
        for f in fpn_feats:
            bn, c, h, w = f.shape
            assert c == self.embed_dims
            flats.append(f.flatten(2).transpose(1, 2))         # (BN, h*w, c)
            levels_hw.append((h, w))
        feat_flatten = torch.cat(flats, dim=1)                  # (BN, sum_HW, c)

        spatial_shapes = torch.tensor(levels_hw, device=imgs.device, dtype=torch.long)
        # level_start_index: cumulative h*w offsets, with leading 0
        starts = [0]
        for h, w in levels_hw[:-1]:
            starts.append(starts[-1] + h * w)
        level_start_index = torch.tensor(starts, device=imgs.device, dtype=torch.long)

        return feat_flatten, spatial_shapes, level_start_index, levels_hw


# ────────────────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────────────────
def _self_test():
    import time
    print("S1.7a R50+FPN backbone self-test")

    device = "cuda"
    B, N_cam = 1, 6
    H, W = 256, 704

    # ── 1. Build with pretrained weights ──────────────────────────────────
    print(f"  building backbone (R50 + FPN, embed_dims=768, ImageNet1k pretrained)…")
    backbone = R50FPNBackbone(embed_dims=768, num_outs=4, pretrained=True).to(device)
    n_back = sum(p.numel() for p in backbone.backbone.parameters())
    n_fpn = sum(p.numel() for p in backbone.neck.parameters())
    print(f"  param counts: ResNet50 = {n_back/1e6:.2f} M, FPN = {n_fpn/1e6:.2f} M, "
          f"total = {(n_back + n_fpn)/1e6:.2f} M")

    # ── 2. Synthetic forward ──────────────────────────────────────────────
    imgs = torch.rand(B, N_cam, 3, H, W, device=device)
    backbone.eval()
    with torch.no_grad():
        t0 = time.time()
        feat, ss, lsi, hw = backbone(imgs)
        torch.cuda.synchronize()
        dt = time.time() - t0
    print(f"  synthetic forward: feat={tuple(feat.shape)}, "
          f"levels_hw={hw}, sum_HW={feat.shape[1]}, dt={dt*1000:.1f} ms")
    assert feat.shape[0] == B * N_cam
    assert feat.shape[2] == 768
    assert feat.shape[1] == sum(h * w for h, w in hw)
    assert ss.shape == (4, 2)
    assert lsi.shape == (4,)

    # Expected level shapes for 256×704 input with ResNet/FPN strides 4/8/16/32:
    expected = [(64, 176), (32, 88), (16, 44), (8, 22)]
    assert hw == expected, f"FPN levels: got {hw}, expected {expected}"
    print(f"  level shapes match expected: {hw}")

    # ── 3. Pretrained weight loaded? Check first conv weight isn't random ──
    w0 = backbone.backbone.conv1.weight
    # ImageNet pretrained R50 conv1 has a known statistical signature
    # (orange-ish edge filters); checking the std is in a reasonable range.
    std = w0.std().item()
    assert 0.05 < std < 0.5, f"conv1.weight std={std:.4f} — likely not pretrained"
    print(f"  conv1.weight std = {std:.4f}  → pretrained weights look loaded")

    # ── 4. Real-data forward (one frame from S1.7b loader) ────────────────
    print(f"  loading 1 real nuScenes frame for backbone test…")
    from ...datasets.nusc_loader import NuScenesLoader
    loader = NuScenesLoader(T=1, verbose=False)
    seq = loader[0]
    real_imgs = seq[0]['imgs'].to(device)                      # (1, 6, 3, 256, 704)
    backbone.eval()
    with torch.no_grad():
        feat_r, ss_r, lsi_r, hw_r = backbone(real_imgs)
    print(f"  real-data forward: feat={tuple(feat_r.shape)}, "
          f"feat range [{feat_r.min().item():.3f}, {feat_r.max().item():.3f}]")
    assert feat_r.shape == feat.shape

    # ── 5. End-to-end with segmentor (real images → segmentor → loss) ─────
    print(f"  end-to-end with S2GOSegmentor (1 frame, real imgs + real lidar)…")
    from ..segmentor import S2GOSegmentor
    seg = S2GOSegmentor(
        K=900, J=10, embed_dims=768,
        num_layers=2, num_heads=12, num_groups=12,
        num_levels=4, num_cams=6,
        num_pts=4, feedforward_channels=2048,
        T_queue=1,
    ).to(device)

    backbone.train()                # backbone in train mode for grad flow
    seg.train()

    feat_t, ss_t, lsi_t, _ = backbone(real_imgs)
    frame = {
        'lidar_pts':         seq[0]['lidar_pts'].to(device),
        'feat_flatten':      feat_t,
        'spatial_shapes':    ss_t,
        'level_start_index': lsi_t,
        'lidar2img':         seq[0]['lidar2img'].to(device),
        'pad_h':             256, 'pad_w': 704,
        'ego_pose':          seq[0]['ego_pose'].to(device),
        'ego_pose_inv':      seq[0]['ego_pose_inv'].to(device),
        'timestamp':         seq[0]['timestamp'].to(device),
        'prev_exists':       seq[0]['prev_exists'].to(device),
    }
    seg.reset_memory()
    out = seg.forward_one_frame(frame)
    assert out.gaussians.means.shape == (1, 9000, 3)
    L = out.gaussians.means.norm() + out.parent.opa.sum() + feat_t.sum() * 0.01
    L.backward()
    # Verify backbone got gradient on UNFROZEN stages.
    # frozen_stages=1 (paper default) freezes stem + layer1; layer2-4 train.
    conv1_grad = backbone.backbone.conv1.weight.grad
    layer4_grad = backbone.backbone.layer4[-1].conv1.weight.grad
    fpn_grad = backbone.neck.fpn_convs[0].conv.weight.grad
    assert conv1_grad is None, "frozen conv1 should NOT have grad"
    assert layer4_grad is not None and layer4_grad.norm().item() > 0, \
        "unfrozen layer4 got no gradient"
    assert fpn_grad is not None and fpn_grad.norm().item() > 0, "FPN got no gradient"
    print(f"  end-to-end OK: layer4.grad ‖·‖ = {layer4_grad.norm().item():.4e}, "
          f"fpn.grad ‖·‖ = {fpn_grad.norm().item():.4e}")
    print(f"  frozen-stages behavior OK: conv1 (frozen) has no grad, layer4 (trainable) does")
    print(f"  output Gaussians = {tuple(out.gaussians.means.shape)}")

    print("\nS1.7a R50+FPN backbone self-test PASSED.")


if __name__ == "__main__":
    _self_test()

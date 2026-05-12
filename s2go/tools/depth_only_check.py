"""Single-frame depth-only diagnostic.

Tests whether L_depth can drive the model toward a recognizable scene given:
  - one frame (no temporal queue)
  - only L_depth at dt=0 active (no denoise, no rgb, no warp)
  - constant LR (no cosine decay)
  - backbone in eval() mode (frozen BN stats)
  - dropout=0 in the decoder (deterministic)

Threshold: L_depth should drop below ~0.5 m within ~500 iters. If it plateaus
above 2 m, there's a structural problem with the depth pathway (Gaussian
state can't represent the scene geometry).

FPS is seeded each forward so the LiDAR anchor pattern is stable across iters
(otherwise the random_start=True default makes init_xyz jitter, which could
alone prevent convergence — see hacks.md H1).

Run:
    cd /media/skr/storage/self_driving/S2GO
    python -m s2go.tools.depth_only_check
"""
import torch
import torch.nn as nn
from torch.optim import AdamW

from ..datasets.nusc_loader import NuScenesLoader
from ..models.backbone.r50_fpn import R50FPNBackbone
from ..models.segmentor import S2GOSegmentor
from ..render.gsplat_wrapper import render
from ..losses.pretrain_loss import DepthRenderLoss


def _gn(linear: nn.Linear, sl: slice) -> float:
    if linear.weight.grad is None:
        return 0.0
    g = linear.weight.grad[sl]
    return float(g.norm().item())


def main():
    device = "cuda"
    torch.manual_seed(0)

    print("Loading 1 sequence with T=1 (single frame)…")
    loader = NuScenesLoader(T=1, verbose=False)
    seq = loader[0]
    f = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
         for k, v in seq[0].items()}

    print("Building backbone + T=1 segmentor (paper-spec arch)…")
    backbone = R50FPNBackbone(embed_dims=768, num_outs=4, pretrained=True).to(device)
    seg = S2GOSegmentor(K=900, J=10, embed_dims=768,
                        num_layers=6, num_pts=13,
                        feedforward_channels=3072,
                        T_queue=1, dropout=0.0).to(device)
    backbone.eval()
    seg.train()                  # decoder needs train mode for grad flow

    optim = AdamW(list(backbone.parameters()) + list(seg.parameters()), lr=1e-4)
    dep_loss = DepthRenderLoss()
    parent_head = seg.parent_refiner.head
    child_head  = seg.child_head.head

    print(f"\n  iter | L_depth (m) | gnorm_par_off | gnorm_ch_off | gnorm_scale "
          f"| gnorm_opa | gnorm_velocity | opa_mean | scale_l2 ")
    print(f"  -----+-------------+---------------+--------------+-------------"
          f"+-----------+----------------+----------+----------")
    N_ITERS = 500
    log_iters = {0, 1, 2, 5, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499}
    for i in range(N_ITERS):
        torch.manual_seed(42)        # stable FPS anchors across iters

        feat, ss, lsi, _ = backbone(f['imgs'])
        f_w = {**f, 'feat_flatten': feat, 'spatial_shapes': ss,
               'level_start_index': lsi, 'pad_h': 256, 'pad_w': 704}
        outputs = seg([f_w])
        out = outputs[0]

        # gsplat needs fp32
        G_fp32 = out.gaussians._replace(
            means=out.gaussians.means.float(),
            scales=out.gaussians.scales.float(),
            rotations=out.gaussians.rotations.float(),
            opacities=out.gaussians.opacities.float(),
            colors=None,                            # render_mode='D' ignores colors
            velocity=out.gaussians.velocity.float(),
        )
        viewmats = f['viewmats'][0].float()
        Ks       = f['cam_K'][0].float()
        _, depth, _ = render(G_fp32, viewmats=viewmats, Ks=Ks,
                             height=256, width=704, render_mode='D')
        L = dep_loss(depth, f['lidar_depth'][0])

        optim.zero_grad()
        L.backward()

        po  = _gn(parent_head, slice(0, 3))
        opa_par = _gn(parent_head, slice(3, 4))
        vel = _gn(parent_head, slice(4, 7))
        co  = _gn(child_head,  slice(0, 3))
        sc  = _gn(child_head,  slice(3, 6))
        opa_ch = _gn(child_head,  slice(10, 11))
        opa_total = (opa_par ** 2 + opa_ch ** 2) ** 0.5

        opa_mean = float(out.gaussians.opacities.detach().float().mean().item())
        scale_l2 = float(out.gaussians.scales.detach().float().norm(dim=-1).mean().item())

        optim.step()

        if i in log_iters:
            print(f"  {i:>4} | {L.item():>11.4f} | {po:>13.2e} | {co:>12.2e} "
                  f"| {sc:>11.2e} | {opa_total:>9.2e} | {vel:>14.2e} "
                  f"| {opa_mean:>8.3f} | {scale_l2:>8.3f}")

    print(f"\nFinal L_depth: {L.item():.4f} m  (target: < 0.5 m)")


if __name__ == "__main__":
    main()

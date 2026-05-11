"""Single-frame denoise-only diagnostic.

Tests whether the parent_offset head can drive L_denoise → ~0 when given:
  - one frame (no temporal queue)
  - only L_denoise active (no depth, RGB, or warp)

Threshold: L_denoise should drop below 0.01 m within ~50 iters. If it plateaus
above that, the position pathway has a connectivity/loss-form problem.

FPS is seeded each forward so the denoise target is stable across iters
(otherwise the random_start=True default makes the target jitter, which alone
could prevent convergence — see hacks.md H1).
"""
import torch
import torch.nn as nn
from torch.optim import AdamW

from ..datasets.nusc_loader import NuScenesLoader
from ..models.backbone.r50_fpn import R50FPNBackbone
from ..models.segmentor import S2GOSegmentor
from ..losses.pretrain_loss import DenoiseLoss


def main():
    device = "cuda"
    torch.manual_seed(0)

    print("Loading 1 sequence with T=1 (single frame)…")
    loader = NuScenesLoader(T=1, verbose=False)
    seq = loader[0]
    f = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
          for k, v in seq[0].items()}

    import os
    DISABLE_XATTN = os.environ.get('DISABLE_XATTN', '0') == '1'

    print("Building backbone + T=1 segmentor (paper-spec arch)…")
    backbone = R50FPNBackbone(embed_dims=768, num_outs=4, pretrained=True).to(device)
    seg = S2GOSegmentor(K=900, J=10, embed_dims=768,
                         num_layers=6, num_pts=13,
                         feedforward_channels=3072,
                         T_queue=1, dropout=0.0).to(device)
    # Freeze BN running stats + disable backbone training-mode noise.
    backbone.eval()
    seg.train()  # decoder needs train mode for grad flow, dropout=0.0 makes it deterministic

    if DISABLE_XATTN:
        # Test 2: bypass deformable cross-attn in every decoder layer. After
        # the bypass, each layer is effectively self-attn → norm → FFN → norm,
        # with no image-feature input. Eliminates any competing pull on
        # parent_offset from image features.
        print("Cross-attention DISABLED (Test 2)")
        for layer in seg.decoder.layers:
            layer.cross_attn = nn.Identity()
            # Replace the residual call with a no-op: the layer's forward does
            # `query = norm2(query + cross_attn(query, ...))`. We need
            # cross_attn(...) → 0 instead of identity. Wrap as a callable:
            class _Zero(nn.Module):
                def forward(self, query, *args, **kwargs):
                    return torch.zeros_like(query)
            layer.cross_attn = _Zero()

    optim = AdamW(list(backbone.parameters()) + list(seg.parameters()), lr=1e-4)
    den_loss = DenoiseLoss()
    parent_head = seg.parent_refiner.head

    print(f"\n  iter | L_denoise (m) | gnorm_parent_offset | gnorm_velocity | gnorm_child_offset")
    print(f"  -----+---------------+---------------------+----------------+--------------------")
    N_ITERS = 500
    for i in range(N_ITERS):
        # Seed FPS each iter so the denoise target is deterministic
        torch.manual_seed(42)

        feat, ss, lsi, _ = backbone(f['imgs'])
        f_w = {**f, 'feat_flatten': feat, 'spatial_shapes': ss,
                'level_start_index': lsi, 'pad_h': 256, 'pad_w': 704}
        outputs = seg([f_w])
        out = outputs[0]
        L = den_loss(out.anchors_xyz, out.refined_xyz)

        optim.zero_grad()
        L.backward()

        def _g(slc):
            g = parent_head.weight.grad
            return float(g[slc].norm().item()) if g is not None else 0.0
        po = _g(slice(0, 3))         # parent offset head rows
        vel = _g(slice(4, 7))         # velocity head rows
        ch = seg.child_head.head.weight.grad
        co = float(ch[0:3].norm().item()) if ch is not None else 0.0

        optim.step()
        log_iters = [0, 1, 2, 3, 4, 5, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 499]
        if i in log_iters:
            print(f"  {i:>4} | {L.item():>13.5f} | {po:>19.2e} | "
                  f"{vel:>14.2e} | {co:>18.2e}")

    print(f"\nFinal L_denoise: {L.item():.5f} m  (target: < 0.01 m)")


if __name__ == "__main__":
    main()

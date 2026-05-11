"""Minimal Stage-1 overfit training test (S1.7e + S1.7f).

What this proves:
  - Real nuScenes data flows through the full Stage-1 stack (backbone
    → segmentor → gsplat → losses) and returns a finite scalar loss.
  - loss.backward() updates weights through every block.
  - Eq. 8 losses (denoise, depth, rgb) decrease monotonically over 50 iters
    on 4 fixed T=4 sequences (T0/T1-tier per Stage1_design.md §7a).

What this does NOT prove:
  - Full-scale paper-rep convergence (that's T2, ~12 hr on Part 1).
  - Stage 2 / occupancy mIoU.

Run:
    cd /media/skr/storage/self_driving/S2GO
    python -m s2go.tools.overfit
"""
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..datasets.nusc_loader import NuScenesLoader
from ..models.backbone.r50_fpn import R50FPNBackbone
from ..models.segmentor import S2GOSegmentor
from ..render.gsplat_wrapper import render
from ..losses.pretrain_loss import DenoiseLoss, DepthRenderLoss, RGBRenderLoss


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def to_device(d: dict, device) -> dict:
    """Move all tensor values in a frame dict onto the given device."""
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in d.items()}


def compute_stage1_loss(outputs, sequence, lambdas, losses, device):
    """Render each frame's gaussians, compute Eq. 8 weighted sum.

    outputs:  list of T FrameOutput from S2GOSegmentor
    sequence: list of T frame dicts (already on device)
    lambdas:  (lambda_denoise, lambda_depth, lambda_rgb)
    losses:   (DenoiseLoss, DepthRenderLoss, RGBRenderLoss) instances
    """
    den_loss, dep_loss, rgb_loss = losses
    lam_d, lam_dp, lam_rgb = lambdas
    L_d, L_dp, L_rgb = 0.0, 0.0, 0.0

    for t, (out, frame) in enumerate(zip(outputs, sequence)):
        # Use the loader's proper LIDAR_TOP→cam decomposition (S1.7-fix step):
        #   lidar2img = K_4x4 @ viewmat  →  pass viewmat + K separately to gsplat.
        # Our Gaussians are in LIDAR_TOP frame (the lifter samples from raw
        # lidar_pts), so this viewmat correctly maps them into each camera's
        # frame — rendered RGB/depth then align with the actual camera images.
        viewmats = frame['viewmats'][0]                     # (N_cam, 4, 4)
        Ks       = frame['cam_K'][0]                        # (N_cam, 3, 3)

        # gsplat's fully_fused_projection requires fp32; if outer autocast is on
        # (T2 mixed precision), cast the Gaussians + cam matrices to fp32 here
        # and run render outside autocast.
        with torch.cuda.amp.autocast(enabled=False):
            G_fp32 = out.gaussians._replace(
                means=out.gaussians.means.float(),
                scales=out.gaussians.scales.float(),
                rotations=out.gaussians.rotations.float(),
                opacities=out.gaussians.opacities.float(),
                colors=(out.gaussians.colors.float()
                         if out.gaussians.colors is not None else None),
                velocity=out.gaussians.velocity.float(),
            )
            rgb, depth, _alpha = render(
                G_fp32, viewmats=viewmats.float(), Ks=Ks.float(),
                height=256, width=704, render_mode='RGB+D')

        # Per-frame loss accumulation
        L_d  = L_d  + den_loss(out.anchors_xyz, out.refined_xyz)
        L_dp = L_dp + dep_loss(depth, frame['lidar_depth'][0])
        L_rgb = L_rgb + rgb_loss(rgb, frame['imgs'][0].permute(0, 2, 3, 1))

    T = len(outputs)
    L_total = (lam_d * L_d + lam_dp * L_dp + lam_rgb * L_rgb) / T
    return L_total, {'denoise': (L_d / T).item(),
                      'depth':   (L_dp / T).item(),
                      'rgb':     (L_rgb / T).item()}


# ────────────────────────────────────────────────────────────────────────────
# Overfit run
# ────────────────────────────────────────────────────────────────────────────
def main(n_iters: int = 50, n_overfit: int = 4, log_every: int = 5,
         lr: float = 4e-4, lr_backbone_mult: float = 0.25,
         num_layers: int = 2, num_pts: int = 4,
         feedforward_channels: int = 2048,
         mixed_precision: bool = False,
         amp_dtype: str = 'bf16',
         rgb_ssim_weight: float = 0.15,
         history_path: str = None):
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.empty_cache()

    print(f"S1.7e/f overfit run — {n_iters} iters on {n_overfit} fixed sequences\n")

    # ── Data: load N_overfit fixed T=4 sequences once, cache on device ────
    print("[1/4] Loading and caching overfit data…")
    loader = NuScenesLoader(T=4, verbose=False)
    sequences = []
    for i in range(n_overfit):
        seq = loader[i]                             # 4 frame-dicts, CPU
        seq_gpu = [to_device(f, device) for f in seq]
        sequences.append(seq_gpu)
    print(f"    {n_overfit} sequences × T=4 frames cached on {device}")

    # ── Model: backbone + segmentor (T0-tier sizing for the 3060) ─────────
    print("[2/4] Building model…")
    backbone = R50FPNBackbone(embed_dims=768, num_outs=4, pretrained=True).to(device)
    seg = S2GOSegmentor(
        K=900, J=10, embed_dims=768,
        num_layers=num_layers,        # paper-spec: 6 (T2); we default 2 (T0/T1)
        num_heads=12, num_groups=12,
        num_levels=4, num_cams=6,
        num_pts=num_pts,              # paper-spec: 13 (T2); we default 4 (T0/T1)
        feedforward_channels=feedforward_channels,  # paper-spec: 3072 (T2); 2048 (T0/T1)
        T_queue=4,
    ).to(device)
    n_back = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_seg  = sum(p.numel() for p in seg.parameters() if p.requires_grad)
    print(f"    backbone: {n_back/1e6:.1f} M trainable, segmentor: {n_seg/1e6:.1f} M trainable, "
          f"total: {(n_back+n_seg)/1e6:.1f} M")

    # ── Optimizer: AdamW with backbone-lr-scaled paramwise group (paper §B) ─
    print("[3/4] Building optimizer (AdamW, lr=4e-4, backbone ×0.25)…")
    optim = AdamW([
        {'params': [p for p in backbone.parameters() if p.requires_grad],
         'lr': lr * lr_backbone_mult},
        {'params': seg.parameters(), 'lr': lr},
    ], weight_decay=0.01)
    scheduler = CosineAnnealingLR(optim, T_max=n_iters)

    # Stage-1 losses (Eq. 8); λ defaults from Stage1_design.md D6
    rgb_l1_w = 1.0 - rgb_ssim_weight if rgb_ssim_weight > 0 else 1.0
    losses = (DenoiseLoss(), DepthRenderLoss(),
                RGBRenderLoss(l1_weight=rgb_l1_w, ssim_weight=rgb_ssim_weight))
    lambdas = (10.0, 1.0, 1.0)
    print(f"    RGB loss form: {rgb_l1_w:.2f}·L1 + {rgb_ssim_weight:.2f}·(1-SSIM)"
          f"{' (SSIM disabled)' if rgb_ssim_weight == 0 else ''}")

    # Mixed precision (paper §B) — autocast. fp16 needs GradScaler; bf16 has
    # fp32-equivalent range so no scaling needed.
    use_bf16 = mixed_precision and amp_dtype == 'bf16'
    use_fp16 = mixed_precision and amp_dtype == 'fp16'
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    if mixed_precision:
        kind = 'bf16 autocast' if use_bf16 else 'fp16 autocast + GradScaler'
        print(f"    mixed precision: ENABLED ({kind})")
    else:
        print(f"    mixed precision: OFF (fp32)")

    # ── Train loop ────────────────────────────────────────────────────────
    print(f"[4/4] Training for {n_iters} iters…\n")
    print(f"  iter |   total |   L_den |  L_dep |  L_rgb |   lr   | sec | mem MB")
    print(f"  -----+---------+---------+--------+--------+--------+-----+-------")
    seg.train(); backbone.train()
    history = []
    t_start = time.time()
    for i in range(n_iters):
        seq = sequences[i % n_overfit]                 # cycle through the 4

        # Backbone + FPN run in FP32 — torch 2.0's FPN upsample_nearest2d
        # has no bf16 kernel. The backbone is small relative to the segmentor,
        # so fp32 here doesn't materially affect memory.
        sequence_with_feat = []
        for f in seq:
            feat, ss, lsi, _ = backbone(f['imgs'])
            sequence_with_feat.append({
                **f,
                'feat_flatten':       feat,
                'spatial_shapes':     ss,
                'level_start_index':  lsi,
                'pad_h': 256, 'pad_w': 704,
            })

        # Segmentor (T-frame loop with memory queue) under autocast — this is
        # where the memory savings live (6-layer decoder, num_pts=13 cross-attn).
        with torch.cuda.amp.autocast(enabled=mixed_precision, dtype=autocast_dtype):
            outputs = seg(sequence_with_feat)
            L, parts = compute_stage1_loss(outputs, sequence_with_feat, lambdas, losses, device)

        # Backward + step (paper §B: gradient clip max_norm=35)
        optim.zero_grad()
        scaler.scale(L).backward()
        scaler.unscale_(optim)
        gnorm = nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(seg.parameters()), max_norm=35.0)
        scaler.step(optim)
        scaler.update()
        scheduler.step()

        history.append({**parts, 'total': L.item(), 'gnorm': gnorm.item()})

        if i % log_every == 0 or i == n_iters - 1:
            elapsed = time.time() - t_start
            mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  {i:>4} | {L.item():>7.3f} | {parts['denoise']:>7.3f} | "
                  f"{parts['depth']:>6.3f} | {parts['rgb']:>6.4f} | "
                  f"{scheduler.get_last_lr()[0]:.1e} | {elapsed:>3.0f} | {mem:>6.0f}")

    print()
    # ── Verify monotonic-ish decrease ─────────────────────────────────────
    first_5 = sum(h['total'] for h in history[:5]) / 5
    last_5  = sum(h['total'] for h in history[-5:]) / 5
    drop = first_5 - last_5
    print(f"Final summary:")
    print(f"  loss avg  first 5 iters: {first_5:.3f}")
    print(f"  loss avg  last  5 iters: {last_5:.3f}")
    print(f"  Δ = {drop:+.3f}  ({'GOOD' if drop > 0 else 'BAD: loss did not drop'})")
    den_drop = (sum(h['denoise'] for h in history[:5]) -
                sum(h['denoise'] for h in history[-5:])) / 5
    print(f"  L_denoise drop: {den_drop:+.3f} m  (target: significantly positive)")
    if drop > 0:
        print("\nS1.7e/f overfit smoke PASSED.")
    else:
        print("\nS1.7e/f overfit smoke FAILED — loss not decreasing.")

    # Optional history dump (json) for later plotting
    if history_path:
        import json
        with open(history_path, 'w') as fp:
            json.dump(history, fp, indent=2)
        print(f"  history saved to {history_path}")
    return history


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-iters", type=int, default=50)
    p.add_argument("--num-layers", type=int, default=2,
                    help="paper-spec T2: 6; T0/T1 default: 2")
    p.add_argument("--num-pts", type=int, default=4,
                    help="paper-spec T2: 13; T0/T1 default: 4")
    p.add_argument("--feedforward-channels", type=int, default=2048,
                    help="paper-spec T2: 3072; T0/T1 default: 2048")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--mixed-precision", action="store_true",
                    help="enable autocast. REQUIRED for T2 on 12 GB GPUs.")
    p.add_argument("--amp-dtype", choices=['bf16', 'fp16'], default='bf16',
                    help="mixed-precision dtype (default bf16 — fp32 range, fp16 memory)")
    p.add_argument("--rgb-ssim-weight", type=float, default=0.15,
                    help="SSIM weight in L_rgb; set to 0 for pure-L1 (saves ~250 MB GPU mem)")
    p.add_argument("--history-path", type=str, default=None)
    a = p.parse_args()
    main(n_iters=a.n_iters, log_every=a.log_every,
         num_layers=a.num_layers, num_pts=a.num_pts,
         feedforward_channels=a.feedforward_channels,
         mixed_precision=a.mixed_precision, amp_dtype=a.amp_dtype,
         rgb_ssim_weight=a.rgb_ssim_weight,
         history_path=a.history_path)

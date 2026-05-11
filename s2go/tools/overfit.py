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


def _head_grad_norm(linear: nn.Linear, row_slice: slice) -> float:
    """L2 norm of the (weight ∪ bias) gradient for a row-slice of a Linear head.

    Used to attribute total gradient flow to a specific output channel-group of
    a multi-head Linear (e.g. row 0:3 of ParentRefiner.head = offset channels).
    Returns 0.0 if grad is None (e.g., before first backward, or if that slice
    receives no signal at all — the case that gives us the velocity A/B proof).
    """
    if linear.weight.grad is None:
        return 0.0
    w_g = linear.weight.grad[row_slice]
    b_g = linear.bias.grad[row_slice] if linear.bias is not None else None
    sq = (w_g ** 2).sum()
    if b_g is not None:
        sq = sq + (b_g ** 2).sum()
    return float(sq.sqrt().item())


def _gaussian_state_stats(G) -> dict:
    """mean / std / min / max for the diagnostic-relevant Gaussian channels.

    Operates on the LAST forward's Gaussians (typical use is the last frame
    of the sequence, since that's what most-recently produced loss signal).
    """
    op = G.opacities.detach().float().flatten()
    # scale is (B, N, 3) → take per-Gaussian L2 norm
    sc = G.scales.detach().float().norm(dim=-1).flatten()
    v  = G.velocity.detach().float().norm(dim=-1).flatten()

    def _stats(t):
        return {
            'mean': float(t.mean().item()),
            'std':  float(t.std().item()),
            'min':  float(t.min().item()),
            'max':  float(t.max().item()),
        }
    return {
        'opacity':       _stats(op),
        'scale_l2':      _stats(sc),
        'velocity_mag':  _stats(v),
    }


def _queue_propagator_stats(seg, last_prop) -> dict:
    """Memory-queue + propagator diagnostics: count, uniqueness, opacity spread.

    Note: queue buffers may be bf16/fp16 if the segmentor ran under autocast.
    `torch.unique(…, dim=…)` is not implemented for bf16, so we cast to fp32
    before the uniqueness check.
    """
    q_rp = seg.queue.memory_reference_point     # (B, L, 3) or None
    if q_rp is None:
        memory_count = 0
        unique_ref = 0
    else:
        rp = q_rp[0].detach().float()            # (L, 3) first batch, fp32
        # "memory_count" = slots that are non-zero (zero-init padding before fill)
        nonzero_mask = (rp.abs().sum(dim=-1) > 0)
        memory_count = int(nonzero_mask.sum().item())
        # "unique_ref_points" = distinct (xyz) tuples among the populated rows
        if memory_count > 0:
            populated = rp[nonzero_mask]
            unique_ref = int(torch.unique(populated.round(decimals=4), dim=0).shape[0])
        else:
            unique_ref = 0
    # Top-k opacity stats from the last frame's propagator output
    topk_opa = last_prop.opa.detach().float().flatten()
    return {
        'memory_count':         memory_count,
        'unique_ref_points':    unique_ref,
        'topk_opacity_min':     float(topk_opa.min().item()),
        'topk_opacity_max':     float(topk_opa.max().item()),
        'topk_opacity_mean':    float(topk_opa.mean().item()),
    }


def _warp_gaussians_means(G, dt: float, T_t_to_n: torch.Tensor):
    """Position-only warp: shift by velocity·dt, then SE(3)-transform to neighbor frame.

    Implements the description in S2GO §3.3.3 (after Eq. 8):
        "the rendering supervision is done on current and neighboring keyframes
         (+/- 0.5s) by moving the Gaussians with predicted velocities v and
         accounting for ego-motion."

    The paper does not specify whether scale/rotation are also transformed; we
    apply translation only (position + velocity·dt). For short ±0.5s windows
    on nuScenes (~5-15 m/s ego speed; small rotations), this is the simplest
    reading consistent with the prose.

    Args:
        G:        current-frame Gaussians; means in t-frame LIDAR_TOP coords
        dt:       signed seconds (e.g. +0.5 for next keyframe, -0.5 for prev)
        T_t_to_n: (1, 4, 4) transform from t-LIDAR coords to neighbor-LIDAR coords
                  = ego_pose_neighbor_inv @ ego_pose_t  (in homogeneous form)

    Returns:
        Gaussians with means replaced; all other fields untouched.
    """
    means_shifted = G.means + G.velocity * dt                              # (B, K*J, 3)
    homog = torch.cat([means_shifted,
                       torch.ones_like(means_shifted[..., :1])], dim=-1)   # (B, K*J, 4)
    means_warped = (T_t_to_n.unsqueeze(1) @ homog.unsqueeze(-1)).squeeze(-1)[..., :3]
    return G._replace(means=means_warped)


def compute_stage1_loss(outputs, sequence, lambdas, losses, device,
                        warp_dts=(-0.5, +0.5), use_rgb: bool = True):
    """Render each frame's gaussians (+ ±0.5 s warped renders), compute Eq. 8.

    outputs:  list of T FrameOutput from S2GOSegmentor
    sequence: list of T frame dicts (already on device)
    lambdas:  (lambda_denoise, lambda_depth, lambda_rgb)
    losses:   (DenoiseLoss, DepthRenderLoss, RGBRenderLoss) instances
    warp_dts: tuple of ±dt seconds for neighbor renders. Empty → t=0 only
              (the pre-fix behavior). Default (-0.5, +0.5) implements the
              paper's "current and neighboring keyframes (+/- 0.5s)" prose.

    Velocity-supervision rationale: at t=0 only, ∂L/∂velocity is zero (velocity
    is unused), so the velocity head receives no gradient. With dt ≠ 0 renders,
    velocity appears as `means + v·dt` and gets a signal through L_depth/L_rgb.
    """
    den_loss, dep_loss, rgb_loss = losses
    lam_d, lam_dp, lam_rgb = lambdas
    L_d = 0.0
    # Three-bucket split by dt for L_depth / L_rgb: dt=-0.5, dt=0, dt=+0.5.
    # Used for observability only; the backward signal uses (sum / total_count)
    # — same gradient as before the split.
    L_dp  = {-0.5: 0.0, 0.0: 0.0, +0.5: 0.0}
    L_rgb = {-0.5: 0.0, 0.0: 0.0, +0.5: 0.0}
    n_dt  = {-0.5: 0,   0.0: 0,   +0.5: 0}

    T = len(outputs)
    for t, (out, frame) in enumerate(zip(outputs, sequence)):
        viewmats = frame['viewmats'][0]                     # (N_cam, 4, 4)
        Ks       = frame['cam_K'][0]                        # (N_cam, 3, 3)

        # gsplat's fully_fused_projection requires fp32; cast Gaussians once
        # here and reuse for all renders at this t.
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

            # ── (a) t=0 render (always) ──────────────────────────────
            render_mode = 'RGB+D' if use_rgb else 'D'
            rgb, depth, _ = render(
                G_fp32, viewmats=viewmats.float(), Ks=Ks.float(),
                height=256, width=704, render_mode=render_mode)
            L_dp[0.0]  = L_dp[0.0]  + dep_loss(depth, frame['lidar_depth'][0])
            if use_rgb:
                L_rgb[0.0] = L_rgb[0.0] + rgb_loss(rgb, frame['imgs'][0].permute(0, 2, 3, 1))
            n_dt[0.0] += 1

            # ── (b) ±0.5s warped renders (where the neighbor exists) ──
            ego_t = frame['ego_pose'][0].float()                       # (4, 4)
            for dt_sec in warp_dts:
                n_idx = t + (1 if dt_sec > 0 else -1)
                if not (0 <= n_idx < T):
                    continue
                neighbor = sequence[n_idx]
                ego_n_inv = neighbor['ego_pose_inv'][0].float()         # (4, 4)
                T_t_to_n = (ego_n_inv @ ego_t).unsqueeze(0)             # (1, 4, 4)
                G_warped = _warp_gaussians_means(G_fp32, dt_sec, T_t_to_n)
                rgb_w, depth_w, _ = render(
                    G_warped, viewmats=neighbor['viewmats'][0].float(),
                    Ks=neighbor['cam_K'][0].float(),
                    height=256, width=704, render_mode=render_mode)
                L_dp[dt_sec]  = L_dp[dt_sec]  + dep_loss(depth_w, neighbor['lidar_depth'][0])
                if use_rgb:
                    L_rgb[dt_sec] = L_rgb[dt_sec] + rgb_loss(rgb_w, neighbor['imgs'][0].permute(0, 2, 3, 1))
                n_dt[dt_sec] += 1

        # Denoise is t-only (no warp dimension)
        L_d = L_d + den_loss(out.anchors_xyz, out.refined_xyz)

    # Aggregate. Gradient uses sum / total — same as before this split.
    n_total = max(sum(n_dt.values()), 1)
    L_d_avg     = L_d / T
    L_dp_total_t  = sum(L_dp.values()) / n_total
    if use_rgb:
        L_rgb_total_t = sum(L_rgb.values()) / n_total
        L_total = lam_d * L_d_avg + lam_dp * L_dp_total_t + lam_rgb * L_rgb_total_t
    else:
        # RGB compute skipped — L_rgb[dt] are still Python 0.0, can't .item().
        # Drop the term from the objective entirely and report NaN downstream.
        L_total = lam_d * L_d_avg + lam_dp * L_dp_total_t

    def _avg(v, n):
        if n == 0 or not hasattr(v, 'item'):
            return float('nan')
        return (v / n).item()
    rgb_combined = (sum(L_rgb.values()) / n_total).item() if use_rgb else float('nan')
    return L_total, {
        'denoise':       L_d_avg.item(),
        'depth':         L_dp_total_t.item(),
        'rgb':           rgb_combined,
        # Three-bucket split (NaN if dt absent for boundary frames OR RGB off):
        'depth_minus':   _avg(L_dp[-0.5],  n_dt[-0.5]),
        'depth_t0':      _avg(L_dp[0.0],   n_dt[0.0]),
        'depth_plus':    _avg(L_dp[+0.5],  n_dt[+0.5]),
        'rgb_minus':     _avg(L_rgb[-0.5], n_dt[-0.5]),
        'rgb_t0':        _avg(L_rgb[0.0],  n_dt[0.0]),
        'rgb_plus':      _avg(L_rgb[+0.5], n_dt[+0.5]),
        # Render counts per bucket (for normalization audit):
        'n_minus':       n_dt[-0.5],
        'n_t0':          n_dt[0.0],
        'n_plus':        n_dt[+0.5],
    }


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
         use_checkpoint: bool = False,
         warp_dts=(-0.5, +0.5),
         use_rgb: bool = True,
         full_data: bool = False,
         history_path: str = None,
         save_path: str = None):
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.empty_cache()

    loader = NuScenesLoader(T=4, verbose=False)
    n_dataset = len(loader)

    if full_data:
        print(f"Stage-1 run — {n_iters} iters STREAMING from {n_dataset} Part-1 sequences\n")
        print("[1/4] Streaming loader (sequences loaded on-demand per iter)…")
        sequences = None     # signal to load per-iter below
    else:
        print(f"S1.7e/f overfit run — {n_iters} iters on {n_overfit} fixed sequences\n")
        print("[1/4] Loading and caching overfit data…")
        sequences = []
        for i in range(n_overfit):
            seq = loader[i]                         # 4 frame-dicts, CPU
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
        use_checkpoint=use_checkpoint,
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
    # Recipe: depth + denoise (+ warps). RGB dropped for "first pass" — saves
    # ~0.3 mIoU at paper-spec (Table 3 (d)→(e)) but removes λ_rgb/SSIM tuning,
    # RGB-channel render activation, and the muddy-blob debug noise.
    lambdas = (10.0, 1.0, 1.0 if use_rgb else 0.0)
    if use_rgb:
        print(f"    RGB loss form: {rgb_l1_w:.2f}·L1 + {rgb_ssim_weight:.2f}·(1-SSIM)"
              f"{' (SSIM disabled)' if rgb_ssim_weight == 0 else ''}")
    else:
        print(f"    RGB loss DROPPED (--no-rgb): render_mode='D', λ_rgb=0")

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

    # Head-slice indices into the two output Linear layers. ParentRefiner.head
    # is Linear(d, 7): rows 0:3 offset, 3:4 opa, 4:7 velocity. ChildGaussianHead.
    # head is Linear(d, 14) in Stage 1: 0:3 offset, 3:6 scale, 6:10 rotation,
    # 10:11 opa, 11:14 rgb.
    parent_head = seg.parent_refiner.head      # Linear(d, 7)
    child_head  = seg.child_head.head          # Linear(d, 14|11)
    HEAD_SLICES = {
        'parent_offset':   (parent_head, slice(0, 3)),
        'parent_opacity':  (parent_head, slice(3, 4)),
        'velocity':        (parent_head, slice(4, 7)),
        'child_offset':    (child_head,  slice(0, 3)),
        'scale':           (child_head,  slice(3, 6)),
        'rotation':        (child_head,  slice(6, 10)),
        'child_opacity':   (child_head,  slice(10, 11)),
    }
    if seg.child_head.mode == 'rgb':
        HEAD_SLICES['rgb'] = (child_head, slice(11, 14))

    # ── Train loop ────────────────────────────────────────────────────────
    print(f"[4/4] Training for {n_iters} iters…\n")
    print(f"  iter |   total |  L_den |  L_dep |  L_rgb |    t0/-/+  depth     |     t0/-/+  rgb     |  ‖∂v‖   |   lr   | sec | mem MB")
    print(f"  -----+---------+--------+--------+--------+----------------------+----------------------+---------+--------+-----+-------")
    seg.train(); backbone.train()
    history = []
    t_start = time.time()

    def _fmt_split(v_t0, v_minus, v_plus, fmt):
        def _f(v):
            return (fmt.format(v) if v == v else "  --  ")    # NaN-safe via x==x
        return f"{_f(v_t0)}/{_f(v_minus)}/{_f(v_plus)}"

    for i in range(n_iters):
        if sequences is None:
            # Streaming: load fresh sequence from disk + move to GPU
            seq_cpu = loader[i % n_dataset]
            seq = [to_device(f, device) for f in seq_cpu]
        else:
            seq = sequences[i % n_overfit]             # cycle through the cached set

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

        with torch.cuda.amp.autocast(enabled=mixed_precision, dtype=autocast_dtype):
            outputs = seg(sequence_with_feat)
            L, parts = compute_stage1_loss(outputs, sequence_with_feat, lambdas, losses,
                                            device, warp_dts=warp_dts, use_rgb=use_rgb)

        # Backward + step (paper §B: gradient clip max_norm=35)
        optim.zero_grad()
        scaler.scale(L).backward()
        scaler.unscale_(optim)

        # ── Diagnostics (after backward, before clip) ────────────────────
        # Per-head grad norms (weight + bias) for every output slice. Zero =
        # that head receives no gradient (e.g., velocity with --no-warp).
        head_grads = {f'gnorm_{name}': _head_grad_norm(lin, sl)
                       for name, (lin, sl) in HEAD_SLICES.items()}
        # Gaussian-state distribution stats on the last forward (last frame
        # of the sequence — most-recent active state).
        last_out = outputs[-1]
        gauss_stats = _gaussian_state_stats(last_out.gaussians)
        # Memory-queue + propagator state after the sequence's last frame.
        queue_stats = _queue_propagator_stats(seg, last_out.prop)

        gnorm = nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(seg.parameters()), max_norm=35.0)
        scaler.step(optim)
        scaler.update()
        scheduler.step()

        # Everything to history (full detail, every iter).
        history.append({
            **parts,
            'total':       L.item(),
            'gnorm':       gnorm.item(),
            'lr':          scheduler.get_last_lr()[0],
            **head_grads,
            **{f'gauss_{k}_{stat}': v[stat]
                for k, v in gauss_stats.items() for stat in v},
            **queue_stats,
        })

        if i % log_every == 0 or i == n_iters - 1:
            elapsed = time.time() - t_start
            mem = torch.cuda.max_memory_allocated() / 1024**2
            depth_split = _fmt_split(parts['depth_t0'], parts['depth_minus'],
                                      parts['depth_plus'], "{:>6.3f}")
            rgb_split   = _fmt_split(parts['rgb_t0'], parts['rgb_minus'],
                                      parts['rgb_plus'], "{:>6.4f}")
            print(f"  {i:>4} | {L.item():>7.3f} | {parts['denoise']:>6.3f} | "
                  f"{parts['depth']:>6.3f} | {parts['rgb']:>6.4f} | "
                  f"{depth_split} | {rgb_split} | "
                  f"{head_grads['gnorm_velocity']:>7.2e} | "
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

    # Optional weights checkpoint (end-of-training, weights-only).
    # Stored as a single .pt with both module state_dicts + arch metadata so a
    # later loader can verify shape/config compatibility before load_state_dict.
    if save_path:
        ckpt = {
            'backbone':  backbone.state_dict(),
            'segmentor': seg.state_dict(),
            'config': {
                'K': 900, 'J': 10, 'embed_dims': 768, 'T_queue': 4,
                'num_layers':           num_layers,
                'num_pts':              num_pts,
                'feedforward_channels': feedforward_channels,
                'rgb_ssim_weight':      rgb_ssim_weight,
                'use_checkpoint':       use_checkpoint,
                'warp_dts':             list(warp_dts),
                'iters_trained':        n_iters,
            },
        }
        torch.save(ckpt, save_path)
        sz_mb = sum(t.numel() * t.element_size() for d in [ckpt['backbone'], ckpt['segmentor']]
                     for t in d.values()) / 1024**2
        print(f"  weights saved to {save_path}  ({sz_mb:.1f} MB)")
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
    p.add_argument("--use-checkpoint", action="store_true",
                    help="enable gradient checkpointing on temporal decoder layers "
                         "(~½ activation memory at ~30%% wall-time cost; needed for T2 on 12 GB)")
    p.add_argument("--no-warp", action="store_true",
                    help="disable ±0.5s neighbor renders (velocity will be un-supervised). "
                         "default: warp ON, matching S2GO §3.3.3 prose.")
    p.add_argument("--no-rgb", action="store_true",
                    help="drop L_rgb entirely (λ_rgb=0, gsplat render_mode='D'). "
                         "Recipe-justified first-pass simplification (paper Table 3 (d)→(e) "
                         "is −0.3 mIoU); removes SSIM memory + RGB tuning unknowns.")
    p.add_argument("--full-data", action="store_true",
                    help="stream all 3,121 Part-1 sequences instead of cycling the cached 4. "
                         "Adds ~0.5-1s/iter for disk I/O but exposes the model to real diversity.")
    p.add_argument("--history-path", type=str, default=None)
    p.add_argument("--save-path", type=str, default=None,
                    help="if set, write final {backbone, segmentor} weights "
                         "(+ arch config) to this .pt at end of training.")
    a = p.parse_args()
    main(n_iters=a.n_iters, log_every=a.log_every,
         num_layers=a.num_layers, num_pts=a.num_pts,
         feedforward_channels=a.feedforward_channels,
         mixed_precision=a.mixed_precision, amp_dtype=a.amp_dtype,
         rgb_ssim_weight=a.rgb_ssim_weight,
         use_checkpoint=a.use_checkpoint,
         warp_dts=() if a.no_warp else (-0.5, +0.5),
         use_rgb=(not a.no_rgb),
         full_data=a.full_data,
         history_path=a.history_path,
         save_path=a.save_path)

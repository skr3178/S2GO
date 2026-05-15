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
import math
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR, LambdaLR

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
                        warp_dts=(-0.5, +0.5), use_rgb: bool = True,
                        denoise_only: bool = False,
                        depth_only: bool = False):
    """Render each frame's gaussians (+ ±0.5 s warped renders), compute Eq. 8.

    outputs:  list of T FrameOutput from S2GOSegmentor
    sequence: list of T frame dicts (already on device)
    lambdas:  (lambda_denoise, lambda_depth, lambda_rgb)
    losses:   (DenoiseLoss, DepthRenderLoss, RGBRenderLoss) instances
    warp_dts: tuple of ±dt seconds for neighbor renders. Empty → t=0 only
              (the pre-fix behavior). Default (-0.5, +0.5) implements the
              paper's "current and neighboring keyframes (+/- 0.5s)" prose.
    depth_only: Paper Table 3 ablation 1 — L_depth at dt=0 only. L_denoise is
              not computed at all (den_loss call is skipped); the 'denoise'
              field in the returned diagnostics is NaN and the L_den column
              in the train log reads 'nan'. Caller should set use_rgb=False
              and warp_dts=() (both auto-set when --depth-only is passed).

    Velocity-supervision rationale: at t=0 only, ∂L/∂velocity is zero (velocity
    is unused), so the velocity head receives no gradient. With dt ≠ 0 renders,
    velocity appears as `means + v·dt` and gets a signal through L_depth/L_rgb.
    """
    den_loss, dep_loss, rgb_loss = losses
    lam_d, lam_dp, lam_rgb = lambdas
    L_d = 0.0
    L_dp  = {-0.5: 0.0, 0.0: 0.0, +0.5: 0.0}
    L_rgb = {-0.5: 0.0, 0.0: 0.0, +0.5: 0.0}
    n_dt  = {-0.5: 0,   0.0: 0,   +0.5: 0}

    T = len(outputs)

    if denoise_only:
        # Fast path: no rendering at all. Only L_denoise on each frame.
        for out in outputs:
            L_d = L_d + den_loss(out.anchors_xyz, out.refined_xyz)
        L_d_avg = L_d / T
        L_total = lam_d * L_d_avg
        return L_total, {
            'denoise':     L_d_avg.item(),
            'depth':       float('nan'),
            'rgb':         float('nan'),
            'depth_minus': float('nan'), 'depth_t0': float('nan'), 'depth_plus': float('nan'),
            'rgb_minus':   float('nan'), 'rgb_t0':   float('nan'), 'rgb_plus':   float('nan'),
            'n_minus':     0, 'n_t0': 0, 'n_plus': 0,
        }

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

        # Denoise: skipped entirely in depth_only mode (no compute, no log).
        # Otherwise contributes to the objective as a training term.
        if not depth_only:
            L_d = L_d + den_loss(out.anchors_xyz, out.refined_xyz)

    # Aggregate. Gradient uses sum / total — same as before this split.
    n_total = max(sum(n_dt.values()), 1)
    L_dp_total_t  = sum(L_dp.values()) / n_total
    if depth_only:
        # Pure L_depth objective. Denoise not computed at all → NaN.
        L_total = lam_dp * L_dp_total_t
        denoise_diag = float('nan')
    elif use_rgb:
        L_d_avg = L_d / T
        L_rgb_total_t = sum(L_rgb.values()) / n_total
        L_total = lam_d * L_d_avg + lam_dp * L_dp_total_t + lam_rgb * L_rgb_total_t
        denoise_diag = L_d_avg.item()
    else:
        # RGB compute skipped — L_rgb[dt] are still Python 0.0, can't .item().
        # Drop the term from the objective entirely and report NaN downstream.
        L_d_avg = L_d / T
        L_total = lam_d * L_d_avg + lam_dp * L_dp_total_t
        denoise_diag = L_d_avg.item()

    def _avg(v, n):
        if n == 0 or not hasattr(v, 'item'):
            return float('nan')
        return (v / n).item()
    rgb_combined = (sum(L_rgb.values()) / n_total).item() if use_rgb else float('nan')
    return L_total, {
        'denoise':       denoise_diag,
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
         grad_clip_max_norm: float = 10.0,
         num_layers: int = 2, num_pts: int = 4,
         feedforward_channels: int = 2048,
         mixed_precision: bool = False,
         amp_dtype: str = 'bf16',
         rgb_ssim_weight: float = 0.15,
         use_checkpoint: bool = False,
         warp_dts=(-0.5, +0.5),
         use_rgb: bool = True,
         full_data: bool = False,
         denoise_only: bool = False,
         depth_only: bool = False,
         barebones: bool = False,
         t_seq: int = 4,
         t_queue: int = 4,
         lr_schedule: str = 'constant',
         lr_min: float = 1e-5,
         warmup_iters: int = 500,
         freeze_backbone: bool = False,
         limit_data: int = None,
         history_path: str = None,
         save_path: str = None,
         save_best: bool = False,
         save_best_window: int = 50,
         save_best_cooldown: int = 100,
         grad_accum_steps: int = 1,
         eval_every: int = 0,
         eval_seqs=(0, 100, 5000, 20000),
         train_tokens_json: str = None,
         val_tokens_json: str = None):
    assert not (denoise_only and depth_only), \
        "--denoise-only and --depth-only are mutually exclusive"
    # Denoise-only convenience: overrides
    if denoise_only:
        use_rgb = False
        warp_dts = ()
    # Depth-only convenience: paper Table 3 ablation 1 (LiDAR+ε, depth-only, dt=0).
    # No L_denoise gradient, no L_rgb, no warps → velocity stays un-supervised.
    if depth_only:
        use_rgb = False
        warp_dts = ()
    # Barebones: paper Table 4 row 1 (Propagation=None) + Table 5 row 1 (Velocity=None
    # in both pretrain and occ-est). Single-frame loader, no temporal queue, depth-only.
    # Implies: T_seq=1, T_queue=1, depth_only=True, use_rgb=False, warp_dts=().
    if barebones:
        depth_only = True
        use_rgb = False
        warp_dts = ()
        t_seq = 1
        t_queue = 1
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.empty_cache()

    loader = NuScenesLoader(
        T=t_seq,
        scene_tokens_json=train_tokens_json,
        verbose=False,
    )
    n_dataset = len(loader)
    if limit_data is not None and limit_data > 0:
        n_dataset = min(n_dataset, int(limit_data))
    if train_tokens_json:
        print(f"    [train scope] curated tokens from {train_tokens_json} → "
              f"{n_dataset} T={t_seq} sequences")
    # Separate val loader if a curated val token list is provided. The periodic
    # eval pass (below, gated on --eval-every > 0) iterates val_loader indices
    # 0..len(eval_seqs)-1 instead of pulling from the train loader by index.
    val_loader = None
    if val_tokens_json:
        val_loader = NuScenesLoader(
            T=t_seq,
            scene_tokens_json=val_tokens_json,
            verbose=False,
        )
        # Guard: train ∩ val must be empty at the sequence level.
        train_start = set(loader.start_tokens)
        val_start = set(val_loader.start_tokens)
        overlap = len(train_start & val_start)
        print(f"    [val scope] curated tokens from {val_tokens_json} → "
              f"{len(val_loader)} T={t_seq} sequences "
              f"(train∩val={overlap}, expected 0)")
        if overlap > 0:
            raise RuntimeError(
                f"train ∩ val overlap = {overlap} sequences — refusing to run. "
                "Check that train_tokens_json and val_tokens_json reference "
                "disjoint scene sets."
            )

    if full_data:
        epochs_implied = n_iters / max(n_dataset, 1)
        print(f"Stage-1 run — {n_iters} iters STREAMING from {n_dataset} sequences "
              f"(≈ {epochs_implied:.1f} epochs)\n")
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
        T_queue=t_queue,
        use_checkpoint=use_checkpoint,
    ).to(device)
    n_back = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_seg  = sum(p.numel() for p in seg.parameters() if p.requires_grad)
    print(f"    backbone: {n_back/1e6:.1f} M trainable, segmentor: {n_seg/1e6:.1f} M trainable, "
          f"total: {(n_back+n_seg)/1e6:.1f} M")

    # Optional: freeze backbone entirely (eval mode + no grad).
    if freeze_backbone:
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        print("    backbone FROZEN (eval mode, no grad)")

    # ── Optimizer: AdamW with backbone-lr-scaled paramwise group (paper §B) ─
    print("[3/4] Building optimizer (AdamW, lr=4e-4, backbone ×0.25)…")
    if freeze_backbone:
        optim = AdamW(seg.parameters(), lr=lr, weight_decay=0.01)
    else:
        optim = AdamW([
            {'params': [p for p in backbone.parameters() if p.requires_grad],
             'lr': lr * lr_backbone_mult},
            {'params': seg.parameters(), 'lr': lr},
        ], weight_decay=0.01)
    # Gradient accumulation: N micro-iters per optimizer step (effective batch ≈ N).
    # The scheduler ticks per OPTIMIZER STEP, not per micro-iter, so T_max is in
    # optimizer steps. n_iters is still the total micro-iter budget (samples seen).
    if grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    n_optim_steps = (n_iters + grad_accum_steps - 1) // grad_accum_steps

    # LR schedule. For short diagnostic runs (50-500 iters), cosine-to-zero
    # makes the last 20% of iters effectively useless because lr→0. Default is
    # 'constant' so LR stays at its peak throughout. Use 'cosine' (decays to
    # lr_min) for a paper-spec multi-epoch run where decay matches horizon.
    if lr_schedule == 'cosine':
        scheduler = CosineAnnealingLR(optim, T_max=n_optim_steps, eta_min=lr_min)
        print(f"    LR schedule: cosine (T_max={n_optim_steps} optim-steps, eta_min={lr_min:.0e})")
    elif lr_schedule == 'constant':
        # factor=1.0, total_iters=n_optim_steps keeps LR at peak the entire run.
        scheduler = ConstantLR(optim, factor=1.0, total_iters=n_optim_steps)
        print(f"    LR schedule: constant (peak lr held for all {n_optim_steps} optim-steps)")
    elif lr_schedule == 'warmup_cosine':
        # Linear warmup 0 → peak over `warmup_iters` micro-iters, then cosine
        # decay to peak × 0.10 over the remaining (n_iters - warmup_iters)
        # micro-iters. Scheduler ticks once per optim step, so convert both
        # warmup and total to optim-step units (float, since fractional).
        warmup_optim = warmup_iters / grad_accum_steps
        if warmup_optim >= n_optim_steps:
            raise ValueError(f"warmup_iters={warmup_iters} ≥ n_iters={n_iters} "
                              f"after dividing by grad_accum_steps={grad_accum_steps}; "
                              f"warmup spans the whole run.")
        def _warmup_cosine_lambda(s):
            if s < warmup_optim:
                return s / warmup_optim
            progress = min((s - warmup_optim) / (n_optim_steps - warmup_optim), 1.0)
            cosine_mul = 0.5 * (1.0 + math.cos(math.pi * progress))
            return 0.10 + 0.90 * cosine_mul
        scheduler = LambdaLR(optim, lr_lambda=_warmup_cosine_lambda)
        print(f"    LR schedule: warmup_cosine "
              f"(warmup {warmup_iters} micro-iters ≈ {warmup_optim:.1f} optim-steps, "
              f"cosine to peak×0.10 over remaining {n_optim_steps - warmup_optim:.1f})")
    else:
        raise ValueError(f"unknown lr_schedule '{lr_schedule}' "
                         f"(use 'cosine', 'constant', or 'warmup_cosine')")
    if grad_accum_steps > 1:
        print(f"    gradient accumulation: {grad_accum_steps} micro-iters per "
              f"optimizer step → {n_optim_steps} optim-steps over {n_iters} micro-iters "
              f"(effective batch ≈ {grad_accum_steps})")

    # Stage-1 losses (Eq. 8); λ defaults from Stage1_design.md D6
    rgb_l1_w = 1.0 - rgb_ssim_weight if rgb_ssim_weight > 0 else 1.0
    losses = (DenoiseLoss(), DepthRenderLoss(),
                RGBRenderLoss(l1_weight=rgb_l1_w, ssim_weight=rgb_ssim_weight))
    if denoise_only:
        lambdas = (1.0, 0.0, 0.0)
    elif depth_only:
        # Paper Table 3 ablation 1: L_depth only at dt=0.
        lambdas = (0.0, 1.0, 0.0)
    else:
        # Recipe: depth + denoise (+ warps). RGB dropped for "first pass" — saves
        # ~0.3 mIoU at paper-spec (Table 3 (d)→(e)) but removes λ_rgb/SSIM tuning,
        # RGB-channel render activation, and the muddy-blob debug noise.
        lambdas = (10.0, 1.0, 1.0 if use_rgb else 0.0)
    if denoise_only:
        print(f"    DENOISE-ONLY (--denoise-only): λ=(1, 0, 0), no rendering, no warps")
    elif depth_only:
        print(f"    DEPTH-ONLY (--depth-only): λ=(0, 1, 0), L_depth at dt=0 only, "
              f"no warps, no RGB (paper Table 3 ablation 1).")
        if barebones:
            print(f"      BAREBONES: T_seq=1, T_queue=1 — paper Table 4 row 1 "
                  f"(Propagation=None) + Table 5 row 1 (Velocity=None).")
        print(f"      trains: child {{offset,scale,rotation,opacity}}, "
              f"parent {{offset,opacity}}.")
        print(f"      does NOT train: velocity (no warps → no grad path).")
    elif use_rgb:
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
    n_skipped = 0
    current_sample_token = "(cached)"     # default; overridden in streaming branch

    # Grad-accum window state. `window_skipped` latches True as soon as any
    # micro-iter in the current window produces non-finite loss; once latched,
    # we abandon the whole window at its end (no optimizer step, grads zeroed).
    window_skipped = False
    nan_zero_dict = {f'gnorm_{name}': float('nan') for name in HEAD_SLICES}

    # ── Save-best tracking ───────────────────────────────────────────────
    # Track the lowest moving-average (window=save_best_window) of L_total
    # over finite (non-skipped) iters. Save weights to `<save_path>_best.pt`
    # whenever the smoothed loss reaches a new low, rate-limited by
    # `save_best_cooldown` iters between saves (to avoid 413 MB writes on
    # every iter when loss is trending down rapidly).
    best_smoothed_loss = float('inf')
    last_best_save_iter = -10**9
    n_best_saves = 0

    # ── Eval-best tracking (option B: held-out BEV nn_dist driven save) ──
    # When --eval-every N > 0, every N micro-iters we run forward on the
    # held-out --eval-seqs, compute pooled BEV nn_dist_mean (refined query
    # vs nearest LiDAR point), and save weights to `<save_path>_eval_best.pt`
    # whenever the metric reaches a new low. Independent of train-loss best.
    best_eval_nn_dist = float('inf')
    n_eval_saves = 0
    last_eval_dist = float('nan')

    def _build_ckpt():
        """Build the same checkpoint dict used by the end-of-training save."""
        return {
            'backbone':  backbone.state_dict(),
            'segmentor': seg.state_dict(),
            'config': {
                'K': 900, 'J': 10, 'embed_dims': 768,
                'T_queue': t_queue, 'T_seq': t_seq,
                'num_layers':           num_layers,
                'num_pts':              num_pts,
                'feedforward_channels': feedforward_channels,
                'rgb_ssim_weight':      rgb_ssim_weight,
                'use_checkpoint':       use_checkpoint,
                'warp_dts':             list(warp_dts),
                'iters_trained':        n_iters,
            },
        }

    def _fmt_split(v_t0, v_minus, v_plus, fmt):
        def _f(v):
            return (fmt.format(v) if v == v else "  --  ")    # NaN-safe via x==x
        return f"{_f(v_t0)}/{_f(v_minus)}/{_f(v_plus)}"

    for i in range(n_iters):
        if sequences is None:
            # Streaming: load fresh sequence from disk + move to GPU
            seq_cpu = loader[i % n_dataset]
            # Seed RNG by sample_token so FPS produces the SAME anchors every
            # time we revisit this sample across epochs. Without this, the
            # denoise target jitters per-iter (hacks.md H1) and the model
            # can't converge on streaming data.
            tok = seq_cpu[0].get('_sample_token', str(i))
            current_sample_token = tok    # for NaN-guard diagnostics
            torch.manual_seed(hash(tok) & 0x7FFFFFFF)
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
                                            device, warp_dts=warp_dts, use_rgb=use_rgb,
                                            denoise_only=denoise_only,
                                            depth_only=depth_only)

        # ── Grad-accum bookkeeping ───────────────────────────────────────
        is_accum_start = (i % grad_accum_steps == 0)
        is_accum_end   = ((i + 1) % grad_accum_steps == 0) or (i == n_iters - 1)
        if is_accum_start:
            optim.zero_grad(set_to_none=True)
            window_skipped = False

        # Backward (scaled by 1/N so accumulated grad is the per-sample mean,
        # not the sum — keeps the effective LR consistent with paper-spec
        # batched training). Skip backward if the window is already poisoned
        # to save compute and avoid touching grad buffers further.
        L_micro = L.item()
        L_finite = (L_micro == L_micro) and not (L_micro == float('inf') or L_micro == float('-inf'))
        if not window_skipped and L_finite:
            scaler.scale(L / grad_accum_steps).backward()
        elif not L_finite and not window_skipped:
            window_skipped = True
            n_skipped += 1
            print(f"  [skip] iter {i}: non-finite micro-loss "
                  f"(L={L_micro:.3g}, token={current_sample_token})", flush=True)
            optim.zero_grad(set_to_none=True)       # nuke any poisoned grads
        else:
            n_skipped += 1
            print(f"  [skip] iter {i}: window already poisoned "
                  f"(token={current_sample_token})", flush=True)

        # ── Forward-state diagnostics (every micro-iter; don't need acc-grad) ─
        # Gaussian-state distribution stats on the last forward (last frame
        # of the sequence — most-recent active state).
        last_out = outputs[-1]
        gauss_stats = _gaussian_state_stats(last_out.gaussians)
        # Memory-queue + propagator state after the sequence's last frame.
        queue_stats = _queue_propagator_stats(seg, last_out.prop)

        # ── Window-end: clip, NaN-check accumulated grads, step, schedule ─
        if is_accum_end:
            if window_skipped:
                head_grads = dict(nan_zero_dict)
                gnorm_val  = float('nan')
                skipped    = True
            else:
                scaler.unscale_(optim)
                # Per-head grad norms (computed once per optimizer step, on
                # the accumulated gradient).
                head_grads = {f'gnorm_{name}': _head_grad_norm(lin, sl)
                              for name, (lin, sl) in HEAD_SLICES.items()}
                gnorm = nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(seg.parameters()),
                    max_norm=grad_clip_max_norm)
                gnorm_val = gnorm.item()
                if not torch.isfinite(gnorm).item():
                    # Accumulated grads went non-finite even though every
                    # micro-iter loss was finite — overflow happened in the
                    # gsplat backward (most likely under bf16). Skip step.
                    n_skipped += 1
                    print(f"  [skip-window] iter {i}: non-finite accumulated gnorm "
                          f"(gnorm={gnorm_val:.3g}, token={current_sample_token})", flush=True)
                    optim.zero_grad(set_to_none=True)
                    skipped = True
                else:
                    scaler.step(optim)
                    scaler.update()
                    skipped = False
            scheduler.step()    # one tick per optimizer step
        else:
            # Mid-window: no optimizer state to report; carry NaNs in history
            head_grads = dict(nan_zero_dict)
            gnorm_val  = float('nan')
            skipped    = window_skipped

        # Everything to history (full detail, every micro-iter). gnorm and
        # head_grads are NaN for mid-window iters (only meaningful at window-end).
        history.append({
            **parts,
            'total':       L_micro,
            'gnorm':       gnorm_val,
            'lr':          scheduler.get_last_lr()[0],
            'skipped':     skipped,
            'window_end':  is_accum_end,
            'sample_token': current_sample_token,
            **head_grads,
            **{f'gauss_{k}_{stat}': v[stat]
                for k, v in gauss_stats.items() for stat in v},
            **queue_stats,
        })

        # ── Save-best logic ──────────────────────────────────────────────
        # Compute smoothed loss over the last `save_best_window` finite iters;
        # save weights when smoothed reaches a new low (with cooldown). Gate on
        # window-end (skipped is meaningful then; mid-window we may carry stale
        # window_skipped flag).
        if save_best and save_path is not None and is_accum_end and not skipped \
                and len(history) >= save_best_window:
            recent_finite = [h['total'] for h in history[-save_best_window:]
                              if not h['skipped'] and h['total'] == h['total']]
            if len(recent_finite) >= save_best_window // 2:
                smoothed = sum(recent_finite) / len(recent_finite)
                cooldown_ok = (i - last_best_save_iter) >= save_best_cooldown
                if smoothed < best_smoothed_loss and cooldown_ok:
                    best_smoothed_loss = smoothed
                    last_best_save_iter = i
                    n_best_saves += 1
                    best_path = save_path.replace('.pt', '_best.pt')
                    if best_path == save_path:               # save_path had no ".pt"
                        best_path = save_path + '_best.pt'
                    ckpt_best = _build_ckpt()
                    # Tag with iter + smoothed loss for later identification
                    ckpt_best['config']['iters_trained'] = i + 1
                    ckpt_best['config']['best_smoothed_loss'] = smoothed
                    ckpt_best['config']['best_window']        = save_best_window
                    torch.save(ckpt_best, best_path)
                    print(f"  [best #{n_best_saves}] iter {i}: smoothed L "
                          f"({save_best_window}-iter MA) = {smoothed:.4f} m → "
                          f"saved {best_path}", flush=True)

        # ── Periodic held-out eval (option B) ────────────────────────────
        # Triggered at window-end every `eval_every` micro-iters. Runs forward
        # on each `eval_seqs` index, computes BEV nn_dist (refined query →
        # nearest LiDAR point), pools across seqs, and conditionally saves
        # `<save_path>_eval_best.pt` when the metric reaches a new low.
        # Independent of the train-loss save-best above.
        if eval_every > 0 and is_accum_end and (i + 1) % eval_every == 0:
            bb_was_training = backbone.training
            seg_was_training = seg.training
            backbone.eval(); seg.eval()
            all_nn = []
            # If a separate val_loader is configured, pull eval sequences from
            # it (curated val pool); otherwise fall back to indexing the train
            # loader (legacy behaviour for old overfit / smoke-test runs).
            eval_loader = val_loader if val_loader is not None else loader
            # When val_loader is set, eval_seqs is interpreted as indices into
            # val_loader (which is usually 50-500 long). Clip to valid range.
            n_eval = len(eval_loader)
            effective_eval_seqs = [idx % n_eval for idx in eval_seqs]
            with torch.no_grad():
                for ev_idx in effective_eval_seqs:
                    ev_seq_cpu = eval_loader[ev_idx]
                    ev_seq = [to_device(f, device) for f in ev_seq_cpu]
                    ev_seq_feat = []
                    for f_ev in ev_seq:
                        feat_, ss_, lsi_, _ = backbone(f_ev['imgs'])
                        ev_seq_feat.append({
                            **f_ev,
                            'feat_flatten':       feat_,
                            'spatial_shapes':     ss_,
                            'level_start_index':  lsi_,
                            'pad_h': 256, 'pad_w': 704,
                        })
                    with torch.cuda.amp.autocast(enabled=mixed_precision,
                                                  dtype=autocast_dtype):
                        ev_outputs = seg(ev_seq_feat)
                    refined = ev_outputs[-1].refined_xyz[0].float()
                    lidar = ev_seq_feat[-1]['lidar_pts'][0].float()
                    d = torch.cdist(refined, lidar).min(dim=1).values
                    all_nn.append(d)
            pooled = torch.cat(all_nn).mean().item()
            last_eval_dist = pooled
            if bb_was_training and not freeze_backbone:
                backbone.train()
            if seg_was_training:
                seg.train()
            print(f"  [eval @ iter {i+1}] BEV nn_dist_mean pooled over "
                  f"{len(eval_seqs)} seqs = {pooled:.4f} m", flush=True)
            if save_best and save_path is not None and pooled < best_eval_nn_dist - 1e-6:
                best_eval_nn_dist = pooled
                n_eval_saves += 1
                eval_path = save_path.replace('.pt', '_eval_best.pt')
                if eval_path == save_path:                # save_path had no ".pt"
                    eval_path = save_path + '_eval_best.pt'
                ckpt_eval = _build_ckpt()
                ckpt_eval['config']['iters_trained']       = i + 1
                ckpt_eval['config']['best_eval_nn_dist']   = pooled
                ckpt_eval['config']['eval_seqs']           = list(eval_seqs)
                torch.save(ckpt_eval, eval_path)
                print(f"  [eval-best #{n_eval_saves}] iter {i+1}: BEV nn_dist "
                      f"{pooled:.4f} m → saved {eval_path}", flush=True)

        # Print log line at every Kth optimizer step (window-end). With N=1
        # this is every K micro-iters, matching the old behavior. With N>1
        # it's every K window-ends (so the gnorm column always has a
        # meaningful value, never NaN-from-mid-window).
        optim_step_idx = (i + 1) // grad_accum_steps - 1 if is_accum_end else -1
        if (is_accum_end and optim_step_idx % log_every == 0) or i == n_iters - 1:
            elapsed = time.time() - t_start
            mem = torch.cuda.max_memory_allocated() / 1024**2
            depth_split = _fmt_split(parts['depth_t0'], parts['depth_minus'],
                                      parts['depth_plus'], "{:>6.3f}")
            rgb_split   = _fmt_split(parts['rgb_t0'], parts['rgb_minus'],
                                      parts['rgb_plus'], "{:>6.4f}")
            print(f"  {i:>4} | {L_micro:>7.3f} | {parts['denoise']:>6.3f} | "
                  f"{parts['depth']:>6.3f} | {parts['rgb']:>6.4f} | "
                  f"{depth_split} | {rgb_split} | "
                  f"{head_grads['gnorm_velocity']:>7.2e} | "
                  f"{scheduler.get_last_lr()[0]:.1e} | {elapsed:>3.0f} | {mem:>6.0f}")

    print()
    # ── Verify monotonic-ish decrease ─────────────────────────────────────
    # Use only finite, non-skipped iters for the summary so NaN/inf can't
    # poison the verdict.
    finite_hist = [h for h in history if not h.get('skipped', False)
                                          and h['total'] == h['total']]
    first_5 = sum(h['total'] for h in finite_hist[:5]) / max(len(finite_hist[:5]), 1)
    last_5  = sum(h['total'] for h in finite_hist[-5:]) / max(len(finite_hist[-5:]), 1)
    drop = first_5 - last_5
    print(f"Final summary:")
    print(f"  iters total       : {n_iters}")
    print(f"  iters skipped (NaN/inf guard fired): {n_skipped}"
          f"  ({100.0*n_skipped/n_iters:.1f}%)")
    print(f"  loss avg  first 5 (finite) iters: {first_5:.3f}")
    print(f"  loss avg  last  5 (finite) iters: {last_5:.3f}")
    print(f"  Δ = {drop:+.3f}  ({'GOOD' if drop > 0 else 'BAD: loss did not drop'})")
    den_drop = (sum(h['denoise'] for h in finite_hist[:5]) -
                sum(h['denoise'] for h in finite_hist[-5:])) / max(len(finite_hist[:5]), 1)
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
        ckpt = _build_ckpt()
        torch.save(ckpt, save_path)
        sz_mb = sum(t.numel() * t.element_size() for d in [ckpt['backbone'], ckpt['segmentor']]
                     for t in d.values()) / 1024**2
        print(f"  weights saved to {save_path}  ({sz_mb:.1f} MB)")
    if save_best and save_path is not None:
        best_path = save_path.replace('.pt', '_best.pt')
        if best_path == save_path:
            best_path = save_path + '_best.pt'
        if n_best_saves > 0:
            print(f"  best-checkpoint saves during training: {n_best_saves} "
                  f"(final best smoothed L = {best_smoothed_loss:.4f} m at iter "
                  f"{last_best_save_iter}) → {best_path}")
        else:
            print(f"  save-best was enabled but no qualifying improvement was "
                  f"seen (no {best_path} written)")
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
    p.add_argument("--denoise-only", action="store_true",
                    help="run pure L_denoise (no depth, no rgb, no warp, no render). "
                         "Fastest path; isolates the position/coord-frame pathway.")
    p.add_argument("--depth-only", action="store_true",
                    help="run pure L_depth at dt=0 (no denoise grad, no rgb, no warp). "
                         "Paper Table 3 ablation 1 (LiDAR+ε, depth-only, dt=0). Trains "
                         "child offset/scale/rotation/opacity + parent offset/opacity; "
                         "velocity stays un-supervised. Mutually exclusive with --denoise-only.")
    p.add_argument("--barebones", action="store_true",
                    help="explicit alias for the (now default) barebones recipe: "
                         "T_seq=1, T_queue=1, depth-only loss. Kept for clarity; "
                         "running with NO recipe flag now gives barebones implicitly.")
    p.add_argument("--full-recipe", action="store_true",
                    help="opt OUT of barebones default. Restores paper-spec: T_seq=4, "
                         "T_queue=4, full Eq. 8 (denoise + depth + rgb + ±0.5s warps). "
                         "Mutually exclusive with --denoise-only / --depth-only / --barebones.")
    p.add_argument("--t-seq", type=int, default=1,
                    help="number of frames per loader sequence. Default 1 "
                         "(barebones; single-frame, no temporal context). "
                         "Paper-spec is 4 (auto-set by --full-recipe).")
    p.add_argument("--t-queue", type=int, default=1,
                    help="memory queue length for temporal propagation. Default 1 "
                         "(barebones; Figure 6 leftmost point ≈ 18.4 mIoU). "
                         "Paper-spec is 4 (auto-set by --full-recipe).")
    p.add_argument("--lr", type=float, default=4e-4,
                    help="peak segmentor LR (backbone gets ×0.25). Default 4e-4 "
                         "(paper-spec). Lower (e.g. 2e-4 or 1e-4) recommended for "
                         "streaming + bf16 to reduce gradient explosion risk.")
    p.add_argument("--grad-clip", type=float, default=10.0,
                    help="max_norm for clip_grad_norm_. Default 10 (tightened from "
                         "paper's 35 after the iter-2919 bf16 gradient-explosion "
                         "incident). Lower = more aggressive overflow protection.")
    p.add_argument("--lr-schedule", choices=['constant', 'cosine', 'warmup_cosine'],
                    default='constant',
                    help="LR schedule. 'constant' (default for short diagnostic runs) "
                         "holds peak LR for all n_iters. 'cosine' decays to --lr-min "
                         "over n_iters; appropriate for paper-spec multi-epoch runs. "
                         "'warmup_cosine' linearly warms 0→peak over --warmup-iters "
                         "micro-iters, then cosine to peak×0.10 over the remainder.")
    p.add_argument("--lr-min", type=float, default=1e-5,
                    help="minimum LR for the cosine schedule. Ignored when "
                         "--lr-schedule=constant. Default 1e-5 (avoid lr=0 starvation).")
    p.add_argument("--warmup-iters", type=int, default=500,
                    help="warmup length in MICRO-ITERS for --lr-schedule=warmup_cosine. "
                         "Ignored for other schedules. Default 500.")
    p.add_argument("--freeze-backbone", action="store_true",
                    help="freeze backbone (eval mode + no grad). Reduces optimizer noise; "
                         "useful for denoise-only diagnostic runs.")
    p.add_argument("--limit-data", type=int, default=None,
                    help="cap streaming-loader effective size to first N sequences. "
                         "Use with --full-data + small N for multi-epoch revisits "
                         "(tests per-sample FPS-seed hypothesis cheaply).")
    p.add_argument("--history-path", type=str, default=None)
    p.add_argument("--save-path", type=str, default=None,
                    help="if set, write final {backbone, segmentor} weights "
                         "(+ arch config) to this .pt at end of training.")
    p.add_argument("--save-best", action="store_true",
                    help="also save weights to '<save-path>_best.pt' whenever "
                         "the moving-average L_total over the last "
                         "--save-best-window iters reaches a new low. Useful "
                         "when streaming loss oscillates: the final-iter "
                         "weights may not be the lowest-loss point. Requires "
                         "--save-path.")
    p.add_argument("--save-best-window", type=int, default=50,
                    help="window size (iters) for the moving-average loss used "
                         "by --save-best. Filters per-scene noise.")
    p.add_argument("--save-best-cooldown", type=int, default=100,
                    help="min iters between consecutive best-saves (rate limit "
                         "so we don't write 413 MB of weights on every improving "
                         "iter during a fast-descent phase).")
    p.add_argument("--grad-accum-steps", type=int, default=1,
                    help="number of micro-iters per optimizer step. Effective "
                         "batch ≈ this value (per-sample loss is scaled by 1/N "
                         "and gradients accumulated over N micro-iters). "
                         "Default 1 (no accumulation). Use 16 to approximate "
                         "paper-spec effective batch on a single-sample loader.")
    p.add_argument("--eval-every", type=int, default=0,
                    help="run held-out eval every N micro-iters (must align with "
                         "an accum window-end). 0 = disabled (default). When "
                         "combined with --save-best, also writes "
                         "`<save-path>_eval_best.pt` whenever pooled BEV "
                         "nn_dist_mean reaches a new low. Cheap (~1-2 s per "
                         "eval seq).")
    p.add_argument("--eval-seqs", type=int, nargs="+",
                    default=[0, 100, 5000, 20000],
                    help="loader sequence indices used for periodic eval. "
                         "Default [0, 100, 5000, 20000] matches the overnight "
                         "eval set for direct comparison. When --val-tokens-json "
                         "is set, these indices are into the val_loader instead "
                         "of the train loader, and are mod-len'd to fit.")
    p.add_argument("--train-tokens-json", type=str, default=None,
                    help="optional JSON with a list of scene tokens to restrict "
                         "training to (e.g. dataset_stats/curated_train_270/tokens.json). "
                         "Filters at scene-level inside the loader's start-token scan.")
    p.add_argument("--val-tokens-json", type=str, default=None,
                    help="optional JSON with a list of scene tokens used to build "
                         "a SEPARATE val NuScenesLoader. When set, the periodic "
                         "eval pass (gated on --eval-every) draws sequences from "
                         "this val_loader instead of the train loader. "
                         "Refuses to start if train ∩ val sequences overlap.")
    a = p.parse_args()

    # ── Resolve recipe defaults ────────────────────────────────────────────
    # Mutually exclusive recipe flags
    n_recipe = sum([a.denoise_only, a.depth_only, a.barebones, a.full_recipe])
    if n_recipe > 1:
        p.error("Pass at most one of --denoise-only / --depth-only / --barebones / --full-recipe")

    if a.full_recipe:
        # Paper-spec full Eq. 8 recipe — overrides barebones defaults.
        a.t_seq = 4
        a.t_queue = 4
        a.depth_only = False
        # Leave --no-warp / --no-rgb as-is so user can still subtract pieces.
    elif a.denoise_only:
        pass     # denoise_only path handles its own overrides inside main()
    else:
        # No recipe flag (or --barebones / --depth-only): treat as barebones.
        # depth_only=True is the gate that disables denoise compute + RGB + warps.
        a.depth_only = True

    main(n_iters=a.n_iters, log_every=a.log_every,
         lr=a.lr, grad_clip_max_norm=a.grad_clip,
         num_layers=a.num_layers, num_pts=a.num_pts,
         feedforward_channels=a.feedforward_channels,
         mixed_precision=a.mixed_precision, amp_dtype=a.amp_dtype,
         rgb_ssim_weight=a.rgb_ssim_weight,
         use_checkpoint=a.use_checkpoint,
         warp_dts=() if a.no_warp else (-0.5, +0.5),
         use_rgb=(not a.no_rgb),
         full_data=a.full_data,
         denoise_only=a.denoise_only,
         depth_only=a.depth_only,
         barebones=a.barebones,
         t_seq=a.t_seq,
         t_queue=a.t_queue,
         lr_schedule=a.lr_schedule,
         lr_min=a.lr_min,
         warmup_iters=a.warmup_iters,
         freeze_backbone=a.freeze_backbone,
         limit_data=a.limit_data,
         history_path=a.history_path,
         save_path=a.save_path,
         save_best=a.save_best,
         save_best_window=a.save_best_window,
         save_best_cooldown=a.save_best_cooldown,
         grad_accum_steps=a.grad_accum_steps,
         eval_every=a.eval_every,
         eval_seqs=tuple(a.eval_seqs),
         train_tokens_json=a.train_tokens_json,
         val_tokens_json=a.val_tokens_json)

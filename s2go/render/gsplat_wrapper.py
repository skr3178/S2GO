"""gsplat_wrapper — render flat Gaussians to multi-camera depth + RGB images.

Wraps gsplat.rasterization with the input/output convention used by S2GO:
  - Input:  Gaussians namedtuple (Stage1_design.md §3 contract)
  - Output: (rgb, depth, alpha) per camera

gsplat 1.5.3 batches over CAMERAS (C), not over Gaussian sets (B). For B > 1,
the caller loops over batch elements; for nuScenes (6 cameras per frame at
B=1), one call renders all 6 cams.

S1.3 — End-to-end forward integration test in __main__:
    LiDAR pts → S2GOLifter → ParentRefiner → ChildGaussianHead
              → assemble_gaussians → gsplat render → (D, RGB)

Run:
    cd /media/skr/storage/self_driving/S2GO
    python -m s2go.render.gsplat_wrapper
"""
from typing import Optional
import torch
from gsplat import rasterization

from ..models.encoder.assembly import Gaussians


def render(gaussians: Gaussians,
           viewmats: torch.Tensor,
           Ks: torch.Tensor,
           height: int,
           width: int,
           render_mode: str = "RGB+D",
           near_plane: float = 0.1,
           far_plane: float = 80.0):
    """Render a single Gaussian set to C cameras.

    Args:
        gaussians:    Gaussians (B=1 expected; caller loops over batch otherwise)
        viewmats:     (C, 4, 4)  world→cam  (OpenCV convention: cam looks +Z)
        Ks:           (C, 3, 3)  pinhole intrinsics
        height,width: image dims (paper §B: 256, 704)
        render_mode:  'RGB+D' (alpha-weighted depth, our default) or 'RGB+ED'
                       (expected/normalized depth — closer to true surface depth)
        near_plane, far_plane: clipping (paper-implicit defaults; near=0.1m
                       skips Gaussians inside camera; far=80m matches nuScenes
                       point-cloud range)

    Returns:
        rgb:    (C, H, W, 3)
        depth:  (C, H, W)         only if render_mode contains D / ED
        alpha:  (C, H, W, 1)
    """
    if gaussians.colors is None:
        raise ValueError("gsplat render needs colors; pass with_rgb=True to assembler")
    if gaussians.means.dim() == 3:
        # Strip leading batch axis if B==1
        assert gaussians.means.shape[0] == 1, \
            "gsplat single-call expects unbatched Gaussians; loop over B in caller"
        means = gaussians.means.squeeze(0)
        quats = gaussians.rotations.squeeze(0)
        scales = gaussians.scales.squeeze(0)
        opacities = gaussians.opacities.squeeze(0).squeeze(-1)        # (N,)
        colors = gaussians.colors.squeeze(0)
    else:
        means, quats, scales = gaussians.means, gaussians.rotations, gaussians.scales
        opacities = gaussians.opacities.squeeze(-1) if gaussians.opacities.dim() == 2 \
                    else gaussians.opacities
        colors = gaussians.colors

    renders, alphas, _meta = rasterization(
        means=means, quats=quats, scales=scales,
        opacities=opacities, colors=colors,
        viewmats=viewmats, Ks=Ks,
        width=width, height=height,
        render_mode=render_mode,
        near_plane=near_plane, far_plane=far_plane,
    )
    rgb = renders[..., :3]                         # (C, H, W, 3)
    depth = renders[..., 3] if render_mode in ("RGB+D", "RGB+ED") else None
    return rgb, depth, alphas


# ────────────────────────────────────────────────────────────────────────────
# Self-test — S1.3 end-to-end integration: pts → lifter → heads → assemble → render
# ────────────────────────────────────────────────────────────────────────────
def _self_test():
    import os
    from PIL import Image
    from ..models.lifter.s2go_lifter import S2GOLifter
    from ..models.encoder.heads import ParentRefiner, ChildGaussianHead
    from ..models.encoder.assembly import assemble_gaussians

    torch.manual_seed(0)
    device = "cuda"
    B, M, K, J, d = 1, 10_000, 900, 10, 768
    H, W = 256, 704
    N_CAM = 6

    print(f"S1.3 integration: B={B}, M={M}, K={K}, J={J}, d={d}, "
          f"image={W}×{H}, N_cam={N_CAM}")

    # ── Synthetic LiDAR + cameras ─────────────────────────────────────────
    pts = torch.empty((B, M, 3), device=device)
    pts[..., 0].uniform_(-50.0, 50.0)
    pts[..., 1].uniform_(-50.0, 50.0)
    pts[..., 2].uniform_(-3.0, 5.0)

    # 6 synthetic cameras around the origin (rough nuScenes-like layout).
    # Identity-rotation cameras placed at +Y, -Y, +X, -X, +Y(rear), origin
    # — for a smoke test only. Real nuScenes intrinsics come in S1.5+.
    fy = 0.5 * H / torch.tan(torch.tensor(60.0) * torch.pi / 360)
    Ks = torch.tensor([[fy, 0., W/2], [0., fy, H/2], [0., 0., 1.]],
                       device=device).unsqueeze(0).expand(N_CAM, -1, -1).contiguous()
    viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(N_CAM, 1, 1)
    # Translate cameras a bit so they don't fully overlap.
    viewmats[:, :3, 3] = torch.tensor([
        [0,  0, 0], [2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0], [0, 0, 2],
    ], dtype=torch.float32, device=device)

    # ── Build modules ─────────────────────────────────────────────────────
    lifter = S2GOLifter(K=K, embed_dims=d, eps=1.0).to(device)
    parent_refiner = ParentRefiner(embed_dims=d).to(device)
    child_head = ChildGaussianHead(embed_dims=d, J=J, mode='rgb').to(device)

    # ── Forward: pts → lifter → parent → child → assemble ─────────────────
    anchors_xyz, init_xyz, init_feat = lifter(pts)
    print(f"  lifter   OK: anchors {tuple(anchors_xyz.shape)}, "
          f"init_xyz {tuple(init_xyz.shape)}, init_feat {tuple(init_feat.shape)}")

    # Without a temporal decoder, we synthesise anchor_embed (S1.5 will replace
    # this with proper positional embedding of init_xyz fed through the decoder).
    anchor_embed = torch.randn_like(init_feat) * 0.1
    parent = parent_refiner(init_feat, anchor_embed)
    children = child_head(parent.feat)
    print(f"  refiner  OK: parent.offset {tuple(parent.offset.shape)}, "
          f"velocity {tuple(parent.velocity.shape)}")
    print(f"  child    OK: offset {tuple(children.offset.shape)}, "
          f"rgb {tuple(children.rgb.shape)}")

    G = assemble_gaussians(init_xyz, parent, children, with_rgb=True)
    assert G.means.shape == (B, K * J, 3),     f"means {G.means.shape}"
    assert G.scales.shape == (B, K * J, 3),    f"scales {G.scales.shape}"
    assert G.rotations.shape == (B, K * J, 4), f"rotations {G.rotations.shape}"
    assert G.opacities.shape == (B, K * J, 1), f"opacities {G.opacities.shape}"
    assert G.colors.shape == (B, K * J, 3),    f"colors {G.colors.shape}"
    assert G.velocity.shape == (B, K * J, 3),  f"velocity {G.velocity.shape}"
    print(f"  assemble OK: K·J = {K*J} Gaussians flattened, all 6 attribute shapes correct")

    # ── Render through gsplat (single-batch wrapper) ──────────────────────
    rgb, depth, alpha = render(G, viewmats=viewmats, Ks=Ks, height=H, width=W,
                                render_mode="RGB+D")
    assert rgb.shape == (N_CAM, H, W, 3),  f"rgb shape {rgb.shape}"
    assert depth.shape == (N_CAM, H, W),   f"depth shape {depth.shape}"
    assert alpha.shape == (N_CAM, H, W, 1), f"alpha shape {alpha.shape}"
    print(f"  render   OK: rgb {tuple(rgb.shape)}, depth {tuple(depth.shape)}, "
          f"alpha {tuple(alpha.shape)}")
    print(f"  render   ranges: rgb∈[{rgb.min().item():.3f}, {rgb.max().item():.3f}], "
          f"depth∈[{depth.min().item():.3f}, {depth.max().item():.3f}] m, "
          f"alpha∈[{alpha.min().item():.3f}, {alpha.max().item():.3f}]")

    # ── Backward: gradient through entire stack ───────────────────────────
    target_depth = torch.full_like(depth, 5.0)
    target_rgb = torch.zeros_like(rgb)
    loss = (rgb - target_rgb).abs().mean() + 0.1 * (depth - target_depth).abs().mean()
    loss.backward()

    # Verify gradient reached every leaf parameter that should be trainable.
    failed = []
    for module_name, module in [('lifter', lifter),
                                  ('parent_refiner', parent_refiner),
                                  ('child_head', child_head)]:
        for p_name, p in module.named_parameters():
            if p.grad is None or p.grad.norm().item() == 0:
                failed.append(f"{module_name}.{p_name}")
    assert not failed, f"missing/zero grad: {failed}"
    print(f"  grad     OK: every learnable param in lifter+parent+child got non-zero grad")

    # ── Save eyeball image (cam 0) ────────────────────────────────────────
    out_dir = "out/s1.3_integration"
    os.makedirs(out_dir, exist_ok=True)
    rgb_img = (rgb[0].detach().cpu().clamp(0, 1) * 255).byte().numpy()
    Image.fromarray(rgb_img).save(f"{out_dir}/rgb_cam0.png")
    d = depth[0].detach().cpu()
    d_vis = ((d - d.min()) / (d.max() - d.min() + 1e-9) * 255).byte().numpy()
    Image.fromarray(d_vis).save(f"{out_dir}/depth_cam0.png")
    print(f"\n  Debug images saved to {out_dir}/  (rgb_cam0.png, depth_cam0.png)")

    print("\nS1.3 integration self-test PASSED.")


if __name__ == "__main__":
    _self_test()

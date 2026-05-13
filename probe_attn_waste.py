#!/usr/bin/env python
"""Probe: how much of the trained model's softmax attention mass currently lands
on geometrically invalid sample sites? Loads our 5000-iter T2 checkpoint, runs
one nuScenes frame, captures per-layer attention + projection, reports per-layer
table of (invalid-site fraction, wasted attention mass, per-camera distribution).

Read-only — does not modify the checkpoint or training state.
"""
import os, sys, types, torch, numpy as np

REPO = "/home/satya/skr/S2GO/S2GO"
CKPT = f"{REPO}/out/s2go_small_t2_half_5000iter_nockpt/best.pt"
sys.path.insert(0, REPO)
os.chdir(REPO)
device = "cuda"

from s2go.models.backbone.r50_fpn import R50FPNBackbone
from s2go.models.segmentor import S2GOSegmentor
from s2go.models.encoder.temporal_decoder import DeformableCrossAttention
from s2go.datasets.nusc_loader import NuScenesLoader

# ─── build + load ─────────────────────────────────────────────────────────
print("Building model (T2 paper-spec)…")
backbone = R50FPNBackbone(embed_dims=768, num_outs=4, pretrained=False).to(device)
seg = S2GOSegmentor(
    K=900, J=10, embed_dims=768,
    num_layers=6, num_heads=12, num_groups=12,
    num_levels=4, num_cams=6, num_pts=13,
    feedforward_channels=3072,
    T_queue=1, use_checkpoint=False,
).to(device)

print(f"Loading checkpoint: {CKPT}")
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
backbone.load_state_dict(ckpt['backbone'])
seg.load_state_dict(ckpt['segmentor'])
backbone.eval(); seg.eval()
for p in list(backbone.parameters()) + list(seg.parameters()):
    p.requires_grad = False

# ─── monkey-patch DeformableCrossAttention.forward to capture intermediates ─
orig_forward = DeformableCrossAttention.forward
def patched(self, query, query_pos, feat_flatten, reference_points,
            spatial_shapes, level_start_index, lidar2img, pad_h, pad_w):
    B, N_q = reference_points.shape[:2]
    # mirror the original forward up through softmax + projection so we can store the tensors
    offsets = self.learnable_fc(query).reshape(B, N_q, self.num_pts, 3)
    key_points = reference_points.unsqueeze(-2) + offsets

    l2i_flat = lidar2img[..., :3, :].flatten(-2)
    cam_embed = self.cam_embed(l2i_flat)
    feat_pos = (query + query_pos).unsqueeze(2) + cam_embed.unsqueeze(1)
    # Compute raw logits once; produce BOTH the pre-fix (no mask) and post-fix
    # (with -1e4 mask) softmax distributions so we can A/B them side-by-side.
    w_raw = self.weights_fc(feat_pos).reshape(B, N_q, -1, self.num_groups)

    pts_h = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
    pts_2d_pre = torch.matmul(
        lidar2img[:, :, None, None],
        pts_h[:, None, ..., None]
    ).squeeze(-1)                                                   # (B, N_cam, N, num_pts, 4)
    pts_z = pts_2d_pre[..., 2]                                      # (B, N_cam, N, num_pts)
    pts_2d = pts_2d_pre[..., :2] / pts_2d_pre[..., 2:3].clamp(min=1e-5)
    pts_xy_norm = torch.stack([
        pts_2d[..., 0] / float(pad_w),
        pts_2d[..., 1] / float(pad_h),
    ], dim=-1)                                                      # (B, N_cam, N, num_pts, 2)

    # Build validity in the same 312-site layout used by the softmax.
    valid_cqp = (
        (pts_z > 1e-5) &
        (pts_xy_norm[..., 0] >= 0.0) & (pts_xy_norm[..., 0] <= 1.0) &
        (pts_xy_norm[..., 1] >= 0.0) & (pts_xy_norm[..., 1] <= 1.0)
    )                                                               # (B, N_cam, N_q, num_pts)
    valid_sites = valid_cqp.unsqueeze(2).expand(-1, -1, self.num_levels, -1, -1)
    valid_sites = valid_sites.permute(0, 3, 1, 2, 4).contiguous()
    valid_sites = valid_sites.reshape(B, N_q, -1)                    # (B, N_q, Ncam*L*P)

    # PRE-fix: plain softmax of raw logits
    w_pre = w_raw.softmax(dim=-2)
    # POST-fix: mask invalid sites with -1e4, then softmax
    w_post = w_raw.masked_fill(~valid_sites.unsqueeze(-1), -1e4).softmax(dim=-2)

    self._probe = {
        'weights_pre':  w_pre.detach().float().cpu(),
        'weights_post': w_post.detach().float().cpu(),
        'pts_z':        pts_z.detach().float().cpu(),
        'pts_xy_norm':  pts_xy_norm.detach().float().cpu(),
    }
    # call original to keep the real output (so downstream model state is unchanged)
    return orig_forward(self, query, query_pos, feat_flatten, reference_points,
                        spatial_shapes, level_start_index, lidar2img, pad_h, pad_w)

DeformableCrossAttention.forward = patched

ca_modules = [m for m in seg.modules() if isinstance(m, DeformableCrossAttention)]
print(f"Hooked {len(ca_modules)} DeformableCrossAttention layers")

# ─── load one nuScenes frame ──────────────────────────────────────────────
print("Loading one nuScenes frame…")
loader = NuScenesLoader(T=1, verbose=False)
seq = loader[0]
data = {}
for k, v in seq[0].items():
    if isinstance(v, torch.Tensor):
        data[k] = v.to(device)
    else:
        data[k] = v

# ─── run backbone + segmentor (mimicking overfit.py) ──────────────────────
print("Running forward…")
with torch.no_grad():
    feat, ss, lsi, _ = backbone(data['imgs'])
    data['feat_flatten']     = feat
    data['spatial_shapes']   = ss
    data['level_start_index'] = lsi
    data['pad_h'] = 256
    data['pad_w'] = 704
    _ = seg([data])

# ─── analyze ──────────────────────────────────────────────────────────────
print("\n" + "="*100)
print(f"{'Layer':>5} | {'Invalid sites':>14} | "
      f"{'Wasted (PRE-fix)':>17} | {'Wasted (POST-fix)':>18} | {'Δ attn on valid':>16}")
print("-"*100)

rows = []
for i, mod in enumerate(ca_modules):
    p = mod._probe
    z   = p['pts_z']            # (B, N_cam, N, num_pts)
    xy  = p['pts_xy_norm']      # (B, N_cam, N, num_pts, 2)
    w_pre  = p['weights_pre']   # (B, N_q, 312, num_groups)
    w_post = p['weights_post']  # (B, N_q, 312, num_groups)
    B, N_cam, N_q, num_pts = z.shape
    num_levels = 4

    valid_z  = z > 1e-5
    valid_x  = (xy[..., 0] >= 0) & (xy[..., 0] <= 1)
    valid_y  = (xy[..., 1] >= 0) & (xy[..., 1] <= 1)
    valid    = valid_z & valid_x & valid_y
    inv_frac = (~valid).float().mean().item()

    valid_312 = valid.unsqueeze(2).expand(-1, -1, num_levels, -1, -1)
    valid_312 = valid_312.permute(0, 3, 1, 2, 4).contiguous()
    valid_312 = valid_312.reshape(B, N_q, N_cam * num_levels * num_pts)
    invalid_312 = (~valid_312).float()

    wasted_pre  = (w_pre  * invalid_312.unsqueeze(-1)).sum(dim=2).mean().item()
    wasted_post = (w_post * invalid_312.unsqueeze(-1)).sum(dim=2).mean().item()
    sharpening = (1.0 - wasted_pre) and (1.0 - wasted_post) / (1.0 - wasted_pre)

    rows.append((i, inv_frac, wasted_pre, wasted_post, sharpening))
    print(f"{i:>5} | {inv_frac:>13.2%} | {wasted_pre:>16.2%} | "
          f"{wasted_post:>17.4%} | {sharpening:>15.2f}×")

print("-"*100)
avg_inv = np.mean([r[1] for r in rows])
avg_pre = np.mean([r[2] for r in rows])
avg_post = np.mean([r[3] for r in rows])
print(f"{'mean':>5} | {avg_inv:>13.2%} | {avg_pre:>16.2%} | {avg_post:>17.4%} |")

print("\n" + "="*100)
print(f"Mean invalid-site fraction (geometry)            : {avg_inv:.2%}")
print(f"Mean wasted attention mass — PRE-fix (no mask)   : {avg_pre:.2%}")
print(f"Mean wasted attention mass — POST-fix (-1e4 mask): {avg_post:.4%}")
print(f"Effective attention sharpening on valid sites    : "
      f"{(1.0 - avg_post) / (1.0 - avg_pre):.2f}×")
print("="*100)

"""TemporalDecoder — block [D] in architecture.md, port of StreamPETR.

Three classes, all rolled into one file:
  1. TemporalSelfAttention  — Flash self-attn with past-query K,V concat
  2. DeformableCrossAttention — port of DeformableFeatureAggregationCuda
                                 (Sparse4D 3D-keypoint pattern + multi-cam projection)
  3. TemporalDecoderLayer    — self → cross → FFN block (with Add+Norm)
  4. TemporalDecoder         — N-layer wrapper

Source references:
  - StreamPETR/.../petr_transformer.py:35-175  (PETRMultiheadFlashAttention — abandoned;
                                                  we use flash_attn_func directly for cleaner code)
  - StreamPETR/.../petr_transformer.py:513-797 (PETRTemporalDecoderLayer)
  - StreamPETR/.../detr3d_transformer.py:480-562 (DeformableFeatureAggregationCuda)

mmcv 2.x compat patches:
  - BaseModule → nn.Module
  - @ATTENTION.register_module() → dropped (we instantiate directly, no registry)
  - mmcv.runner.* / mmcv.utils.* / mmdet.models.utils.builder.* → not used
  - flash_attn integration: cast Q,K,V to fp16 around the flash call (paper §B uses
    mixed precision; flash_attn_func only accepts fp16/bf16)

Run self-test:
    cd /media/skr/storage/self_driving/S2GO
    python -m s2go.models.encoder.temporal_decoder
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from flash_attn import flash_attn_func
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction


# ────────────────────────────────────────────────────────────────────────────
# 1. Self-attention with past-query concat (the StreamPETR pattern)
# ────────────────────────────────────────────────────────────────────────────
class TemporalSelfAttention(nn.Module):
    """Self-attn over current queries Q, with K,V = concat(current, past_memory).

    Uses flash_attn_func directly (paper §B: Flash Attention, Dao 2022).
    Inputs are cast to fp16 for the kernel call; output is cast back to input dtype.

    Args:
        embed_dims: query feature dim (768)
        num_heads:  multi-head count (12 → 64-dim heads)
        dropout:    attention dropout (paper-implicit 0.1)
    """

    def __init__(self, embed_dims: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        assert embed_dims % num_heads == 0, \
            f"embed_dims {embed_dims} must be divisible by num_heads {num_heads}"
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.o_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout_p = dropout

    def forward(self, query: torch.Tensor,
                query_pos: Optional[torch.Tensor] = None,
                temp_memory: Optional[torch.Tensor] = None,
                temp_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query:       (B, K, d) current queries
            query_pos:   (B, K, d) positional embedding to add to current
            temp_memory: (B, L, d) past queries from MemoryQueue (None = no temporal)
            temp_pos:    (B, L, d) past positional embedding

        Returns:
            (B, K, d) attended current queries (residual is NOT added here — the
            decoder layer wraps this with Add+Norm)
        """
        B, K, d = query.shape
        # Apply positional encodings to Q and K only — V is *content* and must
        # not have positional embeddings baked in (standard transformer
        # convention; StreamPETR also keeps V free of position). See hacks.md
        # H6 / Stage1_fix_proposed.md Fix #1.
        q_in = query + query_pos if query_pos is not None else query
        if temp_memory is not None:
            past_k = temp_memory + temp_pos if temp_pos is not None else temp_memory
            k_in = torch.cat([q_in,   past_k     ], dim=1)            # (B, K+L, d) — with pos
            v_in = torch.cat([query,  temp_memory], dim=1)            # (B, K+L, d) — content only
        else:
            k_in = q_in
            v_in = query                                              # content only

        # Project Q, K, V
        Q  = self.q_proj(q_in)
        Kp = self.k_proj(k_in)
        Vp = self.v_proj(v_in)

        # Reshape for multi-head: (B, S, num_heads, head_dim)
        S = k_in.shape[1]                                              # same for k_in / v_in
        Q  = Q.reshape(B, K, self.num_heads, self.head_dim)
        Kp = Kp.reshape(B, S, self.num_heads, self.head_dim)
        Vp = Vp.reshape(B, S, self.num_heads, self.head_dim)

        # flash_attn requires fp16/bf16
        orig_dtype = Q.dtype
        if orig_dtype == torch.float32:
            Q, Kp, Vp = Q.half(), Kp.half(), Vp.half()
        out = flash_attn_func(Q, Kp, Vp,
                                dropout_p=self.dropout_p if self.training else 0.0)
        out = out.to(orig_dtype).reshape(B, K, d)
        return self.o_proj(out)


# ────────────────────────────────────────────────────────────────────────────
# 2. Deformable cross-attention (Sparse4D 3D-keypoint + multi-cam projection)
# ────────────────────────────────────────────────────────────────────────────
class DeformableCrossAttention(nn.Module):
    """Port of StreamPETR's DeformableFeatureAggregationCuda (detr3d_transformer.py:480).

    For each query, generate `num_pts=13` learnable 3D offsets relative to the
    query's reference point. Project all (K queries × num_pts × num_cams=6)
    points to 2D image planes via `lidar2img`. Sample image features at those
    2D positions across `num_levels=4` FPN scales using mmcv's
    MultiScaleDeformableAttnFunction (the Zhu 2020 CUDA kernel).

    Differences from StreamPETR's class:
      - BaseModule → nn.Module
      - `img_metas` dict replaced by explicit `pad_h, pad_w` scalars (cleaner API)
      - Reference points are passed in WORLD coords (not normalized [0,1] —
        StreamPETR's `get_global_pos` step is skipped since our lifter already
        produces world-coord queries)
    """

    def __init__(self, embed_dims: int = 768,
                 num_groups: int = 12,
                 num_levels: int = 4,
                 num_cams: int = 6,
                 num_pts: int = 13,
                 dropout: float = 0.1,
                 im2col_step: int = 64,
                 bias: float = 1.0):
        super().__init__()
        assert embed_dims % num_groups == 0, "embed_dims must be divisible by num_groups"
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.group_dims = embed_dims // num_groups
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.num_pts = num_pts
        self.im2col_step = im2col_step
        self.bias = bias

        # Per-query: 3D offsets for keypoints (initialized small)
        self.learnable_fc = nn.Linear(embed_dims, num_pts * 3)
        # Per-(query, cam): attention weights over (groups × levels × pts)
        self.weights_fc = nn.Linear(embed_dims, num_groups * num_levels * num_pts)
        # Camera-pose embedding (12 = flattened 3×4 of lidar2img top rows)
        self.cam_embed = nn.Sequential(
            nn.Linear(12, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
        )
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.drop = nn.Dropout(dropout)

        # Initialize weights to match StreamPETR's `init_weight`
        nn.init.constant_(self.weights_fc.weight, 0.0)
        nn.init.constant_(self.weights_fc.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        # Fix #4 (Stage1_fix_proposed): zero-init learnable_fc.weight so initial
        # 3D keypoint offsets are exactly the uniform-±bias bias term, not
        # Kaiming-random projections of query features (which dominate the bias
        # at init when query_norm ~1 → offsets unbounded). With weight=0, the
        # network learns offsets-from-baseline rather than fighting random init.
        nn.init.constant_(self.learnable_fc.weight, 0.0)
        nn.init.uniform_(self.learnable_fc.bias, -bias, bias)

    def forward(self, query: torch.Tensor,
                query_pos: torch.Tensor,
                feat_flatten: torch.Tensor,
                reference_points: torch.Tensor,
                spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor,
                lidar2img: torch.Tensor,
                pad_h: int,
                pad_w: int) -> torch.Tensor:
        """
        Args:
            query:             (B, K, d) — current queries
            query_pos:         (B, K, d) — positional embedding (added to query for weights)
            feat_flatten:      (B*N_cam, sum_HW, d) — multi-scale FPN, flattened over levels
            reference_points:  (B, K, 3) — query positions in WORLD coords (lidar frame)
            spatial_shapes:    (num_levels, 2) — (h, w) per level
            level_start_index: (num_levels,) — flat-index offsets per level
            lidar2img:         (B, N_cam, 4, 4) — projection matrices
            pad_h, pad_w:      input image dimensions (for normalizing 2D points to [0,1])

        Returns:
            (B, K, d) — image-feature contribution to add residually onto query
        """
        B, N_q = reference_points.shape[:2]

        # ── 3D keypoints: ref_pt + learnable per-query offsets ─────────────
        offsets = self.learnable_fc(query).reshape(B, N_q, self.num_pts, 3)
        key_points = reference_points.unsqueeze(-2) + offsets         # (B, N, num_pts, 3)

        # ── Project 3D keypoints → 2D per camera (hoisted earlier so the
        #    projection-validity mask is available BEFORE the attention softmax).
        pts_h = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        pts_2d_h = torch.matmul(
            lidar2img[:, :, None, None],                              # (B, N_cam, 1, 1, 4, 4)
            pts_h[:, None, ..., None]                                  # (B, 1,    N, num_pts, 4, 1)
        ).squeeze(-1)                                                  # (B, N_cam, N, num_pts, 4)
        pts_z = pts_2d_h[..., 2]                                       # depth in camera frame
        pts_2d = pts_2d_h[..., :2] / pts_2d_h[..., 2:3].clamp(min=1e-5)
        pts_2d_norm = torch.stack([
            pts_2d[..., 0] / float(pad_w),
            pts_2d[..., 1] / float(pad_h),
        ], dim=-1)                                                     # (B, N_cam, N, num_pts, 2)

        # ── Projection-validity mask: a sample site is valid iff (a) the 3D
        #    keypoint is in front of the camera AND (b) its 2D projection
        #    lands inside the image. Without masking, ~80% of softmax
        #    probability mass leaks to sites the camera physically cannot see
        #    (verified empirically by probe_attn_waste.py).
        valid_cqp = (
            (pts_z > 1e-5) &
            (pts_2d_norm[..., 0] >= 0.0) & (pts_2d_norm[..., 0] <= 1.0) &
            (pts_2d_norm[..., 1] >= 0.0) & (pts_2d_norm[..., 1] <= 1.0)
        )                                                              # (B, N_cam, N_q, num_pts)

        # ── Attention weights (per-cam, per-(group, level, point)) ─────────
        l2i_flat = lidar2img[..., :3, :].flatten(-2)                  # (B, N_cam, 12)
        cam_embed = self.cam_embed(l2i_flat)                          # (B, N_cam, d)
        feat_pos = (query + query_pos).unsqueeze(2) + cam_embed.unsqueeze(1)
        # feat_pos: (B, N, N_cam, d)
        weights = self.weights_fc(feat_pos)                           # (B, N, N_cam, groups*levels*pts)
        # Reshape into (B, N_q, N_cam*levels*pts, num_groups) so that softmax
        # across the second-to-last dim normalizes within each (query, group).
        weights = weights.reshape(B, N_q, -1, self.num_groups)

        # Build validity in the same 312-site layout:
        #   site_index = cam * (num_levels * num_pts) + level * num_pts + pt
        # (validity is shared across feature levels — the same 3D point is
        # sampled at all 4 scales)
        valid_sites = valid_cqp.unsqueeze(2).expand(-1, -1, self.num_levels, -1, -1)
        valid_sites = valid_sites.permute(0, 3, 1, 2, 4).contiguous()
        valid_sites = valid_sites.reshape(B, N_q, -1)                  # (B, N_q, Ncam*L*P)

        # Mask invalid sites with -1e4 so their post-softmax mass ≈ 0.
        # bf16/fp16 safe: exp(-1e4) underflows to 0.
        weights = weights.masked_fill(~valid_sites.unsqueeze(-1), -1e4)
        weights = weights.softmax(dim=-2)
        # → (B, N, N_cam*levels*pts, num_groups)
        weights = weights.reshape(B, N_q, self.num_cams, -1, self.num_groups)
        # → (B, N, N_cam, levels*pts, num_groups)
        weights = weights.permute(0, 2, 1, 4, 3).contiguous()
        # → (B, N_cam, N, num_groups, levels*pts)
        weights = weights.flatten(end_dim=1)
        # → (B*N_cam, N, num_groups, levels*pts)
        # MS-Deform-Attn expects (N, Lq, M, L, P); reshape final dim
        weights = weights.reshape(B * self.num_cams, N_q, self.num_groups,
                                    self.num_levels, self.num_pts)

        # ── Reshape sampling coords for mmcv (pts_2d_norm computed above) ──
        pts_2d_norm = pts_2d_norm.flatten(end_dim=1)                   # (B*N_cam, N, num_pts, 2)
        # Expand for (groups, levels): each group/level samples the same 2D points
        pts_2d_norm = pts_2d_norm[:, :, None, None, :, :].expand(
            -1, -1, self.num_groups, self.num_levels, -1, -1).contiguous()
        # → (B*N_cam, N, num_groups, num_levels, num_pts, 2)

        # ── Reshape feat_flatten for the CUDA op: (BN, S, M, group_dims) ──
        bn, num_value, _ = feat_flatten.shape
        feat_4d = feat_flatten.reshape(bn, num_value, self.num_groups, self.group_dims)

        # ── Call mmcv's MultiScaleDeformableAttnFunction (Zhu 2020) ────────
        sampled = MultiScaleDeformableAttnFunction.apply(
            feat_4d, spatial_shapes, level_start_index,
            pts_2d_norm, weights, self.im2col_step,
        )
        # sampled: (B*N_cam, N, embed_dims)  — flat over groups × group_dims
        sampled = sampled.reshape(B, self.num_cams, N_q, self.embed_dims).sum(dim=1)
        # sum across cameras: (B, N, d)

        out = self.output_proj(sampled)
        out = self.drop(out)
        return out


# ────────────────────────────────────────────────────────────────────────────
# 3. Decoder layer (self → cross → FFN, with Add+Norm)
# ────────────────────────────────────────────────────────────────────────────
class TemporalDecoderLayer(nn.Module):
    """One block of the 6-layer temporal transformer (Figure 1 inset)."""

    def __init__(self, embed_dims: int = 768,
                 num_heads: int = 12,
                 num_groups: int = 12,
                 num_levels: int = 4,
                 num_cams: int = 6,
                 num_pts: int = 13,
                 feedforward_channels: int = 3072,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = TemporalSelfAttention(embed_dims, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.cross_attn = DeformableCrossAttention(embed_dims, num_groups, num_levels,
                                                     num_cams, num_pts, dropout)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(self, query, query_pos, temp_memory, temp_pos,
                feat_flatten, reference_points, spatial_shapes,
                level_start_index, lidar2img, pad_h, pad_w):
        # 1. Self-attn (with past-query K,V concat) — Add + Norm
        sa_out = self.self_attn(query, query_pos, temp_memory, temp_pos)
        query = self.norm1(query + sa_out)
        # 2. Cross-attn (deformable, multi-cam) — Add + Norm
        ca_out = self.cross_attn(query, query_pos, feat_flatten, reference_points,
                                   spatial_shapes, level_start_index, lidar2img,
                                   pad_h, pad_w)
        query = self.norm2(query + ca_out)
        # 3. FFN — Add + Norm
        ff_out = self.ffn(query)
        query = self.norm3(query + ff_out)
        return query


class TemporalDecoder(nn.Module):
    """N-layer temporal transformer wrapper (paper §B: 6 layers, embed=768)."""

    def __init__(self, num_layers: int = 6, use_checkpoint: bool = False, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            TemporalDecoderLayer(**layer_kwargs) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint

    def forward(self, query, query_pos, temp_memory, temp_pos,
                feat_flatten, reference_points, spatial_shapes,
                level_start_index, lidar2img, pad_h, pad_w,
                return_intermediate: bool = False):
        intermediates = []
        for layer in self.layers:
            if self.use_checkpoint and self.training and query.requires_grad:
                # Wrap into a closure capturing the two ints (pad_h, pad_w) and
                # the non-Tensor None possibility for temp_memory/temp_pos. The
                # checkpoint helper requires use_reentrant=False to handle None.
                def _run(q, qp, tm, tp, ff, rp, ss, lsi, l2i, layer=layer):
                    return layer(q, qp, tm, tp, ff, rp, ss, lsi, l2i, pad_h, pad_w)
                query = checkpoint(_run,
                                   query, query_pos, temp_memory, temp_pos,
                                   feat_flatten, reference_points, spatial_shapes,
                                   level_start_index, lidar2img,
                                   use_reentrant=False)
            else:
                query = layer(query, query_pos, temp_memory, temp_pos,
                                feat_flatten, reference_points, spatial_shapes,
                                level_start_index, lidar2img, pad_h, pad_w)
            if return_intermediate:
                intermediates.append(query)
        if return_intermediate:
            return torch.stack(intermediates, dim=0)              # (num_layers, B, K, d)
        return query                                                # (B, K, d)


# ────────────────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────────────────
def _self_test():
    torch.manual_seed(0)
    device = "cuda"
    B, K, L, d = 1, 900, 1024, 768
    N_cam = 6
    num_pts = 13
    pad_h, pad_w = 256, 704

    # Synthetic FPN with 4 levels (typical strides 4, 8, 16, 32 on 256×704)
    levels_hw = [(64, 176), (32, 88), (16, 44), (8, 22)]
    sum_hw = sum(h * w for h, w in levels_hw)
    print(f"S1.5b temporal decoder: B={B}, K={K}, L={L}, d={d}, N_cam={N_cam}, "
          f"FPN levels={levels_hw}, sum_HW={sum_hw}")

    # ── Synthesize inputs ─────────────────────────────────────────────────
    query = torch.randn(B, K, d, device=device)
    query_pos = torch.randn(B, K, d, device=device) * 0.1
    temp_memory = torch.randn(B, L, d, device=device)
    temp_pos = torch.randn(B, L, d, device=device) * 0.1
    # Reference points in nuScenes-like world coords
    reference_points = torch.empty(B, K, 3, device=device)
    reference_points[..., 0].uniform_(-30.0, 30.0)
    reference_points[..., 1].uniform_(-30.0, 30.0)
    reference_points[..., 2].uniform_(-1.0, 3.0)
    # FPN feat_flatten: (B*N_cam, sum_HW, d)
    feat_flatten = torch.randn(B * N_cam, sum_hw, d, device=device) * 0.1
    spatial_shapes = torch.tensor(levels_hw, device=device, dtype=torch.long)
    # level_start_index: 0, sum_l0, sum_l0+l1, ...
    level_start_index = torch.tensor(
        [0] + [sum(h * w for h, w in levels_hw[:i + 1]) for i in range(len(levels_hw) - 1)],
        device=device, dtype=torch.long)
    # Synthetic lidar2img: identity-ish with mild rotation per cam
    lidar2img = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, N_cam, -1, -1).contiguous()
    # Add a focal length to make the projection produce in-image points
    lidar2img = lidar2img.clone()
    lidar2img[..., 0, 0] = 200.0   # fx
    lidar2img[..., 1, 1] = 200.0   # fy
    lidar2img[..., 0, 2] = pad_w / 2.0
    lidar2img[..., 1, 2] = pad_h / 2.0
    # Push cam center back so points at z=0..3 are in front
    lidar2img[..., 2, 3] = 5.0

    # ── 1. TemporalSelfAttention alone ────────────────────────────────────
    sa = TemporalSelfAttention(embed_dims=d, num_heads=12).to(device)
    out_sa = sa(query, query_pos, temp_memory, temp_pos)
    assert out_sa.shape == (B, K, d), f"self-attn out {out_sa.shape}"
    # Without temp_memory should also work
    out_sa_no_mem = sa(query, query_pos, temp_memory=None)
    assert out_sa_no_mem.shape == (B, K, d)
    print(f"  TemporalSelfAttention   OK: {tuple(out_sa.shape)} with past, "
          f"{tuple(out_sa_no_mem.shape)} without past")

    # ── 2. DeformableCrossAttention alone ────────────────────────────────
    ca = DeformableCrossAttention(embed_dims=d, num_groups=12, num_levels=4,
                                    num_cams=N_cam, num_pts=num_pts).to(device)
    out_ca = ca(query, query_pos, feat_flatten, reference_points,
                  spatial_shapes, level_start_index, lidar2img, pad_h, pad_w)
    assert out_ca.shape == (B, K, d), f"cross-attn out {out_ca.shape}"
    print(f"  DeformableCrossAttention OK: {tuple(out_ca.shape)}")

    # ── 3. Single decoder layer ──────────────────────────────────────────
    layer = TemporalDecoderLayer(embed_dims=d, num_heads=12, num_groups=12,
                                   num_levels=4, num_cams=N_cam, num_pts=num_pts,
                                   feedforward_channels=3072).to(device)
    out_layer = layer(query, query_pos, temp_memory, temp_pos,
                        feat_flatten, reference_points, spatial_shapes,
                        level_start_index, lidar2img, pad_h, pad_w)
    assert out_layer.shape == (B, K, d), f"layer out {out_layer.shape}"
    print(f"  TemporalDecoderLayer    OK: {tuple(out_layer.shape)}")

    # ── 4. Full 6-layer decoder ──────────────────────────────────────────
    decoder = TemporalDecoder(num_layers=6, embed_dims=d, num_heads=12, num_groups=12,
                                num_levels=4, num_cams=N_cam, num_pts=num_pts,
                                feedforward_channels=3072).to(device)
    out = decoder(query, query_pos, temp_memory, temp_pos,
                    feat_flatten, reference_points, spatial_shapes,
                    level_start_index, lidar2img, pad_h, pad_w)
    assert out.shape == (B, K, d), f"decoder out {out.shape}"
    print(f"  TemporalDecoder (6L)    OK: {tuple(out.shape)}")

    # ── 5. return_intermediate ──────────────────────────────────────────
    out_inter = decoder(query, query_pos, temp_memory, temp_pos,
                          feat_flatten, reference_points, spatial_shapes,
                          level_start_index, lidar2img, pad_h, pad_w,
                          return_intermediate=True)
    assert out_inter.shape == (6, B, K, d), f"intermediate {out_inter.shape}"
    print(f"  return_intermediate     OK: {tuple(out_inter.shape)}  (6 layers stacked)")

    # ── 6. Param count ───────────────────────────────────────────────────
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  param count: {n_params/1e6:.2f} M  "
          f"(rough estimate: 6 layers × ~7M = ~42M for the temporal decoder)")

    # ── 7. Gradient flow end-to-end ──────────────────────────────────────
    query_g = query.detach().clone().requires_grad_(True)
    feat_g = feat_flatten.detach().clone().requires_grad_(True)
    out_g = decoder(query_g, query_pos, temp_memory, temp_pos,
                      feat_g, reference_points, spatial_shapes,
                      level_start_index, lidar2img, pad_h, pad_w)
    loss = out_g.sum()
    loss.backward()
    assert query_g.grad is not None and query_g.grad.norm().item() > 0, \
        "no grad on query input"
    assert feat_g.grad is not None and feat_g.grad.norm().item() > 0, \
        "no grad on feat_flatten input"
    # Check at least one trainable param in each sub-module
    assert decoder.layers[0].self_attn.q_proj.weight.grad is not None, "self-attn q_proj no grad"
    assert decoder.layers[0].cross_attn.learnable_fc.weight.grad is not None, "cross-attn no grad"
    assert decoder.layers[5].ffn[0].weight.grad is not None, "ffn no grad"
    print(f"  gradient flow OK: ‖∂L/∂query‖={query_g.grad.norm().item():.2e}, "
          f"‖∂L/∂feat‖={feat_g.grad.norm().item():.2e}")

    print("\nS1.5b temporal decoder self-test PASSED.")


if __name__ == "__main__":
    _self_test()

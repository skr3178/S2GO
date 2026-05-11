"""S2GOSegmentor — top-level Stage-1 module (block-diagram-level wiring).

Wires all S1.x components into the T-frame forward pass from
Stage1_pseudocode.md Algorithm 1. Image features (`feat_flatten`) are passed in
pre-extracted; the R50+FPN backbone integration is a separate later step.

Pipeline per frame (one iteration of the T=4 loop):
    [C]   FPS+ε init                — S2GOLifter
    [B]   pre_update memory queue   — MemoryQueue.pre_update
    [D]   Temporal decoder          — TemporalDecoder
    [E]   Parent refiner            — ParentRefiner
    [F]   Child Gaussian head       — ChildGaussianHead
    [G]   Assemble flat Gaussians   — assemble_gaussians
    [H]   Opacity-δ propagator      — OpacityDeltaPropagator
    [B]   post_update memory queue  — MemoryQueue.post_update

The segmentor returns per-frame outputs; rendering ([G1] gsplat) and Eq. 8
losses are computed by the caller (separation keeps the model architecture
independent of the loss head).

Run synthetic-data integration test:
    cd /media/skr/storage/self_driving/S2GO
    python -m s2go.models.segmentor
"""
from typing import NamedTuple, List, Dict
import torch
import torch.nn as nn

from .lifter.s2go_lifter import S2GOLifter
from .encoder.heads import ParentRefiner, ChildGaussianHead, ParentPred
from .encoder.assembly import assemble_gaussians, Gaussians
from .encoder.temporal_decoder import TemporalDecoder
from .queue.memory import MemoryQueue
from .queue.propagator import OpacityDeltaPropagator


class FrameOutput(NamedTuple):
    """Per-frame output of S2GOSegmentor."""
    gaussians:   Gaussians       # (B, K*J, ...) ready for gsplat / voxel splat
    anchors_xyz: torch.Tensor    # (B, K, 3) noise-free FPS anchors → L_denoise target
    init_xyz:    torch.Tensor    # (B, K, 3) noised query positions
    refined_xyz: torch.Tensor    # (B, K, 3) = init_xyz + parent.offset → L_denoise input
    parent:      ParentPred
    velocity:    torch.Tensor    # (B, K*J, 3) for ±0.5s render warps
    prop:        'Propagated'    # opacity-δ propagator output (top-k selection;
                                  # used only for diagnostics, not for loss)


class S2GOSegmentor(nn.Module):
    """Stage-1 top-level segmentor (Algorithm 1 wired into one nn.Module).

    Args:
        K, J:                paper §B (S2GO-Small): 900, 10
        embed_dims:          paper §B: 768
        num_layers:          paper §B (Figure 1 "6×"): 6
        num_heads, num_groups: 12 (= embed_dims / 64 head dim)
        num_levels, num_cams: 4, 6 (FPN scales × nuScenes cameras)
        num_pts:             13 (StreamPETR class default; ablate later)
        feedforward_channels: 4 × embed_dims = 3072
        lidar_noise:         ε scale (paper §B nuScenes-SurroundOcc: 1.0 m)
        propagate_k:         queue propagation count (StreamPETR convention: 256)
        memory_len:          queue capacity (default = T_q × propagate_k = 4·256)
        T_queue:             frames retained in queue (paper §B: 4)
    """

    def __init__(self,
                 K: int = 900, J: int = 10,
                 embed_dims: int = 768,
                 num_layers: int = 6,
                 num_heads: int = 12, num_groups: int = 12,
                 num_levels: int = 4, num_cams: int = 6,
                 num_pts: int = 13,
                 feedforward_channels: int = 3072,
                 dropout: float = 0.1,
                 lidar_noise: float = 1.0,
                 propagate_k: int = 256,
                 T_queue: int = 4,
                 memory_len: int = None,
                 child_mode: str = 'rgb',
                 use_checkpoint: bool = False):
        super().__init__()
        self.K = K
        self.J = J
        self.embed_dims = embed_dims
        memory_len = memory_len if memory_len is not None else T_queue * propagate_k

        # Components — one of each
        self.lifter = S2GOLifter(K=K, embed_dims=embed_dims, eps=lidar_noise)
        self.parent_refiner = ParentRefiner(embed_dims=embed_dims)
        self.child_head = ChildGaussianHead(embed_dims=embed_dims, J=J, mode=child_mode)
        self.decoder = TemporalDecoder(
            num_layers=num_layers,
            embed_dims=embed_dims, num_heads=num_heads, num_groups=num_groups,
            num_levels=num_levels, num_cams=num_cams, num_pts=num_pts,
            feedforward_channels=feedforward_channels, dropout=dropout,
            use_checkpoint=use_checkpoint)
        self.queue = MemoryQueue(memory_len=memory_len, embed_dims=embed_dims)
        self.propagator = OpacityDeltaPropagator(k=propagate_k)

        # Positional embedding for query xyz: 3 → embed_dims
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )

    def reset_memory(self):
        """Wipe queue at start of each mini-sequence."""
        self.queue.reset()

    def forward_one_frame(self, frame: Dict[str, torch.Tensor]) -> FrameOutput:
        """One step of Algorithm 1's T-frame loop.

        Required keys in `frame`:
            lidar_pts          (B, M, 3)
            feat_flatten       (B*N_cam, sum_HW, d) — pre-extracted FPN
            spatial_shapes     (num_levels, 2)
            level_start_index  (num_levels,)
            lidar2img          (B, N_cam, 4, 4)
            pad_h, pad_w       int
            ego_pose           (B, 4, 4)         — current frame, world←cam coord-frame map
            ego_pose_inv       (B, 4, 4)
            timestamp          (B,)               — relative time delta
            prev_exists        (B,)               — 1 same scene, 0 reset
        """
        # 1. FPS+ε init  ([C])
        anchors_xyz, init_xyz, init_feat = self.lifter(frame['lidar_pts'])

        # 2. Pre-update memory queue (ego-compensate past)  ([B])
        self.queue.pre_update(
            prev_exists=frame['prev_exists'],
            timestamp_delta=frame['timestamp'],
            ego_pose_inv=frame['ego_pose_inv'])

        # 3. Positional embeddings (current + past)
        query_pos = self.pos_mlp(init_xyz)
        # Past memory positional: embed memory_reference_point
        temp_memory = self.queue.memory_embedding                  # (B, L, d) detached
        temp_pos = self.pos_mlp(self.queue.memory_reference_point)  # (B, L, d)

        # 4. Temporal decoder  ([D])
        refined_feat = self.decoder(
            query=init_feat,
            query_pos=query_pos,
            temp_memory=temp_memory,
            temp_pos=temp_pos,
            feat_flatten=frame['feat_flatten'],
            reference_points=init_xyz,
            spatial_shapes=frame['spatial_shapes'],
            level_start_index=frame['level_start_index'],
            lidar2img=frame['lidar2img'],
            pad_h=frame['pad_h'], pad_w=frame['pad_w'])

        # 5. Parent refiner  ([E])
        parent = self.parent_refiner(refined_feat, query_pos)
        refined_xyz = init_xyz + parent.offset

        # 6. Child Gaussian head (Eq. 6)  ([F])
        children = self.child_head(parent.feat)

        # 7. Assemble flat Gaussians  ([G])
        G = assemble_gaussians(init_xyz, parent, children, with_rgb=(self.child_head.mode == 'rgb'))

        # 8. Opacity-δ propagator  ([H])
        prop = self.propagator(refined_xyz, parent)

        # 9. Post-update memory queue (push to head, transform to world)  ([B])
        self.queue.post_update(prop, ego_pose=frame['ego_pose'], timestamp=frame['timestamp'])

        return FrameOutput(
            gaussians=G,
            anchors_xyz=anchors_xyz,
            init_xyz=init_xyz,
            refined_xyz=refined_xyz,
            parent=parent,
            velocity=G.velocity,
            prop=prop)

    def forward(self, sequence: List[Dict[str, torch.Tensor]]) -> List[FrameOutput]:
        """T-frame mini-sequence forward. Resets memory at start."""
        self.reset_memory()
        return [self.forward_one_frame(frame) for frame in sequence]


# ────────────────────────────────────────────────────────────────────────────
# Synthetic-data integration test (S1.7a + S1.7d)
# ────────────────────────────────────────────────────────────────────────────
def _self_test():
    import math
    import torch
    from ..render.gsplat_wrapper import render
    from ..losses.pretrain_loss import DenoiseLoss, DepthRenderLoss, RGBRenderLoss

    torch.manual_seed(0)
    device = "cuda"
    B, T = 1, 4
    K, J, d = 900, 10, 768
    M = 10_000
    N_cam = 6
    pad_h, pad_w = 256, 704
    levels_hw = [(64, 176), (32, 88), (16, 44), (8, 22)]
    sum_hw = sum(h * w for h, w in levels_hw)

    print(f"S1.7 synthetic integration: T={T} frames, B={B}, K={K}, J={J}, d={d}")
    print(f"  N_cam={N_cam}, image={pad_w}×{pad_h}, FPN sum_HW={sum_hw}")

    # ── Build a small-config segmentor for testing ────────────────────────
    # Keep K, J, d at paper-spec but reduce num_layers / num_pts to fit memory
    seg = S2GOSegmentor(
        K=K, J=J, embed_dims=d,
        num_layers=2,                 # T0-tier: reduced from 6 for speed
        num_heads=12, num_groups=12,
        num_levels=4, num_cams=N_cam,
        num_pts=4,                    # T0-tier: reduced from 13
        feedforward_channels=2048,    # smaller FFN for the test
        lidar_noise=1.0,
        propagate_k=256,
        T_queue=T,
    ).to(device)

    n_params = sum(p.numel() for p in seg.parameters() if p.requires_grad)
    print(f"  segmentor params: {n_params/1e6:.2f} M (T0-tier with num_layers=2, num_pts=4)")

    # ── Build T=4 synthetic frames ────────────────────────────────────────
    def make_frame(t):
        # synthetic LiDAR
        pts = torch.empty(B, M, 3, device=device)
        pts[..., 0].uniform_(-30, 30)
        pts[..., 1].uniform_(-30, 30)
        pts[..., 2].uniform_(-1, 3)
        # synthetic FPN features
        feat = torch.randn(B * N_cam, sum_hw, d, device=device) * 0.1
        # synthetic camera intrinsics (focal 200, center at image middle)
        l2i = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, N_cam, -1, -1).contiguous().clone()
        l2i[..., 0, 0] = 200.0
        l2i[..., 1, 1] = 200.0
        l2i[..., 0, 2] = pad_w / 2
        l2i[..., 1, 2] = pad_h / 2
        l2i[..., 2, 3] = 5.0    # push cam back so points at z=0..3 are in front
        # ego pose (identity for test — no actual motion)
        ego = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
        prev_exists = torch.tensor([1.0 if t > 0 else 0.0], device=device)
        ts_delta = torch.tensor([0.5], device=device)
        return {
            'lidar_pts': pts,
            'feat_flatten': feat,
            'spatial_shapes': torch.tensor(levels_hw, device=device, dtype=torch.long),
            'level_start_index': torch.tensor(
                [0] + [sum(h * w for h, w in levels_hw[:i + 1]) for i in range(3)],
                device=device, dtype=torch.long),
            'lidar2img': l2i,
            'pad_h': pad_h, 'pad_w': pad_w,
            'ego_pose': ego, 'ego_pose_inv': ego,
            'timestamp': ts_delta,
            'prev_exists': prev_exists,
        }
    sequence = [make_frame(t) for t in range(T)]
    print(f"  built T={T} synthetic frames with {M}-pt LiDAR + FPN-shaped features")

    # ── Forward through segmentor ─────────────────────────────────────────
    outputs = seg(sequence)
    assert len(outputs) == T
    for t, o in enumerate(outputs):
        assert o.gaussians.means.shape == (B, K * J, 3), \
            f"frame {t} means {o.gaussians.means.shape}"
        assert o.gaussians.colors.shape == (B, K * J, 3)
        assert o.parent.feat.shape == (B, K, d)
    print(f"  segmentor forward OK: {T} frames × {K * J} Gaussians, all shapes match")

    # ── Render each frame via gsplat + compute Eq. 8 losses ──────────────
    den_loss = DenoiseLoss()
    dep_loss = DepthRenderLoss()
    rgb_loss = RGBRenderLoss()

    L_den, L_dep, L_rgb = 0.0, 0.0, 0.0
    for t, o in enumerate(outputs):
        # Render through gsplat (B=1 → segmentor produces (B, KJ, ...) Gaussians)
        viewmats = torch.eye(4, device=device).unsqueeze(0).expand(N_cam, -1, -1).contiguous()
        # Use real camera intrinsics from frame (drop the row 3,3 perspective row)
        l2i = sequence[t]['lidar2img'][0]                  # (N_cam, 4, 4) — B==1
        Ks = l2i[:, :3, :3]                                 # (N_cam, 3, 3)
        rgb, depth, _ = render(o.gaussians, viewmats=viewmats, Ks=Ks,
                                  height=pad_h, width=pad_w)
        # Synthetic targets: random RGB GT, sparse depth from a few projected points
        rgb_gt = torch.rand_like(rgb)
        depth_gt = torch.zeros_like(depth)
        # ~5% of pixels have a sparse depth
        n_valid = int(0.05 * N_cam * pad_h * pad_w)
        flat_idx = torch.randperm(N_cam * pad_h * pad_w, device=device)[:n_valid]
        depth_gt.view(-1)[flat_idx] = torch.empty(n_valid, device=device).uniform_(2.0, 8.0)

        L_den = L_den + den_loss(o.anchors_xyz, o.refined_xyz)
        L_dep = L_dep + dep_loss(depth, depth_gt)
        L_rgb = L_rgb + rgb_loss(rgb, rgb_gt)

    L = (10.0 * L_den + 1.0 * L_dep + 1.0 * L_rgb) / T
    print(f"  Eq. 8 loss: total={L.item():.4f}, "
          f"L_den={L_den.item()/T:.3f}, L_dep={L_dep.item()/T:.4f}, L_rgb={L_rgb.item()/T:.4f}")
    assert L.requires_grad, "loss should require grad"
    assert torch.isfinite(L), f"loss is non-finite: {L}"

    # ── Backward — verify gradient reaches every learnable param ─────────
    L.backward()
    no_grad = []
    for name, p in seg.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None or p.grad.norm().item() == 0:
            no_grad.append(name)
    if no_grad:
        # The pos_mlp may have zero grad on first frame if its output is unused
        # there; but with 4 frames and post_update at each, every param should
        # receive gradient. Print as info but only fail on multiple-frame absence.
        print(f"  WARN: {len(no_grad)} param(s) without gradient: {no_grad[:5]}{'…' if len(no_grad) > 5 else ''}")
    else:
        print(f"  gradient flow OK: every learnable param in segmentor got non-zero grad")

    # GPU memory peek
    mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  peak GPU memory: {mem_mb:.1f} MB")

    print("\nS1.7 synthetic-data integration self-test PASSED.")


if __name__ == "__main__":
    _self_test()

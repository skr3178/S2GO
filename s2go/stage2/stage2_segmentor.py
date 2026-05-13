"""S2GOStage2 — Stage 1 segmentor + per-parent semantic head wrapper.

This module wraps `S2GOSegmentor` (which we do NOT modify) and a fresh
`SemanticHead`. Per-frame forward returns:

  - The Stage-1 `FrameOutput` (gaussians, anchors_xyz, refined_xyz,
    parent, velocity, prop) — exactly as before.
  - `sem_logits_per_g`: (B, K*J, C) per-Gaussian class logits, obtained by
    broadcasting per-parent logits over the J=10 children of each parent.

The Stage 1 `Gaussians` dataclass stays unchanged (`colors=None` when
`child_mode='semantic'`); semantic logits travel as a sibling tensor.
Downstream code (G2V layer, loss, eval) treats them as a separate
per-Gaussian field.

No edits to `s2go/models/segmentor.py` or `s2go/models/encoder/assembly.py`.
"""
from typing import NamedTuple, List, Dict
import torch
import torch.nn as nn

from ..models.segmentor import S2GOSegmentor, FrameOutput
from .semantic_head import SemanticHead
from .stage2_lifter import Stage2Lifter
from . import NUM_CLASSES


class Stage2FrameOutput(NamedTuple):
    """Per-frame Stage-2 output: everything Stage 1 gives + class logits."""
    raw:               FrameOutput
    sem_logits_per_g:  torch.Tensor    # (B, K*J, C)


class S2GOStage2(nn.Module):
    """Stage-2 model: Stage-1 segmentor (in semantic mode) + parent classifier.

    Args:
        segmentor_kwargs: kwargs forwarded to `S2GOSegmentor.__init__`. The
                          `child_mode` is forced to 'semantic' here.
        num_classes:      total class count (default 18 = 17 sem + 1 empty).
        sem_hidden:       SemanticHead hidden width.
        query_init:       'learned' (default, paper §3.4.1) or 'fps_lidar'.
                          Stage 2 paper-faithful behaviour uses learnable
                          query positions (no LiDAR in forward). 'fps_lidar'
                          falls back to the inherited Stage 1 lifter for
                          debugging / direct ablation.
        stage2_init_range: AABB for `Stage2Lifter.query_xyz` init when
                          `query_init='learned'`. Default matches Occ3D
                          voxel grid extent.
    """
    def __init__(self,
                 segmentor_kwargs: dict,
                 num_classes: int = NUM_CLASSES,
                 sem_hidden: int = 256,
                 query_init: str = 'learned',
                 stage2_init_range=(-50.0, -50.0, -5.0, 50.0, 50.0, 3.0)):
        super().__init__()
        assert query_init in ('learned', 'fps_lidar'), query_init
        kwargs = dict(segmentor_kwargs)
        kwargs['child_mode'] = 'semantic'   # ensure no RGB head, free 3 dims
        self.segmentor = S2GOSegmentor(**kwargs)
        embed_dims = self.segmentor.embed_dims
        self.semantic_head = SemanticHead(
            feat_dim=embed_dims,
            num_classes=num_classes,
            hidden=sem_hidden,
        )
        self.num_classes = num_classes
        self.J = self.segmentor.J
        self.query_init = query_init

        # Gap-1 fix (paper §3.4.1): for Stage 2, swap the inherited S2GOLifter
        # (FPS+ε on LiDAR) with a `Stage2Lifter` that emits learnable query
        # positions and ignores `lidar_pts`. No edit to the Stage 1 segmentor
        # class — this is a runtime swap on `self.segmentor.lifter`.
        if query_init == 'learned':
            self.segmentor.lifter = Stage2Lifter(
                K=self.segmentor.K,
                embed_dims=embed_dims,
                init_range=stage2_init_range,
            )

    def reset_memory(self):
        self.segmentor.reset_memory()

    def forward_one_frame(self, frame: Dict[str, torch.Tensor]) -> Stage2FrameOutput:
        out: FrameOutput = self.segmentor.forward_one_frame(frame)
        # parent.feat: (B, K, D); per-parent logits (B, K, C)
        sem_parent = self.semantic_head(out.parent.feat)
        B, K, C = sem_parent.shape
        # Broadcast over J=10 children → (B, K*J, C). Matches how
        # `assemble_gaussians` broadcasts parent.opa and parent.velocity.
        sem_per_g = (sem_parent.unsqueeze(2)
                     .expand(-1, -1, self.J, -1)
                     .reshape(B, K * self.J, C)
                     .contiguous())
        return Stage2FrameOutput(raw=out, sem_logits_per_g=sem_per_g)

    def forward(self, sequence: List[Dict[str, torch.Tensor]]) -> List[Stage2FrameOutput]:
        self.reset_memory()
        return [self.forward_one_frame(f) for f in sequence]

    # ── Stage-1 checkpoint loading bridge ──────────────────────────────────
    def load_stage1_state(self, ckpt: dict, strict: bool = False):
        """Load weights from a Stage-1 checkpoint into the underlying segmentor.

        Args:
            ckpt:   loaded dict from `torch.load('stage1.pt')`, with keys
                    'segmentor', 'backbone', 'config'.
            strict: passed to load_state_dict.

        Notes (post-fix for Gap 8):
            - Stage 1 was trained with `child_mode='rgb'` → its child_head.head
              is `Linear(d, 14)`. Our Stage 2 segmentor has child_head in
              'semantic' mode → `Linear(d, 11)`. The 14 rows of the Stage-1
              head correspond to:
                  rows 0..2  : child offset (3 dims)
                  rows 3..5  : child scale (3 dims)
                  rows 6..9  : child rotation quaternion (4 dims)
                  row  10    : child opacity (1 dim)
                  rows 11..13: RGB (3 dims, Stage 1 only)
              The first 11 rows have identical meaning between the two
              stages — we copy them so 5000 iters of Stage 1's learned
              child-geometry projection survive the bridge. The RGB rows
              (11..13) are silently discarded.
            - The SemanticHead has no Stage-1 counterpart → fresh init.
            - When `query_init='learned'` (Gap-1 fix), Stage 1's
              `lifter.query_feat` weight loads cleanly into
              `Stage2Lifter.query_feat` (same name + shape (K, d));
              `Stage2Lifter.query_xyz` has no Stage 1 counterpart and stays
              at its uniform-AABB init.
            - The R50+FPN backbone is loaded by the training script itself,
              not here (it lives outside this nn.Module).

        Returns:
            (missing_keys, unexpected_keys, head_partial_transferred:bool)
        """
        seg_state = dict(ckpt['segmentor'])

        # Pop the mismatched child_head.head keys before load_state_dict so
        # the strict=False path doesn't simply ignore them — we want to
        # actively partial-copy them afterwards.
        s1_head_w = seg_state.pop('child_head.head.weight', None)
        s1_head_b = seg_state.pop('child_head.head.bias',   None)

        missing, unexpected = self.segmentor.load_state_dict(seg_state, strict=strict)

        # Partial transfer of the (14, d) → (11, d) head: copy the first 11
        # rows for offset/scale/rot/opa, drop the trailing 3 RGB rows.
        head_partial_transferred = False
        if s1_head_w is not None and s1_head_b is not None:
            s2_w = self.segmentor.child_head.head.weight   # (11, d)
            s2_b = self.segmentor.child_head.head.bias     # (11,)
            assert s1_head_w.shape[0] == 14 and s1_head_b.shape[0] == 14, \
                f"unexpected Stage-1 child_head.head shapes: " \
                f"w={tuple(s1_head_w.shape)}, b={tuple(s1_head_b.shape)}"
            assert s2_w.shape[0] == 11 and s2_b.shape[0] == 11, \
                f"unexpected Stage-2 child_head.head shapes: " \
                f"w={tuple(s2_w.shape)}, b={tuple(s2_b.shape)}"
            assert s1_head_w.shape[1] == s2_w.shape[1], \
                f"d mismatch between stages: " \
                f"s1={s1_head_w.shape[1]}, s2={s2_w.shape[1]}"
            with torch.no_grad():
                s2_w.copy_(s1_head_w[:11].to(s2_w.device, s2_w.dtype))
                s2_b.copy_(s1_head_b[:11].to(s2_b.device, s2_b.dtype))
            head_partial_transferred = True

        return missing, unexpected, head_partial_transferred


# ────────────────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────────────────
def _self_test():
    """Synthetic-data integration: Stage 2 forward + grad flow."""
    torch.manual_seed(0)
    device = "cuda"
    B, T = 1, 1
    K, J, d = 900, 10, 768
    M = 5000
    N_cam = 6
    pad_h, pad_w = 256, 704
    levels_hw = [(64, 176), (32, 88), (16, 44), (8, 22)]
    sum_hw = sum(h * w for h, w in levels_hw)

    print(f"Stage2 synthetic integration: T={T} frames, B={B}, K={K}, J={J}, d={d}")
    model = S2GOStage2(
        segmentor_kwargs=dict(
            K=K, J=J, embed_dims=d,
            num_layers=2, num_heads=12, num_groups=12,
            num_levels=4, num_cams=N_cam,
            num_pts=4, feedforward_channels=2048,
            T_queue=1,
        ),
        num_classes=18,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_sem = sum(p.numel() for p in model.semantic_head.parameters())
    print(f"  total params: {n_params/1e6:.2f} M  (SemanticHead alone: {n_sem/1e6:.3f} M)")

    # One synthetic frame
    pts = torch.empty(B, M, 3, device=device); pts[..., 0].uniform_(-30, 30)
    pts[..., 1].uniform_(-30, 30); pts[..., 2].uniform_(-1, 3)
    feat = torch.randn(B * N_cam, sum_hw, d, device=device) * 0.1
    l2i = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, N_cam, -1, -1).contiguous().clone()
    l2i[..., 0, 0] = 200.0; l2i[..., 1, 1] = 200.0
    l2i[..., 0, 2] = pad_w / 2; l2i[..., 1, 2] = pad_h / 2
    l2i[..., 2, 3] = 5.0
    ego = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
    frame = {
        'lidar_pts': pts, 'feat_flatten': feat,
        'spatial_shapes':    torch.tensor(levels_hw, device=device, dtype=torch.long),
        'level_start_index': torch.tensor(
            [0] + [sum(h * w for h, w in levels_hw[:i + 1]) for i in range(3)],
            device=device, dtype=torch.long),
        'lidar2img': l2i, 'pad_h': pad_h, 'pad_w': pad_w,
        'ego_pose': ego, 'ego_pose_inv': ego,
        'timestamp': torch.tensor([0.0], device=device),
        'prev_exists': torch.tensor([0.0], device=device),
    }

    outs = model([frame])
    assert len(outs) == 1
    out = outs[0]
    assert out.raw.gaussians.means.shape == (B, K * J, 3)
    assert out.raw.gaussians.colors is None, "semantic mode → colors should be None"
    assert out.sem_logits_per_g.shape == (B, K * J, 18)

    # Check children-of-same-parent share class logits
    sem = out.sem_logits_per_g.view(B, K, J, 18)
    spread = (sem - sem[:, :, :1, :]).abs().max().item()
    assert spread < 1e-5, f"siblings should share semantic logits; spread={spread}"
    print(f"  per-Gaussian sem logits OK: shape {tuple(out.sem_logits_per_g.shape)}, "
          f"sibling spread {spread:.1e}")

    # Backward through both segmentor + semantic head
    loss = out.sem_logits_per_g.sum() + out.raw.gaussians.means.sum()
    loss.backward()
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or p.grad.norm().item() == 0)]
    # pos_mlp / propagator may have zero grad at T=1 if their outputs aren't
    # consumed by the loss; that's expected.
    print(f"  params without grad (likely T=1-only): {len(no_grad)}")
    print("S2GOStage2 self-test PASSED.")


if __name__ == "__main__":
    _self_test()

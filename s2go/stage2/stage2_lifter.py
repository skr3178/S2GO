"""Stage2Lifter — learnable query init (no LiDAR), paper §3.4.1 compliant.

S2GO paper distinguishes the two stages' query initialisation:
  - Stage 1 pretraining uses noised LiDAR points (FPS+ε) — `S2GOLifter`.
  - Stage 2 occupancy training uses **learnable 3D query positions**;
    LiDAR is not consumed in the forward pass ("S2GO only uses RGB images
    during inference" — paper §3.4.1).

This module is a drop-in replacement for `S2GOLifter`. It keeps the same
forward signature (`pts → (anchors_xyz, init_xyz, init_feat)`) so it can
be hot-swapped into `S2GOSegmentor.lifter` after the Stage-1 segmentor
class has been constructed, without editing any Stage 1 file.

The `pts` argument is **accepted and ignored** — it remains in the
signature only so the segmentor's `forward_one_frame` does not need
modification. `anchors_xyz` is returned equal to `init_xyz` (denoise loss
is not computed in Stage 2).

Parameter naming is chosen so the Stage 1 `lifter.query_feat` checkpoint
weights load cleanly (shape `(K, embed_dims)` matches in both lifters);
`lifter.query_xyz` is unique to Stage 2 and starts fresh.
"""
from typing import Tuple

import torch
import torch.nn as nn


class Stage2Lifter(nn.Module):
    """Learnable per-query (xyz, feature) embeddings — Stage 2 paper recipe.

    Args:
        K:           number of queries (S2GO-Small: 900).
        embed_dims:  per-query feature dim (paper §B: 768).
        init_range:  AABB for the initial query-xyz draw, in LIDAR_TOP frame.
                     Default matches the Occ3D voxel grid extent
                     [-50, 50] × [-50, 50] × [-5, 3] m, so queries are
                     uniformly seeded across the scene volume of interest.
        feat_std:    Gaussian σ for the per-query feature init.
    """

    def __init__(self,
                 K: int = 900,
                 embed_dims: int = 768,
                 init_range=(-50.0, -50.0, -5.0, 50.0, 50.0, 3.0),
                 feat_std: float = 0.02):
        super().__init__()
        self.K = int(K)
        self.embed_dims = int(embed_dims)
        x0, y0, z0, x1, y1, z1 = init_range

        # Learnable query positions in the LIDAR_TOP frame. Uniform-AABB init.
        with torch.no_grad():
            q = torch.empty(K, 3)
            q[:, 0].uniform_(x0, x1)
            q[:, 1].uniform_(y0, y1)
            q[:, 2].uniform_(z0, z1)
        self.query_xyz = nn.Parameter(q)

        # Learnable per-query feature — same name + shape as `S2GOLifter.query_feat`
        # so a Stage 1 ckpt's `lifter.query_feat` weight will load cleanly when
        # someone keeps both modules' parameter naming convention.
        self.query_feat = nn.Parameter(torch.empty(K, embed_dims))
        nn.init.normal_(self.query_feat, mean=0.0, std=feat_std)

    def forward(self, pts: torch.Tensor, add_noise: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pts: (B, M, 3) — ACCEPTED AND IGNORED. Kept for signature parity
                 with `S2GOLifter` so `S2GOSegmentor.forward_one_frame` doesn't
                 need modification.
            add_noise: unused. Kept for signature parity.

        Returns:
            anchors_xyz: (B, K, 3)  — same as init_xyz (no denoise loss in Stage 2)
            init_xyz:    (B, K, 3)  — learnable, broadcast across batch
            init_feat:   (B, K, d)  — learnable, broadcast across batch
        """
        del add_noise  # unused
        B = pts.shape[0]
        init_xyz  = self.query_xyz.unsqueeze(0).expand(B, -1, -1).contiguous()
        init_feat = self.query_feat.unsqueeze(0).expand(B, -1, -1).contiguous()
        anchors_xyz = init_xyz  # denoise target unused in Stage 2
        return anchors_xyz, init_xyz, init_feat


# ────────────────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────────────────
def _self_test():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, K, d = 2, 900, 768
    lifter = Stage2Lifter(K=K, embed_dims=d).to(device)

    # Dummy `pts` — the lifter must IGNORE it (the values shouldn't show up
    # anywhere in the output).
    pts = torch.randn(B, 10_000, 3, device=device) * 1e6   # absurd values
    anchors, init_xyz, init_feat = lifter(pts)
    assert anchors.shape == (B, K, 3)
    assert init_xyz.shape == (B, K, 3)
    assert init_feat.shape == (B, K, d)
    assert torch.equal(anchors, init_xyz), \
        "anchors_xyz should equal init_xyz in Stage 2 (no denoise loss)"
    assert torch.equal(init_xyz[0], init_xyz[1]), \
        "init_xyz must be shared across batch (broadcast)"
    assert torch.equal(init_feat[0], init_feat[1]), \
        "init_feat must be shared across batch (broadcast)"
    print(f"Stage2Lifter fwd OK: anchors {tuple(anchors.shape)}, "
          f"init_xyz {tuple(init_xyz.shape)}, init_feat {tuple(init_feat.shape)}")

    # Verify the lifter REALLY ignores `pts` — output range should match
    # `init_range` defaults, not the absurd pts values.
    xyz_min, xyz_max = init_xyz.min().item(), init_xyz.max().item()
    assert -55.0 < xyz_min and xyz_max < 55.0, \
        f"output picked up pts? range = [{xyz_min}, {xyz_max}]"
    print(f"  pts ignored OK: init_xyz range = [{xyz_min:.2f}, {xyz_max:.2f}] "
          f"(matches default init_range)")

    # Grad flow on both parameters
    loss = init_xyz.sum() + init_feat.sum()
    loss.backward()
    for name, p in lifter.named_parameters():
        gn = p.grad.norm().item()
        assert gn > 0, f"no grad on {name}"
        print(f"  ∇{name}: norm={gn:.3e}, shape={tuple(p.shape)}")
    print("Stage2Lifter self-test PASSED.")


if __name__ == "__main__":
    _self_test()

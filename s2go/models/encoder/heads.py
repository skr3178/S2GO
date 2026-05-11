"""ParentRefiner + ChildGaussianHead — Eq. 6 hierarchical decode (S1.2).

Per S2GO Eq. 6, each parent query i predicts:
   - position offset      o^i      (3)
   - opacity              a^i      (1)
   - velocity             v^i      (3)        — broadcast to children
and J = 10 child Gaussians per parent, each with:
   - position offset      o^{i,j}  (3, relative to parent)
   - scale                s^{i,j}  (3)
   - rotation (quaternion) r^{i,j} (4)
   - opacity              a^{i,j}  (1)
   - color (Stage 1 only) c^{i,j}  (3)

Each Gaussian's final attributes:
   means     = p^i + o^i + o^{i,j}
   opacity   = a^i · a^{i,j}
   scale     = s^{i,j}
   rotation  = r^{i,j}
   colors    = c^{i,j}             (Stage 1)   or shared parent class (Stage 2)
   velocity  = v^i                  (broadcast over J children)

This file follows Stage1_design.md §5 verbatim. The MLP trunk (`linear_relu_ln`)
mirrors GF-2's V2 refiner pattern (utils.py:49) so a future weight-port from
the V2 refiner is mechanical.

Run self-test:
    conda activate /media/skr/storage/conda_envs/selfocc
    python s2go/models/encoder/heads.py
"""
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_relu_ln(embed_dims: int, in_loops: int, out_loops: int):
    """GF-2 V2 trunk pattern (verbatim from utils.py:49) — `out_loops` outer
    iterations, each containing `in_loops` (Linear+ReLU) blocks, capped by LN."""
    layers = []
    in_dim = embed_dims
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(in_dim, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            in_dim = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class ParentPred(NamedTuple):
    """Output of ParentRefiner."""
    offset:   torch.Tensor   # (B, K, 3)
    opa:      torch.Tensor   # (B, K, 1)  in [0, 1]
    velocity: torch.Tensor   # (B, K, 3)  m/s, no activation
    feat:     torch.Tensor   # (B, K, d)  trunk hidden — fed to ChildGaussianHead


class ChildPred(NamedTuple):
    """Output of ChildGaussianHead (Stage 1 with RGB)."""
    offset: torch.Tensor   # (B, K, J, 3) relative to parent
    scale:  torch.Tensor   # (B, K, J, 3) in [scale_range[0], scale_range[1]] meters
    rot:    torch.Tensor   # (B, K, J, 4) unit-norm quaternion
    opa:    torch.Tensor   # (B, K, J, 1) in [0, 1]
    rgb:    torch.Tensor   # (B, K, J, 3) in [0, 1]   — None for Stage 2


class ParentRefiner(nn.Module):
    """Per-parent: offset (3) + opacity (1) + velocity (3). 7-dim head.

    Args:
        embed_dims: query feature dim (paper §B: 768)
        unit_xyz: parent offset bound, in meters (default [4, 4, 1] — matches GF-2 V2 config)
    """
    def __init__(self, embed_dims: int = 768, unit_xyz=(4.0, 4.0, 1.0)):
        super().__init__()
        self.embed_dims = embed_dims
        self.register_buffer("unit_xyz", torch.tensor(unit_xyz, dtype=torch.float32))
        self.trunk = nn.Sequential(*linear_relu_ln(embed_dims, in_loops=2, out_loops=2))
        self.head = nn.Linear(embed_dims, 3 + 1 + 3)   # offset, opa, velo

    def forward(self, instance_feature: torch.Tensor, anchor_embed: torch.Tensor) -> ParentPred:
        """
        Args:
            instance_feature: (B, K, d) per-query feature (post-decoder)
            anchor_embed:     (B, K, d) positional embedding of init_xyz

        Returns:
            ParentPred(offset, opa, velocity, feat)
        """
        h = self.trunk(instance_feature + anchor_embed)              # (B, K, d)
        out = self.head(h)                                            # (B, K, 7)
        # offset bounded by unit_xyz: 2·sigmoid(x) − 1 ∈ [−1, 1] then scaled.
        offset = (2.0 * torch.sigmoid(out[..., :3]) - 1.0) * self.unit_xyz
        opa = torch.sigmoid(out[..., 3:4])
        velocity = out[..., 4:7]                                      # m/s, no activation
        return ParentPred(offset=offset, opa=opa, velocity=velocity, feat=h)


class ChildGaussianHead(nn.Module):
    """For each parent feature (B, K, d), decode J children: 14 dims × J.

    Output per child:
       offset (3) + scale (3) + rot (4) + opa (1) + rgb (3) = 14 dims  (Stage 1)
       offset (3) + scale (3) + rot (4) + opa (1)            = 11 dims  (Stage 2)

    Args:
        embed_dims:       parent feature dim
        J:                children per parent (S2GO-Small: 10)
        mode:             'rgb' (Stage 1) or 'semantic' (Stage 2)
        num_classes:      only used when mode='semantic' (paper: 17 fg + 1 empty = 18)
        child_unit_xyz:   child offset bound, m (default [1, 1, 0.5] — children stay near parent)
        scale_range:      (min, max) Gaussian scale, m (default GF-2 V2: [0.01, 2.5])
    """
    def __init__(self, embed_dims: int = 768, J: int = 10,
                 mode: str = 'rgb',
                 num_classes: int = 18,
                 child_unit_xyz=(1.0, 1.0, 0.5),
                 scale_range=(0.01, 2.5)):
        super().__init__()
        assert mode in ('rgb', 'semantic')
        self.embed_dims = embed_dims
        self.J = J
        self.mode = mode
        self.num_classes = num_classes
        self.register_buffer("child_unit_xyz",
                              torch.tensor(child_unit_xyz, dtype=torch.float32))
        self.scale_min, self.scale_max = float(scale_range[0]), float(scale_range[1])

        # Expand parent feature into J child slots (one nn.Linear per parent
        # producing embed_dims*J outputs, matching Stage1_design.md §5).
        self.expand = nn.Linear(embed_dims, embed_dims * J)
        # Per-child final projection. RGB is Stage-1-only; Stage-2 emits a parent-
        # shared semantic class instead (handled separately at the parent level).
        # The child head only differs in its last 3 dims.
        if mode == 'rgb':
            self.head = nn.Linear(embed_dims, 3 + 3 + 4 + 1 + 3)        # 14
        else:  # 'semantic'
            self.head = nn.Linear(embed_dims, 3 + 3 + 4 + 1)             # 11
        # Stage-2 semantic head is per-PARENT (not per-child), so we attach a
        # separate parent-level classifier outside this module — see Stage1_design §5.

    def forward(self, parent_feat: torch.Tensor) -> ChildPred:
        """
        Args:
            parent_feat: (B, K, d) — output of ParentRefiner.feat

        Returns:
            ChildPred(offset, scale, rot, opa, rgb=None-if-semantic)
        """
        B, K, d = parent_feat.shape
        # (B, K, d) -> (B, K, J*d) -> (B, K, J, d)
        x = self.expand(parent_feat).view(B, K, self.J, d)
        out = self.head(x)                                              # (B, K, J, 14|11)

        offset = (2.0 * torch.sigmoid(out[..., :3]) - 1.0) * self.child_unit_xyz
        scale = self.scale_min + (self.scale_max - self.scale_min) * torch.sigmoid(out[..., 3:6])
        rot = F.normalize(out[..., 6:10], dim=-1)
        opa = torch.sigmoid(out[..., 10:11])
        if self.mode == 'rgb':
            rgb = torch.sigmoid(out[..., 11:14])
        else:
            rgb = None                       # Stage 2 — per-parent class is set elsewhere
        return ChildPred(offset=offset, scale=scale, rot=rot, opa=opa, rgb=rgb)


# ────────────────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────────────────
def _self_test():
    torch.manual_seed(0)
    device = "cuda"
    B, K, J, d = 2, 900, 10, 768

    # Synthetic inputs mimicking what comes out of the temporal decoder.
    instance_feature = torch.randn(B, K, d, device=device)
    anchor_embed = torch.randn(B, K, d, device=device)

    parent_refiner = ParentRefiner(embed_dims=d).to(device)
    child_head = ChildGaussianHead(embed_dims=d, J=J, mode='rgb').to(device)

    # ── Parent forward ────────────────────────────────────────────────────
    parent = parent_refiner(instance_feature, anchor_embed)
    assert parent.offset.shape == (B, K, 3),      f"parent.offset {parent.offset.shape}"
    assert parent.opa.shape == (B, K, 1),         f"parent.opa {parent.opa.shape}"
    assert parent.velocity.shape == (B, K, 3),    f"parent.velocity {parent.velocity.shape}"
    assert parent.feat.shape == (B, K, d),        f"parent.feat {parent.feat.shape}"
    # Offset bound: per axis, in [−unit_xyz, +unit_xyz]
    unit_xyz = torch.tensor([4.0, 4.0, 1.0], device=device)
    assert (parent.offset.abs() <= unit_xyz + 1e-5).all(), \
        f"parent.offset exceeds unit_xyz: max per-axis = {parent.offset.abs().amax(dim=(0,1))}"
    assert (parent.opa >= 0).all() and (parent.opa <= 1).all(), "parent.opa out of [0,1]"
    print(f"  parent shapes OK: offset {tuple(parent.offset.shape)}, opa {tuple(parent.opa.shape)}, "
          f"velocity {tuple(parent.velocity.shape)}, feat {tuple(parent.feat.shape)}")
    print(f"  parent ranges OK: |offset|≤unit_xyz, opa∈[{parent.opa.min().item():.3f}, "
          f"{parent.opa.max().item():.3f}], velocity∈[{parent.velocity.min().item():.2f}, "
          f"{parent.velocity.max().item():.2f}] m/s")

    # ── Child forward ─────────────────────────────────────────────────────
    children = child_head(parent.feat)
    assert children.offset.shape == (B, K, J, 3), f"child.offset {children.offset.shape}"
    assert children.scale.shape == (B, K, J, 3),  f"child.scale  {children.scale.shape}"
    assert children.rot.shape == (B, K, J, 4),    f"child.rot    {children.rot.shape}"
    assert children.opa.shape == (B, K, J, 1),    f"child.opa    {children.opa.shape}"
    assert children.rgb.shape == (B, K, J, 3),    f"child.rgb    {children.rgb.shape}"

    child_unit = torch.tensor([1.0, 1.0, 0.5], device=device)
    assert (children.offset.abs() <= child_unit + 1e-5).all(), \
        f"child.offset exceeds child_unit_xyz"
    assert (children.scale >= 0.01 - 1e-5).all() and (children.scale <= 2.5 + 1e-5).all(), \
        f"child.scale outside [0.01, 2.5]"
    rot_norms = children.rot.norm(dim=-1)
    assert (rot_norms - 1.0).abs().max().item() < 1e-5, \
        f"child.rot not unit norm: max |‖q‖−1| = {(rot_norms - 1.0).abs().max().item():.4e}"
    assert (children.opa >= 0).all() and (children.opa <= 1).all(), "child.opa out of [0,1]"
    assert (children.rgb >= 0).all() and (children.rgb <= 1).all(), "child.rgb out of [0,1]"
    print(f"  child shapes  OK: offset {tuple(children.offset.shape)}, "
          f"scale {tuple(children.scale.shape)}, rot {tuple(children.rot.shape)}, "
          f"opa {tuple(children.opa.shape)}, rgb {tuple(children.rgb.shape)}")
    print(f"  child ranges  OK: |offset|≤[1,1,0.5], scale∈[0.01,2.5], ‖rot‖=1, "
          f"opa∈[0,1], rgb∈[0,1]")

    # ── Gradient flow through both ────────────────────────────────────────
    loss = (parent.offset.sum() + parent.opa.sum() + parent.velocity.pow(2).sum()
            + children.offset.sum() + children.scale.sum() + children.opa.sum()
            + children.rgb.sum())
    loss.backward()
    grads_ok = True
    for name, p in parent_refiner.named_parameters():
        if p.grad is None or p.grad.norm().item() == 0:
            print(f"  FAIL parent_refiner.{name}: grad missing or zero")
            grads_ok = False
    for name, p in child_head.named_parameters():
        if p.grad is None or p.grad.norm().item() == 0:
            print(f"  FAIL child_head.{name}: grad missing or zero")
            grads_ok = False
    assert grads_ok, "gradient flow broken"
    print(f"  gradient flow OK (all params in both modules received non-zero grad)")

    # ── Stage-2 mode (no RGB) sanity ──────────────────────────────────────
    child_head_s2 = ChildGaussianHead(embed_dims=d, J=J, mode='semantic').to(device)
    s2 = child_head_s2(parent.feat.detach())
    assert s2.rgb is None, "Stage-2 mode should return rgb=None"
    assert s2.offset.shape == (B, K, J, 3) and s2.scale.shape == (B, K, J, 3)
    print(f"  Stage-2 mode (no RGB) OK: rgb is None, other shapes match")

    # ── Param counts (sanity vs Stage1_design.md §5 estimates) ────────────
    n_parent = sum(p.numel() for p in parent_refiner.parameters())
    n_child = sum(p.numel() for p in child_head.parameters())
    print(f"  param counts: parent_refiner = {n_parent/1e6:.2f} M,  "
          f"child_head = {n_child/1e6:.2f} M  "
          f"(Stage1_design §5 estimate: child ≈ 5.9 M at d=768, J=10)")

    print("\nS1.2 heads self-test PASSED.")


if __name__ == "__main__":
    _self_test()

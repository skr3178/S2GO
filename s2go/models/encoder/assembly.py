"""assemble_gaussians — combine ParentPred + ChildPred into flat per-Gaussian
tensors, per S2GO Eq. 6 and Stage1_design.md §3 shape contract.

For each query i (i = 1..K) and child j (j = 1..J):
    means_{i,j}     = p^i + o^i + o^{i,j}
    opacity_{i,j}   = a^i · a^{i,j}
    scale_{i,j}     = s^{i,j}
    rotation_{i,j}  = r^{i,j}                  (already unit-norm)
    color_{i,j}     = c^{i,j}                  (Stage 1)   or shared parent class (Stage 2)
    velocity_{i,j}  = v^i                       (broadcast over J children)

Output is flattened to (B, K·J, ...) so it feeds gsplat directly.
"""
from typing import Optional, NamedTuple
import torch

from .heads import ParentPred, ChildPred


class Gaussians(NamedTuple):
    """Flat per-Gaussian tensors (Eq. 6 assembled).

    All tensors are (B, K*J, ...). 'colors' is None if with_rgb=False (Stage 2).
    """
    means:     torch.Tensor   # (B, K*J, 3)
    scales:    torch.Tensor   # (B, K*J, 3)
    rotations: torch.Tensor   # (B, K*J, 4)
    opacities: torch.Tensor   # (B, K*J, 1)
    colors:    Optional[torch.Tensor]   # (B, K*J, 3)  Stage 1 only
    velocity:  torch.Tensor   # (B, K*J, 3)


def assemble_gaussians(init_xyz: torch.Tensor,
                        parent: ParentPred,
                        children: ChildPred,
                        with_rgb: bool = True) -> Gaussians:
    """Eq. 6 assembly: K parents × J children → K·J flat Gaussians.

    Args:
        init_xyz:  (B, K, 3)   — query position from S2GOLifter
        parent:    ParentPred  — output of ParentRefiner (offset, opa, velocity, feat)
        children:  ChildPred   — output of ChildGaussianHead
        with_rgb:  whether to include RGB (Stage 1 = True, Stage 2 = False)

    Returns:
        Gaussians namedtuple. All tensors are (B, K*J, ...) contiguous.
    """
    B, K, _ = init_xyz.shape
    J = children.offset.shape[2]

    # means = init_xyz + parent.offset + child.offset
    # broadcast init_xyz and parent.offset over J: (B,K,1,3)
    means = (init_xyz.unsqueeze(2) +
              parent.offset.unsqueeze(2) +
              children.offset)                   # (B, K, J, 3)
    means = means.reshape(B, K * J, 3).contiguous()

    scales    = children.scale.reshape(B, K * J, 3).contiguous()
    rotations = children.rot.reshape(B, K * J, 4).contiguous()

    # opacity = parent.opa * child.opa, broadcast parent over J
    opacities = (parent.opa.unsqueeze(2) * children.opa).reshape(B, K * J, 1).contiguous()

    if with_rgb:
        assert children.rgb is not None, "with_rgb=True requires children.rgb"
        colors = children.rgb.reshape(B, K * J, 3).contiguous()
    else:
        colors = None

    # velocity broadcast over J children
    velocity = (parent.velocity.unsqueeze(2)
                .expand(-1, -1, J, -1)
                .reshape(B, K * J, 3)
                .contiguous())

    return Gaussians(means=means, scales=scales, rotations=rotations,
                      opacities=opacities, colors=colors, velocity=velocity)

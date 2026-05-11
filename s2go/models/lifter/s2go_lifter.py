"""S2GOLifter — Stage-1 query initialization (FPS+ε on LiDAR points, S2GO Eq. 7).

For each batch element:
  1. Sample K furthest points from M LiDAR points (FPS).
  2. Add per-axis uniform noise ε ~ U(-σ, σ).
  3. Combine with a learnable shared feature embedding.

The noise-free FPS anchors are returned separately because they are the target
of the L_denoise loss (Stage1_pseudocode.md Algorithm 1 line 21).

Stage-2 uses a different lifter mode ('learnable'); only the 'fps_eps' mode is
implemented here. Dual-mode unification will happen when Stage 2 starts.

Run self-test:
    conda activate /media/skr/storage/conda_envs/selfocc
    python s2go/models/lifter/s2go_lifter.py
"""
import torch
import torch.nn as nn
from torch_cluster import fps


class S2GOLifter(nn.Module):
    """FPS+ε lifter (Stage 1).

    Args:
        K: number of queries (S2GO-Small: 900)
        embed_dims: query feature dimension (paper §B: 768)
        eps: ε scale in meters (paper §B: 1.0 for nuScenes-SurroundOcc;
             scale by smaller scene extent for Occ3D / KITTI)

    forward(pts) returns:
        anchors_xyz: (B, K, 3)  noise-free FPS anchors  (target for L_denoise)
        init_xyz:    (B, K, 3)  anchors_xyz + ε         (query positions)
        init_feat:   (B, K, d)  shared learnable embedding broadcast over batch
    """

    def __init__(self, K=900, embed_dims=768, eps=1.0):
        super().__init__()
        self.K = K
        self.embed_dims = embed_dims
        self.eps = eps
        # Shared learnable query feature, broadcast across the K queries.
        # GF-2 LifterV2 uses zero-init then xavier; we xavier directly.
        self.query_feat = nn.Parameter(torch.empty(K, embed_dims))
        nn.init.xavier_uniform_(self.query_feat)

    def forward(self, pts: torch.Tensor, add_noise: bool = True):
        """
        Args:
            pts:        (B, M, 3) LiDAR points
            add_noise:  if False, init_xyz == anchors_xyz (for unit tests)

        Returns:
            anchors_xyz, init_xyz, init_feat
        """
        B, M, _ = pts.shape
        device = pts.device
        if M < self.K:
            raise ValueError(f"need M >= K, got M={M}, K={self.K}")

        # Vectorized FPS across the batch: flatten and pass batch indices.
        flat = pts.reshape(B * M, 3)
        batch_idx = torch.arange(B, device=device).repeat_interleave(M)
        # ratio = K/M gives ≈ K samples per batch element (rounding can yield K±1).
        global_idx = fps(flat, batch=batch_idx, ratio=self.K / M)

        # Convert global flat indices → per-batch local indices, pad/truncate to K.
        anchors_list = []
        for b in range(B):
            mask = (global_idx >= b * M) & (global_idx < (b + 1) * M)
            local_idx = global_idx[mask] - b * M
            if local_idx.numel() > self.K:
                local_idx = local_idx[: self.K]
            elif local_idx.numel() < self.K:
                # Pad by repeating the last index — fps rounding rarely gives < K.
                pad = local_idx[-1:].expand(self.K - local_idx.numel())
                local_idx = torch.cat([local_idx, pad])
            anchors_list.append(pts[b, local_idx])
        anchors_xyz = torch.stack(anchors_list, dim=0)            # (B, K, 3)

        if add_noise:
            noise = torch.empty_like(anchors_xyz).uniform_(-self.eps, self.eps)
            init_xyz = anchors_xyz + noise
        else:
            init_xyz = anchors_xyz.clone()

        init_feat = self.query_feat.unsqueeze(0).expand(B, -1, -1).contiguous()
        return anchors_xyz, init_xyz, init_feat


# ────────────────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────────────────
def _self_test():
    torch.manual_seed(0)
    device = "cuda"
    B, M, K, d = 2, 10_000, 900, 768
    eps = 1.0

    # Synthetic point cloud: random points in a 100×100×8 m volume (nuScenes-ish extent).
    pts = torch.empty((B, M, 3), device=device)
    pts[..., 0].uniform_(-50.0, 50.0)
    pts[..., 1].uniform_(-50.0, 50.0)
    pts[..., 2].uniform_(-3.0, 5.0)

    lifter = S2GOLifter(K=K, embed_dims=d, eps=eps).to(device)

    # ── Forward with noise (training path) ──────────────────────────────────
    anchors, init_xyz, init_feat = lifter(pts, add_noise=True)
    assert anchors.shape == (B, K, 3), f"anchors shape {anchors.shape}"
    assert init_xyz.shape == (B, K, 3), f"init_xyz shape {init_xyz.shape}"
    assert init_feat.shape == (B, K, d), f"init_feat shape {init_feat.shape}"

    # ε bound: ‖init_xyz - anchors‖_∞ ≤ ε per axis
    delta = (init_xyz - anchors).abs()
    assert delta.max().item() <= eps + 1e-6, \
        f"ε violated: max |Δ| = {delta.max().item():.4f} > {eps}"
    # And not trivially zero (noise actually added)
    assert delta.max().item() > 0.1, \
        f"noise too small? max |Δ| = {delta.max().item():.4f}"
    print(f"  shapes  OK: anchors {tuple(anchors.shape)}, init_xyz {tuple(init_xyz.shape)}, "
          f"init_feat {tuple(init_feat.shape)}")
    print(f"  ε bound OK: max |init_xyz − anchors| = {delta.max().item():.4f} m  (eps={eps})")

    # ── Forward without noise (eval / unit test path) ───────────────────────
    a2, p2, _ = lifter(pts, add_noise=False)
    assert torch.allclose(a2, p2), "add_noise=False should give init_xyz == anchors"
    print(f"  no-noise OK: init_xyz == anchors_xyz")

    # ── FPS sanity: anchors should be more spread out than random sampling ─
    # Mean nearest-neighbour distance among anchors vs random subset of points.
    def mean_nn_dist(x):                                              # x: (K, 3)
        d = torch.cdist(x, x)
        d.fill_diagonal_(float("inf"))
        return d.min(dim=1).values.mean().item()
    fps_nn = mean_nn_dist(anchors[0])
    rand_nn = mean_nn_dist(pts[0, torch.randperm(M)[:K]])
    assert fps_nn > 1.5 * rand_nn, \
        f"FPS not better-spread than random: fps_nn={fps_nn:.3f} vs rand_nn={rand_nn:.3f}"
    print(f"  FPS spread OK: anchor mean-NN-dist = {fps_nn:.3f} m  "
          f"(random = {rand_nn:.3f} m, ratio {fps_nn/rand_nn:.2f}×)")

    # ── init_feat is shared across batch (broadcast) ────────────────────────
    assert torch.equal(init_feat[0], init_feat[1]), "init_feat should be shared across batch"
    print(f"  init_feat shared across batch OK")

    # ── Gradient flow: init_feat.grad should be populated after a backward ─
    loss = init_feat.sum() + (init_xyz - anchors).pow(2).mean()
    loss.backward()
    assert lifter.query_feat.grad is not None, "no grad on query_feat"
    assert lifter.query_feat.grad.norm().item() > 0, "zero grad on query_feat"
    print(f"  query_feat grad OK: ‖∂L/∂query_feat‖ = {lifter.query_feat.grad.norm().item():.4e}")

    print("\nS1.1 lifter self-test PASSED.")


if __name__ == "__main__":
    _self_test()

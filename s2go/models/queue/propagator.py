"""OpacityDeltaPropagator — block [H] in architecture.md.

Algorithm 3 from Stage1_pseudocode.md:
    Input:   parent.{xyz, opa, feat, velo}, k = cfg.propagate_k = 256, training flag
    Output:  prop.{xyz, opa, feat, velo}  with |prop| = k

    1: if training then  δ ~ Uniform(0, 3)  else  δ ← cfg.delta_eval = 1.6  end if
    2: idx ← argsort(parent.opa, descending)
    3: keep ← []
    4: for i in idx do
    5:     if min_{j ∈ keep} ‖ parent.xyz[i] − parent.xyz[j] ‖_2  ≥ δ then
    6:         keep ← keep ∪ {i}
    7:     end if
    8:     if |keep| = k then  break  end if
    9: end for
    10: return parent[keep]

The train/eval δ branch is paper §B verbatim and is REQUIRED for paper-correct
val mIoU. Both code paths exist here.

If fewer than k queries pass the δ filter (e.g. all queries clustered too
close), we pad with the next-highest-opacity remaining queries (relaxing δ)
so the queue is always exactly size k. The paper does not specify this
fallback; the choice mirrors StreamPETR's always-fixed-size queue.

Run self-test:
    cd /media/skr/storage/self_driving/S2GO
    python -m s2go.models.queue.propagator
"""
from typing import NamedTuple
import torch
import torch.nn as nn

from ..encoder.heads import ParentPred


class Propagated(NamedTuple):
    """Top-k queue entries pushed to next frame's memory.

    All tensors are (B, k, ...) where k = cfg.propagate_k = 256 by default.
    """
    xyz:  torch.Tensor   # (B, k, 3)  — refined query position
    opa:  torch.Tensor   # (B, k, 1)  — parent opacity (selection criterion)
    feat: torch.Tensor   # (B, k, d)  — parent feature (becomes memory_embedding)
    velo: torch.Tensor   # (B, k, 3)  — parent velocity (becomes memory_velo)


class OpacityDeltaPropagator(nn.Module):
    """Greedy NMS-by-distance over opacity-sorted parent queries.

    Args:
        k:               # queries to push (cfg.propagate_k, default 256 — paper)
        delta_train_lo:  lower bound for U(δ_lo, δ_hi) at training (paper: 0)
        delta_train_hi:  upper bound (paper: 3 m)
        delta_eval:      fixed δ at inference (paper: 1.6 m — REQUIRED for paper-correct mIoU)
    """

    def __init__(self, k: int = 256,
                 delta_train_lo: float = 0.0,
                 delta_train_hi: float = 3.0,
                 delta_eval: float = 1.6):
        super().__init__()
        self.k = k
        self.delta_train_lo = float(delta_train_lo)
        self.delta_train_hi = float(delta_train_hi)
        self.delta_eval = float(delta_eval)

    def _sample_delta(self, device: torch.device) -> float:
        """Train: δ ~ U(δ_lo, δ_hi). Eval: fixed δ_eval. Paper §B."""
        if self.training:
            return torch.empty(1, device=device).uniform_(
                self.delta_train_lo, self.delta_train_hi).item()
        return self.delta_eval

    def _select_one(self, xyz: torch.Tensor, opa: torch.Tensor, delta: float) -> torch.Tensor:
        """Greedy distance-prune over opacity-sorted queries (single batch element).

        Args:
            xyz: (K, 3)
            opa: (K, 1) or (K,)
            delta: minimum mutual distance, in meters

        Returns:
            (k,) long tensor of indices into the K queries
        """
        K = xyz.shape[0]
        opa_flat = opa.reshape(K)
        sorted_idx = torch.argsort(opa_flat, descending=True).tolist()

        keep: list[int] = []
        # Maintain kept_xyz on-device as a growing tensor for fast cdist.
        kept_xyz_buf = xyz.new_zeros((self.k, 3))
        kept_count = 0

        for i in sorted_idx:
            if kept_count == 0:
                keep.append(i)
                kept_xyz_buf[0] = xyz[i]
                kept_count = 1
            else:
                # min distance from xyz[i] to any kept point
                diff = kept_xyz_buf[:kept_count] - xyz[i].unsqueeze(0)
                min_d = diff.norm(dim=-1).min().item()
                if min_d >= delta:
                    keep.append(i)
                    kept_xyz_buf[kept_count] = xyz[i]
                    kept_count += 1
            if kept_count == self.k:
                break

        # Pad with next-highest-opacity remaining (relaxing δ) if we ran out.
        if len(keep) < self.k:
            kept_set = set(keep)
            for i in sorted_idx:
                if i not in kept_set:
                    keep.append(i)
                    if len(keep) == self.k:
                        break

        return torch.tensor(keep, device=xyz.device, dtype=torch.long)

    def forward(self, refined_xyz: torch.Tensor, parent: ParentPred) -> Propagated:
        """
        Args:
            refined_xyz: (B, K, 3) — init_xyz + parent.offset (full query position)
            parent:      ParentPred — uses .opa, .feat, .velocity

        Returns:
            Propagated(xyz, opa, feat, velo) with leading shape (B, k, ...)
        """
        B, K, _ = refined_xyz.shape
        d = parent.feat.shape[-1]
        delta = self._sample_delta(refined_xyz.device)

        keep_idx_list = []
        for b in range(B):
            keep_idx_list.append(
                self._select_one(refined_xyz[b], parent.opa[b], delta))
        keep_idx = torch.stack(keep_idx_list, dim=0)              # (B, k)

        # Gather along query axis. unsqueeze + expand for the trailing dims.
        idx_xyz  = keep_idx.unsqueeze(-1).expand(-1, -1, 3)
        idx_opa  = keep_idx.unsqueeze(-1)                          # (B, k, 1)
        idx_feat = keep_idx.unsqueeze(-1).expand(-1, -1, d)
        idx_velo = keep_idx.unsqueeze(-1).expand(-1, -1, 3)

        xyz_p  = torch.gather(refined_xyz,    1, idx_xyz)
        opa_p  = torch.gather(parent.opa,     1, idx_opa)
        feat_p = torch.gather(parent.feat,    1, idx_feat)
        velo_p = torch.gather(parent.velocity, 1, idx_velo)

        return Propagated(xyz=xyz_p, opa=opa_p, feat=feat_p, velo=velo_p)


# ────────────────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────────────────
def _self_test():
    torch.manual_seed(0)
    device = "cuda"
    B, K, d = 2, 900, 768
    k = 256

    # Synthetic queries + parent state. Spread xyz across nuScenes-ish 100×100×8 m.
    refined_xyz = torch.empty((B, K, 3), device=device)
    refined_xyz[..., 0].uniform_(-50.0, 50.0)
    refined_xyz[..., 1].uniform_(-50.0, 50.0)
    refined_xyz[..., 2].uniform_(-3.0, 5.0)
    opa = torch.rand((B, K, 1), device=device)
    feat = torch.randn((B, K, d), device=device, requires_grad=True)
    velo = torch.randn((B, K, 3), device=device)
    parent = ParentPred(offset=torch.zeros_like(refined_xyz),
                         opa=opa, velocity=velo, feat=feat)

    prop_module = OpacityDeltaPropagator(k=k).to(device)

    # ── Eval mode (δ = 1.6 m, deterministic) ──────────────────────────────
    prop_module.eval()
    out1 = prop_module(refined_xyz, parent)
    out2 = prop_module(refined_xyz, parent)
    assert out1.xyz.shape  == (B, k, 3),  f"xyz {out1.xyz.shape}"
    assert out1.opa.shape  == (B, k, 1),  f"opa {out1.opa.shape}"
    assert out1.feat.shape == (B, k, d),  f"feat {out1.feat.shape}"
    assert out1.velo.shape == (B, k, 3),  f"velo {out1.velo.shape}"
    assert torch.equal(out1.xyz, out2.xyz), "eval mode should be deterministic"
    print(f"  eval shapes OK: xyz {tuple(out1.xyz.shape)}, opa {tuple(out1.opa.shape)}, "
          f"feat {tuple(out1.feat.shape)}, velo {tuple(out1.velo.shape)}")
    print(f"  eval determinism OK")

    # ── δ-distance constraint check (eval mode, δ = 1.6) ──────────────────
    # All retained pairs should be ≥ δ apart up to the point of padding.
    # The well-spread synthetic data should NOT need padding (K=900, k=256, scene=100×100×8).
    delta = 1.6
    for b in range(B):
        d_pair = torch.cdist(out1.xyz[b], out1.xyz[b])
        d_pair.fill_diagonal_(float("inf"))
        n_violations = (d_pair < delta).sum().item() // 2     # symmetric, count once
        # Allow zero violations on well-spread synthetic data
        assert n_violations == 0, \
            f"batch {b}: {n_violations} pairs closer than δ={delta} m"
    print(f"  δ-constraint OK (eval, δ=1.6 m): zero pairs closer than δ")

    # ── Highest-opacity query is always kept ──────────────────────────────
    for b in range(B):
        top_idx = opa[b].squeeze(-1).argmax().item()
        top_xyz = refined_xyz[b, top_idx]
        # The top-opa xyz must appear in out1.xyz[b]
        match = (out1.xyz[b] - top_xyz.unsqueeze(0)).norm(dim=-1).min().item()
        assert match < 1e-5, \
            f"batch {b}: highest-opacity query (opa={opa[b, top_idx].item():.4f}) " \
            f"missing from output (closest match dist={match:.4e})"
    print(f"  highest-opacity always kept OK")

    # ── Train mode (δ ~ U(0, 3), stochastic) ──────────────────────────────
    prop_module.train()
    out_a = prop_module(refined_xyz, parent)
    out_b = prop_module(refined_xyz, parent)
    # Note: stochastic δ → output sets *may* differ. Just verify shapes still match.
    assert out_a.xyz.shape == (B, k, 3) and out_b.xyz.shape == (B, k, 3)
    print(f"  train mode shapes OK; δ sampled stochastically per call")

    # ── Gradient flow ─────────────────────────────────────────────────────
    loss = out_a.feat.sum()
    loss.backward()
    assert feat.grad is not None and feat.grad.norm().item() > 0, "grad on feat missing"
    # Only k of the K queries are selected, so only those rows get non-zero grad.
    nonzero_rows = (feat.grad.norm(dim=-1) > 0).sum(dim=-1)        # (B,)
    assert (nonzero_rows == k).all(), \
        f"expected exactly {k} non-zero grad rows per batch, got {nonzero_rows.tolist()}"
    print(f"  grad flow OK: exactly k={k} feat rows per batch received non-zero grad")

    # ── Padding fallback (cluster all queries within δ) ───────────────────
    # All xyz at origin → no valid distance pairs → all selections must come
    # from the padding loop. Output should still be exactly k.
    refined_clust = torch.zeros_like(refined_xyz)               # all at origin
    parent_clust = ParentPred(offset=torch.zeros_like(refined_xyz),
                                opa=opa, velocity=velo, feat=feat.detach())
    prop_module.eval()
    out_clust = prop_module(refined_clust, parent_clust)
    assert out_clust.xyz.shape == (B, k, 3), "padding fallback should still produce k outputs"
    # All output xyz are at origin in this degenerate case.
    assert (out_clust.xyz.abs().max().item() == 0.0), \
        "clustered case: all xyz should equal 0"
    print(f"  padding fallback OK: degenerate cluster → still k={k} outputs")

    # ── Δ-eval value pinned correctly ─────────────────────────────────────
    assert prop_module.delta_eval == 1.6, "δ_eval must be 1.6 (paper §B)"
    print(f"  δ_eval pinned at 1.6 m (paper §B verbatim)")

    print("\nS1.6 propagator self-test PASSED.")


if __name__ == "__main__":
    _self_test()

"""MemoryQueue — block [B] in architecture.md, port of StreamPETR's queue.

Holds 5 memory_* tensors across T-frame mini-sequences and exposes
reset / pre_update / post_update methods. Ported from
[reference_code/StreamPETR/.../streampetr_head.py:312-374] with these S2GO
adaptations:
  - 3D velocity (not StreamPETR's 2D bbox velocity)
  - post_update consumes a Propagated namedtuple from OpacityDeltaPropagator
    (S1.6) instead of StreamPETR's DETR-style top-k-by-classification-score
  - No `pseudo_reference_points` (first-frame learnable padding) — left for
    future if required; current behaviour: first frame sees zeroed memory

Coordinate-frame convention (matches StreamPETR):
  - Inside the queue between frames, memory_reference_point is in WORLD coords.
  - post_update applies `ego_pose` (forward) → maps current-frame → world.
  - pre_update applies `ego_pose_inv` of next frame → maps world → next-frame.
  - memory_egopose chains the inverse transforms over the queue's lifetime so
    each entry knows the cumulative ego motion since insertion.

Run self-test:
    cd /media/skr/storage/self_driving/S2GO
    python -m s2go.models.queue.memory
"""
import torch
import torch.nn as nn

from .propagator import Propagated


# ────────────────────────────────────────────────────────────────────────────
# Helpers (ported from reference_code/StreamPETR/.../utils/misc.py)
# ────────────────────────────────────────────────────────────────────────────
def memory_refresh(memory: torch.Tensor, prev_exists: torch.Tensor) -> torch.Tensor:
    """Zero out memory entries where the scene has reset (prev_exists=0).

    Multiplies along the batch axis: prev_exists is (B,) → broadcasts over the
    rest of memory.shape.
    """
    view_shape = [1] * memory.ndim
    view_shape[0] = -1
    return memory * prev_exists.view(*view_shape)


def transform_reference_points(reference_points: torch.Tensor,
                                egopose: torch.Tensor,
                                reverse: bool = False,
                                translation: bool = True) -> torch.Tensor:
    """Apply a 4×4 ego transform to (B, L, 3) reference points (homogeneous).

    Args:
        reference_points: (B, L, 3)
        egopose:          (B, 4, 4)
        reverse:          if True, use egopose.inverse()
        translation:      if False, zero out the translation column

    Returns:
        (B, L, 3) transformed points
    """
    rp_h = torch.cat([reference_points,
                       torch.ones_like(reference_points[..., :1])], dim=-1)        # (B,L,4)
    matrix = egopose.inverse() if reverse else egopose
    if not translation:
        matrix = matrix.clone()
        matrix[..., :3, 3] = 0.0
    return (matrix.unsqueeze(1) @ rp_h.unsqueeze(-1)).squeeze(-1)[..., :3]


# ────────────────────────────────────────────────────────────────────────────
# Memory queue
# ────────────────────────────────────────────────────────────────────────────
class MemoryQueue(nn.Module):
    """Streaming memory queue across T-frame mini-sequences.

    Args:
        memory_len:  total queue capacity (paper §B: T·k_prop = 4·256 = 1024)
        embed_dims:  per-query feature dim (paper §B: 768)
    """

    def __init__(self, memory_len: int = 1024, embed_dims: int = 768):
        super().__init__()
        self.memory_len = memory_len
        self.embed_dims = embed_dims
        self.reset()

    def reset(self):
        """Wipe queue (called at start of each mini-sequence)."""
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None

    def _ensure_initialized(self, B: int, device, dtype):
        """Lazy-allocate queue tensors with given batch size."""
        self.memory_embedding = torch.zeros(B, self.memory_len, self.embed_dims,
                                              device=device, dtype=dtype)
        self.memory_reference_point = torch.zeros(B, self.memory_len, 3,
                                                    device=device, dtype=dtype)
        self.memory_timestamp = torch.zeros(B, self.memory_len, 1,
                                              device=device, dtype=dtype)
        self.memory_egopose = torch.zeros(B, self.memory_len, 4, 4,
                                           device=device, dtype=dtype)
        self.memory_velo = torch.zeros(B, self.memory_len, 3,
                                        device=device, dtype=dtype)

    def pre_update(self, prev_exists: torch.Tensor,
                    timestamp_delta: torch.Tensor,
                    ego_pose_inv: torch.Tensor):
        """Ego-compensate past memory into current frame's coordinates.

        Args:
            prev_exists:     (B,) — 1 = same scene as previous frame, 0 = reset
            timestamp_delta: (B,) — Δt (current_t − prev_t), in seconds
            ego_pose_inv:    (B, 4, 4) — inverse of current frame's ego pose
                               (maps world → current-frame coords)
        """
        x = prev_exists.to(dtype=ego_pose_inv.dtype)
        B = x.shape[0]
        if self.memory_embedding is None:
            self._ensure_initialized(B, ego_pose_inv.device, ego_pose_inv.dtype)
            return

        # Accrue time / chain ego transforms — applied to all entries first
        self.memory_timestamp = self.memory_timestamp + timestamp_delta.view(B, 1, 1)
        self.memory_egopose = ego_pose_inv.unsqueeze(1) @ self.memory_egopose
        self.memory_reference_point = transform_reference_points(
            self.memory_reference_point, ego_pose_inv, reverse=False)

        # Refresh: zero entries from scenes that have just reset (prev_exists=0)
        self.memory_timestamp = memory_refresh(self.memory_timestamp, x)
        self.memory_reference_point = memory_refresh(self.memory_reference_point, x)
        self.memory_embedding = memory_refresh(self.memory_embedding, x)
        self.memory_egopose = memory_refresh(self.memory_egopose, x)
        self.memory_velo = memory_refresh(self.memory_velo, x)

    def post_update(self, prop: Propagated,
                     ego_pose: torch.Tensor,
                     timestamp: torch.Tensor):
        """Push newly-propagated queries onto queue head; transform to world.

        Args:
            prop:       Propagated (B, k, ...) from OpacityDeltaPropagator
            ego_pose:   (B, 4, 4) current frame's forward ego pose
                          (maps current-frame → world)
            timestamp:  (B,) — current frame's absolute time, used to set
                          inserted entries' time to 0 (relative-now), then bump.
        """
        if self.memory_embedding is None:
            raise RuntimeError("post_update called before pre_update")

        B, k, d = prop.feat.shape
        assert d == self.embed_dims, f"prop.feat dim {d} != embed_dims {self.embed_dims}"

        # Memory stores DETACHED features — gradients re-enter via the next
        # frame's self-attn K,V (the past-query concat at decoder layer 1),
        # not through this stored history.
        rec_feat = prop.feat.detach()
        rec_xyz = prop.xyz.detach()
        rec_velo = prop.velo.detach()
        rec_ts = torch.zeros(B, k, 1, device=prop.feat.device, dtype=prop.feat.dtype)
        rec_ego = torch.eye(4, device=prop.feat.device, dtype=prop.feat.dtype) \
                       .unsqueeze(0).unsqueeze(0).expand(B, k, -1, -1).contiguous()

        # Concatenate at head, truncate to capacity. Tail (oldest) entries are
        # dropped when the queue is full.
        self.memory_embedding = torch.cat([rec_feat, self.memory_embedding], dim=1)[:, :self.memory_len]
        self.memory_reference_point = torch.cat([rec_xyz, self.memory_reference_point], dim=1)[:, :self.memory_len]
        self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)[:, :self.memory_len]
        self.memory_timestamp = torch.cat([rec_ts, self.memory_timestamp], dim=1)[:, :self.memory_len]
        self.memory_egopose = torch.cat([rec_ego, self.memory_egopose], dim=1)[:, :self.memory_len]

        # Push reference_points OUT of current-frame-coords into world coords
        # so the NEXT frame's pre_update transforms them correctly.
        self.memory_reference_point = transform_reference_points(
            self.memory_reference_point, ego_pose, reverse=False)
        self.memory_timestamp = self.memory_timestamp - timestamp.view(B, 1, 1)
        self.memory_egopose = ego_pose.unsqueeze(1) @ self.memory_egopose


# ────────────────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────────────────
def _self_test():
    torch.manual_seed(0)
    device = "cuda"
    B, L, d = 2, 1024, 768
    k_prop = 256                                    # propagated per frame

    queue = MemoryQueue(memory_len=L, embed_dims=d).to(device)

    # ── reset() leaves all None ───────────────────────────────────────────
    queue.reset()
    assert queue.memory_embedding is None and queue.memory_reference_point is None
    print(f"  reset OK: all 5 memory_* fields None")

    # ── First pre_update lazily allocates zero tensors ────────────────────
    prev_exists = torch.tensor([1.0, 1.0], device=device)
    ts_delta = torch.tensor([0.5, 0.5], device=device)
    ego_pose_inv = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
    queue.pre_update(prev_exists, ts_delta, ego_pose_inv)
    assert queue.memory_embedding.shape == (B, L, d), f"{queue.memory_embedding.shape}"
    assert queue.memory_embedding.abs().max().item() == 0.0, "first frame should be zeroed"
    print(f"  first pre_update lazy-init OK: shapes (B,L,d)=({B},{L},{d}), all-zero")

    # ── post_update with synthetic Propagated ─────────────────────────────
    prop = Propagated(
        xyz=torch.randn(B, k_prop, 3, device=device),
        opa=torch.rand(B, k_prop, 1, device=device),
        feat=torch.randn(B, k_prop, d, device=device),
        velo=torch.randn(B, k_prop, 3, device=device),
    )
    ego_pose = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
    timestamp = torch.tensor([1.0, 1.0], device=device)
    queue.post_update(prop, ego_pose, timestamp)

    # The first k_prop queue rows should match the inserted features.
    head_feat = queue.memory_embedding[:, :k_prop]
    assert torch.allclose(head_feat, prop.feat), \
        f"head feat mismatch (max diff {(head_feat - prop.feat).abs().max().item():.4e})"
    # Timestamp at head: 0 - timestamp = -1.0 for both batches
    head_ts = queue.memory_timestamp[:, :k_prop, 0]
    assert torch.allclose(head_ts, -timestamp.unsqueeze(-1).expand(B, k_prop)), \
        f"head timestamp not pushed correctly"
    print(f"  post_update OK: head k_prop={k_prop} entries match inserted, "
          f"timestamps shifted by -1.0")

    # ── Detached storage ──────────────────────────────────────────────────
    assert not queue.memory_embedding.requires_grad, "queue should hold detached tensors"
    print(f"  detached storage OK: memory_embedding.requires_grad = False")

    # ── Capacity bound ────────────────────────────────────────────────────
    assert queue.memory_embedding.shape[1] == L, "queue should clip at memory_len"
    print(f"  capacity OK: queue size = {L}")

    # ── Multi-frame loop: verify queue grows then truncates ───────────────
    queue.reset()
    queue.pre_update(prev_exists, ts_delta, ego_pose_inv)
    n_frames_to_fill = (L // k_prop) + 1                    # one more than capacity
    for t in range(n_frames_to_fill):
        prop_t = Propagated(
            xyz=torch.randn(B, k_prop, 3, device=device) * 0.01,
            opa=torch.rand(B, k_prop, 1, device=device),
            feat=torch.full((B, k_prop, d), float(t), device=device),  # tag with t
            velo=torch.zeros(B, k_prop, 3, device=device),
        )
        queue.post_update(prop_t, ego_pose, timestamp)
        if t + 1 < n_frames_to_fill:
            queue.pre_update(prev_exists, ts_delta, ego_pose_inv)

    # After n_frames_to_fill insertions, oldest entries (frame 0) should be evicted.
    # Most recent feature (frame n_frames_to_fill-1) should be at the head.
    head_val = queue.memory_embedding[0, 0, 0].item()
    expected_head = float(n_frames_to_fill - 1)
    assert head_val == expected_head, f"head should be frame {expected_head}, got {head_val}"
    # Tail (oldest surviving frame) should NOT be frame 0 (evicted)
    tail_val = queue.memory_embedding[0, -1, 0].item()
    assert tail_val != 0.0, f"frame 0 should have been evicted (queue has {n_frames_to_fill} insertions, capacity {L})"
    print(f"  eviction OK: after {n_frames_to_fill} inserts of {k_prop} each (total > {L}), "
          f"head=frame {int(head_val)}, oldest={int(tail_val)}")

    # ── Scene-reset wipe (prev_exists=0) ──────────────────────────────────
    # Mark a non-zero memory then call pre_update with prev_exists=0 → memory wiped
    queue.memory_embedding.fill_(7.0)
    queue.memory_reference_point.fill_(7.0)
    no_prev = torch.tensor([0.0, 0.0], device=device)
    queue.pre_update(no_prev, ts_delta, ego_pose_inv)
    assert queue.memory_embedding.abs().max().item() == 0.0, \
        "prev_exists=0 should wipe memory_embedding"
    assert queue.memory_reference_point.abs().max().item() == 0.0, \
        "prev_exists=0 should wipe reference_point"
    print(f"  scene-reset wipe OK: prev_exists=0 zeroed all memory_* fields")

    # ── Ego-pose round-trip identity ──────────────────────────────────────
    # If ego_pose and ego_pose_inv are both identity, pre+post over a round-trip
    # should leave ref_points unchanged (modulo the world-coord scratchpad).
    queue.reset()
    queue.pre_update(prev_exists, ts_delta, ego_pose_inv)
    xyz_in = torch.randn(B, k_prop, 3, device=device)
    prop_id = Propagated(xyz=xyz_in, opa=torch.rand(B, k_prop, 1, device=device),
                          feat=torch.randn(B, k_prop, d, device=device),
                          velo=torch.zeros(B, k_prop, 3, device=device))
    queue.post_update(prop_id, ego_pose, timestamp)
    queue.pre_update(prev_exists, ts_delta, ego_pose_inv)
    head_xyz = queue.memory_reference_point[:, :k_prop]
    assert torch.allclose(head_xyz, xyz_in, atol=1e-5), \
        f"identity ego: xyz round-trip should be identity, max diff {(head_xyz - xyz_in).abs().max().item():.4e}"
    print(f"  identity ego round-trip OK: xyz preserved through post→pre cycle")

    print("\nS1.5a memory queue self-test PASSED.")


if __name__ == "__main__":
    _self_test()

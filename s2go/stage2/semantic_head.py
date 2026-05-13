"""SemanticHead — per-parent class classifier (Stage 2).

Consumes parent features (B, K, D) from S2GOSegmentor's ParentRefiner and
emits class logits (B, K, C). The Stage 2 segmentor wrapper broadcasts
these to children: each of the J=10 children of parent k inherits the
same class distribution. This mirrors how RGB is currently per-child
(Stage 1 ChildPred.rgb) while semantic is per-parent (paper §3.4).

Initialization detail: the head's output bias is set so that at iter 0,
softmax(logits) ≈ uniform-over-non-empty + tiny prior toward EMPTY_CLASS_ID.
This keeps the initial KL/CE loss bounded (no log(0) surprises) and lets
gradient signal flow through both occupied and free voxels from iter 1.
"""
import torch
import torch.nn as nn

from . import NUM_CLASSES, EMPTY_CLASS_ID


class SemanticHead(nn.Module):
    """Two-layer MLP: parent_feat (B, K, D) -> class logits (B, K, C).

    Args:
        feat_dim:     parent feature dim (segmentor default: 768).
        num_classes:  total class count incl. empty (Stage 2 default: 18).
        hidden:       hidden width (default: 256, ≈ feat_dim/3).
        empty_id:     class index biased upward at init.
    """
    def __init__(self,
                 feat_dim: int = 768,
                 num_classes: int = NUM_CLASSES,
                 hidden: int = 256,
                 empty_id: int = EMPTY_CLASS_ID):
        super().__init__()
        self.num_classes = num_classes
        self.empty_id = empty_id
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, num_classes)

        # Output bias prior: small positive bump on the empty class.
        # log(2.0) for empty, 0 elsewhere → softmax ≈ ~10% empty, ~5% per other
        # class at init (for C=18). Keeps initial loss in a sane range and
        # avoids saturating any one logit before training begins.
        nn.init.zeros_(self.fc2.bias)
        with torch.no_grad():
            self.fc2.bias[empty_id] = float(torch.log(torch.tensor(2.0)))

    def forward(self, parent_feat: torch.Tensor) -> torch.Tensor:
        """parent_feat: (B, K, D)  ->  logits: (B, K, C)."""
        return self.fc2(self.act(self.fc1(parent_feat)))


# ────────────────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────────────────
def _self_test():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, K, D, C = 2, 900, 768, NUM_CLASSES
    head = SemanticHead(feat_dim=D, num_classes=C).to(device)

    x = torch.randn(B, K, D, device=device)
    logits = head(x)
    assert logits.shape == (B, K, C), f"got {logits.shape}"

    probs = logits.softmax(dim=-1)
    p_empty_mean = probs[..., EMPTY_CLASS_ID].mean().item()
    p_other_mean = probs[..., :-1].mean().item()
    print(f"SemanticHead OK: shape {tuple(logits.shape)}")
    print(f"  init probs — empty: {p_empty_mean:.3f}, "
          f"avg-non-empty: {p_other_mean:.3f}  (should be empty > non-empty)")
    assert p_empty_mean > p_other_mean, "empty-class prior not biased upward"

    # Grad flow
    loss = logits.sum()
    loss.backward()
    for n, p in head.named_parameters():
        assert p.grad is not None and p.grad.norm().item() > 0, f"no grad: {n}"
    print("SemanticHead self-test PASSED.")


if __name__ == "__main__":
    _self_test()

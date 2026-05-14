"""Canonical post-install verification for local_aggregate_s2go.

Run after `pip install -e .` on any new PC:

    python kernel/localagg_s2go/check_install.py

Exits non-zero on any failure with a diagnostic hint.

The four checks:
  1. The package is registered with pip (editable install).
  2. The Python wrapper module is importable.
  3. The compiled `.so` loads (no symbol / SM-mismatch errors).
  4. An actual forward pass runs on the GPU (end-to-end functional).

Most cross-machine issues (different GPU SM, gcc-13 host compiler, stale
.so after a source edit) show up at step 3 or 4.
"""
from __future__ import annotations
import importlib
import sys
import traceback


def _fail(step: str, hint: str) -> None:
    print(f"  FAIL — {step}")
    print(f"  hint: {hint}")
    sys.exit(1)


def main() -> None:
    print(f"Verifying local_aggregate_s2go install...\n")

    # ── 1. pip thinks it's installed ──────────────────────────────────────
    try:
        import importlib.metadata as md
        dist = md.distribution("local-aggregate-s2go")
        loc = dist.locate_file("").as_posix() if hasattr(dist, "locate_file") else "<unknown>"
        print(f"  [1/4] pip-installed     ✓   version={dist.version}  location={loc}")
    except Exception as e:
        _fail(f"step 1: pip can't see the package ({e})",
              "run `pip install -e .` from this directory inside the conda env")

    # ── 2. wrapper module importable ──────────────────────────────────────
    try:
        m = importlib.import_module("local_aggregate_s2go")
        print(f"  [2/4] wrapper import    ✓   {m.__file__}")
    except Exception as e:
        _fail(f"step 2: `import local_aggregate_s2go` raised {type(e).__name__}: {e}",
              "the conda env may not be the one pip installed into — check `which python`")

    # ── 3. compiled .so loads ─────────────────────────────────────────────
    try:
        from local_aggregate_s2go import LocalAggregator   # noqa: F401
        print(f"  [3/4] .so load          ✓   LocalAggregator import OK")
    except ImportError as e:
        _fail(f"step 3: .so failed to load — ImportError: {e}",
              "PyTorch ABI mismatch (expected torch==2.0.0+cu118), or libcudart not on "
              "linker path. Verify with `ldd .../local_aggregate_s2go/_C*.so`")
    except RuntimeError as e:
        if "kernel image" in str(e).lower() or "compute capability" in str(e).lower():
            _fail(f"step 3: GPU SM mismatch — {e}",
                  "the bundled .so is built for SM 8.6 (RTX 3060). On a different GPU, "
                  "rebuild with: `pip install -e .` (or with CC=gcc-11 on Ubuntu 23+)")
        _fail(f"step 3: RuntimeError {e}", "see full traceback above")

    # ── 4. end-to-end forward on cuda ─────────────────────────────────────
    try:
        import torch
    except Exception as e:
        _fail(f"step 4: torch import failed ({e})", "env not activated?")

    if not torch.cuda.is_available():
        _fail("step 4: torch.cuda.is_available() == False",
              "no CUDA-capable GPU visible; check `nvidia-smi`")

    dev = torch.device("cuda")
    torch.manual_seed(0)

    try:
        from local_aggregate_s2go import LocalAggregator

        # Move the module to the GPU after construction — `pc_min` is a
        # registered buffer on CPU by default, so `.to(dev)` is required.
        agg = LocalAggregator(
            scale_multiplier=3.0,
            H=10, W=10, D=10,
            pc_min=[-1.0, -1.0, -1.0],
            grid_size=0.2,
            radii_min=1,
        ).to(dev)

        N, P = 4, 3
        means     = torch.zeros(1, N, 3, device=dev)
        opas      = torch.full((1, N), 0.5, device=dev)
        classes   = torch.randn(1, N, 18, device=dev)
        scales    = torch.full((1, N, 3), 0.05, device=dev)
        sigma_inv = (torch.eye(3, device=dev) / (0.05 ** 2)).expand(1, N, 3, 3).contiguous()
        pts       = torch.zeros(1, P, 3, device=dev)

        sem, occ, density = agg(pts, means, opas, classes, scales, sigma_inv)
        torch.cuda.synchronize()

        assert sem.shape == (P, 18), f"sem.shape {sem.shape} != ({P}, 18)"
        assert occ.shape == (P,),    f"occ.shape {occ.shape} != ({P},)"
        assert density.shape == (P,), f"density.shape {density.shape} != ({P},)"
        assert torch.isfinite(sem).all(), "sem has NaN/Inf"
        assert torch.isfinite(occ).all(), "occ has NaN/Inf"

        print(f"  [4/4] forward on cuda   ✓   sem={tuple(sem.shape)}  "
              f"occ={tuple(occ.shape)}  occ.mean={occ.mean().item():.3f}")
    except Exception as e:
        traceback.print_exc()
        _fail(f"step 4: end-to-end forward raised {type(e).__name__}: {e}",
              "the kernel loads but doesn't run — usually a CUDA-runtime / driver "
              "mismatch. Check `nvidia-smi` driver version supports CUDA 11.8 runtime "
              "(driver >= 520.61).")

    print("\nkernel install verified ✓")


if __name__ == "__main__":
    main()

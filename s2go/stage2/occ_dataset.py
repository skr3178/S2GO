"""Stage2OccLoader — adds dense Occ3D voxel GT to NuScenesLoader output.

Wraps `s2go.datasets.nusc_loader.NuScenesLoader` (no edits) and yields the
same T-frame dicts plus two extra per-frame fields:

  - 'sem_voxel_gt': (Vx, Vy, Vz) int64 — class id per voxel, EMPTY_CLASS_ID
                    for voxels not labelled in the sparse .npy
  - 'occ_voxel_gt': (Vx, Vy, Vz) bool — True where sem_voxel_gt != EMPTY

GT source: data/nuscenes_occ/nuscenes_occ/samples/*.npy
  Each .npy is a sparse (M, 4) int64 array (vx, vy, vz, class_id) in the
  LIDAR_TOP frame, voxel-quantized to a 200×200×16 grid with voxel size
  0.5m and pc_range [-50,-50,-5, 50,50,3] (verified at P1).

The filename ↔ sample mapping: each sample's LIDAR_TOP `sample_data` row
has a `filename` like `samples/LIDAR_TOP/<scene>__LIDAR_TOP__<ts>.pcd.bin`.
The corresponding .npy is named `<basename(filename)>.npy` in the occ
samples dir.

Sequence selection: pass `restrict_to_covered=True` (default) to only
yield loader sequences whose every frame has matching GT. Useful for
training; eval and debugging can opt out via `restrict_to_covered=False`.

No edits to `s2go/datasets/nusc_loader.py`.
"""
import os
from typing import List, Dict, Optional, Sequence

import numpy as np
import torch

from ..datasets.nusc_loader import NuScenesLoader
from . import GRID_SHAPE, EMPTY_CLASS_ID


# Per-PC override via S2GO_DATA_ROOT env var (see s2go/datasets/nusc_loader.py).
# Keeps this file byte-identical across machines so `git pull` never
# conflicts on a hardcoded path. Fallback = canonical host-1 path.
_S2GO_DATA_ROOT = os.environ.get(
    "S2GO_DATA_ROOT", "/media/skr/storage/self_driving/S2GO/data")
DEFAULT_OCC_ROOT = os.path.join(
    _S2GO_DATA_ROOT, "nuscenes_occ", "nuscenes_occ", "samples")


class Stage2OccLoader:
    """T-frame sequences enriched with dense voxel GT.

    Args:
        nusc_loader: an existing `NuScenesLoader` (constructed by caller).
        occ_root:    directory containing the .npy GT files.
        grid_shape:  (Vx, Vy, Vz). Must match what the .npy data was built for.
        empty_id:    class id used for fill (paper convention: 17).
        restrict_to_covered: if True, only expose indices whose every frame
                              has matching GT (recommended for training).
    """

    def __init__(self,
                 nusc_loader: NuScenesLoader,
                 occ_root: str = DEFAULT_OCC_ROOT,
                 grid_shape=GRID_SHAPE,
                 empty_id: int = EMPTY_CLASS_ID,
                 restrict_to_covered: bool = True):
        self.loader = nusc_loader
        self.occ_root = occ_root
        self.grid_shape = tuple(grid_shape)
        self.empty_id = int(empty_id)

        assert os.path.isdir(occ_root), f"occ_root not found: {occ_root}"
        self._existing = set(os.listdir(occ_root))

        if restrict_to_covered:
            self.indices = self._scan_covered_indices()
            if not self.indices:
                raise RuntimeError(
                    f"No loader sequences had GT in all T={nusc_loader.T} frames. "
                    f"Check occ_root and the loader's T."
                )
        else:
            self.indices = list(range(len(nusc_loader)))

    # ──────────────────────────────────────────────────────────────────────
    def _scan_covered_indices(self) -> List[int]:
        """Walk every loader sequence; keep those whose every frame has GT."""
        covered = []
        for i, start_tok in enumerate(self.loader.start_tokens):
            sample = self.loader.nusc.get('sample', start_tok)
            ok = True
            for _ in range(self.loader.T):
                sd = self.loader.nusc.get('sample_data',
                                            sample['data']['LIDAR_TOP'])
                npy_name = os.path.basename(sd['filename']) + '.npy'
                if npy_name not in self._existing:
                    ok = False
                    break
                if sample['next'] == '':
                    if _ < self.loader.T - 1:
                        ok = False
                    break
                sample = self.loader.nusc.get('sample', sample['next'])
            if ok:
                covered.append(i)
        return covered

    def _lidar_npy_name(self, sample) -> str:
        sd = self.loader.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        return os.path.basename(sd['filename']) + '.npy'

    def _load_voxel_gt(self, npy_name: str):
        """Load sparse (M, 4) → dense (Vx, Vy, Vz) int64 + (Vx,Vy,Vz) bool."""
        path = os.path.join(self.occ_root, npy_name)
        sparse = np.load(path)                              # (M, 4) int64
        Vx, Vy, Vz = self.grid_shape
        dense = np.full((Vx, Vy, Vz), self.empty_id, dtype=np.int64)
        # Clip to grid bounds defensively (data is well-formed in practice)
        vx = np.clip(sparse[:, 0], 0, Vx - 1)
        vy = np.clip(sparse[:, 1], 0, Vy - 1)
        vz = np.clip(sparse[:, 2], 0, Vz - 1)
        dense[vx, vy, vz] = sparse[:, 3]
        occ_mask = dense != self.empty_id
        return (torch.from_numpy(dense),
                torch.from_numpy(occ_mask))

    # ──────────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int) -> List[Dict]:
        loader_idx = self.indices[i]
        frames: List[Dict] = self.loader[loader_idx]

        # Walk the underlying sample chain again to resolve each frame's
        # lidar token (NuScenesLoader doesn't expose this directly).
        start_tok = frames[0].get('_sample_token')
        assert start_tok is not None, "loader must include _sample_token"
        sample = self.loader.nusc.get('sample', start_tok)
        for t, frame in enumerate(frames):
            npy_name = self._lidar_npy_name(sample)
            sem_dense, occ_mask = self._load_voxel_gt(npy_name)
            frame['sem_voxel_gt'] = sem_dense.unsqueeze(0)           # (1, Vx, Vy, Vz)
            frame['occ_voxel_gt'] = occ_mask.unsqueeze(0)            # (1, Vx, Vy, Vz)
            if t + 1 < len(frames):
                sample = self.loader.nusc.get('sample', sample['next'])
        return frames

    @property
    def covered_indices(self) -> List[int]:
        return list(self.indices)


# ────────────────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────────────────
def _self_test():
    print("Stage2OccLoader self-test (uses real nuScenes + local Occ3D npy)")
    base = NuScenesLoader(T=1, verbose=False)
    occ = Stage2OccLoader(base, restrict_to_covered=True)
    print(f"  base loader: {len(base)} sequences "
          f"({len(occ)} have GT coverage; first idx = {occ.indices[0]})")

    frames = occ[0]
    assert len(frames) == 1
    f = frames[0]
    sem = f['sem_voxel_gt']
    mask = f['occ_voxel_gt']
    assert sem.shape == (1, *GRID_SHAPE),   f"sem shape {sem.shape}"
    assert mask.shape == (1, *GRID_SHAPE),  f"mask shape {mask.shape}"
    assert sem.dtype == torch.int64
    assert mask.dtype == torch.bool

    n_occ = int(mask.sum().item())
    classes_seen = sorted(sem[mask].unique().tolist())
    print(f"  frame[0]: occupied voxels = {n_occ} "
          f"({100 * n_occ / mask.numel():.2f}% of grid)")
    print(f"  classes present in this frame: {classes_seen}")
    assert n_occ > 0, "GT had no occupied voxels — file corruption?"
    assert all(0 <= c <= 16 for c in classes_seen), \
        f"unexpected class ids: {classes_seen}"
    print("Stage2OccLoader self-test PASSED.")


if __name__ == "__main__":
    _self_test()

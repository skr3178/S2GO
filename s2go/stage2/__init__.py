"""S2GO Stage-2 — semantic occupancy training (paper §3.4).

Pure additive module. Stage 1 (`s2go/models/`, `s2go/tools/overfit.py`,
`s2go/tools/stage1_eval.py`) is not modified. Stage 2:

  - Adds a per-parent SemanticHead on top of `S2GOSegmentor.parent.feat`.
  - Broadcasts parent semantic logits to children → per-Gaussian (B, K*J, C).
  - Splats Gaussians + per-Gaussian class probs into a 200×200×16 voxel grid
    via the pure-torch G2V reference (CUDA backend deferred).
  - Computes voxel-space losses: occupancy BCE + (KL | CE+Lovász) semantic.
  - Loads dense semantic occupancy GT from the local nuScenes-Occ3D dump.

Voxel grid constants — pinned to the local GT data extent (verified
2026-05-12 by scanning 30 random .npy files; col-min/max consistent with
SurroundOcc generation conventions [-50,50] × [-50,50] × [-5,3] m).
"""

# Grid extent — LIDAR_TOP frame, matches SurroundOcc/GaussianFormer convention
# and the local .npy data at data/nuscenes_occ/nuscenes_occ/samples/.
PC_RANGE      = (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0)   # (x0, y0, z0, x1, y1, z1)
VOXEL_SIZE    = (0.5, 0.5, 0.5)                          # m, isotropic
GRID_SHAPE    = (200, 200, 16)                           # (Vx, Vy, Vz)

# Class scheme — 17 semantic (0..16) + 1 empty (17). Matches
# GaussianFormer-2's `config/prob/nuscenes_gs12800.py` (empty_label=17,
# num_classes=18). Class IDs 0..15 are SurroundOcc's 16 named classes,
# id 16 is the unnamed "other" catch-all observed in the .npy data
# (verified: union of class ids over 30 random files = {0..16}).
NUM_CLASSES    = 18
EMPTY_CLASS_ID = 17

# Class names — first 16 from SurroundOcc; 17th is "other"; 18th is "empty".
CLASS_NAMES = (
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'other', 'empty',
)
assert len(CLASS_NAMES) == NUM_CLASSES

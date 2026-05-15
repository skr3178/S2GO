"""Minimal nuScenes T-frame loader for Stage-1 integration tests.

Loads contiguous T-keyframe mini-sequences from a nuScenes split. Each frame
provides exactly the fields S2GOSegmentor.forward expects:
    imgs           (B=1, N_cam=6, 3, H, W)   resized to (256, 704)
    lidar_pts      (B=1, M, 3)                in LIDAR_TOP frame
    lidar2img      (B=1, N_cam, 4, 4)         LIDAR_TOP coords → image plane
    ego_pose       (B=1, 4, 4)                LIDAR_TOP-frame ego pose (world←ego)
    ego_pose_inv   (B=1, 4, 4)
    timestamp      (B=1,)                      Δt to previous keyframe (s)
    prev_exists    (B=1,)                      0 at sequence head, 1 thereafter

Images are resized anisotropically to (256, 704) and intrinsics are scaled
accordingly. Center-crop preprocessing comes later when paper-correct numbers
are needed; for now this matches paper-§B image size and is sufficient for
smoke tests.

Run self-test (loads 1 mini-sequence from Part-1 data):
    cd /media/skr/storage/self_driving/S2GO
    python -m s2go.datasets.nusc_loader
"""
import os
import sys
from typing import List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# nuscenes-devkit is not pip-installed; add the local sdk path.
NUSC_SDK = "/media/skr/storage/self_driving/S2GO/nuscenes/nuscenes-devkit/python-sdk"
if NUSC_SDK not in sys.path:
    sys.path.insert(0, NUSC_SDK)

from nuscenes.nuscenes import NuScenes                                       # noqa: E402
from nuscenes.utils.data_classes import LidarPointCloud                      # noqa: E402
from pyquaternion import Quaternion                                           # noqa: E402


CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def _quat_trans_to_4x4(quat: list, trans: list) -> np.ndarray:
    """Build a 4×4 homogeneous transform from a quaternion + translation."""
    T = np.eye(4)
    T[:3, :3] = Quaternion(quat).rotation_matrix
    T[:3, 3] = np.asarray(trans)
    return T


class NuScenesLoader(Dataset):
    """Yields T-frame mini-sequences. __getitem__ returns a list of T dicts."""

    def __init__(self,
                 dataroot: str = "/media/skr/storage/self_driving/S2GO/data/nuscenes",
                 version: str = "v1.0-trainval",
                 T: int = 4,
                 image_size=(256, 704),
                 max_lidar_points: int = 35_000,
                 scene_tokens=None,
                 scene_tokens_json: str = None,
                 verbose: bool = False):
        """
        scene_tokens / scene_tokens_json: optional whitelist of scene tokens to
            include. If given, only samples whose scene_token is in this set
            contribute T-keyframe sequences. Either pass a Python iterable
            (`scene_tokens`) or a path to a JSON file containing a list of
            tokens (`scene_tokens_json`). The two are mutually exclusive.
        """
        self.dataroot = dataroot
        self.T = T
        self.image_h, self.image_w = image_size
        self.max_lidar_points = max_lidar_points
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        if scene_tokens is not None and scene_tokens_json is not None:
            raise ValueError("pass at most one of scene_tokens / scene_tokens_json")
        if scene_tokens_json is not None:
            import json as _json
            with open(scene_tokens_json, 'r') as f:
                scene_tokens = _json.load(f)
        self._scene_whitelist = set(scene_tokens) if scene_tokens is not None else None

        # Find sample tokens that begin a T-keyframe contiguous run within the
        # same scene AND whose every sensor file is on disk (Part-1 has only
        # ~15% of trainval extracted). If a scene_token whitelist is provided,
        # only samples whose scene is in the whitelist contribute.
        self.start_tokens: List[str] = []
        for sample in self.nusc.sample:
            if self._scene_whitelist is not None and \
               sample['scene_token'] not in self._scene_whitelist:
                continue
            if not self._sequence_exists(sample['token']):
                continue
            self.start_tokens.append(sample['token'])
        if verbose:
            extra = f" (whitelist={len(self._scene_whitelist)} scenes)" \
                    if self._scene_whitelist is not None else ""
            print(f"NuScenesLoader: {len(self.start_tokens)} usable T={T}-frame sequences{extra}")

    # ─────────────────────────────────────────────────────────────────────
    def _all_files_present(self, sample) -> bool:
        for sd_token in sample['data'].values():
            sd = self.nusc.get('sample_data', sd_token)
            if not os.path.exists(os.path.join(self.dataroot, sd['filename'])):
                return False
        return True

    def _sequence_exists(self, start_token: str) -> bool:
        """Walk T-1 nexts; verify all in same scene + all files on disk."""
        cur = self.nusc.get('sample', start_token)
        scene_token = cur['scene_token']
        for _ in range(self.T):
            if not self._all_files_present(cur):
                return False
            if cur['scene_token'] != scene_token:
                return False
            if cur['next'] == '':
                # Last frame of scene — only OK if this is the LAST frame in the seq.
                return _ == self.T - 1
            nxt_tok = cur['next']
            cur = self.nusc.get('sample', nxt_tok)
        return True

    # ─────────────────────────────────────────────────────────────────────
    def _load_lidar(self, sample_data) -> np.ndarray:
        """Load LiDAR_TOP point cloud → (M, 3) in LIDAR_TOP coords."""
        path = os.path.join(self.dataroot, sample_data['filename'])
        pc = LidarPointCloud.from_file(path)
        pts = pc.points[:3].T                                    # (M, 3)
        if pts.shape[0] > self.max_lidar_points:
            idx = np.random.choice(pts.shape[0], self.max_lidar_points, replace=False)
            pts = pts[idx]
        return pts.astype(np.float32)

    def _load_image_and_intrinsic(self, cam_data) -> tuple:
        """Load image (3, H, W) in [0,1] float, return scaled intrinsic K."""
        img = Image.open(os.path.join(self.dataroot, cam_data['filename'])).convert('RGB')
        orig_w, orig_h = img.size
        img = img.resize((self.image_w, self.image_h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)                              # (3, H, W)
        sx = self.image_w / orig_w
        sy = self.image_h / orig_h
        K_orig = np.asarray(self.nusc.get(
            'calibrated_sensor', cam_data['calibrated_sensor_token']
        )['camera_intrinsic'], dtype=np.float32)
        K = K_orig.copy()
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        return arr, K

    def _build_lidar2img(self, lidar_sd, cam_sd) -> tuple:
        """Compose LIDAR_TOP → world → ego(cam_t) → cam → image.

        Returns:
            lidar2img: (4, 4)  full projection (K_4x4 @ viewmat) — for cross-attn
            viewmat:   (4, 4)  LIDAR_TOP → cam extrinsic (geometric transform only)
            K_3x3:     (3, 3)  camera intrinsic (original image dims)
        """
        # LIDAR_TOP → ego(lidar_t)
        l_calib = self.nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        T_lidar_egolidar = _quat_trans_to_4x4(l_calib['rotation'], l_calib['translation'])

        # ego(lidar_t) → world
        l_ego = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        T_egolidar_world = _quat_trans_to_4x4(l_ego['rotation'], l_ego['translation'])

        # world → ego(cam_t)
        c_ego = self.nusc.get('ego_pose', cam_sd['ego_pose_token'])
        T_world_egocam = np.linalg.inv(_quat_trans_to_4x4(c_ego['rotation'], c_ego['translation']))

        # ego(cam_t) → cam
        c_calib = self.nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        T_egocam_cam = np.linalg.inv(_quat_trans_to_4x4(c_calib['rotation'], c_calib['translation']))

        # Geometric extrinsic: LIDAR_TOP → cam coords (gsplat's "viewmat")
        viewmat = T_egocam_cam @ T_world_egocam @ T_egolidar_world @ T_lidar_egolidar

        # Camera intrinsic 3×3 (in original-image pixel coords)
        K_3x3 = np.asarray(c_calib['camera_intrinsic'], dtype=np.float64)
        K_4x4 = np.eye(4)
        K_4x4[:3, :3] = K_3x3

        lidar2img = K_4x4 @ viewmat
        return (lidar2img.astype(np.float32),
                viewmat.astype(np.float32),
                K_3x3.astype(np.float32))

    def _scale_lidar2img_for_resize(self, l2i: np.ndarray, sx: float, sy: float) -> np.ndarray:
        """Apply image-resize scale to the projection matrix's first 2 rows."""
        out = l2i.copy()
        out[0] *= sx
        out[1] *= sy
        return out

    def _project_lidar_to_cams(self,
                                 pts: np.ndarray,
                                 lidar2img: np.ndarray,
                                 H: int, W: int) -> np.ndarray:
        """Project LiDAR points onto N_cam image planes → sparse depth maps.

        Args:
            pts:       (M, 3) LiDAR points in LIDAR_TOP frame
            lidar2img: (N_cam, 4, 4) projection matrices (already scaled to (H, W))

        Returns:
            depth_maps: (N_cam, H, W) — 0 where no LiDAR return, depth (m) elsewhere.
                        Multiple LiDAR points on the same pixel → keep the closest
                        (z-min, i.e. nearest surface).
        """
        N_cam = lidar2img.shape[0]
        M = pts.shape[0]
        pts_h = np.concatenate([pts, np.ones((M, 1), dtype=pts.dtype)], axis=-1)  # (M, 4)
        depth_maps = np.zeros((N_cam, H, W), dtype=np.float32)

        for cam_idx in range(N_cam):
            proj = pts_h @ lidar2img[cam_idx].T          # (M, 4)
            z = proj[:, 2]
            # Keep points strictly in front of camera
            in_front = z > 0.1
            if not in_front.any():
                continue
            x_ndc = proj[in_front, 0] / z[in_front]
            y_ndc = proj[in_front, 1] / z[in_front]
            z_v = z[in_front]
            ix = np.round(x_ndc).astype(np.int32)
            iy = np.round(y_ndc).astype(np.int32)
            in_image = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            ix, iy, z_v = ix[in_image], iy[in_image], z_v[in_image]

            # Z-min sort: assign farther points first, so closest writes last (and wins).
            order = np.argsort(-z_v)
            depth_maps[cam_idx, iy[order], ix[order]] = z_v[order]
        return depth_maps

    # ─────────────────────────────────────────────────────────────────────
    def _load_frame(self, sample, prev_exists: bool, prev_timestamp: float) -> Dict:
        """Build one frame dict for S2GOSegmentor."""
        # LiDAR
        lidar_sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pts = self._load_lidar(lidar_sd)                          # (M, 3)

        # 6 cameras
        imgs, l2is, viewmats, Ks = [], [], [], []
        for cam in CAM_NAMES:
            cam_sd = self.nusc.get('sample_data', sample['data'][cam])
            img, _K_resized = self._load_image_and_intrinsic(cam_sd)
            l2i_orig, viewmat, K_orig = self._build_lidar2img(lidar_sd, cam_sd)
            # Rescale projection-side rows to account for image resize.
            orig_w, orig_h = cam_sd['width'], cam_sd['height']
            sx, sy = self.image_w / orig_w, self.image_h / orig_h
            l2i = self._scale_lidar2img_for_resize(l2i_orig, sx, sy)
            # K also scales with resize (viewmat does NOT — it's geometric)
            K_resized = K_orig.copy()
            K_resized[0, 0] *= sx; K_resized[0, 2] *= sx
            K_resized[1, 1] *= sy; K_resized[1, 2] *= sy
            imgs.append(img)
            l2is.append(l2i)
            viewmats.append(viewmat)
            Ks.append(K_resized)

        # Ego pose (use LiDAR-frame ego as the per-frame reference, matches StreamPETR)
        l_ego = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        ego = _quat_trans_to_4x4(l_ego['rotation'], l_ego['translation']).astype(np.float32)

        # Timestamp (Δt to previous keyframe, in seconds; nusc stores microseconds)
        cur_ts = sample['timestamp'] / 1e6
        ts_delta = (cur_ts - prev_timestamp) if prev_exists else 0.0

        # Project LiDAR onto each camera → sparse depth GT for L_depth (Eq. 8 term 2)
        l2is_arr = np.stack(l2is)                                                # (6, 4, 4)
        lidar_depth = self._project_lidar_to_cams(pts, l2is_arr,
                                                     H=self.image_h, W=self.image_w)

        return {
            'imgs':         torch.from_numpy(np.stack(imgs)).unsqueeze(0),       # (1, 6, 3, H, W)
            'lidar_pts':    torch.from_numpy(pts).unsqueeze(0),                  # (1, M, 3)
            'lidar2img':    torch.from_numpy(l2is_arr).unsqueeze(0),              # (1, 6, 4, 4)
            'viewmats':     torch.from_numpy(np.stack(viewmats)).unsqueeze(0),    # (1, 6, 4, 4)  LIDAR_TOP→cam
            'cam_K':        torch.from_numpy(np.stack(Ks)).unsqueeze(0),          # (1, 6, 3, 3)  resized intrinsic
            'lidar_depth':  torch.from_numpy(lidar_depth).unsqueeze(0),           # (1, 6, H, W)
            'ego_pose':     torch.from_numpy(ego).unsqueeze(0),                  # (1, 4, 4)
            'ego_pose_inv': torch.from_numpy(np.linalg.inv(ego)).unsqueeze(0).float(),
            'timestamp':    torch.tensor([ts_delta], dtype=torch.float32),       # (1,)
            'prev_exists':  torch.tensor([float(prev_exists)], dtype=torch.float32),
            '_sample_token': sample['token'],
            '_cur_ts':       cur_ts,
        }

    # ─────────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.start_tokens)

    def __getitem__(self, idx) -> List[Dict]:
        tok = self.start_tokens[idx]
        sample = self.nusc.get('sample', tok)
        frames: List[Dict] = []
        prev_ts = 0.0
        for t in range(self.T):
            frame = self._load_frame(sample, prev_exists=(t > 0), prev_timestamp=prev_ts)
            prev_ts = frame['_cur_ts']
            frames.append(frame)
            if t + 1 < self.T:
                sample = self.nusc.get('sample', sample['next'])
        return frames


# ────────────────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────────────────
def _self_test():
    print("S1.7b nuScenes loader self-test")
    loader = NuScenesLoader(T=4, verbose=True)
    print(f"  dataset size: {len(loader)} usable T=4 sequences")
    assert len(loader) > 0, "no usable sequences found — check Part 1 extraction"

    seq = loader[0]
    assert len(seq) == 4, f"expected T=4 frames, got {len(seq)}"
    print(f"  loaded sequence #0 — {len(seq)} frames, scene "
          f"'{loader.nusc.get('sample', seq[0]['_sample_token'])['scene_token'][:8]}…'")

    # ── Shape checks for each frame ───────────────────────────────────────
    for t, f in enumerate(seq):
        assert f['imgs'].shape         == (1, 6, 3, 256, 704),  f"frame {t}: imgs {f['imgs'].shape}"
        assert f['lidar2img'].shape    == (1, 6, 4, 4),         f"frame {t}: l2i  {f['lidar2img'].shape}"
        assert f['ego_pose'].shape     == (1, 4, 4),            f"frame {t}: ego  {f['ego_pose'].shape}"
        assert f['ego_pose_inv'].shape == (1, 4, 4)
        assert f['lidar_pts'].dim() == 3 and f['lidar_pts'].shape[0] == 1
        # prev_exists pattern: 0 on first frame, 1 thereafter
        expected_pe = 1.0 if t > 0 else 0.0
        assert f['prev_exists'].item() == expected_pe, \
            f"frame {t} prev_exists: got {f['prev_exists'].item()}, expected {expected_pe}"
    print(f"  shapes OK across all 4 frames  imgs={tuple(seq[0]['imgs'].shape)}, "
          f"lidar_pts.M={seq[0]['lidar_pts'].shape[1]}–{seq[3]['lidar_pts'].shape[1]}")

    # ── lidar_depth verification ──────────────────────────────────────────
    for t, f in enumerate(seq):
        assert f['lidar_depth'].shape == (1, 6, 256, 704), \
            f"frame {t}: lidar_depth {f['lidar_depth'].shape}"
        ld = f['lidar_depth'][0]                                       # (6, H, W)
        n_valid = (ld > 0).sum().item()
        n_total = ld.numel()
        valid_d = ld[ld > 0]
        if t == 0:
            print(f"  lidar_depth: {n_valid:,}/{n_total:,} px have LiDAR returns "
                  f"({100*n_valid/n_total:.2f}%); depth range "
                  f"[{valid_d.min().item():.2f}, {valid_d.max().item():.2f}] m")
    assert n_valid > 1000, "expected at least 1k valid depth pixels per frame"

    # ── Image value range sanity ──────────────────────────────────────────
    img0 = seq[0]['imgs'][0, 0]                              # (3, H, W) of CAM_FRONT
    assert img0.min() >= 0.0 and img0.max() <= 1.0
    print(f"  CAM_FRONT image range: [{img0.min():.3f}, {img0.max():.3f}]  (expected [0,1])")

    # ── Δt across consecutive keyframes (~0.5s for nuScenes 2 Hz) ─────────
    dts = [seq[t]['timestamp'].item() for t in range(1, 4)]
    print(f"  Δt between consecutive keyframes: {dts}  (~0.5s expected for nuScenes 2 Hz)")
    assert all(0.4 < dt < 0.7 for dt in dts), \
        f"unexpected keyframe spacing: {dts}"

    # ── lidar2img projection sanity: project a few in-front points → image ─
    pts = seq[0]['lidar_pts'][0].numpy()                          # (M, 3)
    in_front = pts[(pts[:, 0] > 0.5) & (pts[:, 0] < 30)][:50]     # CAM_FRONT roughly +x
    in_front_h = np.concatenate([in_front, np.ones((in_front.shape[0], 1))], axis=1)
    l2i_front = seq[0]['lidar2img'][0, 0].numpy()                  # CAM_FRONT
    proj = (l2i_front @ in_front_h.T).T
    proj_2d = proj[:, :2] / np.clip(proj[:, 2:3], 1e-5, None)
    in_image = ((proj_2d[:, 0] >= 0) & (proj_2d[:, 0] < 704) &
                (proj_2d[:, 1] >= 0) & (proj_2d[:, 1] < 256) &
                (proj[:, 2] > 0)).sum()
    print(f"  lidar2img projection sanity: {in_image}/50 forward LiDAR points "
          f"land in the CAM_FRONT image (expected: many)")
    assert in_image > 5, "lidar2img produces almost no in-image points — check matrix composition"

    # ── Save eyeball debug images ─────────────────────────────────────────
    out_dir = "out/s1.7b_loader"
    os.makedirs(out_dir, exist_ok=True)
    for cam_idx, cam in enumerate(CAM_NAMES):
        img = (seq[0]['imgs'][0, cam_idx].permute(1, 2, 0) * 255).byte().numpy()
        Image.fromarray(img).save(f"{out_dir}/frame0_{cam}.png")
    # Also save with projected LiDAR points overlaid on CAM_FRONT
    img_front = (seq[0]['imgs'][0, 0].permute(1, 2, 0) * 255).byte().numpy().copy()
    in_image_proj = proj_2d[
        (proj_2d[:, 0] >= 0) & (proj_2d[:, 0] < 704) &
        (proj_2d[:, 1] >= 0) & (proj_2d[:, 1] < 256) & (proj[:, 2] > 0)
    ].astype(int)
    for x, y in in_image_proj:
        img_front[max(0, y-1):y+2, max(0, x-1):x+2] = [255, 255, 0]
    Image.fromarray(img_front).save(f"{out_dir}/frame0_CAM_FRONT_with_lidar_overlay.png")
    # Save lidar_depth as a colormap-ish grayscale (depth-normalized)
    ld_front = seq[0]['lidar_depth'][0, 0].numpy()
    if ld_front.max() > 0:
        ld_vis = np.where(ld_front > 0,
                            255 - (ld_front / ld_front.max() * 255).astype(np.uint8),
                            0).astype(np.uint8)
        Image.fromarray(ld_vis).save(f"{out_dir}/frame0_CAM_FRONT_lidar_depth.png")
    print(f"  debug images saved to {out_dir}/")

    print("\nS1.7b nuScenes loader self-test PASSED.")


if __name__ == "__main__":
    _self_test()

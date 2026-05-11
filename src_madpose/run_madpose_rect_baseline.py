"""
MADPose baseline with fisheye rectification.

Rectifies fisheye images to a pinhole view, remaps SuperGlue keypoints into
rectified image coordinates, then runs MADPose hybrid depth+epipolar.
Depth values are invariant to projection (same 3D point, same z), so depth
maps are sampled at the original fisheye keypoint locations.

Rectification backends (--calib-type):
  aria      (default) Aria FISHEYE624 via projectaria_tools
  kitti360  KITTI-360 MEI model via cv2.omnidir; requires --kitti-calib <image_02.yaml>

Usage:
  python run_madpose_rect_baseline.py \
      --image0   <pair_dir>/image0.jpg \
      --image1   <pair_dir>/image1.jpg \
      --mkpts0   <results_dir>/mkpts1.npy \
      --mkpts1   <results_dir>/mkpts2.npy \
      --depth0   <results_dir>/image0_depth.npy \
      --depth1   <results_dir>/image1_depth.npy \
      --output   <results_dir>/result_madpose_rect.json \
      [--calib-type aria|kitti360] [--kitti-calib <image_02.yaml>]
"""

import argparse
import json
import cv2
import numpy as np
import yaml
import madpose
from madpose.utils import get_depths
from projectaria_tools.core import calibration
from projectaria_tools.core.sophus import SE3


# ── Aria FISHEYE624 calibration ───────────────────────────────────────────────
ARIA_RGB_1408_PARAMS = [
    611.0, 704.0, 704.0,
    -0.1,  0.07, -0.02,  0.002,
     0.0,   0.0,
     0.0,   0.0,  0.0,   0.0,
]
ARIA_RGB_1408_VALID_RADIUS = 704.0
ARIA_RGB_1408_MAX_ANGLE    = 1.5
ARIA_RECT_SIZE  = 512
ARIA_RECT_FOCAL = 280.0


def build_aria_src_calib(params=None, w=1408, h=1408):
    if params is None:
        params = ARIA_RGB_1408_PARAMS
    T = SE3.from_matrix(np.eye(4))
    return calibration.CameraCalibration(
        'camera-rgb', calibration.FISHEYE624,
        np.array(params, dtype=np.float64),
        T, w, h, ARIA_RGB_1408_VALID_RADIUS, ARIA_RGB_1408_MAX_ANGLE, ''
    )


def rectify_aria(img_bgr, src_calib, dst_calib):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rect_rgb = calibration.distort_by_calibration(img_rgb, dst_calib, src_calib)
    return cv2.cvtColor(rect_rgb, cv2.COLOR_RGB2BGR)


def remap_keypoints_aria(mkpts, src_calib, dst_calib, rect_size):
    """Map fisheye pixel coords → rectified pixel coords using projectaria_tools."""
    remapped, valid = [], []
    for pt in mkpts:
        try:
            ray  = src_calib.unproject_no_checks(pt)
            pt2d = dst_calib.project(ray)
            in_bounds = (0 <= pt2d[0] < rect_size) and (0 <= pt2d[1] < rect_size)
            remapped.append(pt2d)
            valid.append(in_bounds)
        except Exception:
            remapped.append([0.0, 0.0])
            valid.append(False)
    return np.array(remapped, dtype=np.float32), np.array(valid, dtype=bool)


# ── KITTI-360 MEI calibration ─────────────────────────────────────────────────
KITTI_RECT_SIZE  = 512
KITTI_RECT_FOCAL = 500.0


def load_kitti360_calib(calib_yaml):
    with open(calib_yaml) as f:
        raw = f.read().replace('%YAML:1.0', '')
    c = yaml.safe_load(raw)
    xi     = float(c['mirror_parameters']['xi'])
    k1     = float(c['distortion_parameters']['k1'])
    k2     = float(c['distortion_parameters']['k2'])
    p1     = float(c['distortion_parameters']['p1'])
    p2     = float(c['distortion_parameters']['p2'])
    gamma1 = float(c['projection_parameters']['gamma1'])
    gamma2 = float(c['projection_parameters']['gamma2'])
    u0     = float(c['projection_parameters']['u0'])
    v0     = float(c['projection_parameters']['v0'])
    K = np.array([[gamma1, 0, u0], [0, gamma2, v0], [0, 0, 1]], dtype=np.float64)
    D = np.array([k1, k2, p1, p2], dtype=np.float64)
    return K, D, xi


def rectify_kitti360(img_bgr, K, D, xi, rect_size=KITTI_RECT_SIZE, rect_focal=KITTI_RECT_FOCAL):
    K_new = np.array([[rect_focal, 0, rect_size / 2],
                      [0, rect_focal, rect_size / 2],
                      [0, 0, 1]], dtype=np.float64)
    return cv2.omnidir.undistortImage(
        img_bgr, K, D, np.array([xi]),
        cv2.omnidir.RECTIFY_PERSPECTIVE,
        Knew=K_new, new_size=(rect_size, rect_size)
    )


def remap_keypoints_kitti360(mkpts, K, D, xi, rect_size, rect_focal):
    """Map fisheye pixel coords → rectified pinhole coords using cv2.omnidir."""
    K_new = np.array([[rect_focal, 0, rect_size / 2],
                      [0, rect_focal, rect_size / 2],
                      [0, 0, 1]], dtype=np.float64)
    pts = mkpts.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.omnidir.undistortPoints(
        pts, K, D, np.array([xi]),
        cv2.omnidir.RECTIFY_PERSPECTIVE,
        np.eye(3), K_new
    ).reshape(-1, 2).astype(np.float32)
    valid = ((undist[:, 0] >= 0) & (undist[:, 0] < rect_size) &
             (undist[:, 1] >= 0) & (undist[:, 1] < rect_size))
    return undist, valid


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    image0 = cv2.imread(args.image0)
    image1 = cv2.imread(args.image1)
    if image0 is None or image1 is None:
        raise FileNotFoundError(f"Could not read images: {args.image0}, {args.image1}")

    mkpts0 = np.load(args.mkpts0)
    mkpts1 = np.load(args.mkpts1)
    depth_map0 = np.load(args.depth0)
    depth_map1 = np.load(args.depth1)

    # ── Build calibrations and rectify ────────────────────────────────────────
    if args.calib_type == 'aria':
        fisheye624_params = ARIA_RGB_1408_PARAMS
        rect_size  = ARIA_RECT_SIZE
        rect_focal = ARIA_RECT_FOCAL
        if args.calib:
            with open(args.calib) as f:
                c = json.load(f)
            fisheye624_params = c.get("fisheye624_params", fisheye624_params)
            rect_size  = int(c.get("rect_size",  rect_size))
            rect_focal = float(c.get("rect_focal", rect_focal))
        src_calib = build_aria_src_calib(fisheye624_params)
        dst_calib = calibration.get_linear_camera_calibration(
            rect_size, rect_size, rect_focal, 'camera-rgb'
        )
        rect0 = rectify_aria(image0, src_calib, dst_calib)
        rect1 = rectify_aria(image1, src_calib, dst_calib)
        rk0, valid0 = remap_keypoints_aria(mkpts0, src_calib, dst_calib, rect_size)
        rk1, valid1 = remap_keypoints_aria(mkpts1, src_calib, dst_calib, rect_size)
        print(f"[Aria] Rectified to {rect_size}x{rect_size}, focal={rect_focal}")
    else:
        if not args.kitti_calib:
            raise ValueError("--kitti-calib required when --calib-type kitti360")
        K, D, xi = load_kitti360_calib(args.kitti_calib)
        rect_size  = KITTI_RECT_SIZE
        rect_focal = KITTI_RECT_FOCAL
        rect0 = rectify_kitti360(image0, K, D, xi, rect_size, rect_focal)
        rect1 = rectify_kitti360(image1, K, D, xi, rect_size, rect_focal)
        rk0, valid0 = remap_keypoints_kitti360(mkpts0, K, D, xi, rect_size, rect_focal)
        rk1, valid1 = remap_keypoints_kitti360(mkpts1, K, D, xi, rect_size, rect_focal)
        print(f"[KITTI-360] Rectified to {rect_size}x{rect_size}, focal={rect_focal}")

    valid = valid0 & valid1
    rk0, rk1 = rk0[valid], rk1[valid]

    # Depths sampled at original fisheye keypoint locations (z-invariant)
    d0 = get_depths(image0, depth_map0, mkpts0[valid])
    d1 = get_depths(image1, depth_map1, mkpts1[valid])

    print(f"Keypoints after rectification filter: {valid.sum()} / {len(valid)}")

    if len(rk0) < 8:
        print(f"WARNING: only {len(rk0)} matches after remapping — writing null result")
        with open(args.output, "w") as f:
            json.dump({"R_est": None, "t_est": None,
                       "error": "too few matches after rect remap"}, f, indent=2)
        return

    pp = np.array([(rect_size - 1) / 2.0, (rect_size - 1) / 2.0])

    options = madpose.HybridLORansacOptions()
    options.min_num_iterations = 100
    options.max_num_iterations = 1000
    options.final_least_squares = True
    options.threshold_multiplier = 5.0
    options.num_lo_steps = 4
    options.squared_inlier_thresholds = [16.0 ** 2, 1.0 ** 2]
    options.data_type_weights = [1.0, 1.0]
    options.random_seed = 0

    est_config = madpose.EstimatorConfig()
    est_config.min_depth_constraint = True
    est_config.use_shift = True
    est_config.ceres_num_threads = 8

    pose, stats = madpose.HybridEstimatePoseScaleOffsetSharedFocal(
        rk0, rk1,
        d0, d1,
        [depth_map0.min(), depth_map1.min()],
        pp, pp,
        options, est_config,
    )

    R_est = pose.R()
    t_est = pose.t()
    focal  = pose.focal
    print(f"MADPose+rect R=\n{R_est}\nt={t_est}\nfocal={focal:.1f} (expected ~{rect_focal:.0f})")

    with open(args.output, "w") as f:
        json.dump({
            "R_est":  R_est.tolist(),
            "t_est":  t_est.tolist(),
            "scale":  float(pose.scale),
            "focal":  float(focal),
        }, f, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image0",      required=True)
    p.add_argument("--image1",      required=True)
    p.add_argument("--mkpts0",      required=True)
    p.add_argument("--mkpts1",      required=True)
    p.add_argument("--depth0",      required=True)
    p.add_argument("--depth1",      required=True)
    p.add_argument("--output",      required=True)
    p.add_argument("--calib-type",  default="aria", choices=["aria", "kitti360"])
    p.add_argument("--kitti-calib", default=None,
                   help="Path to KITTI-360 image_02.yaml")
    p.add_argument("--calib",       default=None,
                   help="Optional JSON calib override (aria only)")
    main(p.parse_args())

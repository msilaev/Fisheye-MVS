"""
Baseline: rectify fisheye images to pinhole, then estimate relative pose
via SIFT + Essential matrix.

Rectification backends (--calib-type):
  aria      (default) Aria FISHEYE624 via projectaria_tools
  kitti360  KITTI-360 MEI model via cv2.omnidir; requires --kitti-calib <image_02.yaml>

Output JSON:
  {"scale_est": 1.0, "R_est": [...], "t_est": [...]}
"""

import argparse
import json
import numpy as np
import cv2
import yaml

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


def build_aria_src_calib(params=None, width=1408, height=1408):
    if params is None:
        params = ARIA_RGB_1408_PARAMS
    T = SE3.from_matrix(np.eye(4))
    return calibration.CameraCalibration(
        'camera-rgb', calibration.FISHEYE624,
        np.array(params, dtype=np.float64),
        T, width, height,
        ARIA_RGB_1408_VALID_RADIUS, ARIA_RGB_1408_MAX_ANGLE, ''
    )


def rectify_aria(img_bgr, src_calib, dst_calib):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rect_rgb = calibration.distort_by_calibration(img_rgb, dst_calib, src_calib)
    return cv2.cvtColor(rect_rgb, cv2.COLOR_RGB2BGR)


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
    """Rectify KITTI-360 MEI fisheye to pinhole (pure numpy, no cv2.omnidir required)."""
    k1, k2, p1, p2 = D
    gamma1, gamma2 = K[0, 0], K[1, 1]
    u0, v0 = K[0, 2], K[1, 2]
    cx_r = cy_r = rect_size / 2.0

    us = np.arange(rect_size, dtype=np.float64)
    ug, vg = np.meshgrid(us, us)

    # Rectified normalized coords → 3D ray (X=x_r, Y=y_r, Z=1)
    x_r = (ug - cx_r) / rect_focal
    y_r = (vg - cy_r) / rect_focal
    r_3d = np.sqrt(x_r**2 + y_r**2 + 1.0)

    # MEI projection: x_n = X / (Z + xi*r)
    denom = 1.0 + xi * r_3d
    x_n = x_r / denom
    y_n = y_r / denom

    # Brown-Conrady distortion
    rho2 = x_n**2 + y_n**2
    rad = 1 + k1*rho2 + k2*rho2**2
    x_d = x_n*rad + 2*p1*x_n*y_n + p2*(rho2 + 2*x_n**2)
    y_d = y_n*rad + p1*(rho2 + 2*y_n**2) + 2*p2*x_n*y_n

    # Intrinsics → source pixel in fisheye image
    map_x = (gamma1 * x_d + u0).astype(np.float32)
    map_y = (gamma2 * y_d + v0).astype(np.float32)

    return cv2.remap(img_bgr, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


# ── Shared utilities ──────────────────────────────────────────────────────────
def get_pinhole_K(dst_calib):
    params = dst_calib.get_projection_params()
    if len(params) == 4:
        fx, fy, cx, cy = params
    else:
        fx = fy = params[0]; cx = params[1]; cy = params[2]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def match_sift(gray0, gray1):
    sift = cv2.SIFT_create(nfeatures=4096)
    kp0, des0 = sift.detectAndCompute(gray0, None)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = matcher.knnMatch(des0, des1, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
    return pts0, pts1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image0",      required=True)
    p.add_argument("--image1",      required=True)
    p.add_argument("--output",      required=True)
    p.add_argument("--calib-type",  default="aria", choices=["aria", "kitti360"],
                   help="Rectification backend: aria (default) or kitti360")
    p.add_argument("--kitti-calib", default=None,
                   help="Path to KITTI-360 image_02.yaml (required when --calib-type kitti360)")
    # Legacy JSON calib override for aria
    p.add_argument("--calib",       default=None,
                   help="Optional JSON with fisheye624_params, rect_size, rect_focal (aria only)")
    args = p.parse_args()

    img0 = cv2.imread(args.image0)
    img1 = cv2.imread(args.image1)
    if img0 is None or img1 is None:
        raise FileNotFoundError(f"Could not read: {args.image0}, {args.image1}")

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
        rect0 = rectify_aria(img0, src_calib, dst_calib)
        rect1 = rectify_aria(img1, src_calib, dst_calib)
        K = get_pinhole_K(dst_calib)
        print(f"[Aria] Rectified to {rect_size}x{rect_size}, focal={rect_focal}")
    else:
        if not args.kitti_calib:
            raise ValueError("--kitti-calib required when --calib-type kitti360")
        K_src, D, xi = load_kitti360_calib(args.kitti_calib)
        rect_size  = KITTI_RECT_SIZE
        rect_focal = KITTI_RECT_FOCAL
        rect0 = rectify_kitti360(img0, K_src, D, xi, rect_size, rect_focal)
        rect1 = rectify_kitti360(img1, K_src, D, xi, rect_size, rect_focal)
        K = np.array([[rect_focal, 0, rect_size / 2],
                      [0, rect_focal, rect_size / 2],
                      [0, 0, 1]], dtype=np.float64)
        print(f"[KITTI-360] Rectified to {rect_size}x{rect_size}, focal={rect_focal}")

    gray0 = cv2.cvtColor(rect0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)

    pts0, pts1 = match_sift(gray0, gray1)
    print(f"SIFT matches after ratio test: {len(pts0)}")

    if len(pts0) < 8:
        print("WARNING: Too few matches — writing null result")
        with open(args.output, "w") as f:
            json.dump({"R_est": None, "t_est": None, "error": "too few matches"}, f, indent=2)
        return

    E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, pts0, pts1, K, mask=mask)

    result = {"scale_est": 1.0, "R_est": R.tolist(), "t_est": t.ravel().tolist()}
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved pose to {args.output}")


if __name__ == "__main__":
    main()

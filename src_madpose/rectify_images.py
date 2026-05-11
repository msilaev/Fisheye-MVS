"""
Rectify a pair of fisheye images to a pinhole view and save them for
downstream processing (UniK3D, SuperGlue).

Rectification backends (--calib-type):
  aria      (default) Aria FISHEYE624 via projectaria_tools
  kitti360  KITTI-360 MEI model via cv2.omnidir; requires --kitti-calib <image_02.yaml>

Usage:
  python rectify_images.py \
      --image0   <pair_dir>/image0.jpg \
      --image1   <pair_dir>/image1.jpg \
      --output-dir <rect_dir> \
      [--calib-type aria|kitti360] [--kitti-calib <image_02.yaml>]

Outputs:
  <rect_dir>/image0.jpg       rectified image 0
  <rect_dir>/image1.jpg       rectified image 1
  <rect_dir>/image_pairs.txt  "image0.jpg image1.jpg"
"""

import argparse
import os
import json
import cv2
import numpy as np
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


def build_aria_src_calib(params, w=1408, h=1408):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image0",      required=True)
    p.add_argument("--image1",      required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--calib-type",  default="aria", choices=["aria", "kitti360"],
                   help="Rectification backend: aria (default) or kitti360")
    p.add_argument("--kitti-calib", default=None,
                   help="Path to KITTI-360 image_02.yaml (required when --calib-type kitti360)")
    # Legacy JSON calib override for aria
    p.add_argument("--calib",       default=None,
                   help="Optional JSON with fisheye624_params, rect_size, rect_focal (aria only)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
        def do_rectify(img_bgr):
            return rectify_aria(img_bgr, src_calib, dst_calib)
        print(f"[Aria] Rectifying to {rect_size}x{rect_size}, focal={rect_focal}")
    else:
        if not args.kitti_calib:
            raise ValueError("--kitti-calib required when --calib-type kitti360")
        K, D, xi = load_kitti360_calib(args.kitti_calib)
        rect_size  = KITTI_RECT_SIZE
        rect_focal = KITTI_RECT_FOCAL
        def do_rectify(img_bgr):
            return rectify_kitti360(img_bgr, K, D, xi, rect_size, rect_focal)
        print(f"[KITTI-360] Rectifying to {rect_size}x{rect_size}, focal={rect_focal}")

    for src_path, out_name in [(args.image0, "image0.jpg"), (args.image1, "image1.jpg")]:
        img = cv2.imread(src_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {src_path}")
        rect = do_rectify(img)
        out_path = os.path.join(args.output_dir, out_name)
        cv2.imwrite(out_path, rect)
        print(f"Saved {out_path}  ({rect.shape[1]}x{rect.shape[0]})")

    pairs_path = os.path.join(args.output_dir, "image_pairs.txt")
    with open(pairs_path, "w") as f:
        f.write("image0.jpg image1.jpg\n")
    print(f"Saved {pairs_path}")


if __name__ == "__main__":
    main()

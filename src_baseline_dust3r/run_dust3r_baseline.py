"""
Baseline: DUSt3R relative pose estimation.

Supports two modes:
  --raw       feed raw fisheye images directly (default)
  --rectify   undistort fisheye → pinhole first, then run DUSt3R

Rectification backends (--calib-type):
  aria      (default) Aria FISHEYE624 via projectaria_tools
  kitti360  KITTI-360 MEI model via cv2.omnidir; requires --kitti-calib <image_02.yaml>

Output JSON matches the format of pose_estimation_procrustes.py:
  {"scale_est": 1.0, "R_est": [...], "t_est": [...]}
"""

import argparse
import json
import tempfile
import os
import numpy as np
import cv2
import yaml

# DUSt3R imports — install dust3r before running
try:
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
except ImportError as e:
    raise ImportError(
        "dust3r not installed. Run: pip install dust3r  "
        "or clone https://github.com/naver/dust3r"
    ) from e

DUST3R_MODEL = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"

# ── Aria FISHEYE624 rectification ─────────────────────────────────────────────
from projectaria_tools.core import calibration as aria_calib
from projectaria_tools.core.sophus import SE3

ARIA_RGB_1408_PARAMS = [611.0, 704.0, 704.0,
                        -0.1, 0.07, -0.02, 0.002,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ARIA_RECT_SIZE  = 512
ARIA_RECT_FOCAL = 280.0


def build_aria_calibs(rect_size=ARIA_RECT_SIZE, rect_focal=ARIA_RECT_FOCAL, params=None):
    if params is None:
        params = ARIA_RGB_1408_PARAMS
    T = SE3.from_matrix(np.eye(4))
    src = aria_calib.CameraCalibration(
        'camera-rgb', aria_calib.FISHEYE624,
        np.array(params, dtype=np.float64),
        T, 1408, 1408, 704.0, 1.5, ''
    )
    dst = aria_calib.get_linear_camera_calibration(rect_size, rect_size, rect_focal, 'camera-rgb')
    return src, dst


def rectify_aria(img_bgr, src_calib, dst_calib):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rect_rgb = aria_calib.distort_by_calibration(img_rgb, dst_calib, src_calib)
    return cv2.cvtColor(rect_rgb, cv2.COLOR_RGB2BGR)


# ── KITTI-360 MEI rectification ───────────────────────────────────────────────
KITTI_RECT_SIZE  = 512
KITTI_RECT_FOCAL = 500.0


def load_kitti360_calib(calib_yaml):
    """Parse KITTI-360 MEI fisheye calibration yaml."""
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


# ── DUSt3R inference ──────────────────────────────────────────────────────────
def run_dust3r(img_path0, img_path1, device="cuda"):
    model = AsymmetricCroCo3DStereo.from_pretrained(DUST3R_MODEL)
    model = model.to(device).eval()

    images = load_images([img_path0, img_path1], size=512)
    pairs  = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)
    poses = scene.get_im_poses()
    P0 = poses[0].detach().cpu().numpy()
    P1 = poses[1].detach().cpu().numpy()

    R_rel = P1[:3, :3] @ P0[:3, :3].T
    t_rel = P1[:3, 3] - R_rel @ P0[:3, 3]
    return R_rel, t_rel


def rectify_image(img_path, calib_type, aria_src=None, aria_dst=None,
                  kitti_K=None, kitti_D=None, kitti_xi=None):
    """Rectify image to a temp file. Returns temp file path."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    if calib_type == 'aria':
        rect = rectify_aria(img, aria_src, aria_dst)
    else:
        rect = rectify_kitti360(img, kitti_K, kitti_D, kitti_xi)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, rect)
    return tmp.name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image0",      required=True)
    p.add_argument("--image1",      required=True)
    p.add_argument("--output",      required=True)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--rectify",     action="store_true",
                   help="Rectify fisheye images to pinhole before running DUSt3R")
    p.add_argument("--calib-type",  default="aria", choices=["aria", "kitti360"],
                   help="Rectification backend: aria (default) or kitti360")
    p.add_argument("--kitti-calib", default=None,
                   help="Path to KITTI-360 image_02.yaml (required when --calib-type kitti360)")
    args = p.parse_args()

    img0, img1 = args.image0, args.image1
    tmp_files = []

    if args.rectify:
        if args.calib_type == 'aria':
            src_c, dst_c = build_aria_calibs()
            print(f"[Aria] Rectifying to {ARIA_RECT_SIZE}x{ARIA_RECT_SIZE}, focal={ARIA_RECT_FOCAL}")
            img0 = rectify_image(args.image0, 'aria', aria_src=src_c, aria_dst=dst_c)
            img1 = rectify_image(args.image1, 'aria', aria_src=src_c, aria_dst=dst_c)
        else:
            if not args.kitti_calib:
                raise ValueError("--kitti-calib required when --calib-type kitti360")
            K, D, xi = load_kitti360_calib(args.kitti_calib)
            print(f"[KITTI-360] Rectifying to {KITTI_RECT_SIZE}x{KITTI_RECT_SIZE}, focal={KITTI_RECT_FOCAL}")
            img0 = rectify_image(args.image0, 'kitti360', kitti_K=K, kitti_D=D, kitti_xi=xi)
            img1 = rectify_image(args.image1, 'kitti360', kitti_K=K, kitti_D=D, kitti_xi=xi)
        tmp_files = [img0, img1]

    try:
        R, t = run_dust3r(img0, img1, device=args.device)
    finally:
        for f in tmp_files:
            os.unlink(f)

    result = {"scale_est": 1.0, "R_est": R.tolist(), "t_est": t.tolist()}
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved pose to {args.output}")


if __name__ == "__main__":
    main()

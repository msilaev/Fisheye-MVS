"""
Fisheye Essential Matrix baseline.

Uses the known fisheye calibration to unproject SuperGlue keypoint matches
directly to unit-sphere rays, then estimates the essential matrix via the
spherical epipolar constraint  r2^T E r1 = 0  with RANSAC.

No rectification — the full fisheye FOV is used.
Translation is recovered up to scale; scale_est is always 1.0.

Supported calibration backends (--calib-type):
  aria      Aria FISHEYE624 via projectaria_tools (default)
  kitti360  KITTI-360 MEI model; requires --kitti-calib <image_02.yaml>

Usage:
  python run_fisheye_emat_baseline.py \
      --mkpts0    <results_dir>/mkpts1.npy \
      --mkpts1    <results_dir>/mkpts2.npy \
      --output    <results_dir>/result_fisheye_emat.json \
      [--calib-type aria|kitti360] [--kitti-calib <yaml>] \
      [--mask <fisheye_mask.png>] [--size-x 1408] [--size-y 1408]

Output JSON:
  {"scale_est": 1.0, "R_est": [[...]], "t_est": [...], "inliers": N}
"""

import argparse
import json

import cv2
import numpy as np
from PIL import Image

from projectaria_tools.core import calibration
from projectaria_tools.core.sophus import SE3


# ── Aria FISHEYE624 defaults ──────────────────────────────────────────────────

ARIA_RGB_1408_PARAMS = [
    611.0, 704.0, 704.0,
    -0.1,  0.07, -0.02,  0.002,
     0.0,   0.0,
     0.0,   0.0,  0.0,   0.0,
]
ARIA_RGB_1408_VALID_RADIUS = 704.0
ARIA_RGB_1408_MAX_ANGLE    = 1.5


def build_aria_calib(params=None, width=1408, height=1408):
    if params is None:
        params = ARIA_RGB_1408_PARAMS
    T = SE3.from_matrix(np.eye(4))
    return calibration.CameraCalibration(
        'camera-rgb', calibration.FISHEYE624,
        np.array(params, dtype=np.float64),
        T, width, height,
        ARIA_RGB_1408_VALID_RADIUS, ARIA_RGB_1408_MAX_ANGLE, ''
    )


def unproject_aria(mkpts, calib):
    """(N,2) pixel (x,y) → (N,3) unit rays via Aria FISHEYE624."""
    rays = np.zeros((len(mkpts), 3), dtype=np.float64)
    for i, pt in enumerate(mkpts):
        r = calib.unproject(pt)
        if r is not None:
            r = np.asarray(r, dtype=np.float64)
            n = np.linalg.norm(r)
            if n > 0:
                rays[i] = r / n
                continue
        rays[i] = [0.0, 0.0, 1.0]   # fallback: forward ray
    return rays


# ── KITTI-360 MEI model ───────────────────────────────────────────────────────

def load_kitti360_calib(calib_yaml):
    import yaml
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


def unproject_kitti360_mei(mkpts, K, D, xi):
    """(N,2) pixel (x,y) → (N,3) unit rays via KITTI-360 MEI model.

    Pipeline (inverse of forward projection):
      pixel → K^-1 + undo Brown-Conrady  → (x_n, y_n) normalized undistorted
            → MEI inverse (unit-sphere lift)  → unit ray (mx, my, mz)
    """
    # Undo Brown-Conrady distortion and K normalisation
    pts = mkpts.reshape(-1, 1, 2).astype(np.float32)
    pts_undist = cv2.undistortPoints(
        pts, K.astype(np.float32), D.astype(np.float32)
    ).reshape(-1, 2).astype(np.float64)

    x_n, y_n = pts_undist[:, 0], pts_undist[:, 1]

    # MEI inverse: (x_n, y_n) → unit-sphere point (mx, my, mz)
    # Derivation: x_n = mx/(mz+xi), mx^2+my^2+mz^2=1
    # → (rho^2+1)*u^2 - 2*xi*u + (xi^2-1) = 0,  u = mz+xi
    # → u = (xi + sqrt(1 + rho^2*(1-xi^2))) / (rho^2+1)
    rho2 = x_n**2 + y_n**2
    disc = np.maximum(1.0 + rho2 * (1.0 - xi**2), 0.0)
    u = (xi + np.sqrt(disc)) / (rho2 + 1.0)
    mx = x_n * u
    my = y_n * u
    mz = u - xi

    rays = np.stack([mx, my, mz], axis=1)
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    return rays / np.maximum(norms, 1e-8)


# ── Mask and filtering ────────────────────────────────────────────────────────

def load_fisheye_mask(mask_path, size_x, size_y):
    """Load fisheye validity mask → bool array (size_x, size_y) indexed [x, y]."""
    mask = Image.open(mask_path).resize((size_x, size_y), Image.NEAREST)
    arr = np.array(mask, dtype=np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr[..., 0]           # use first channel
    return (arr.T > 0.5)            # transpose → (W, H) = (x, y) indexing


def apply_mask(mkpts0, mkpts1, mask):
    """Keep only pairs where both keypoints fall in the valid mask region."""
    valid = []
    W, H = mask.shape
    for i, (p0, p1) in enumerate(zip(mkpts0.astype(int), mkpts1.astype(int))):
        if (0 <= p0[0] < W and 0 <= p0[1] < H and mask[p0[0], p0[1]] and
                0 <= p1[0] < W and 0 <= p1[1] < H and mask[p1[0], p1[1]]):
            valid.append(i)
    return np.array(valid, dtype=int)


# ── Essential matrix on unit sphere ──────────────────────────────────────────

def rays_to_normalized(rays):
    """Unit rays → (x/z, y/z) normalized image coords for cv2.findEssentialMat.

    The spherical epipolar constraint r2^T E r1 = 0 is equivalent to the
    standard essential-matrix constraint when the rays are treated as
    perspective-normalized coordinates with focal=1, pp=(0,0).
    Valid only for forward-facing rays (rz > 0).
    """
    z = rays[:, 2]
    valid = z > 1e-3
    pts = np.zeros((len(rays), 2), dtype=np.float64)
    pts[valid, 0] = rays[valid, 0] / z[valid]
    pts[valid, 1] = rays[valid, 1] / z[valid]
    return pts, valid


def estimate_essential(pts0, pts1, ransac_thresh=0.01, confidence=0.999):
    """Estimate E from normalized coords; return (R, t, n_inliers) or None."""
    E, mask = cv2.findEssentialMat(
        pts0, pts1,
        focal=1.0, pp=(0.0, 0.0),
        method=cv2.RANSAC,
        prob=confidence,
        threshold=ransac_thresh,
    )
    if E is None:
        return None, None, 0
    n_in, R, t, _ = cv2.recoverPose(
        E, pts0, pts1,
        focal=1.0, pp=(0.0, 0.0),
        mask=mask,
    )
    return R, t.ravel(), n_in


# ── Main ──────────────────────────────────────────────────────────────────────

def write_result(path, R, t, n_inliers=None, error=None):
    obj = {"scale_est": 1.0,
           "R_est": R.tolist() if R is not None else None,
           "t_est": t.tolist() if t is not None else None}
    if n_inliers is not None:
        obj["inliers"] = int(n_inliers)
    if error is not None:
        obj["error"] = error
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mkpts0",      required=True, help="mkpts1.npy from SuperGlue (image 0)")
    p.add_argument("--mkpts1",      required=True, help="mkpts2.npy from SuperGlue (image 1)")
    p.add_argument("--output",      required=True)
    p.add_argument("--calib-type",  default="aria", choices=["aria", "kitti360"])
    p.add_argument("--kitti-calib", default=None,   help="KITTI-360 image_02.yaml")
    p.add_argument("--calib",       default=None,   help="JSON calib override (aria only)")
    p.add_argument("--mask",        default=None,   help="Fisheye validity mask .png")
    p.add_argument("--size-x",      type=int, default=1408)
    p.add_argument("--size-y",      type=int, default=1408)
    p.add_argument("--ransac-thresh", type=float, default=0.01,
                   help="Sampson distance threshold for RANSAC (normalized coords)")
    args = p.parse_args()

    mkpts0 = np.load(args.mkpts0)   # (N, 2) float, pixel (x, y)
    mkpts1 = np.load(args.mkpts1)
    print(f"Loaded {len(mkpts0)} SuperGlue matches")

    # Mask filter
    if args.mask:
        mask = load_fisheye_mask(args.mask, args.size_x, args.size_y)
        idx = apply_mask(mkpts0, mkpts1, mask)
        mkpts0, mkpts1 = mkpts0[idx], mkpts1[idx]
        print(f"After mask filter: {len(mkpts0)} matches")

    if len(mkpts0) < 8:
        print("WARNING: too few matches")
        write_result(args.output, None, None, error="too few matches")
        return

    # Unproject to unit rays
    if args.calib_type == 'aria':
        fisheye624_params = ARIA_RGB_1408_PARAMS
        if args.calib:
            with open(args.calib) as f:
                c = json.load(f)
            fisheye624_params = c.get("fisheye624_params", fisheye624_params)
        calib = build_aria_calib(fisheye624_params)
        rays0 = unproject_aria(mkpts0, calib)
        rays1 = unproject_aria(mkpts1, calib)
        print(f"[Aria] Unprojected {len(rays0)} points via FISHEYE624")
    else:
        if not args.kitti_calib:
            raise ValueError("--kitti-calib required for kitti360")
        K, D, xi = load_kitti360_calib(args.kitti_calib)
        rays0 = unproject_kitti360_mei(mkpts0, K, D, xi)
        rays1 = unproject_kitti360_mei(mkpts1, K, D, xi)
        print(f"[KITTI-360] Unprojected {len(rays0)} points via MEI (xi={xi:.3f})")

    # Convert to normalized coords (valid for forward-facing rays only)
    pts_n0, valid0 = rays_to_normalized(rays0)
    pts_n1, valid1 = rays_to_normalized(rays1)
    valid = valid0 & valid1
    pts_n0, pts_n1 = pts_n0[valid], pts_n1[valid]
    print(f"Forward-facing rays: {valid.sum()} / {len(valid)}")

    if len(pts_n0) < 8:
        print("WARNING: too few forward-facing rays")
        write_result(args.output, None, None, error="too few valid rays")
        return

    # Essential matrix + pose recovery
    R, t, n_in = estimate_essential(pts_n0, pts_n1,
                                     ransac_thresh=args.ransac_thresh)
    if R is None:
        print("WARNING: essential matrix estimation failed")
        write_result(args.output, None, None, error="E estimation failed")
        return

    print(f"Inliers: {n_in} / {len(pts_n0)}")
    write_result(args.output, R, t, n_inliers=n_in)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

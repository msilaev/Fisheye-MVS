"""
Run Procrustes Sim(3) pose estimation on a sampled pair and evaluate against GT.
Writes rotation_error_est and translation_error_est back into the test set JSON.

Differences from pose_estimation_procrustes.py:
  - Reads GT cam-to-world poses from --cam2w-1 / --cam2w-2
  - Optionally corrects for 90° CW image rotation (--rotate-gt, use for ADT)
  - Appends errors to --sample-t-R-test-db-path at --sample-ind
"""

import argparse
import json
import numpy as np
from PIL import Image
from scipy.linalg import orthogonal_procrustes
from utils import compute_pose_error


# ── Fisheye mask ──────────────────────────────────────────────────────────────

def get_mask(fisheye_mask_path, size_x, size_y):
    mask = Image.open(fisheye_mask_path)
    mask = mask.resize((size_x, size_y), Image.NEAREST)
    mask = np.array(mask) / 255.0
    mask = mask.transpose(2, 1, 0)
    mask = mask[2, :, :]
    return (mask > 0.5).astype(bool)


def filter_mkpts_by_mask(mask, mkpts):
    valid = [i for i, p in enumerate(mkpts) if mask[p[0], p[1]]]
    return np.array(valid, dtype=int)


def filter_cloud_at_mkpts(path_P, mkpts, mask):
    P = np.load(path_P)
    P = P[0].transpose(0, 2, 1)           # (3, W, H)
    pts = P[:, mkpts[:, 0], mkpts[:, 1]]  # (3, N)
    return pts.T                           # (N, 3)


# ── Procrustes ────────────────────────────────────────────────────────────────

def procrustes_sim3(data1, data2):
    mtx1 = np.array(data1, dtype=np.float64, copy=True)
    mtx2 = np.array(data2, dtype=np.float64, copy=True)

    mu1 = mtx1.mean(0);  mu2 = mtx2.mean(0)
    mtx1 -= mu1;          mtx2 -= mu2

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Degenerate point set")

    mtx1 /= norm1;  mtx2 /= norm2
    R, s = orthogonal_procrustes(mtx2, mtx1)
    scale = s * norm2 / norm1
    t = scale * np.dot(-mu1, R.T) + mu2
    return R, t, scale


# ── GT pose ───────────────────────────────────────────────────────────────────

# 90° CW rotation correction for ADT (images are stored sideways)
_Rz_90     = np.array([[0, -1, 0], [1,  0, 0], [0, 0, 1]], dtype=float)
_Rz_neg90  = np.array([[0,  1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)


def load_gt_pose(cam2w_1_path, cam2w_2_path, rotate):
    T1 = np.loadtxt(cam2w_1_path)
    T2 = np.loadtxt(cam2w_2_path)
    T_rel = np.linalg.inv(T2) @ T1
    R_gt = T_rel[:3, :3]
    t_gt = T_rel[:3, 3]
    if rotate:
        R_gt = _Rz_neg90.T @ R_gt @ _Rz_90.T
        t_gt = t_gt @ _Rz_90.T
    return R_gt, t_gt


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # Load and filter keypoints
    mkpts1 = np.load(args.mkpts1)
    mkpts2 = np.load(args.mkpts2)

    mask = get_mask(args.fisheye_mask_path, args.size_x, args.size_y)

    valid1 = filter_mkpts_by_mask(mask, mkpts1)
    valid2 = filter_mkpts_by_mask(mask, mkpts2)
    shared  = np.intersect1d(valid1, valid2)

    mkpts1 = mkpts1[shared]
    mkpts2 = mkpts2[shared]

    P1 = filter_cloud_at_mkpts(args.point1, mkpts1, mask)
    P2 = filter_cloud_at_mkpts(args.point2, mkpts2, mask)

    # Distance threshold
    dist = np.linalg.norm(P1, axis=1)
    keep = dist < args.distance_threshold
    P1, P2 = P1[keep], P2[keep]

    # Estimated pose
    R_est, t_est, scale_est = procrustes_sim3(P1, P2)

    # GT pose
    R_gt, t_gt = load_gt_pose(args.cam2w_1, args.cam2w_2, args.rotate_gt)

    d_t, d_R = compute_pose_error(R_gt, t_gt, R_est, t_est)
    print(f"rotation_error = {d_R:.3f}°,  translation_error = {d_t:.4f}m")

    # Baseline (identity) for reference
    d_t0, d_R0 = compute_pose_error(R_gt, t_gt, np.eye(3), np.zeros(3))
    print(f"identity baseline: rotation = {d_R0:.3f}°, translation = {d_t0:.4f}m")

    # Write errors back to test set JSON
    if args.sample_t_R_test_db_path and args.sample_ind is not None:
        with open(args.sample_t_R_test_db_path) as f:
            test_set = json.load(f)
        test_set[args.sample_ind]["translation_error_est"] = float(d_t)
        test_set[args.sample_ind]["rotation_error_est"]    = float(d_R)
        with open(args.sample_t_R_test_db_path, "w") as f:
            json.dump(test_set, f, indent=2)
        print(f"Updated {args.sample_t_R_test_db_path} at index {args.sample_ind}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--point1",               required=True)
    p.add_argument("--point2",               required=True)
    p.add_argument("--mkpts1",               required=True)
    p.add_argument("--mkpts2",               required=True)
    p.add_argument("--cam2w-1",              required=True,
                   help="GT cam-to-world pose for image 0 (.txt)")
    p.add_argument("--cam2w-2",              required=True,
                   help="GT cam-to-world pose for image 1 (.txt)")
    p.add_argument("--fisheye-mask-path",    required=True)
    p.add_argument("--distance-threshold",   type=float, required=True)
    p.add_argument("--size-x",               type=int,   required=True)
    p.add_argument("--size-y",               type=int,   required=True)
    p.add_argument("--sample-t-R-test-db-path", default=None)
    p.add_argument("--sample-ind",           type=int,   default=None)
    p.add_argument("--rotate-gt",            action="store_true",
                   help="Apply 90° CW rotation correction to GT (use for ADT)")
    main(p.parse_args())

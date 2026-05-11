#!/usr/bin/env python3
"""Compute multiview consistency metrics for merged point clouds.

This script is intended for per-pair folders produced by the rendering pipeline,
for example:

  experiments/<EXP>/rendered_example/<pair_name>/
      image0_points.npy
      image1_points.npy
      image0.txt
      image1.txt
      result_procrustes_ransac.json
      result_madpose.json
      result_madpose_rect.json
      ...

It reports:
- pose consistency: rotation / translation error against GT
- geometric consistency: nearest-neighbour distances and Chamfer distance
- overlap consistency: inlier / overlap ratio below a distance threshold

Example:
    python src_metrics/compute_cloud_consistency.py \
        --pair-dir /path/to/rendered_example/pair_custom_... \
        --dataset adt \
        --mask assets/fisheye_masks/MaskADT_rot.png \
        --distance-threshold 1000 \
        --overlap-threshold 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


DEFAULT_METHODS = [
    ("GT", None),
    ("Ours", "result_procrustes.json"),
    ("Ours+RANSAC", "result_procrustes_ransac.json"),
    ("MADPose", "result_madpose.json"),
    ("MADPose+rect", "result_madpose_rect.json"),
    ("Fisheye-Emat", "result_fisheye_emat.json"),
    ("Pinhole", "result_pinhole.json"),
    ("DUSt3R", "result_dust3r.json"),
    ("DUSt3R+rect", "result_dust3r_rect.json"),
]


def angle_error_mat(R1: np.ndarray, R2: np.ndarray) -> float:
    cos = (np.trace(R1.T @ R2) - 1.0) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.rad2deg(np.abs(np.arccos(cos))))


def compute_pose_error(R_gt: np.ndarray, t_gt: np.ndarray,
                       R_est: np.ndarray, t_est: np.ndarray) -> Tuple[float, float]:
    error_t = float(np.linalg.norm(t_gt - t_est))
    error_R = angle_error_mat(R_est, R_gt)
    return error_t, error_R


def load_cloud(path: Path) -> np.ndarray:
    """Load UniK3D point cloud .npy into shape (N, 3)."""
    P = np.load(path)
    if P.ndim == 4 and P.shape[0] == 1 and P.shape[1] == 3:
        P = P[0].transpose(0, 2, 1).reshape(3, -1).T
    elif P.ndim == 2 and P.shape[1] == 3:
        pass
    else:
        raise ValueError(f"Unexpected point cloud shape {P.shape} in {path}")
    return P.astype(np.float64, copy=False)


def load_gt_pose(cam2w0: Path, cam2w1: Path) -> Tuple[np.ndarray, np.ndarray]:
    T0 = np.loadtxt(cam2w0)
    T1 = np.loadtxt(cam2w1)
    rel = np.linalg.inv(T1) @ T0
    return rel[:3, :3], rel[:3, 3]


def load_method_result(json_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    R = data.get("R_est")
    t = data.get("t_est")
    if R is None or t is None:
        return None

    scale = data.get("scale_est")
    if scale is None:
        scale = data.get("scale", 1.0)

    return np.array(R, dtype=np.float64), np.array(t, dtype=np.float64), float(scale)


def load_mask(mask_path: Path, size: int, dataset: str) -> np.ndarray:
    """Load fisheye mask to a flat boolean array matching cloud ordering."""
    resample = getattr(Image, "Resampling", Image).NEAREST
    mask = Image.open(mask_path).convert("RGB")
    mask = mask.resize((size, size), resample)
    mask = np.array(mask) / 255.0
    if dataset == "adt":
        mask = np.rot90(mask, k=1)
    mask = mask.transpose(2, 1, 0).reshape(3, -1).T
    return mask[:, 2] > 0.5


def filter_cloud(P: np.ndarray,
                 distance_threshold: float,
                 mask: Optional[np.ndarray] = None) -> np.ndarray:
    keep = np.linalg.norm(P, axis=1) < distance_threshold
    if mask is not None:
        if len(mask) != len(P):
            raise ValueError(
                f"Mask length {len(mask)} does not match cloud length {len(P)}"
            )
        keep &= mask
    return P[keep]


def maybe_subsample(P: np.ndarray,
                    max_points: Optional[int],
                    seed: int) -> np.ndarray:
    if max_points is None or max_points <= 0 or len(P) <= max_points:
        return P
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(P), size=max_points, replace=False)
    return P[indices]


def nearest_neighbour_distances(tree: cKDTree, points: np.ndarray) -> np.ndarray:
    try:
        distances, _ = tree.query(points, k=1, workers=-1)
    except TypeError:
        distances, _ = tree.query(points, k=1)
    return distances


def symmetric_nn_metrics(Pa: np.ndarray, Pb: np.ndarray,
                         overlap_threshold: float) -> Dict[str, float]:
    if len(Pa) == 0 or len(Pb) == 0:
        raise ValueError("One of the filtered point clouds is empty")

    tree_b = cKDTree(Pb)
    d_ab = nearest_neighbour_distances(tree_b, Pa)

    tree_a = cKDTree(Pa)
    d_ba = nearest_neighbour_distances(tree_a, Pb)

    d_all = np.concatenate([d_ab, d_ba])

    return {
        "n_points_0": int(len(Pa)),
        "n_points_1": int(len(Pb)),
        "mean_nn_0_to_1": float(np.mean(d_ab)),
        "mean_nn_1_to_0": float(np.mean(d_ba)),
        "median_nn_0_to_1": float(np.median(d_ab)),
        "median_nn_1_to_0": float(np.median(d_ba)),
        "p90_nn_0_to_1": float(np.quantile(d_ab, 0.9)),
        "p90_nn_1_to_0": float(np.quantile(d_ba, 0.9)),
        "chamfer_l1": float(np.mean(d_ab) + np.mean(d_ba)),
        "chamfer_l2": float(np.mean(d_ab ** 2) + np.mean(d_ba ** 2)),
        "rmse_symmetric": float(np.sqrt(np.mean(d_all ** 2))),
        "inlier_ratio_0_to_1": float(np.mean(d_ab <= overlap_threshold)),
        "inlier_ratio_1_to_0": float(np.mean(d_ba <= overlap_threshold)),
        "overlap_ratio": float(
            0.5 * (np.mean(d_ab <= overlap_threshold) + np.mean(d_ba <= overlap_threshold))
        ),
    }


def format_metric(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute merged-cloud consistency metrics")
    parser.add_argument("--pair-dir", required=True,
                        help="Pair folder containing image*_points.npy, image*.txt and result JSONs")
    parser.add_argument("--dataset", choices=["adt", "kitti"], default="adt")
    parser.add_argument("--mask", default=None,
                        help="Optional fisheye mask PNG to apply before scoring")
    parser.add_argument("--distance-threshold", type=float, default=1000.0,
                        help="Keep only points within this distance from the camera")
    parser.add_argument("--overlap-threshold", type=float, default=0.05,
                        help="Distance threshold (metres) for overlap / inlier ratio")
    parser.add_argument("--max-points", type=int, default=None,
                        help="Optional random subsampling cap per cloud before NN scoring")
    parser.add_argument("--output", default=None,
                        help="Optional path to save the metrics JSON")
    args = parser.parse_args()

    pair_dir = Path(args.pair_dir)
    point0_path = pair_dir / "image0_points.npy"
    point1_path = pair_dir / "image1_points.npy"
    cam2w0_path = pair_dir / "image0.txt"
    cam2w1_path = pair_dir / "image1.txt"

    if not point0_path.exists() or not point1_path.exists():
        raise FileNotFoundError(
            f"Expected point clouds at {point0_path} and {point1_path}. "
            "Use a rendered_example/<pair_name>/ folder."
        )

    P0_raw = load_cloud(point0_path)
    P1_raw = load_cloud(point1_path)

    mask = None
    if args.mask:
        size = 1408 if args.dataset == "adt" else 1400
        mask = load_mask(Path(args.mask), size=size, dataset=args.dataset)

    P0 = filter_cloud(P0_raw, args.distance_threshold, mask)
    P1 = filter_cloud(P1_raw, args.distance_threshold, mask)
    P0 = maybe_subsample(P0, args.max_points, seed=0)
    P1 = maybe_subsample(P1, args.max_points, seed=1)

    R_gt, t_gt = load_gt_pose(cam2w0_path, cam2w1_path)

    summary: Dict[str, Dict[str, float]] = {}

    print()
    print(f"Pair: {pair_dir.name}")
    print(f"Filtered points: image0={len(P0)}, image1={len(P1)}")
    if args.max_points:
        print(f"Subsample cap: {args.max_points} points per cloud")
    print(f"Overlap threshold: {args.overlap_threshold} m")
    print()
    print(
        f"{'Method':<15} {'dR(deg)':>10} {'dt(m)':>10} {'ChamferL1':>12} "
        f"{'RMSE':>10} {'Overlap':>10}"
    )
    print("-" * 72)

    for method_name, json_name in DEFAULT_METHODS:
        if json_name is None:
            R_est, t_est, scale = R_gt, t_gt, 1.0
        else:
            result = load_method_result(pair_dir / json_name)
            if result is None:
                continue
            R_est, t_est, scale = result

        P0_aligned = scale * (P0 @ R_est.T) + t_est
        metrics = symmetric_nn_metrics(P0_aligned, P1, args.overlap_threshold)
        dt_err, dR_err = compute_pose_error(R_gt, t_gt, R_est, t_est)

        metrics["rotation_error_deg"] = dR_err
        metrics["translation_error"] = dt_err
        metrics["scale_est"] = float(scale)
        summary[method_name] = metrics

        print(
            f"{method_name:<15} {format_metric(dR_err, 2):>10} {format_metric(dt_err, 4):>10} "
            f"{format_metric(metrics['chamfer_l1'], 4):>12} {format_metric(metrics['rmse_symmetric'], 4):>10} "
            f"{format_metric(metrics['overlap_ratio'], 3):>10}"
        )

    if not summary:
        raise RuntimeError("No valid method result JSONs were found in the pair directory")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print()
        print(f"Saved metrics JSON to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

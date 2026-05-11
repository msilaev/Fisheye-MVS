"""
MADPose baseline: hybrid depth+epipolar pose estimation.
Uses UniK3D depth maps + SuperGlue keypoint matches already produced
by earlier pipeline steps.

Usage:
  python run_madpose_baseline.py \
      --image0     <pair_dir>/image0.jpg \
      --image1     <pair_dir>/image1.jpg \
      --mkpts0     <results_dir>/mkpts1.npy \
      --mkpts1     <results_dir>/mkpts2.npy \
      --depth0     <results_dir>/image0_depth.npy \
      --depth1     <results_dir>/image1_depth.npy \
      --output     <results_dir>/result_madpose.json
"""

import argparse
import json
import sys

import cv2
import numpy as np
import madpose
from madpose.utils import get_depths


def main(args):
    image0 = cv2.imread(args.image0)
    image1 = cv2.imread(args.image1)
    if image0 is None or image1 is None:
        raise FileNotFoundError(f"Could not read images: {args.image0}, {args.image1}")

    mkpts0 = np.load(args.mkpts0)
    mkpts1 = np.load(args.mkpts1)
    depth_map0 = np.load(args.depth0)
    depth_map1 = np.load(args.depth1)

    if len(mkpts0) < 8:
        print(f"WARNING: only {len(mkpts0)} matches — writing null result")
        with open(args.output, "w") as f:
            json.dump({"R_est": None, "t_est": None, "error": "too few matches"}, f, indent=2)
        return

    depth0 = get_depths(image0, depth_map0, mkpts0)
    depth1 = get_depths(image1, depth_map1, mkpts1)

    pp0 = (np.array(image0.shape[:2][::-1]) - 1) / 2
    pp1 = (np.array(image1.shape[:2][::-1]) - 1) / 2

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
        mkpts0, mkpts1,
        depth0, depth1,
        [depth_map0.min(), depth_map1.min()],
        pp0, pp1,
        options, est_config,
    )

    R_est = pose.R()
    t_est = pose.t()
    focal = pose.focal
    print(f"MADPose R=\n{R_est}\nt={t_est}\nfocal={focal:.1f}")

    with open(args.output, "w") as f:
        json.dump({
            "R_est":   R_est.tolist(),
            "t_est":   t_est.tolist(),
            "scale":   float(pose.scale),
            "focal":   float(focal),
        }, f, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image0",  required=True)
    p.add_argument("--image1",  required=True)
    p.add_argument("--mkpts0",  required=True, help="mkpts for image0 (.npy)")
    p.add_argument("--mkpts1",  required=True, help="mkpts for image1 (.npy)")
    p.add_argument("--depth0",  required=True, help="depth map for image0 (.npy)")
    p.add_argument("--depth1",  required=True, help="depth map for image1 (.npy)")
    p.add_argument("--output",  required=True, help="output JSON path")
    main(p.parse_args())

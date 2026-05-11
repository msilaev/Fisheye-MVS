import argparse
import json
import os

import cv2
import numpy as np

import madpose
from madpose.utils import bougnoux_numpy, compute_pose_error, get_depths

def main(args):

    reproj_pix_thres = 16.0
    epipolar_pix_thres = 1.0

    # Weight for epipolar error
    epipolar_weight = 1.0

    # Ransac Options and Estimator Configs
    options = madpose.HybridLORansacOptions()
    options.min_num_iterations = 100
    options.max_num_iterations = 1000
    options.final_least_squares = True
    options.threshold_multiplier = 5.0
    options.num_lo_steps = 4
    options.squared_inlier_thresholds = [reproj_pix_thres**2, epipolar_pix_thres**2]
    options.data_type_weights = [1.0, epipolar_weight]
    options.random_seed = 0

    est_config = madpose.EstimatorConfig()
    est_config.min_depth_constraint = True
    est_config.use_shift = True
    est_config.ceres_num_threads = 8

    with open(args.image_pair_file, "r") as f:
        for line in f:
            image0_name, image1_name = line.split()
            break

    print(f"Image 1 path {os.path.join(args.image_input_dir, image0_name)}")
    print(f"Image 2 path {os.path.join(args.image_input_dir, image1_name)}")

    image0 = cv2.imread(os.path.join(args.image_input_dir, image0_name))
    image1 = cv2.imread(os.path.join(args.image_input_dir, image1_name))

    matches_0_file = args.matches_0_file
    matches_1_file = args.matches_1_file

    mkpts0 = np.load(matches_0_file)
    mkpts1 = np.load(matches_1_file)

    depth_0_file = args.depth_0_file
    depth_1_file = args.depth_1_file

    depth_map0 = np.load(depth_0_file)
    depth_map1 = np.load(depth_1_file)

    print(f"mkpts0 file: {matches_0_file}")
    print(f"mkpts0: {mkpts0.shape}")
    print(f"depth_map0: {depth_map0.shape}")

    # Query the depth priors of the keypoints
    depth0 = get_depths(image0, depth_map0, mkpts0)
    depth1 = get_depths(image1, depth_map1, mkpts1)

    #############################################3
    # Save matching points with their depth
    matched_points0 = np.hstack([mkpts0, depth0[:, None]])  # Nx3: x, y, depth
    matched_points1 = np.hstack([mkpts1, depth1[:, None]])

    np.save(args.mp_depth_0_file, matched_points0)
    np.save(args.mp_depth_1_file, matched_points1)

    # Compute the principal points
    pp0 = (np.array(image0.shape[:2][::-1]) - 1) / 2
    pp1 = (np.array(image1.shape[:2][::-1]) - 1) / 2

    print(f"Principal point 1 {pp0}")
    print(f"Principal point 2 {pp1}")

    # Run hybrid estimation
    '''
    pose, stats = madpose.HybridEstimatePoseScaleOffsetTwoFocal(
        mkpts0,
        mkpts1,
        depth0,
        depth1,
        [depth_map0.min(), depth_map1.min()],
        pp0,
        pp1,
        options,
        est_config,
    )
    '''

    # Run hybrid estimation
    pose, stats = madpose.HybridEstimatePoseScaleOffsetSharedFocal(
        mkpts0,
        mkpts1,
        depth0,
        depth1,
        [depth_map0.min(), depth_map1.min()],
        pp0,
        pp1,
        options,
        est_config,
    )
    # rotation and translation of the estimated pose
    R_est, t_est = pose.R(), pose.t()
    # scale and offsets of the affine corrected depth maps
    s_est, o0_est, o1_est = pose.scale, pose.offset0, pose.offset1
    # the estimated two focal lengths
    f0_est = pose.focal

    print(f"R = {R_est}")
    print(f"t = {t_est}")
    print(f"s = {s_est}")
    print(f"f = {f0_est}")


    # -----------------------------------------
    # Save everything as JSON
    # -----------------------------------------

    result = {
        "rotation": R_est.tolist(),
        "translation": t_est.tolist(),
        "scale": float(s_est),
        "offset0": float(o0_est),
        "offset1": float(o1_est),
        "focal": float(f0_est) ,
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.madpose_est_file), exist_ok=True)

    with open(args.madpose_est_file, "w") as f:
        json.dump(result, f, indent=4)

    print(f"[INFO] Saved MADPose results to: {args.madpose_est_file}")



if __name__=="__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument("--mp_depth_0_file", type=str, required=True)
    parser.add_argument("--mp_depth_1_file", type=str, required=True)
    parser.add_argument("--matches_0_file", type=str, required=True)
    parser.add_argument("--matches_1_file", type=str, required=True)

    parser.add_argument("--depth_0_file", type=str, required=True)
    parser.add_argument("--depth_1_file", type=str, required=True)

    parser.add_argument("--image_input_dir", type=str, required=True)

    parser.add_argument("--image_pair_file", type=str, required=True)

    parser.add_argument("--madpose_est_file", type=str, required=True)

    #parser.add_argument("--image0", type=str, required=True)
    #parser.add_argument("--image1", type=str, required=True)

    args=parser.parse_args()

    main(args)

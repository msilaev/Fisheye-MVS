import argparse
from scipy.linalg import orthogonal_procrustes
import numpy as np
import os
import json
from PIL import Image
from utils import compute_pose_error

def save_pose(args, t_est, R_est, scale_est):

    pose={}

    pose["scale_est"] = float(scale_est)
    pose["R_est"] = R_est.tolist()
    pose["t_est"] = t_est.tolist()

    with open(args.remote_transform_result_path, "w") as f:
        json.dump(pose, f, indent=2)
    
    print(f"updated test json to {args.remote_transform_result_path}")


def load_unik3d_cloud(path):
    """Load a Unik3D point cloud (1,3,H,W) and return Nx3."""
    P = np.load(path)
    print(f"Loaded {path}, raw shape: {P.shape}")

    if P.ndim != 4 or P.shape[0] != 1 or P.shape[1] != 3:
        raise ValueError(f"Invalid shape {P.shape}, expected (1,3,H,W)")

    P = P[0]                     # (3,H,W)
    P = P.transpose(0,2,1)
    P = P.reshape(3, -1).T       # → (H*W, 3)

    return P

def procrustes(data1, data2):

    mtx1 = np.array(data1, dtype=np.float64, copy=True)
    mtx2 = np.array(data2, dtype=np.float64, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mu1 = np.mean(mtx1, 0)
    mu2 = np.mean(mtx2, 0)

    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx2, mtx1)
    mtx1 = np.dot(mtx1, R.T) * s

    return (mu1, norm1, mtx1), (mu2, norm2, mtx2), s*norm2/norm1, R, mu1


def filter_mkpts(args, path_P, mkpts):

    P = np.load(path_P)

    mask = get_mask_kitti(args)
    mkpts_mask = []

    for p in mkpts:
        #print("p", p)
        if mask[p[0], p[1]]:
            mkpts_mask.append(p)

    mkpts_mask = np.array(mkpts_mask)

    P_1 = P[0]

    P_1 = P_1.transpose(0,2,1)

    P_1 = P_1[:, mkpts_mask[:, 0], mkpts_mask[:, 1]]
    #P_1 = P_1[:, mkpts[:, 0], mkpts[:, 1]]

    P_1 = P_1.reshape(3,-1).T

    return P_1

def get_mask_kitti(args):

    mask = Image.open(args.remote_fisheye_mask_path)
    mask = mask.resize((1400, 1400), Image.NEAREST)  # NEAREST preserves binary edges

    mask = np.array(mask) / 255.0  # (H, W) or (H, W, 3)
    print(f"mask shape = {mask.shape}")
    mask = mask.transpose(2, 1, 0)

    print(f"mask shape = {mask.shape}")

    #mask = mask.reshape(3, -1).T
    mask = mask[2, :, :]  # use any channel
    mask = (mask > 0.5).astype(bool)  # threshold away black area

    print(f"Mask shape {mask.shape}, mask[0] {mask[0]}")

    return mask

def pose_est(args):

    mkpts1 = np.load(args.mkpts1)
    mkpts2 = np.load(args.mkpts2)

    mkpts1, mkpts2 = get_filtered_mkpts(args, mkpts1, mkpts2)

    P1_filtered = filter_mkpts(args, args.point1, mkpts1)
    P2_filtered = filter_mkpts(args, args.point2, mkpts2)

    distances_P = []
    for p in P1_filtered:
        distances_P.append(np.linalg.norm(p))
    distances_P = np.array(distances_P)

    P1_filtered = P1_filtered[distances_P < args.distance_threshold]
    P2_filtered = P2_filtered[distances_P < args.distance_threshold]

    (mu1, norm1, mtx1), (mu2, norm2, mtx2), scale, R, _ = procrustes(P1_filtered, P2_filtered)
    t= scale * np.dot((- mu1), R.T) + mu2

    return R, t, scale

def get_filtered_mkpts(args, mkpts1, mkpts2 ):

    mask = get_mask_kitti(args)

    valid = []
    for i in range(len(mkpts1)):
        p1 = mkpts1[i]
        p2 = mkpts2[i]

        if mask[p1[0], p1[1]] and mask[p2[0], p2[1]]:
            valid.append(i)

    return mkpts1[valid], mkpts2[valid]


def main(args):  

    R_est, t_est, scale_est = pose_est(args)

    save_pose(args, t_est, R_est, scale_est)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--point1", type=str, required=True, help="Unik3D .npy point cloud 1")
    parser.add_argument("--point2", type=str, required=True, help="Unik3D .npy point cloud 2")

    parser.add_argument("--mkpts1", type=str, required=True, help="1")
    parser.add_argument("--mkpts2", type=str, required=True, help="2")

    parser.add_argument("--img1", type=str, required=True, help="1")
    parser.add_argument("--img2", type=str, required=True, help="2")
    
    parser.add_argument("--distance_threshold", type=float)
    parser.add_argument("--remote_transform_result_path", type=str, required=True)
    parser.add_argument("--remote_fisheye_mask_path", type=str, required=True)

    main(parser.parse_args())


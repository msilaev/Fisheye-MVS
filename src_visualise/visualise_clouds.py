import argparse
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
import open3d as o3d
import numpy as np
import os
import json
import imageio
from PIL import Image

import matplotlib.pyplot as plt

SIZE_X=3115
SIZE_Y=2078


def get_pose(args):

    with open(args.remote_transform_result_path, "r") as f:
        pose = json.load(f)

    R_est = np.array(pose["R_est"])
    t_est = np.array(pose["t_est"])
    scale_est = pose["scale_est"]
    return R_est, t_est, scale_est
    
    

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


def get_mask(args):

    mask = Image.open(args.mask_fisheye)
    mask = mask.resize((1400, 1400), Image.NEAREST)  # NEAREST preserves binary edges

    mask = np.array(mask) / 255.0  # (H, W) or (H, W, 3)
    #print(f"mask shape = {mask.shape}")
    mask = mask.transpose(2, 1, 0)
    mask = mask.reshape(3, -1).T
    mask = mask[:, 2]  # use any channel
    #print(f"Mask shape {mask.shape}, mask[0] {mask[0]}")
    mask = (mask > 0.5).astype(bool)  # threshold away black area

    return mask

def show_clouds(pcd_est, args):

    # --- Visualize ---
    vis = o3d.visualization.Visualizer()
    vis.create_window("Colored Point Cloud")
    vis.add_geometry(pcd_est)

    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0, 0, 0])

    vis.run()
    vis.destroy_window()

def main(args):

    P1 = load_unik3d_cloud(args.point1)
    P2 = load_unik3d_cloud(args.point2)

    img1 = np.array(Image.open(args.img1))/255.0
    img2 = np.array(Image.open(args.img2))/255.0

    img1 = img1.transpose(2, 1,0)
    img2 = img2.transpose(2, 1, 0)

    colors1 = img1.reshape(3, -1).T
    colors2 = img2.reshape(3, -1).T

    mask = get_mask(args)

    P1 = P1[mask]
    P2 = P2[mask]

    colors1 = colors1[mask]
    colors2 = colors2[mask]

    distances_P2 = []
    for p in P2:
        distances_P2.append(np.linalg.norm(p))
    distances_P2 = np.array(distances_P2)

    distances_P1 = []
    for p in P1:
        distances_P1.append(np.linalg.norm(p))
    distances_P1 = np.array(distances_P1)

    P1 = P1[distances_P1 < args.distance_threshold_plt]
    P2 = P2[distances_P2 < args.distance_threshold_plt]

    colors1 = colors1[distances_P1 < args.distance_threshold_plt]
    colors2 = colors2[distances_P2 < args.distance_threshold_plt]

    R_est, t_est, scale_est = get_pose(args)

    print(f"R_est={R_est}")
    print(f"t_est={t_est}")
    print(f"scale_est={scale_est}")

    ############################33
    P1_trans_est = scale_est * np.dot(P1 , R_est.T) + t_est

    #points_gt = np.vstack((P1_trans_gt, P2))
    points_est = np.vstack((P1_trans_est, P2))
    colors = np.vstack((colors1, colors2))

    pcd_est = o3d.geometry.PointCloud()
    pcd_est.points = o3d.utility.Vector3dVector(points_est)
    pcd_est.colors = o3d.utility.Vector3dVector(colors)

    show_clouds(pcd_est, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--point1", type=str, required=True, help="Unik3D .npy point cloud 1")
    parser.add_argument("--point2", type=str, required=True, help="Unik3D .npy point cloud 2")


    parser.add_argument("--img1", type=str, required=True, help="1")
    parser.add_argument("--img2", type=str, required=True, help="2")


    parser.add_argument("--point_size", type=float, default=2.0)

    parser.add_argument("--mask_fisheye", type=str)

    parser.add_argument("--distance_threshold_plt", type=float)

    main(parser.parse_args())


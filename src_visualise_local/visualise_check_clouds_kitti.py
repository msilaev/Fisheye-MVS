import argparse
import os
from scipy.linalg import orthogonal_procrustes
import open3d as o3d
import numpy as np
import json
from PIL import Image
from utils import compute_pose_error

#import matplotlib.pyplot as plt

SIZE_X=3115
SIZE_Y=2078

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

    #P_1 = P_1[:, mkpts_mask[:, 0], mkpts_mask[:, 1]]
    P_1 = P_1[:, mkpts[:, 0], mkpts[:, 1]]

    P_1 = P_1.reshape(3,-1).T

    return P_1

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

def save_camera(vis, filename="camera.json"):
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, params)
    print(f"[INFO] Saved camera to {filename}")

def load_camera(vis, filename="camera.json"):
    ctr = vis.get_view_control()
    params = o3d.io.read_pinhole_camera_parameters(filename)
    try:
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    except TypeError:
        ctr.convert_from_pinhole_camera_parameters(params)
    vis.update_renderer()
    print(f"[INFO] Loaded camera from {filename}")

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

def pose_est(args):

    if args.transform_json:
        with open(args.transform_json) as f:
            data = json.load(f)
        R = np.array(data["R_est"])
        t = np.array(data["t_est"])
        scale = data.get("scale_est", data.get("scale", 1.0))
        return R, t, scale

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

    #mkpts1 = np.load(args.mkpts1)
    #mkpts2 = np.load(args.mkpts2)

    valid = []
    for i in range(len(mkpts1)):
        p1 = mkpts1[i]
        p2 = mkpts2[i]

        if mask[p1[0], p1[1]] and mask[p2[0], p2[1]]:
            valid.append(i)

    return mkpts1[valid], mkpts2[valid]


def pose_gt(args):

    T1 = np.loadtxt(args.cam2w_1)
    T2 = np.loadtxt(args.cam2w_2)

    T_1to2_gt = np.linalg.inv(T2) @ T1

    R_gt = T_1to2_gt[:3, :3]
    t_gt = T_1to2_gt[:3, 3]

    return R_gt, t_gt

def cloud_est():
    pass

def cloud_gt():
    pass


def get_mask_kitti(args):
    '''
    This function filters out the frame of the camera
    :param args:
    :return:
    '''

    mask = Image.open(args.mask_fisheye_kitti)
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


def get_mask(args):

    mask = Image.open(args.mask_fisheye_kitti)
    mask = mask.resize((1400, 1400), Image.NEAREST)  # NEAREST preserves binary edges

    mask = np.array(mask) / 255.0  # (H, W) or (H, W, 3)
    print(f"mask shape = {mask.shape}")
    mask = mask.transpose(2, 1, 0)
    mask = mask.reshape(3, -1).T
    mask = mask[:, 2]  # use any channel
    print(f"Mask shape {mask.shape}, mask[0] {mask[0]}")
    mask = (mask > 0.5).astype(bool)  # threshold away black area

    return mask

def choose_start_pose(pcd_est, args):

    # --- Visualize ---
    vis = o3d.visualization.Visualizer()
    vis.create_window("Colored Point Cloud")
    vis.add_geometry(pcd_est)
    if args.camera_param and os.path.exists(args.camera_param):
        load_camera(vis, args.camera_param)

    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0, 0, 0])

    vis.run()
    if args.camera_param:
        save_camera(vis, args.camera_param)
    vis.destroy_window()


def main(args):

    # --- Load raw clouds ---
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


    ####################################

    distances_P2 = []
    for p in P2:
        distances_P2.append(np.linalg.norm(p))
    distances_P2 = np.array(distances_P2)

    distances_P1 = []
    for p in P1:
        distances_P1.append(np.linalg.norm(p))
    distances_P1 = np.array(distances_P1)

    P1 = P1[distances_P1 < args.distance_threshold]
    P2 = P2[distances_P2 < args.distance_threshold]

    colors1 = colors1[distances_P1 < args.distance_threshold]
    colors2 = colors2[distances_P2 < args.distance_threshold]

    ###################################



    #colors1[:]= [1,0,0]
    #colors2[:] = [0, 0, 1]

    R_gt, t_gt = pose_gt(args)

    R_est, t_est, scale_est = pose_est(args)

    print(f"R_est={R_est}")
    print(f"t_est={t_est}")
    print(f"scale_est={scale_est}")


    print(f"R_gt={R_gt}")
    print(f"t_gt={t_gt}")

    d_t12, d_R12 = compute_pose_error(R_est, t_est, np.eye(3), 0 * t_est)
    print(f"relative distance estimated d_R12 = {d_R12}, d_t12 = {d_t12}")

    d_t12, d_R12 = compute_pose_error(R_gt, t_gt, np.eye(3), 0 * t_est)
    print(f"relative distance GT d_R12 = {d_R12}, d_t12 = {d_t12}")

    d_t, d_R = compute_pose_error(R_gt, t_gt, R_est, t_est)
    print(f"error d_R = {d_R}, d_t = {d_t}")

    ############################33
    P1_trans_est = scale_est * np.dot(P1 , R_est.T) + t_est

    #P1_trans_est = P1
    #P1_trans_gt =  np.dot( P1, R_gt.T) +  t_gt  #scale * np.dot( (P1 - mu1), np.linalg.inv(R_gt).T) + mu2

    #points_gt = np.vstack((P1_trans_gt, P2))
    points_est = np.vstack((P1_trans_est, P2))
    colors = np.vstack((colors1, colors2))

    pcd_gt = o3d.geometry.PointCloud()
    #pcd_gt.points = o3d.utility.Vector3dVector(points_gt)
    pcd_gt.colors = o3d.utility.Vector3dVector(colors)

    pcd_est = o3d.geometry.PointCloud()
    pcd_est.points = o3d.utility.Vector3dVector(points_est)
    pcd_est.colors = o3d.utility.Vector3dVector(colors)

    choose_start_pose(pcd_est, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--point1", type=str, required=True, help="Unik3D .npy point cloud 1")
    parser.add_argument("--point2", type=str, required=True, help="Unik3D .npy point cloud 2")

    parser.add_argument("--mkpts1", type=str, help="1")
    parser.add_argument("--mkpts2", type=str, help="2")
    parser.add_argument("--transform_json", type=str, help="JSON with ready R_est, t_est, scale_est")

    parser.add_argument("--cam2w_1", type=str, required=True, help="1")
    parser.add_argument("--cam2w_2", type=str, required=True, help="2")

    parser.add_argument("--img1", type=str, required=True, help="1")
    parser.add_argument("--img2", type=str, required=True, help="2")

    parser.add_argument("--point_size", type=float, default=2.0)

    parser.add_argument("--img_rendered_gt", type=str)
    parser.add_argument("--img_rendered_est", type=str)

    parser.add_argument("--mask_fisheye_kitti", type=str)

    parser.add_argument("--camera_param", type=str)

    parser.add_argument("--distance_threshold", type=float, default=10.0, help="Max distance from camera to keep points")

    main(parser.parse_args())


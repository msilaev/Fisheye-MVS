import argparse
import json
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

import open3d as o3d
import numpy as np
from PIL import Image

from utils import compute_pose_error

SIZE_X=3115
SIZE_Y=2078

def main(args):

    T1 = np.loadtxt(args.cam2w_1)
    T2 = np.loadtxt(args.cam2w_2)

    T1_inv = np.linalg.inv(T1)
    T_1to2_gt = T2 @ T1_inv

    T_1to2_gt = np.linalg.inv(T2) @ T1

    # Extract Ground Truth Rotation (R_gt) and Translation (t_gt)
    R_gt = T_1to2_gt[:3, :3]
    t_gt = T_1to2_gt[:3, 3]

    print(f"R_GT = {R_gt}")
    print(f"t_GT = {t_gt}")

'''
SEQ_NAME=Apartment_release_clean_seq136_M1292
LOCAL_RESULTS_DIR="/worktmp/THESES/GAUSSIAN-SPLATTING/experiments/
RESULT_MADPOSE/result_madpose_${SEQ_NAME}"
cam2w_1="${LOCAL_IMAGE_DIR}/image0.txt" 
cam2w_2="${LOCAL_IMAGE_DIR}/image1.txt" 
python 
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cam2w_1", type=str, required=True, help="1")
    parser.add_argument("--cam2w_2", type=str, required=True, help="2")

    main(parser.parse_args())



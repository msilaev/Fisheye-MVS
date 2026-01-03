#!/bin/bash
set -e
set -x

###############################################
# 1. Load environment variables
###############################################
if [ -f ".env" ]; then
    source .env
fi

REMOTE_USER="mikhail"
REMOTE_HOST="wks-104983-npc.pit.cs.tut.fi"

###############################################
# 2. Remote configuration
###############################################

REMOTE_DIR="/home/hdd/mikhail/GAUSSIAN-SPLATTING/experiments"
REMOTE_LOG_DIR="${REMOTE_DIR}/log_madpose"
REMOTE_RESULTS_DIR="${REMOTE_DIR}/result_madpose"
REMOTE_SCRIPT_DIR="${REMOTE_DIR}/src_madpose"

IMAGE_DIR="IMAGES_DIR_unik3d"
REMOTE_DIR_EXP="${REMOTE_DIR}/${IMAGE_DIR}"

LOCAL_DIR_EXP="/worktmp/THESES/GAUSSIAN-SPLATTING/experiments"
LOCAL_SRC_DIR="/worktmp/THESES/GAUSSIAN-SPLATTING/MADpose-pipeline/src_visualise"

LOCAL_RESULTS_DIR="/worktmp/THESES/GAUSSIAN-SPLATTING/experiments/RESULT_MADPOSE"
LOCAL_RESULTS_DIR="/worktmp/THESES/GAUSSIAN-SPLATTING/experiments/RESULT_MADPOSE/result_madpose"

LOCAL_RESULTS_DIR="/worktmp/THESES/GAUSSIAN-SPLATTING/experiments/RESULT_MADPOSE/result_madpose_kitti_frames_6360_6367"
#LOCAL_RESULTS_DIR="/worktmp/THESES/GAUSSIAN-SPLATTING/experiments/RESULT_MADPOSE/result_madpose_kitti_frames_8603_8587"

LOCAL_MASK_DIR="/worktmp/THESES/GAUSSIAN-SPLATTING/experiments/RESULT_MADPOSE/fisheye_masks"

image_pair_file="${REMOTE_DIR_EXP}/image_pairs.txt"
image_input_dir="${REMOTE_DIR_EXP}"

LOCAL_IMAGE_DIR="$LOCAL_DIR_EXP/$IMAGE_DIR"

LOCAL_IMAGE_DIR=$LOCAL_RESULTS_DIR
DISTANCE_THRESHOLD=20
DISTANCE_THRESHOLD_PLT=100

###############################################
# 3. Activate conda
###############################################
eval "$(conda shell.bash hook)"
conda activate unik3d

###############################################
# 4. Copy results from remote → local
###############################################
mkdir -p "$LOCAL_RESULTS_DIR"

#scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_RESULTS_DIR}/" "$LOCAL_RESULTS_DIR/"
#scp -r "${REMOTE_USER}@${REMOTE_HOST}:${image_input_dir}/." "$LOCAL_RESULTS_DIR/result_madpose/"

###############################################
# 5. Run visualisation
###############################################
cd $LOCAL_SRC_DIR
python visualise_check_clouds_kitti.py \
    --point1 "${LOCAL_RESULTS_DIR}/image0_points.npy" \
    --point2 "${LOCAL_RESULTS_DIR}/image1_points.npy" \
    --mkpts1 "${LOCAL_RESULTS_DIR}/mkpts1.npy" \
    --mkpts2 "${LOCAL_RESULTS_DIR}/mkpts2.npy" \
    --img1 "${LOCAL_IMAGE_DIR}/image0.png" \
    --img2 "${LOCAL_IMAGE_DIR}/image1.png" \
    --cam2w_1 "${LOCAL_IMAGE_DIR}/image0.txt" \
    --cam2w_2 "${LOCAL_IMAGE_DIR}/image1.txt" \
    --mask_fisheye  "${LOCAL_MASK_DIR}/mask_fisheye_rot.png" \
    --mask_fisheye_kitti  "${LOCAL_MASK_DIR}/MaskKitti360.png" \
    --camera_param  "${LOCAL_RESULTS_DIR}/vis_frames_camera_param.json" \
    --img_rendered_gt "${LOCAL_RESULTS_DIR}/image_rendered_gt.jpg" \
    --img_rendered_est "${LOCAL_RESULTS_DIR}/image_rendered_est.jpg" \
    --distance_threshold $DISTANCE_THRESHOLD \
    --distance_threshold_plt $DISTANCE_THRESHOLD_PLT


  #--point1 "${LOCAL_SRC_DIR}/mkpts1.npy" \
  #--point2 "${LOCAL_SRC_DIR}/mkpts2.npy"
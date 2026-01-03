#!/bin/bash
set -e
set -x

if [ -f ".env" ]; then
    source .env
fi

# Required env vars:
#   REMOTE_USER
#   REMOTE_HOST

if [ -z "$REMOTE_USER" ] || [ -z "$REMOTE_HOST" ]; then
    echo "ERROR: REMOTE_USER or REMOTE_HOST not defined (set in .env)"
    exit 1
fi

DISTANCE_THRESHOLD_PLT=50
REMOTE_DIR_ROOT="/home/hdd/mikhail/GAUSSIAN-SPLATTING/experiments"
EXPERIMENT_NAME="KITTI-360"
IMAGE_DIR="IMAGES_DIR_experiment_1"

REMOTE_RESULTS_DIR="${REMOTE_DIR_ROOT}/${EXPERIMENT_NAME}/results"
remote_transform_result_path="${REMOTE_RESULTS_DIR}/transform_result.json"

REMOTE_SCRIPT_DIR="${REMOTE_DIR_ROOT}/src_remote"
REMOTE_FISHEYE_MASK_DIR="${REMOTE_DIR_ROOT}/fisheye_masks"

LOCAL_ROOT="/worktmp/THESES/GAUSSIAN-SPLATTING/"
LOCAL_SCRIPT_DIR="${LOCAL_ROOT}/Fisheye-MVS/"
LOCAL_SCRIPT_DIR_visualise="${LOCAL_ROOT}/Fisheye-MVS/src_visualise/"
LOCAL_FISHEYE_MASK_DIR="${LOCAL_SCRIPT_DIR}/assets/fisheye_masks"
LOCAL_RESULTS_DIR="${LOCAL_ROOT}/experiments/${EXPERIMENT_NAME}/results"

local_transform_result_path="${LOCAL_RESULTS_DIR}/transform_result.json"


scp "$REMOTE_USER@$REMOTE_HOST:$REMOTE_RESULTS_DIR/" "$LOCAL_RESULTS_DIR"


cd "$LOCAL_SCRIPT_DIR_visualise"
python visualise_clouds.py \
    --point1 "${LOCAL_RESULTS_DIR}/image0_points.npy" \
    --point2 "${LOCAL_RESULTS_DIR}/image1_points.npy" \
    --img1 "${LOCAL_RESULTS_DIR}/image0.png" \
    --img2 "${LOCAL_RESULTS_DIR}/image1.png" \
    --mask_fisheye  "${LOCAL_MASK_DIR}/MaskKitti360.png" \
    --local_transform_result_path "${local_transform_result_path}" \
    --distance_threshold_plt $DISTANCE_THRESHOLD_PLT
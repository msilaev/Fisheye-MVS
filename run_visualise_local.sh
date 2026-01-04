#!/bin/bash
set -e
set -x

sourse .env

# Required env vars:
#   REMOTE_USER
#   REMOTE_HOST

source path_config.sh

LOCAL_ROOT="/worktmp/THESES/GAUSSIAN-SPLATTING/"
REMOTE_DIR_ROOT="/home/hdd/mikhail/GAUSSIAN-SPLATTING/experiments"


LOCAL_SCRIPT_DIR="${LOCAL_ROOT}/Fisheye-MVS/"
LOCAL_SCRIPT_DIR_visualise="${LOCAL_ROOT}/Fisheye-MVS/src_visualise/"
LOCAL_FISHEYE_MASK_DIR="${LOCAL_SCRIPT_DIR}/assets/fisheye_masks"

LOCAL_RESULTS_DIR="${LOCAL_ROOT}/experiments/${EXPERIMENT_NAME}/results"
local_transform_result_path="${LOCAL_RESULTS_DIR}/transform_result.json"

sourse experiment_config_adt.sh

EXPERIMENT_NAME="ADT"
DISTANCE_THRESHOLD_PLT=1000
FISHEYE_MASK_FILE="MaskADT_rot.png"
IMAGE_DIR="IMAGES_DIR_experiment_1"
SIZE_X=1408
SIZE_Y=1408
IMG_EXTENSION="jpg"


EXPERIMENT_NAME="KITTI-360"
DISTANCE_THRESHOLD_PLT=50
FISHEYE_MASK_FILE="MaskKitti360.png"
IMAGE_DIR="IMAGES_DIR_experiment_1"
SIZE_X=1400
SIZE_Y=1400
IMG_EXTENSION="png"

local_fisheye_mask_path="${LOCAL_FISHEYE_MASK_DIR}/${FISHEYE_MASK_FILE}"

REMOTE_RESULTS_DIR="${REMOTE_DIR_ROOT}/${EXPERIMENT_NAME}/results"
REMOTE_IMAGE_DIR="${REMOTE_DIR_ROOT}/${EXPERIMENT_NAME}/${IMAGE_DIR}"
remote_transform_result_path="${REMOTE_RESULTS_DIR}/transform_result.json"
REMOTE_SCRIPT_DIR="${REMOTE_DIR_ROOT}/src_remote"
REMOTE_FISHEYE_MASK_DIR="${REMOTE_DIR_ROOT}/fisheye_masks"

#rm -rf $LOCAL_RESULTS_DIR
#scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_RESULTS_DIR" "$LOCAL_RESULTS_DIR"
#rsync -avz "$REMOTE_USER@$REMOTE_HOST:${REMOTE_IMAGE_DIR}/" "$LOCAL_RESULTS_DIR/"


if [ -f "$CONDA_SETUP" ]; then
    source "$CONDA_SETUP"
else
    eval "$(conda shell.bash hook)"
fi

conda deactivate || true
conda activate 3d_pose_env

cd "$LOCAL_SCRIPT_DIR_visualise"
python visualise_clouds.py \
    --point1 "${LOCAL_RESULTS_DIR}/image0_points.npy" \
    --point2 "${LOCAL_RESULTS_DIR}/image1_points.npy" \
    --img1 "${LOCAL_RESULTS_DIR}/image0.${IMG_EXTENSION}" \
    --img2 "${LOCAL_RESULTS_DIR}/image1.${IMG_EXTENSION}" \
    --mask_fisheye "${local_fisheye_mask_path}" \
    --local_transform_result_path "${local_transform_result_path}" \
    --distance_threshold_plt $DISTANCE_THRESHOLD_PLT \
    --size_x $SIZE_X \
    --size_y $SIZE_Y \




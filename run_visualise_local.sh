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

LOCAL_ROOT="/worktmp/THESES/GAUSSIAN-SPLATTING/"
LOCAL_SCRIPT_DIR="${LOCAL_ROOT}/Fisheye-MVS/"
LOCAL_SCRIPT_DIR_visualise="${LOCAL_ROOT}/Fisheye-MVS/src_visualise/"
LOCAL_FISHEYE_MASK_DIR="${LOCAL_SCRIPT_DIR}/assets/fisheye_masks"
REMOTE_DIR_ROOT="/home/hdd/mikhail/GAUSSIAN-SPLATTING/experiments"


#DISTANCE_THRESHOLD_PLT=50
#EXPERIMENT_NAME="KITTI-360"

DISTANCE_THRESHOLD_PLT=1000
EXPERIMENT_NAME="ADT"

IMAGE_DIR="IMAGES_DIR_experiment_1"
LOCAL_RESULTS_DIR="${LOCAL_ROOT}/experiments/${EXPERIMENT_NAME}/results"
local_transform_result_path="${LOCAL_RESULTS_DIR}/transform_result.json"
REMOTE_FISHEYE_MASK_DIR="${REMOTE_DIR_ROOT}/fisheye_masks"

#local_fisheye_mask_path="${LOCAL_FISHEYE_MASK_DIR}/MaskKitti360.png"
#SIZE_X=1400
#SIZE_Y=1400
#img_type="png"

local_fisheye_mask_path="${LOCAL_FISHEYE_MASK_DIR}/MaskADT_rot.png"
SIZE_X=1408
SIZE_Y=1408
img_type="jpg"


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
conda activate unik3d

cd "$LOCAL_SCRIPT_DIR_visualise"
python visualise_clouds.py \
    --point1 "${LOCAL_RESULTS_DIR}/image0_points.npy" \
    --point2 "${LOCAL_RESULTS_DIR}/image1_points.npy" \
    --img1 "${LOCAL_RESULTS_DIR}/image0.${img_type}" \
    --img2 "${LOCAL_RESULTS_DIR}/image1.${img_type}" \
    --mask_fisheye "${local_fisheye_mask_path}" \
    --local_transform_result_path "${local_transform_result_path}" \
    --distance_threshold_plt $DISTANCE_THRESHOLD_PLT \
    --size_x $SIZE_X \
    --size_y $SIZE_Y \


#python visualise_clouds.py \
#    --point1 "${LOCAL_RESULTS_DIR}/image0_points.npy" \
#    --point2 "${LOCAL_RESULTS_DIR}/image1_points.npy" \
#    --img1 "${LOCAL_RESULTS_DIR}/image0.jpg" \
#    --img2 "${LOCAL_RESULTS_DIR}/image1.jpg" \
#    --mask_fisheye  "${local_fisheye_mask_path}" \
#    --local_transform_result_path "${local_transform_result_path}" \
#    --distance_threshold_plt $DISTANCE_THRESHOLD_PLT \
#    --size_x $SIZE_X \
#    --size_y $SIZE_Y \

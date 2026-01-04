#!/bin/bash
set -e
set -x

source .env

# Required env vars:
#   REMOTE_USER
#   REMOTE_HOST

source path_config.sh
source experiment_config_adt.sh

REMOTE_FISHEYE_MASK_DIR="${REMOTE_DIR_ROOT}/fisheye_masks"
remote_fisheye_mask_path="${REMOTE_FISHEYE_MASK_DIR}/${FISHEYE_MASK_FILE}"

REMOTE_LOG_DIR="${REMOTE_DIR_ROOT}/${EXPERIMENT_NAME}/log_procrustes"
REMOTE_RESULTS_DIR="${REMOTE_DIR_ROOT}/${EXPERIMENT_NAME}/results"
remote_transform_result_path="${REMOTE_RESULTS_DIR}/transform_result.json"

REMOTE_IMAGE_DIR="${REMOTE_DIR_ROOT}/${EXPERIMENT_NAME}/${IMAGE_DIR}"
REMOTE_SCRIPT_DIR="${REMOTE_DIR_ROOT}/src_remote"
REMOTE_SCRIPT_DIR_unik3d="${REMOTE_DIR_ROOT}/src_remote"
REMOTE_SCRIPT_DIR_superglue="${REMOTE_DIR_ROOT}/src_remote"
REMOTE_SCRIPT_DIR_procrustes="${REMOTE_DIR_ROOT}/src_remote"


LOCAL_SCRIPT_DIR="${LOCAL_ROOT}/Fisheye-MVS/"
LOCAL_SCRIPT_DIR_unik3d="${LOCAL_ROOT}/Fisheye-MVS/src_unik3d/"
LOCAL_SCRIPT_DIR_superglue="${LOCAL_ROOT}/Fisheye-MVS/src_superglue/"
LOCAL_SCRIPT_DIR_procrustes="${LOCAL_ROOT}/Fisheye-MVS/src_procrustes/"

LOCAL_IMAGE_DIR="${LOCAL_ROOT}/experiments/${EXPERIMENT_NAME}/${IMAGE_DIR}"

REMOTE_FISHEYE_MASK_DIR="${REMOTE_DIR_ROOT}/fisheye_masks"
LOCAL_FISHEYE_MASK_DIR="${LOCAL_SCRIPT_DIR}/assets/fisheye_masks"

ssh "$REMOTE_USER@$REMOTE_HOST" "
    mkdir -p \
        '$REMOTE_LOG_DIR' \
        '$REMOTE_SCRIPT_DIR_unik3d' \
        '$REMOTE_SCRIPT_DIR_superglue' \
        '$REMOTE_SCRIPT_DIR_procrustes' \
        '$REMOTE_IMAGE_DIR' \
        '$REMOTE_RESULTS_DIR' \
        '$REMOTE_FISHEYE_MASK_DIR'
"

rsync -avz "$LOCAL_FISHEYE_MASK_DIR/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_FISHEYE_MASK_DIR/"

rsync -avz "$LOCAL_SCRIPT_DIR_unik3d/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR_unik3d/"

rsync -avz "$LOCAL_SCRIPT_DIR_superglue/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR_superglue/"

rsync -avz "$LOCAL_SCRIPT_DIR_procrustes/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR_procrustes/"

rsync -avz "$LOCAL_SCRIPT_DIR/run_pipeline_remote.sh" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR/run_pipeline_remote.sh"

rsync -avz "$LOCAL_IMAGE_DIR/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_IMAGE_DIR/"

ssh "$REMOTE_USER@$REMOTE_HOST" \
  "nohup bash '$REMOTE_SCRIPT_DIR/run_pipeline_remote.sh' \
  '$REMOTE_LOG_DIR' \
  '$REMOTE_SCRIPT_DIR_unik3d' \
  '$REMOTE_SCRIPT_DIR_superglue' \
  '$REMOTE_SCRIPT_DIR_procrustes' \
  '$REMOTE_IMAGE_DIR' \
  '$REMOTE_RESULTS_DIR' \
  '$REMOTE_SUPER_GLUE_DIR' \
  '$REMOTE_UNIK3D_DIR' \
  '$remote_fisheye_mask_path' \
  '$remote_transform_result_path' \
  '$DISTANCE_THRESHOLD' \
  '$SIZE_X' \
  '$SIZE_Y' \
  > '$REMOTE_LOG_DIR/run_pipeline_remote.log' 2>&1 &"

echo "[LOCAL] Pipeline started on remote host."
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

DISTANCE_THRESHOLD=50
REMOTE_DIR_ROOT="/home/hdd/mikhail/GAUSSIAN-SPLATTING/experiments"
EXPERIMENT_NAME="KITTI-360"
IMAGE_DIR="IMAGES_DIR_experiment_1"

REMOTE_LOG_DIR="${REMOTE_DIR_ROOT}/${EXPERIMENT_NAME}/log_metrics"
remote_transform_result_path="${REMOTE_DIR_ROOT}/${EXPERIMENT_NAME}/transform_result.json"
REMOTE_RESULTS_DIR="${REMOTE_DIR_ROOT}/${EXPERIMENT_NAME}/results"

REMOTE_IMAGE_DIR="${REMOTE_DIR_ROOT}/${EXPERIMENT_NAME}/${IMAGE_DIR}"


REMOTE_SCRIPT_DIR="${REMOTE_DIR_ROOT}/src_remote"
REMOTE_SCRIPT_DIR_unik3d="${REMOTE_DIR_ROOT}/src_remote"
REMOTE_SCRIPT_DIR_superglue="${REMOTE_DIR_ROOT}/src_remote"
REMOTE_SCRIPT_DIR_procrustes="${REMOTE_DIR_ROOT}/src_remote"
REMOTE_SUPER_GLUE_DIR="/home/hdd/mikhail/GAUSSIAN-SPLATTING/SuperGluePretrainedNetwork"
REMOTE_UNIK3D_DIR="/home/hdd/mikhail/GAUSSIAN-SPLATTING/MVF-UniK3D"


LOCAL_ROOT="/worktmp/THESES/GAUSSIAN-SPLATTING/"
LOCAL_SCRIPT_DIR="${LOCAL_ROOT}/Fisheye-MVS/"
LOCAL_SCRIPT_DIR_unik3d="${LOCAL_ROOT}/Fisheye-MVS/src_unik3d/"
LOCAL_SCRIPT_DIR_superglue="${LOCAL_ROOT}/Fisheye-MVS/src_superglue/"
LOCAL_SCRIPT_DIR_procrustes="${LOCAL_ROOT}/Fisheye-MVS/src_procrustes/"

LOCAL_IMAGE_DIR="${LOCAL_ROOT}/${EXPERIMENT_NAME}/${IMAGE_DIR}"

ssh "$REMOTE_USER@$REMOTE_HOST" "
    mkdir -p \
        '$REMOTE_LOG_DIR' \
        '$REMOTE_SCRIPT_DIR_unik3d' \
        '$REMOTE_SCRIPT_DIR_superglue' \
        '$REMOTE_SCRIPT_DIR_procrustes' \
        '$REMOTE_IMAGE_DIR' \
        '$REMOTE_RESULTS_DIR'
"

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
  '$remote_transform_result_path' \
  '$DISTANCE_THRESHOLD' \
  > '$REMOTE_LOG_DIR/run_pipeline_remote.log' 2>&1 &"

echo "[LOCAL] Pipeline started on remote host."
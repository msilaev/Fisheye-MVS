#!/bin/bash
set -e
set -x

###############################################
# 1. Load environment variables
###############################################
if [ -f ".env" ]; then
    source .env
    echo "loaded"
fi

REMOTE_USER="mikhail"
REMOTE_HOST="<remote-workstation>"

###############################################
# 2. Remote configuration
###############################################
REMOTE_DIR="/home/<remote-user>/GAUSSIAN-SPLATTING/experiments"
DATA_DIR="KITTI-360_data"
SEQUENCE_NAME="2013_05_28_drive_0000_sync"


REMOTE_DATA_DIR="${REMOTE_DIR}/$DATA_DIR/${SEQUENCE_NAME}"
REMOTE_OUT_DIR="${REMOTE_DIR}/$DATA_DIR/${SEQUENCE_NAME}/pair_dataset"

REMOTE_LOG_DIR="${REMOTE_DIR}/log_adt_metrics"

REMOTE_SCRIPT_DIR_metrics="${REMOTE_DIR}/src_metrics_remote"
REMOTE_SCRIPT_DIR_unik3d="${REMOTE_DIR}/src_metrics_remote"
REMOTE_SCRIPT_DIR_superglue="${REMOTE_DIR}/src_metrics_remote"
REMOTE_SCRIPT_DIR_visualise="${REMOTE_DIR}/src_metrics_remote"

IMAGE_DIR="IMAGES_DIR_unik3d_metrics"
REMOTE_DIR_EXP="${REMOTE_DIR}/${IMAGE_DIR}"

REMOTE_RESULTS_DIR="${REMOTE_DIR}/result_madpose_metrics"

##################################
LOCAL_DIR_EXP="/worktmp/THESES/GAUSSIAN-SPLATTING/experiments"

LOCAL_SCRIPT_DIR_metrics="/worktmp/THESES/GAUSSIAN-SPLATTING/MADpose-pipeline/src_metrics_remote_kitti/"
LOCAL_SCRIPT_DIR_unik3d="/worktmp/THESES/GAUSSIAN-SPLATTING/MADpose-pipeline/src_unik3d/"
LOCAL_SCRIPT_DIR_superglue="/worktmp/THESES/GAUSSIAN-SPLATTING/MADpose-pipeline/src_superglue/"
LOCAL_SCRIPT_DIR_visualise="/worktmp/THESES/GAUSSIAN-SPLATTING/MADpose-pipeline/src_visualise/"

#LOCAL_DATA_DIR="/worktmp/THESES/GAUSSIAN-SPLATTING/ADT/projectaria_tools_aria-scenes_data/${SEQUENCE_NAME}/adt_dataset/${SEQUENCE_NAME}"
LOCAL_DIR_KITTI_ROOT="/worktmp/THESES/GAUSSIAN-SPLATTING/KITTI-360"
LOCAL_DATA_DIR="${LOCAL_DIR_KITTI_ROOT}/sampling_dir/"

###############################################
# 3. Prepare remote directories
###############################################
#ssh "$REMOTE_USER@$REMOTE_HOST" "
#  rm -rf '$REMOTE_DATA_DIR'
#"

ssh "$REMOTE_USER@$REMOTE_HOST" "
    mkdir -p \
        '$REMOTE_DATA_DIR' \
        '$REMOTE_OUT_DIR' \
        '$REMOTE_LOG_DIR' \
        '$REMOTE_SCRIPT_DIR_metrics' \
        '$REMOTE_SCRIPT_DIR_unik3d' \
        '$REMOTE_SCRIPT_DIR_superglue' \
        '$REMOTE_SCRIPT_DIR_visualise' \
        '$REMOTE_DIR_EXP' \
        '$REMOTE_RESULTS_DIR'
"

rsync -avz "$LOCAL_SCRIPT_DIR_metrics/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR_metrics/"

rsync -avz "$LOCAL_SCRIPT_DIR_unik3d/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR_unik3d/"

rsync -avz "$LOCAL_SCRIPT_DIR_superglue/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR_superglue/"

rsync -avz "$LOCAL_SCRIPT_DIR_visualise/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR_visualise/"

#rsync -avz "$LOCAL_DATA_DIR/" \
#    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DATA_DIR/"

###############################################
# 7. Start pipeline remotely
###############################################
ssh "$REMOTE_USER@$REMOTE_HOST" \
  "nohup bash '$REMOTE_SCRIPT_DIR_metrics/run_pipeline_metrics_remote.sh' \
  '$REMOTE_DATA_DIR' \
  '$REMOTE_OUT_DIR' \
  '$REMOTE_LOG_DIR' \
  '$REMOTE_SCRIPT_DIR_metrics' \
  '$REMOTE_SCRIPT_DIR_unik3d' \
  '$REMOTE_SCRIPT_DIR_superglue' \
  '$REMOTE_SCRIPT_DIR_visualise' \
  '$REMOTE_DIR_EXP' \
  '$REMOTE_RESULTS_DIR' \
  '$SEQUENCE_NAME' \
  > '$REMOTE_LOG_DIR/run_pipeline_metrics_remote.log' 2>&1 &"

echo "[LOCAL] Pipeline started on remote host."
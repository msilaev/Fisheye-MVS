#!/bin/bash
set -e

###############################################
# 1. Load environment variables
###############################################
if [ -f ".env" ]; then
  source .env
fi

REMOTE_USER="mikhail"
REMOTE_HOST="<remote-workstation>"

###############################################
# 2. Remote configuration
###############################################

REMOTE_DIR="/home/<remote-user>/GAUSSIAN-SPLATTING/experiments"
REMOTE_LOG_DIR="${REMOTE_DIR}/log_madpose"
REMOTE_RESULTS_DIR="${REMOTE_DIR}/result_madpose"
REMOTE_SCRIPT_DIR="${REMOTE_DIR}/src_madpose"

IMAGE_DIR="IMAGES_DIR_unik3d"
REMOTE_DIR_EXP="${REMOTE_DIR}/${IMAGE_DIR}"

#LOCAL_DIR_EXP="/worktmp/THESES/GAUSSIAN-SPLATTING/experiments"
LOCAL_SRC_DIR="/worktmp/THESES/GAUSSIAN-SPLATTING/MADpose-pipeline/src_madpose"
#LOCAL_SRC_DIR="/opt/src_madpose"

DATA_TYPE="pinhole"

image_pair_file="${REMOTE_DIR_EXP}/image_pairs.txt"
image_input_dir="${REMOTE_DIR_EXP}"

###############################################
# 3. Prepare remote directories
###############################################
ssh "$REMOTE_USER@$REMOTE_HOST" "
    mkdir -p \
        '$REMOTE_DIR_EXP' \
        '$REMOTE_LOG_DIR' \
        '$REMOTE_RESULTS_DIR' \
        '$REMOTE_SCRIPT_DIR'
"

###############################################
# 5. Copy pipeline script to remote
###############################################

# Copy pipeline script to remote
rsync -avz "$LOCAL_SRC_DIR/remote_pipeline_madpose.sh" \
      "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR/"

rsync -avz "$LOCAL_SRC_DIR/madpose_inference.py" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR/"

# Run pipeline remotely in background
ssh "$REMOTE_USER@$REMOTE_HOST" \
  "nohup bash '$REMOTE_SCRIPT_DIR/remote_pipeline_madpose.sh' \
  '$image_pair_file' \
  '$image_input_dir' \
  '$DATA_TYPE' \
  '$REMOTE_LOG_DIR' \
  '$REMOTE_RESULTS_DIR' \
  '$REMOTE_SCRIPT_DIR' \
  > '$REMOTE_LOG_DIR/madpose_pipeline.log' 2>&1 &"

echo "[LOCAL] Pipeline started on remote host."


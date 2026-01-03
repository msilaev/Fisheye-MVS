#!/bin/bash
set -e

# ------------------------------
# 1. Load environment variables
# ------------------------------
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

# ------------------------------
# 2. Remote configuration
# ------------------------------
REMOTE_DIR="/home/hdd/mikhail/GAUSSIAN-SPLATTING/experiments"
REMOTE_LOG_DIR="${REMOTE_DIR}/log_madpose"
REMOTE_RESULTS_DIR="${REMOTE_DIR}/result_madpose"
REMOTE_SCRIPT_DIR="${REMOTE_DIR}/script_superglue"

IMAGE_DIR="IMAGES_DIR_unik3d"
REMOTE_DIR_EXP="$REMOTE_DIR/$IMAGE_DIR"

LOCAL_DIR_EXP="/worktmp/THESES/GAUSSIAN-SPLATTING/experiments"
LOCAL_SRC_DIR="/worktmp/THESES/GAUSSIAN-SPLATTING/MADpose-pipeline/src_superglue"

image_pair_file="${REMOTE_DIR_EXP}/image_pairs.txt"
image_input_dir="${REMOTE_DIR_EXP}"

# ADT
SIZE_X=1408
SIZE_Y=1408

# KITTI-360
SIZE_X=1400
SIZE_Y=1400

# ------------------------------
# 4. Ensure remote directory exists
# ------------------------------
ssh "$REMOTE_USER@$REMOTE_HOST" "
    mkdir -p \
        '$REMOTE_DIR_EXP' \
        '$REMOTE_LOG_DIR' \
        '$REMOTE_RESULTS_DIR' \
        '$REMOTE_SCRIPT_DIR'
"

# ------------------------------
# 5. Copy files to remote
# ------------------------------
rsync -avz "$LOCAL_SRC_DIR/remote_pipeline_superglue.sh" \
      "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR/"

rsync -avz "$LOCAL_SRC_DIR/superglue_extract_pairs.py" \
      "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR/"

# ------------------------------
# 6. Start pipeline remotely
# ------------------------------
ssh "$REMOTE_USER@$REMOTE_HOST" \
"nohup bash '$REMOTE_SCRIPT_DIR/remote_pipeline_superglue.sh' \
    '$image_pair_file' \
    '$image_input_dir' \
    '$DATA_TYPE' \
    '$REMOTE_LOG_DIR' \
    '$REMOTE_RESULTS_DIR' \
    '$REMOTE_SCRIPT_DIR' \
    '$SIZE_X' \
    '$SIZE_Y' \
    > '$REMOTE_LOG_DIR/pipeline.log' 2>&1 &"


echo "[LOCAL] Pipeline started on remote host."


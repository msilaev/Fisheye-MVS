#!/bin/bash
# Local launcher: sync scripts to remote, then start the metrics evaluation pipeline.
# Evaluates on a full dataset sequence (many pairs) and writes errors to a JSON.
#
# Usage:
#   bash run_metrics_local.sh          # uses experiment_config_adt.sh
#   DATASET=kitti bash run_metrics_local.sh
set -e
set -x

source .env
# Required: REMOTE_USER, REMOTE_HOST

source path_config.sh

DATASET="${DATASET:-adt}"

if [ "$DATASET" = "kitti" ]; then
    source experiment_config_kitti.sh
    IMG_EXT="png"
    ROTATE_GT=""          # KITTI: no 90° rotation
else
    source experiment_config_adt.sh
    IMG_EXT="jpg"
    ROTATE_GT="true"      # ADT: images are stored 90° CW
fi

# ── Remote paths ──────────────────────────────────────────────────────────────

REMOTE_SCRIPT_DIR_metrics="${REMOTE_PROJECT_DIR}/src_metrics"
REMOTE_SCRIPT_DIR_unik3d="${REMOTE_PROJECT_DIR}/src_unik3d"
REMOTE_SCRIPT_DIR_superglue="${REMOTE_PROJECT_DIR}/src_superglue"

# Dataset directory on remote: must contain poses/ and images/ subdirs
REMOTE_DATA_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/dataset"
REMOTE_OUT_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/pair_dataset"

REMOTE_IMAGE_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/metrics_images"
REMOTE_RESULTS_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/metrics_results"
REMOTE_LOG_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/logs_metrics"

REMOTE_FISHEYE_MASK_DIR="${REMOTE_PROJECT_DIR}/assets/fisheye_masks"
FISHEYE_MASK_PATH="${REMOTE_FISHEYE_MASK_DIR}/${FISHEYE_MASK_FILE}"

# ── Local paths ───────────────────────────────────────────────────────────────

LOCAL_SCRIPT_DIR="${LOCAL_ROOT}/Fisheye-MVS"
LOCAL_FISHEYE_MASK_DIR="${LOCAL_SCRIPT_DIR}/assets/fisheye_masks"

# ── Create remote dirs ────────────────────────────────────────────────────────

ssh "$REMOTE_USER@$REMOTE_HOST" "
    mkdir -p \
        '$REMOTE_OUT_DIR' \
        '$REMOTE_IMAGE_DIR' \
        '$REMOTE_RESULTS_DIR' \
        '$REMOTE_LOG_DIR' \
        '$REMOTE_FISHEYE_MASK_DIR'
"

# ── Sync scripts ──────────────────────────────────────────────────────────────

rsync -avz "${LOCAL_SCRIPT_DIR}/src_metrics/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR_metrics/"

rsync -avz "${LOCAL_SCRIPT_DIR}/src_unik3d/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR_unik3d/"

rsync -avz "${LOCAL_SCRIPT_DIR}/src_superglue/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR_superglue/"

rsync -avz "$LOCAL_FISHEYE_MASK_DIR/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_FISHEYE_MASK_DIR/"

# ── Launch on remote (non-blocking) ──────────────────────────────────────────

ssh "$REMOTE_USER@$REMOTE_HOST" \
  "nohup bash '$REMOTE_SCRIPT_DIR_metrics/run_pipeline_metrics_remote.sh' \
  '$REMOTE_DATA_DIR' \
  '$REMOTE_OUT_DIR' \
  '$REMOTE_LOG_DIR' \
  '$REMOTE_SCRIPT_DIR_metrics' \
  '$REMOTE_SCRIPT_DIR_unik3d' \
  '$REMOTE_SCRIPT_DIR_superglue' \
  '$REMOTE_IMAGE_DIR' \
  '$REMOTE_RESULTS_DIR' \
  '$EXPERIMENT_NAME' \
  '$IMG_EXT' \
  '$FISHEYE_MASK_PATH' \
  '$DISTANCE_THRESHOLD' \
  '$SIZE_X' \
  '$SIZE_Y' \
  '$ROTATE_GT' \
  '$REMOTE_UNIK3D_DIR' \
  '$REMOTE_SUPER_GLUE_DIR' \
  > '$REMOTE_LOG_DIR/run_pipeline_metrics_remote.log' 2>&1 &"

echo "[LOCAL] Metrics pipeline started on remote."
echo "[LOCAL] Sequence: ${EXPERIMENT_NAME}, Dataset: ${DATASET}"
echo "[LOCAL] Logs: ${REMOTE_LOG_DIR}/run_pipeline_metrics_remote.log"
echo "[LOCAL] Results: ${REMOTE_OUT_DIR}/sample_t_R_pair_dataset.json"

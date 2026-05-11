#!/bin/bash
# Local launcher: syncs scripts + images to remote, then submits SLURM jobs.
# Usage: bash sbatch_run_pipeline_local.sh
set -e
set -x

source .env

# Required env vars:
#   REMOTE_USER
#   REMOTE_HOST

source path_config.sh
source experiment_config_adt.sh

# ── Remote paths ──────────────────────────────────────────────────────────────

REMOTE_SCRIPT_DIR="${REMOTE_PROJECT_DIR}"
REMOTE_SCRIPT_DIR_unik3d="${REMOTE_PROJECT_DIR}/src_unik3d"
REMOTE_SCRIPT_DIR_superglue="${REMOTE_PROJECT_DIR}/src_superglue"
REMOTE_SCRIPT_DIR_procrustes="${REMOTE_PROJECT_DIR}/src_procrustes"

REMOTE_IMAGE_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/${IMAGE_DIR}"
REMOTE_RESULTS_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/results"
REMOTE_LOG_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/logs"

REMOTE_FISHEYE_MASK_DIR="${REMOTE_PROJECT_DIR}/assets/fisheye_masks"
remote_fisheye_mask_path="${REMOTE_FISHEYE_MASK_DIR}/${FISHEYE_MASK_FILE}"
remote_transform_result_path="${REMOTE_RESULTS_DIR}/transform_result.json"

# ── Local paths ───────────────────────────────────────────────────────────────

LOCAL_SCRIPT_DIR="${LOCAL_ROOT}/Fisheye-MVS"
LOCAL_SCRIPT_DIR_unik3d="${LOCAL_SCRIPT_DIR}/src_unik3d"
LOCAL_SCRIPT_DIR_superglue="${LOCAL_SCRIPT_DIR}/src_superglue"
LOCAL_SCRIPT_DIR_procrustes="${LOCAL_SCRIPT_DIR}/src_procrustes"

# Source images on local machine:
#   ${LOCAL_ROOT}/experiments/${EXPERIMENT_NAME}/${IMAGE_DIR}/
#   Place image0.jpg and image1.jpg there, along with image_pairs.txt
LOCAL_IMAGE_DIR="${LOCAL_ROOT}/experiments/${EXPERIMENT_NAME}/${IMAGE_DIR}"
LOCAL_FISHEYE_MASK_DIR="${LOCAL_SCRIPT_DIR}/assets/fisheye_masks"

# ── Create remote dirs ────────────────────────────────────────────────────────

ssh "$REMOTE_USER@$REMOTE_HOST" "
    mkdir -p \
        '$REMOTE_LOG_DIR' \
        '$REMOTE_RESULTS_DIR' \
        '$REMOTE_IMAGE_DIR' \
        '$REMOTE_FISHEYE_MASK_DIR'
"

# ── Sync scripts and assets ───────────────────────────────────────────────────

rsync -avz "$LOCAL_FISHEYE_MASK_DIR/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_FISHEYE_MASK_DIR/"

rsync -avz "$LOCAL_SCRIPT_DIR_unik3d/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR_unik3d/"

rsync -avz "$LOCAL_SCRIPT_DIR_superglue/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR_superglue/"

rsync -avz "$LOCAL_SCRIPT_DIR_procrustes/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR_procrustes/"

rsync -avz "$LOCAL_SCRIPT_DIR/sbatch_run_pipeline_remote.sh" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR/sbatch_run_pipeline_remote.sh"

# ── Sync source images ────────────────────────────────────────────────────────
# Expected contents of LOCAL_IMAGE_DIR:
#   image0.jpg / image0.png   — first camera image
#   image1.jpg / image1.png   — second camera image
#   image_pairs.txt           — one line: "image0.jpg image1.jpg"

rsync -avz "$LOCAL_IMAGE_DIR/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_IMAGE_DIR/"

# ── Submit SLURM jobs on remote ───────────────────────────────────────────────

ssh "$REMOTE_USER@$REMOTE_HOST" \
  "bash '$REMOTE_SCRIPT_DIR/sbatch_run_pipeline_remote.sh' \
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
  '$SIZE_Y'"

echo "[LOCAL] SLURM jobs submitted on remote host."
echo "[LOCAL] Logs: ${REMOTE_LOG_DIR}/"

#!/bin/bash
# Download rendered example PNGs from remote.
#
# Usage:
#   bash fetch_rendered_examples.sh kitti [JOB_ID]
#   bash fetch_rendered_examples.sh adt   [JOB_ID]
set -e

DATASET="${1:-adt}"
JOB_ID="${2:-}"

source "$(dirname "$0")/../.env"
source "$(dirname "$0")/../path_config.sh"

if [ "$DATASET" = "kitti" ]; then
    source "$(dirname "$0")/../experiment_config_kitti.sh"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-KITTI-360}"
else
    source "$(dirname "$0")/../experiment_config_adt.sh"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-ADT_seq133}"
fi

REMOTE_OUTPUT_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/eval_full/rendered_example"
REMOTE_OUTPUT_DIR_LEGACY="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/eval_full/rendered_examples"
LOCAL_OUTPUT_DIR="${LOCAL_ROOT}/experiments/${EXPERIMENT_NAME}/rendered_example"
mkdir -p "$LOCAL_OUTPUT_DIR"

# If job ID given, wait for completion
if [ -n "$JOB_ID" ]; then
    echo "[LOCAL] Waiting for job $JOB_ID to finish..."
    while ssh "$REMOTE_USER@$REMOTE_HOST" "squeue -j $JOB_ID -h" 2>/dev/null | grep -q "$JOB_ID"; do
        echo "  still running... (sleep 30s)"
        sleep 30
    done
    echo "[LOCAL] Job $JOB_ID done."
fi

# Pick the new remote directory if present, otherwise fall back to the legacy name.
if ssh "$REMOTE_USER@$REMOTE_HOST" "[ -d '${REMOTE_OUTPUT_DIR}' ]" 2>/dev/null; then
    REMOTE_FETCH_DIR="$REMOTE_OUTPUT_DIR"
elif ssh "$REMOTE_USER@$REMOTE_HOST" "[ -d '${REMOTE_OUTPUT_DIR_LEGACY}' ]" 2>/dev/null; then
    REMOTE_FETCH_DIR="$REMOTE_OUTPUT_DIR_LEGACY"
else
    echo "[ERROR] Neither ${REMOTE_OUTPUT_DIR} nor ${REMOTE_OUTPUT_DIR_LEGACY} exists on remote."
    exit 1
fi

# Download all PNGs
echo "[LOCAL] Downloading rendered examples from ${REMOTE_FETCH_DIR}/ ..."
rsync -av --include="*.png" --exclude="*" \
    "$REMOTE_USER@$REMOTE_HOST:${REMOTE_FETCH_DIR}/" \
    "$LOCAL_OUTPUT_DIR/"

echo "[LOCAL] Images saved to: $LOCAL_OUTPUT_DIR"
ls -lh "$LOCAL_OUTPUT_DIR/"

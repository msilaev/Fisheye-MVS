#!/bin/bash
# Download rendered example PNGs and all per-pair data from remote.
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

REMOTE_EVAL_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/eval_full"
REMOTE_OUTPUT_DIR="${REMOTE_EVAL_DIR}/rendered_example"
REMOTE_OUTPUT_DIR_LEGACY="${REMOTE_EVAL_DIR}/rendered_examples"
LOCAL_OUTPUT_DIR="${LOCAL_ROOT}/experiments/${EXPERIMENT_NAME}/rendered_example"
mkdir -p "$LOCAL_OUTPUT_DIR"

# Shared SSH connection — all subsequent ssh/scp reuse this socket
SSH_SOCKET="/tmp/ssh_mux_${REMOTE_USER}_${REMOTE_HOST}"
ssh -fNM -o ControlMaster=yes -o ControlPath="$SSH_SOCKET" \
    -o ControlPersist=60 "$REMOTE_USER@$REMOTE_HOST"
SSH_OPTS="-o ControlMaster=no -o ControlPath=$SSH_SOCKET"

cleanup() { ssh $SSH_OPTS -O exit "$REMOTE_USER@$REMOTE_HOST" 2>/dev/null || true; }
trap cleanup EXIT

# If job ID given, wait for completion
if [ -n "$JOB_ID" ]; then
    echo "[LOCAL] Waiting for job $JOB_ID to finish..."
    while ssh $SSH_OPTS "$REMOTE_USER@$REMOTE_HOST" "squeue -j $JOB_ID -h" 2>/dev/null | grep -q "$JOB_ID"; do
        echo "  still running... (sleep 30s)"
        sleep 30
    done
    echo "[LOCAL] Job $JOB_ID done."
fi

# Pick the new remote directory if present, otherwise fall back to the legacy name.
if ssh $SSH_OPTS "$REMOTE_USER@$REMOTE_HOST" "[ -d '${REMOTE_OUTPUT_DIR}' ]" 2>/dev/null; then
    REMOTE_FETCH_DIR="$REMOTE_OUTPUT_DIR"
elif ssh $SSH_OPTS "$REMOTE_USER@$REMOTE_HOST" "[ -d '${REMOTE_OUTPUT_DIR_LEGACY}' ]" 2>/dev/null; then
    REMOTE_FETCH_DIR="$REMOTE_OUTPUT_DIR_LEGACY"
else
    echo "[ERROR] Neither ${REMOTE_OUTPUT_DIR} nor ${REMOTE_OUTPUT_DIR_LEGACY} exists on remote."
    exit 1
fi

# Download rendered PNGs and per-pair subdirs.
echo "[LOCAL] Downloading rendered examples from ${REMOTE_FETCH_DIR}/ ..."
scp $SSH_OPTS -r "$REMOTE_USER@$REMOTE_HOST:${REMOTE_FETCH_DIR}/." "$LOCAL_OUTPUT_DIR/"

# Also copy result JSONs from eval_full/results into the same local pair folders
# so each pair's clouds/images/results are kept together in one directory.
REMOTE_RESULTS_DIR="${REMOTE_EVAL_DIR}/results"
echo "[LOCAL] Merging result JSONs from ${REMOTE_RESULTS_DIR}/ into local pair folders ..."
RSYNC_SSH="ssh $SSH_OPTS"
rsync -av -e "$RSYNC_SSH" \
    --include='*/' \
    --include='result_*.json' \
    --exclude='*' \
    "$REMOTE_USER@$REMOTE_HOST:${REMOTE_RESULTS_DIR}/" \
    "$LOCAL_OUTPUT_DIR/"

echo "[LOCAL] Data saved to: $LOCAL_OUTPUT_DIR"
ls -lh "$LOCAL_OUTPUT_DIR/"

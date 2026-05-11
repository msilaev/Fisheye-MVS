#!/bin/bash
# Sync select_pairs_by_rotation.py to remote and run it there.
# Selected pairs are written to REMOTE_EXPERIMENTS_DIR/<EXPERIMENT_NAME>/test_pairs/
#
# Usage:
#   bash select_pairs_local.sh              # ADT (default)
#   DATASET=kitti bash select_pairs_local.sh
set -e
set -x

source .env
# Required: REMOTE_USER, REMOTE_HOST

source path_config.sh

DATASET="${DATASET:-adt}"

if [ "$DATASET" = "kitti" ]; then
    source experiment_config_kitti.sh
    IMG_EXT="png"
    ROTATE_FLAG=""
else
    source experiment_config_adt.sh
    IMG_EXT="jpg"
    ROTATE_FLAG="--rotate"
fi

# ── Remote paths ──────────────────────────────────────────────────────────────

REMOTE_SCRIPT_DIR_metrics="${REMOTE_PROJECT_DIR}/src_metrics"

# Dataset on the shared storage: must contain poses/ and images/ subdirs
# KITTI: .../KITTI-360/<SEQUENCE_NAME>/
# ADT:   .../ADT/<SEQUENCE_NAME>/
REMOTE_DATASET_SEQ="${REMOTE_DATASETS_DIR}/${EXPERIMENT_NAME}/${SEQUENCE_NAME}"

# Output: selected pairs ready for the pipeline
REMOTE_TEST_PAIRS_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/test_pairs"

# ── Sync scripts ──────────────────────────────────────────────────────────────

rsync -avz "${LOCAL_ROOT}/Fisheye-MVS/src_metrics/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRIPT_DIR_metrics/"

# ── Run on remote ─────────────────────────────────────────────────────────────

ssh "$REMOTE_USER@$REMOTE_HOST" "
    mkdir -p '$REMOTE_TEST_PAIRS_DIR'

    module load mamba
    eval \"\$(mamba shell hook --shell bash)\"
    source activate 3d_pose_env

    python '$REMOTE_SCRIPT_DIR_metrics/select_pairs_by_rotation.py' \
        --pose-dir       '${REMOTE_DATASET_SEQ}/poses' \
        --image-dir      '${REMOTE_DATASET_SEQ}/images' \
        --output-dir     '$REMOTE_TEST_PAIRS_DIR' \
        --img-ext        '$IMG_EXT' \
        --max-frame-sep  '${MAX_FRAME_SEP:-100}' \
        --n-per-bin      '${N_PER_BIN:-1}' \
        $ROTATE_FLAG
"

echo "[LOCAL] Pairs selected. Remote location:"
echo "  ${REMOTE_TEST_PAIRS_DIR}/"
echo ""
echo "To run the pipeline on a pair, set IMAGE_DIR in experiment_config_adt.sh to:"
echo "  test_pairs/pair_rot_00-10   (or whichever bin)"
echo "then: bash run_pipeline_local.sh"

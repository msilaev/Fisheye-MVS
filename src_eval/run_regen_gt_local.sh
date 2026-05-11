#!/bin/bash
# Local launcher: sync scripts to remote and submit sbatch_regen_gt.sh
# for all ADT/KITTI experiment directories that have results.
#
# Usage:
#   bash src_eval/run_regen_gt_local.sh              # ADT seq133 (default)
#   DATASET=kitti bash src_eval/run_regen_gt_local.sh
#   EXPERIMENT_NAME=ADT_seq136 bash src_eval/run_regen_gt_local.sh
#   PAIRS_SUBDIR=test_pairs_custom bash src_eval/run_regen_gt_local.sh
set -e
set -x

source .env
source path_config.sh

DATASET="${DATASET:-adt}"
PAIRS_SUBDIR="${PAIRS_SUBDIR:-test_pairs}"  # test_pairs or test_pairs_custom

_EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME:-}"

if [ "$DATASET" = "kitti" ]; then
    source experiment_config_kitti.sh
    CALIB_TYPE="kitti360"
else
    source experiment_config_adt.sh
    CALIB_TYPE="aria"
fi

[ -n "$_EXPERIMENT_NAME_OVERRIDE" ] && EXPERIMENT_NAME="$_EXPERIMENT_NAME_OVERRIDE"

REMOTE_EVAL_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/eval_full"
REMOTE_PAIRS_DIR="${REMOTE_EVAL_DIR}/${PAIRS_SUBDIR}"
REMOTE_RESULTS_DIR="${REMOTE_EVAL_DIR}/results"
REMOTE_SLURM_LOG_DIR="${REMOTE_EVAL_DIR}/slurm_logs"

# Sync updated scripts
scp src_eval/sbatch_regen_gt.sh \
    src_eval/generate_result_gt.py \
    "$REMOTE_USER@$REMOTE_HOST:${REMOTE_PROJECT_DIR}/src_eval/"

ssh "$REMOTE_USER@$REMOTE_HOST" "
    sed -i 's/\r//' '${REMOTE_PROJECT_DIR}/src_eval/sbatch_regen_gt.sh'
    chmod +x '${REMOTE_PROJECT_DIR}/src_eval/sbatch_regen_gt.sh'
    mkdir -p '${REMOTE_SLURM_LOG_DIR}'
"

JOB=$(ssh "$REMOTE_USER@$REMOTE_HOST" "sbatch --parsable \
    --output='${REMOTE_SLURM_LOG_DIR}/regen_gt-%j.log' \
    --error='${REMOTE_SLURM_LOG_DIR}/regen_gt-%j.err' \
    '${REMOTE_PROJECT_DIR}/src_eval/sbatch_regen_gt.sh' \
    '${REMOTE_PAIRS_DIR}' '${REMOTE_RESULTS_DIR}' \
    '${REMOTE_PROJECT_DIR}' '${CALIB_TYPE}'")

echo "[LOCAL] Submitted regen_gt job: $JOB"
echo "[LOCAL] Pairs dir:   ${REMOTE_PAIRS_DIR}"
echo "[LOCAL] Results dir: ${REMOTE_RESULTS_DIR}"
echo "[LOCAL] Calib type:  ${CALIB_TYPE}"
echo "[LOCAL] Monitor: ssh $REMOTE_USER@$REMOTE_HOST 'squeue -j $JOB'"

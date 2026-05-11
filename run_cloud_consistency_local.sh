#!/bin/bash
# Submit cloud-consistency SLURM jobs for all fully-evaluated pairs.
# Re-runs UniK3D on each pair, then scores merged clouds for every method.
#
# Usage:
#   bash run_cloud_consistency_local.sh                        # ADT seq133 (default)
#   DATASET=kitti bash run_cloud_consistency_local.sh          # KITTI-360
#   EXPERIMENT_NAME=ADT_seq136 bash run_cloud_consistency_local.sh
#
# After all jobs complete, fetch results and aggregate:
#   bash fetch_cloud_consistency_local.sh
set -e
set -x

source .env
source path_config.sh

DATASET="${DATASET:-adt}"

_EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME:-}"

if [ "$DATASET" = "kitti" ]; then
    source experiment_config_kitti.sh
else
    source experiment_config_adt.sh
fi

[ -n "$_EXPERIMENT_NAME_OVERRIDE" ] && EXPERIMENT_NAME="$_EXPERIMENT_NAME_OVERRIDE"

OVERLAP_THRESH="${OVERLAP_THRESH:-0.05}"
N_JOBS="${N_JOBS:-10}"

# ── Remote paths ──────────────────────────────────────────────────────────────

REMOTE_EVAL_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/eval_full"
REMOTE_PAIRS_DIR="${REMOTE_EVAL_DIR}/test_pairs"
REMOTE_RESULTS_DIR="${REMOTE_EVAL_DIR}/results"
REMOTE_LOG_DIR="${REMOTE_EVAL_DIR}/logs_cc"
REMOTE_MASK="${REMOTE_PROJECT_DIR}/assets/fisheye_masks/${FISHEYE_MASK_FILE}"

# ── Sync scripts to remote ────────────────────────────────────────────────────

scp -r "${LOCAL_ROOT}/Fisheye-MVS/src_metrics" \
    "$REMOTE_USER@$REMOTE_HOST:${REMOTE_PROJECT_DIR}/"

scp -r "${LOCAL_ROOT}/Fisheye-MVS/src_unik3d" \
    "$REMOTE_USER@$REMOTE_HOST:${REMOTE_PROJECT_DIR}/"

ssh "$REMOTE_USER@$REMOTE_HOST" \
    "mkdir -p '${REMOTE_PROJECT_DIR}/assets/fisheye_masks'"
scp "${LOCAL_ROOT}/Fisheye-MVS/assets/fisheye_masks/"* \
    "$REMOTE_USER@$REMOTE_HOST:${REMOTE_PROJECT_DIR}/assets/fisheye_masks/"

ssh "$REMOTE_USER@$REMOTE_HOST" "
    mkdir -p '$REMOTE_LOG_DIR'
    find '${REMOTE_PROJECT_DIR}' -name '*.sh' -exec sed -i 's/\r//' {} +
    chmod +x '${REMOTE_PROJECT_DIR}/src_metrics/sbatch_cloud_consistency.sh'
"

# ── Build batch lists and submit SLURM jobs ───────────────────────────────────

ssh "$REMOTE_USER@$REMOTE_HOST" "
PAIRS_DIR='${REMOTE_PAIRS_DIR}'
RESULTS_DIR='${REMOTE_RESULTS_DIR}'
LOG_DIR='${REMOTE_LOG_DIR}'
PROJ='${REMOTE_PROJECT_DIR}'
UNIK3D='${REMOTE_UNIK3D_DIR}'
MASK='${REMOTE_MASK}'
BATCH_DIR='${REMOTE_EVAL_DIR}/batch_lists_cc'
SLURM_LOG_DIR='${REMOTE_EVAL_DIR}/slurm_logs_cc'
N_JOBS=${N_JOBS}

mkdir -p \"\$BATCH_DIR\" \"\$SLURM_LOG_DIR\"

# Only include pairs whose evaluation is complete (result_procrustes.json exists)
PAIR_LIST=()
for d in \"\$PAIRS_DIR\"/pair_rot_*; do
    PAIR_NAME=\$(basename \"\$d\")
    if [ -f \"\$RESULTS_DIR/\$PAIR_NAME/result_procrustes.json\" ]; then
        PAIR_LIST+=(\"\$d\")
    fi
done
TOTAL=\${#PAIR_LIST[@]}
echo \"Found \$TOTAL fully-evaluated pairs\"

if [ \$TOTAL -eq 0 ]; then
    echo '[ERROR] No evaluated pairs found. Run run_eval_batch_local.sh first.'
    exit 1
fi

PAIRS_PER_JOB=\$(( (TOTAL + N_JOBS - 1) / N_JOBS ))

batch_idx=0
i=0
while [ \$i -lt \$TOTAL ]; do
    batch_idx=\$((batch_idx + 1))
    LIST_FILE=\"\$BATCH_DIR/batch_cc_\$(printf '%03d' \$batch_idx).txt\"
    > \"\$LIST_FILE\"
    for k in \$(seq 0 \$((PAIRS_PER_JOB - 1))); do
        idx=\$((i + k))
        [ \$idx -ge \$TOTAL ] && break
        echo \"\${PAIR_LIST[\$idx]}\" >> \"\$LIST_FILE\"
    done
    i=\$((i + PAIRS_PER_JOB))

    JOB_ID=\$(sbatch --parsable \
        --output=\"\$SLURM_LOG_DIR/cc_batch-%j.log\" \
        --error=\"\$SLURM_LOG_DIR/cc_batch-%j.err\" \
        \"\$PROJ/src_metrics/sbatch_cloud_consistency.sh\" \
        \"\$LIST_FILE\" \"\$RESULTS_DIR\" \"\$LOG_DIR\" \
        \"\$PROJ\" \"\$UNIK3D\" \"\$MASK\" \
        '${DISTANCE_THRESHOLD}' '${OVERLAP_THRESH}' \
        '${SIZE_X}' '${SIZE_Y}' '${DATASET}')
    echo \"Batch \$batch_idx (\$(wc -l < \"\$LIST_FILE\") pairs): job=\$JOB_ID\"
done

echo ''
echo \"Submitted \$batch_idx batch jobs covering \$TOTAL pairs.\"
"

echo ""
echo "[LOCAL] Cloud consistency jobs submitted."
echo "[LOCAL] Experiment: ${EXPERIMENT_NAME}, Dataset: ${DATASET}"
echo "[LOCAL] Monitor: ssh $REMOTE_USER@$REMOTE_HOST 'squeue -u <remote-user>'"
echo "[LOCAL] When done, run: bash fetch_cloud_consistency_local.sh"

#!/bin/bash
# Full evaluation batch launcher:
#   1. Syncs scripts to remote
#   2. Runs pair selection on remote (select_pairs_by_rotation.py)
#   3. Submits one SLURM job per pair (sbatch_eval_pair.sh)
#
# Usage:
#   bash run_eval_batch_local.sh                                        # ADT seq133 (default)
#   DATASET=kitti bash run_eval_batch_local.sh                          # KITTI-360
#   SEQUENCE_NAME=Apartment_release_clean_seq136_M1292 \
#     EXPERIMENT_NAME=ADT_seq136 bash run_eval_batch_local.sh           # other ADT sequence
#
# After all jobs complete, fetch results and compute errors:
#   bash fetch_eval_results_local.sh
set -e
set -x

source .env
source path_config.sh

DATASET="${DATASET:-adt}"

# Save any env overrides before sourcing config (config would overwrite them)
_SEQUENCE_NAME_OVERRIDE="${SEQUENCE_NAME:-}"
_EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME:-}"

if [ "$DATASET" = "kitti" ]; then
    source experiment_config_kitti.sh
    IMG_EXT="png"
    ROTATE_FLAG=""
else
    source experiment_config_adt.sh
    IMG_EXT="jpg"
    ROTATE_FLAG="--rotate"
fi

# Re-apply overrides after config sourcing
[ -n "$_SEQUENCE_NAME_OVERRIDE"  ] && SEQUENCE_NAME="$_SEQUENCE_NAME_OVERRIDE"
[ -n "$_EXPERIMENT_NAME_OVERRIDE" ] && EXPERIMENT_NAME="$_EXPERIMENT_NAME_OVERRIDE"

N_PER_BIN="${N_PER_BIN:-50}"
MAX_FRAME_SEP="${MAX_FRAME_SEP:-100}"
SEED="${SEED:-42}"

# Dataset dir uses the fixed dataset name (ADT/KITTI-360), not EXPERIMENT_NAME
DATASET_DIR_NAME=$([ "$DATASET" = "kitti" ] && echo "KITTI-360" || echo "ADT")

# ── Remote paths ──────────────────────────────────────────────────────────────

REMOTE_DATASET_SEQ="${REMOTE_DATASETS_DIR}/${DATASET_DIR_NAME}/${SEQUENCE_NAME}"
REMOTE_EVAL_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/eval_full"
REMOTE_PAIRS_DIR="${REMOTE_EVAL_DIR}/test_pairs"
REMOTE_RESULTS_DIR="${REMOTE_EVAL_DIR}/results"
REMOTE_LOG_DIR="${REMOTE_EVAL_DIR}/logs"

REMOTE_MASK="${REMOTE_PROJECT_DIR}/assets/fisheye_masks/${FISHEYE_MASK_FILE}"

# KITTI-360 calibration yaml path on remote (empty for ADT)
if [ "$CALIB_TYPE" = "kitti360" ] && [ -n "${KITTI_CALIB_FILE:-}" ]; then
    REMOTE_KITTI_CALIB="${REMOTE_DATASETS_DIR}/KITTI-360/${KITTI_CALIB_FILE}"
else
    REMOTE_KITTI_CALIB=""
fi

# ── Sync scripts ──────────────────────────────────────────────────────────────

ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p '$REMOTE_PAIRS_DIR' '$REMOTE_RESULTS_DIR' '$REMOTE_LOG_DIR'"

for DIR in src_metrics src_unik3d src_superglue src_procrustes \
           src_baseline_pinhole src_baseline_dust3r src_madpose \
           src_baseline_fisheye src_eval; do
    scp -r "${LOCAL_ROOT}/Fisheye-MVS/${DIR}" \
        "$REMOTE_USER@$REMOTE_HOST:${REMOTE_PROJECT_DIR}/"
done

# Sync only the fisheye_masks subdir from assets (avoid large image files)
ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p '${REMOTE_PROJECT_DIR}/assets/fisheye_masks'"
scp "${LOCAL_ROOT}/Fisheye-MVS/assets/fisheye_masks/"* \
    "$REMOTE_USER@$REMOTE_HOST:${REMOTE_PROJECT_DIR}/assets/fisheye_masks/"

# Fix line endings on all shell scripts
ssh "$REMOTE_USER@$REMOTE_HOST" "
    find '${REMOTE_PROJECT_DIR}' -name '*.sh' -exec sed -i 's/\r//' {} +
    chmod +x '${REMOTE_PROJECT_DIR}/src_eval/sbatch_eval_pair.sh'
"

# ── Select pairs on remote ────────────────────────────────────────────────────

ssh "$REMOTE_USER@$REMOTE_HOST" "
    module load mamba
    eval \"\$(mamba shell hook --shell bash)\"
    source activate 3d_pose_env

    python '${REMOTE_PROJECT_DIR}/src_metrics/select_pairs_by_rotation.py' \
        --pose-dir          '${REMOTE_DATASET_SEQ}/poses' \
        --image-dir         '${REMOTE_DATASET_SEQ}/images' \
        --output-dir        '${REMOTE_PAIRS_DIR}' \
        --img-ext           '${IMG_EXT}' \
        --max-frame-sep     '${MAX_FRAME_SEP}' \
        --n-per-bin         '${N_PER_BIN}' \
        --seed              '${SEED}' \
        --no-copy-images \
        --rotation-bins     ${ROTATION_BINS} \
        --translation-bins  ${TRANSLATION_BINS} \
        ${ROTATE_FLAG}
"

# ── Split pairs into batches and submit one SLURM job per batch ───────────────

N_JOBS="${N_JOBS:-10}"   # max simultaneous jobs

ssh "$REMOTE_USER@$REMOTE_HOST" "
PAIRS_DIR='${REMOTE_PAIRS_DIR}'
RESULTS_DIR='${REMOTE_RESULTS_DIR}'
LOG_DIR='${REMOTE_LOG_DIR}'
PROJ='${REMOTE_PROJECT_DIR}'
UNIK3D='${REMOTE_UNIK3D_DIR}'
SUPERGLUE='${REMOTE_SUPER_GLUE_DIR}'
DUST3R='/scratch/work/<remote-user>/3d/dust3r'
MASK='${REMOTE_MASK}'
BATCH_DIR='${REMOTE_EVAL_DIR}/batch_lists'
SLURM_LOG_DIR='${REMOTE_EVAL_DIR}/slurm_logs'
N_JOBS=${N_JOBS}

mkdir -p \"\$BATCH_DIR\" \"\$SLURM_LOG_DIR\"

# Collect all pair dirs into a list
PAIR_LIST=(\"\$PAIRS_DIR\"/pair_rot_*)
TOTAL=\${#PAIR_LIST[@]}
PAIRS_PER_JOB=\$(( (TOTAL + N_JOBS - 1) / N_JOBS ))
echo \"Total pairs: \$TOTAL, jobs: \$N_JOBS, pairs per job: \$PAIRS_PER_JOB\"

# Write batch list files and submit
batch_idx=0
i=0
while [ \$i -lt \$TOTAL ]; do
    batch_idx=\$((batch_idx + 1))
    LIST_FILE=\"\$BATCH_DIR/batch_\$(printf '%03d' \$batch_idx).txt\"
    > \"\$LIST_FILE\"
    for k in \$(seq 0 \$((PAIRS_PER_JOB - 1))); do
        idx=\$((i + k))
        [ \$idx -ge \$TOTAL ] && break
        echo \"\${PAIR_LIST[\$idx]}\" >> \"\$LIST_FILE\"
    done
    i=\$((i + PAIRS_PER_JOB))

    chmod +x \"\$PROJ/src_eval/sbatch_eval_batch.sh\"
    chmod +x \"\$PROJ/src_eval/sbatch_madpose_only.sh\"
    chmod +x \"\$PROJ/src_eval/sbatch_rect_gpu.sh\"
    chmod +x \"\$PROJ/src_eval/sbatch_fisheye_emat_only.sh\"

    EVAL_JOB=\$(sbatch --parsable \
        --output=\"\$SLURM_LOG_DIR/eval_batch-%j.log\" \
        --error=\"\$SLURM_LOG_DIR/eval_batch-%j.err\" \
        \"\$PROJ/src_eval/sbatch_eval_batch.sh\" \
        \"\$LIST_FILE\" \"\$RESULTS_DIR\" \"\$LOG_DIR\" \
        \"\$PROJ\" \"\$UNIK3D\" \"\$SUPERGLUE\" \"\$DUST3R\" \
        \"\$MASK\" ${DISTANCE_THRESHOLD} ${SIZE_X} ${SIZE_Y} \
        "${CALIB_TYPE}" "${REMOTE_KITTI_CALIB}")
    echo \"Batch \$batch_idx (\$(wc -l < \"\$LIST_FILE\") pairs): eval_job=\$EVAL_JOB\"

    # MADPose (fisheye) — runs on batch-skl after eval_job produces depth/mkpts
    MP_JOB=\$(sbatch --parsable \
        --dependency=afterok:\$EVAL_JOB \
        --output=\"\$SLURM_LOG_DIR/madpose-%j.log\" \
        --error=\"\$SLURM_LOG_DIR/madpose-%j.err\" \
        \"\$PROJ/src_eval/sbatch_madpose_only.sh\" \
        "$LIST_FILE" "$RESULTS_DIR" "$LOG_DIR" "$PROJ" \
        "${CALIB_TYPE}" "${REMOTE_KITTI_CALIB}")
    echo \"  madpose batch \$batch_idx: job=\$MP_JOB (after \$EVAL_JOB)\"

    # Fisheye E-mat — CPU, depends only on mkpts from eval_job
    FEMAT_JOB=\$(sbatch --parsable \
        --dependency=afterok:\$EVAL_JOB \
        --output=\"\$SLURM_LOG_DIR/fisheye_emat-%j.log\" \
        --error=\"\$SLURM_LOG_DIR/fisheye_emat-%j.err\" \
        \"\$PROJ/src_eval/sbatch_fisheye_emat_only.sh\" \
        "$LIST_FILE" "$RESULTS_DIR" "$LOG_DIR" "$PROJ" \
        "$MASK" "${CALIB_TYPE}" "${REMOTE_KITTI_CALIB}" \
        "${SIZE_X}" "${SIZE_Y}")
    echo \"  fisheye_emat batch \$batch_idx: job=\$FEMAT_JOB (after \$EVAL_JOB)\"

    # MADPose+rect: generate rect intermediates on GPU, then run MADPose on CPU
    RECT_JOB=\$(sbatch --parsable \
        --dependency=afterok:\$EVAL_JOB \
        --output=\"\$SLURM_LOG_DIR/rect_gpu-%j.log\" \
        --error=\"\$SLURM_LOG_DIR/rect_gpu-%j.err\" \
        \"\$PROJ/src_eval/sbatch_rect_gpu.sh\" \
        \"\$LIST_FILE\" \"\$RESULTS_DIR\" \"\$LOG_DIR\" \
        \"\$PROJ\" \"\$UNIK3D\" \"\$SUPERGLUE\" \
        "${CALIB_TYPE}" "${REMOTE_KITTI_CALIB}")
    MPRECT_JOB=\$(sbatch --parsable \
        --dependency=afterok:\$RECT_JOB \
        --output=\"\$SLURM_LOG_DIR/madpose_rect-%j.log\" \
        --error=\"\$SLURM_LOG_DIR/madpose_rect-%j.err\" \
        \"\$PROJ/src_eval/sbatch_madpose_only.sh\" \
        "$LIST_FILE" "$RESULTS_DIR" "$LOG_DIR" "$PROJ" \
        "${CALIB_TYPE}" "${REMOTE_KITTI_CALIB}")
    echo \"  rect_gpu batch \$batch_idx: job=\$RECT_JOB (after \$EVAL_JOB)\"
    echo \"  madpose_rect batch \$batch_idx: job=\$MPRECT_JOB (after \$RECT_JOB)\"
done
"

echo ""
echo "[LOCAL] All jobs submitted."
echo "[LOCAL] Monitor with: ssh $REMOTE_USER@$REMOTE_HOST 'squeue -u <remote-user>'"
echo "[LOCAL] Results will be at: ${REMOTE_RESULTS_DIR}/"
echo "[LOCAL] When done, run: bash fetch_eval_results_local.sh"

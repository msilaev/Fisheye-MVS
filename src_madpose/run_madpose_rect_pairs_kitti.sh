#!/bin/bash
# Submit MADPose+rect jobs for specific KITTI-360 pairs on the remote machine.
#
# Run this script on Triton inside the remote Fisheye-MVS checkout.
# It accepts either existing pair names or explicit frame specs like:
#   frames:0000006360|0000006367
#
# Usage on remote:
#   PAIRS='frames:0000006360|0000006367 frames:0000008603|0000008587' \
#     bash ./src_madpose/run_madpose_rect_pairs_kitti.sh
#   bash ./src_madpose/run_madpose_rect_pairs_kitti.sh pair_custom_0000006360__0000006367

set -euo pipefail
set -x

source "$(dirname "$0")/../path_config.sh"
source "$(dirname "$0")/../experiment_config_kitti.sh"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-KITTI-360}"
REMOTE_EVAL_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/eval_full"
REMOTE_PAIRS_DIR="${REMOTE_EVAL_DIR}/test_pairs"
REMOTE_RESULTS_DIR="${REMOTE_EVAL_DIR}/results"
REMOTE_LOG_DIR="${REMOTE_EVAL_DIR}/logs"
REMOTE_PREPARED_PAIRS_DIR="${REMOTE_EVAL_DIR}/test_pairs_custom"
REMOTE_BATCH_DIR="${REMOTE_EVAL_DIR}/batch_lists"
REMOTE_SLURM_LOG_DIR="${REMOTE_EVAL_DIR}/slurm_logs"
REMOTE_PREPARE_SCRIPT="${REMOTE_PROJECT_DIR}/src_visualise/prepare_render_pairs.py"
REMOTE_PAIR_SPECS_FILE="${REMOTE_BATCH_DIR}/madpose_rect_pair_specs_kitti.txt"
REMOTE_PAIR_LIST_FILE="${REMOTE_BATCH_DIR}/madpose_rect_pairs_kitti.txt"
REMOTE_KITTI_CALIB="${REMOTE_DATASETS_DIR}/KITTI-360/${KITTI_CALIB_FILE}"

if [ -n "${PAIRS:-}" ]; then
    IFS=' ' read -r -a SELECTED <<< "$PAIRS"
elif [ "$#" -gt 0 ]; then
    SELECTED=("$@")
else
    echo "[ERROR] Provide specific pairs via PAIRS env var or script arguments"
    exit 1
fi

mkdir -p "${REMOTE_BATCH_DIR}" "${REMOTE_SLURM_LOG_DIR}" "${REMOTE_PREPARED_PAIRS_DIR}" "${REMOTE_RESULTS_DIR}" "${REMOTE_LOG_DIR}"

printf '%s\n' "${SELECTED[@]}" > "${REMOTE_PAIR_SPECS_FILE}"

module load mamba
eval "$(mamba shell hook --shell bash)"
source activate 3d_pose_env

: > "${REMOTE_PAIR_LIST_FILE}"
while IFS= read -r PAIR_SPEC; do
    [ -z "$PAIR_SPEC" ] && continue
    META=$(python "${REMOTE_PREPARE_SCRIPT}" \
        --dataset kitti \
        --pair-spec "$PAIR_SPEC" \
        --default-pairs-dir "${REMOTE_PAIRS_DIR}" \
        --default-results-dir "${REMOTE_RESULTS_DIR}" \
        --output-root "${REMOTE_PREPARED_PAIRS_DIR}" \
        --experiments-root "${REMOTE_EXPERIMENTS_DIR}" \
        --shared-datasets-root "${REMOTE_DATASETS_DIR}" \
        --image-ext "${IMG_EXTENSION}")
    PAIR_DIR=$(printf '%s\n' "$META" | cut -f2)
    printf '%s\n' "$PAIR_DIR" >> "${REMOTE_PAIR_LIST_FILE}"
done < "${REMOTE_PAIR_SPECS_FILE}"

cat "${REMOTE_PAIR_LIST_FILE}"

RECT_JOB=$(sbatch --parsable \
    --output="${REMOTE_SLURM_LOG_DIR}/rect_gpu_custom_kitti-%j.log" \
    --error="${REMOTE_SLURM_LOG_DIR}/rect_gpu_custom_kitti-%j.err" \
    "${REMOTE_PROJECT_DIR}/src_eval/sbatch_rect_gpu.sh" \
    "${REMOTE_PAIR_LIST_FILE}" "${REMOTE_RESULTS_DIR}" "${REMOTE_LOG_DIR}" \
    "${REMOTE_PROJECT_DIR}" "${REMOTE_UNIK3D_DIR}" "${REMOTE_SUPER_GLUE_DIR}" \
    kitti360 "${REMOTE_KITTI_CALIB}")

MP_JOB=$(sbatch --parsable \
    --dependency=afterok:${RECT_JOB} \
    --output="${REMOTE_SLURM_LOG_DIR}/madpose_custom_kitti-%j.log" \
    --error="${REMOTE_SLURM_LOG_DIR}/madpose_custom_kitti-%j.err" \
    "${REMOTE_PROJECT_DIR}/src_eval/sbatch_madpose_only.sh" \
    "${REMOTE_PAIR_LIST_FILE}" "${REMOTE_RESULTS_DIR}" "${REMOTE_LOG_DIR}" \
    "${REMOTE_PROJECT_DIR}" kitti360 "${REMOTE_KITTI_CALIB}")

echo "[REMOTE] Submitted KITTI rect job: ${RECT_JOB}"
echo "[REMOTE] Submitted KITTI madpose job: ${MP_JOB}"
echo "[REMOTE] Monitor with: squeue -j ${RECT_JOB},${MP_JOB}"
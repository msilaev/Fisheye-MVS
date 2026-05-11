#!/bin/bash
# Submit an isolated full evaluation pipeline for specific KITTI-360 pairs on remote.
#
# This runs only the selected pairs and generates all method JSON files needed by
# visualisation (including result_procrustes_ransac.json, result_madpose.json,
# result_madpose_rect.json).
#
# Run on Triton inside remote Fisheye-MVS checkout.
#
# Usage:
#   PAIRS='frames:0000006360|0000006367 frames:0000008603|0000008587' \
#     bash ./src_eval/run_custom_pairs_pipeline_kitti.sh
#   bash ./src_eval/run_custom_pairs_pipeline_kitti.sh pair_custom_0000006360__0000006367

set -eo pipefail
set -x

source "$(dirname "$0")/../path_config.sh"
source "$(dirname "$0")/../experiment_config_kitti.sh"

# Be resilient to CRLF configs copied from Windows.
for _v in EXPERIMENT_NAME DISTANCE_THRESHOLD SIZE_X SIZE_Y IMG_EXTENSION FISHEYE_MASK_FILE CALIB_TYPE KITTI_CALIB_FILE; do
    eval "$_v=\"\${$_v//$'\r'/}\""
done

EXPERIMENT_NAME="${EXPERIMENT_NAME:-KITTI-360}"
REMOTE_EVAL_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/eval_full"
REMOTE_PAIRS_DIR="${REMOTE_EVAL_DIR}/test_pairs"
REMOTE_RESULTS_DIR="${REMOTE_EVAL_DIR}/results"
REMOTE_LOG_DIR="${REMOTE_EVAL_DIR}/logs"
REMOTE_PREPARED_PAIRS_DIR="${REMOTE_EVAL_DIR}/test_pairs_custom"
REMOTE_BATCH_DIR="${REMOTE_EVAL_DIR}/batch_lists"
REMOTE_SLURM_LOG_DIR="${REMOTE_EVAL_DIR}/slurm_logs"
REMOTE_PREPARE_SCRIPT="${REMOTE_PROJECT_DIR}/src_visualise/prepare_render_pairs.py"
REMOTE_PAIR_SPECS_FILE="${REMOTE_BATCH_DIR}/custom_full_pair_specs_kitti.txt"
REMOTE_PAIR_LIST_FILE="${REMOTE_BATCH_DIR}/custom_full_pairs_kitti.txt"
REMOTE_MASK="${REMOTE_PROJECT_DIR}/assets/fisheye_masks/${FISHEYE_MASK_FILE}"
REMOTE_DUST3R_DIR="${REMOTE_DUST3R_DIR:-/scratch/work/<remote-user>/3d/dust3r}"
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
mamba activate 3d_pose_env

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

chmod +x "${REMOTE_PROJECT_DIR}/src_eval/sbatch_eval_batch.sh"
chmod +x "${REMOTE_PROJECT_DIR}/src_eval/sbatch_madpose_only.sh"
chmod +x "${REMOTE_PROJECT_DIR}/src_eval/sbatch_rect_gpu.sh"

EVAL_JOB=$(sbatch --parsable \
    --output="${REMOTE_SLURM_LOG_DIR}/eval_batch_custom_kitti-%j.log" \
    --error="${REMOTE_SLURM_LOG_DIR}/eval_batch_custom_kitti-%j.err" \
    "${REMOTE_PROJECT_DIR}/src_eval/sbatch_eval_batch.sh" \
    "${REMOTE_PAIR_LIST_FILE}" "${REMOTE_RESULTS_DIR}" "${REMOTE_LOG_DIR}" \
    "${REMOTE_PROJECT_DIR}" "${REMOTE_UNIK3D_DIR}" "${REMOTE_SUPER_GLUE_DIR}" \
    "${REMOTE_DUST3R_DIR}" "${REMOTE_MASK}" \
    "${DISTANCE_THRESHOLD}" "${SIZE_X}" "${SIZE_Y}" kitti360 "${REMOTE_KITTI_CALIB}")

MP_JOB=$(sbatch --parsable \
    --dependency=afterok:${EVAL_JOB} \
    --output="${REMOTE_SLURM_LOG_DIR}/madpose_custom_kitti-%j.log" \
    --error="${REMOTE_SLURM_LOG_DIR}/madpose_custom_kitti-%j.err" \
    "${REMOTE_PROJECT_DIR}/src_eval/sbatch_madpose_only.sh" \
    "${REMOTE_PAIR_LIST_FILE}" "${REMOTE_RESULTS_DIR}" "${REMOTE_LOG_DIR}" \
    "${REMOTE_PROJECT_DIR}" kitti360 "${REMOTE_KITTI_CALIB}")

RECT_JOB=$(sbatch --parsable \
    --dependency=afterok:${EVAL_JOB} \
    --output="${REMOTE_SLURM_LOG_DIR}/rect_gpu_custom_kitti-%j.log" \
    --error="${REMOTE_SLURM_LOG_DIR}/rect_gpu_custom_kitti-%j.err" \
    "${REMOTE_PROJECT_DIR}/src_eval/sbatch_rect_gpu.sh" \
    "${REMOTE_PAIR_LIST_FILE}" "${REMOTE_RESULTS_DIR}" "${REMOTE_LOG_DIR}" \
    "${REMOTE_PROJECT_DIR}" "${REMOTE_UNIK3D_DIR}" "${REMOTE_SUPER_GLUE_DIR}" \
    kitti360 "${REMOTE_KITTI_CALIB}")

MPRECT_JOB=$(sbatch --parsable \
    --dependency=afterok:${RECT_JOB} \
    --output="${REMOTE_SLURM_LOG_DIR}/madpose_rect_custom_kitti-%j.log" \
    --error="${REMOTE_SLURM_LOG_DIR}/madpose_rect_custom_kitti-%j.err" \
    "${REMOTE_PROJECT_DIR}/src_eval/sbatch_madpose_only.sh" \
    "${REMOTE_PAIR_LIST_FILE}" "${REMOTE_RESULTS_DIR}" "${REMOTE_LOG_DIR}" \
    "${REMOTE_PROJECT_DIR}" kitti360 "${REMOTE_KITTI_CALIB}")

echo "[REMOTE] Submitted KITTI isolated custom pipeline"
echo "  eval_batch:    ${EVAL_JOB}"
echo "  madpose:       ${MP_JOB} (after eval_batch)"
echo "  rect_gpu:      ${RECT_JOB} (after eval_batch)"
echo "  madpose_rect:  ${MPRECT_JOB} (after rect_gpu)"
echo "[REMOTE] Monitor with: squeue -j ${EVAL_JOB},${MP_JOB},${RECT_JOB},${MPRECT_JOB}"

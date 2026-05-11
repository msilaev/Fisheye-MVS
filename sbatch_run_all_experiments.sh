#!/bin/bash
# Submits the full pipeline + all baselines as dependent SLURM jobs.
# Usage: mirrors sbatch_run_pipeline_remote.sh with extra args for baselines.
set -e
set -x

REMOTE_LOG_DIR="$1"
REMOTE_SCRIPT_DIR_unik3d="$2"
REMOTE_SCRIPT_DIR_superglue="$3"
REMOTE_SCRIPT_DIR_procrustes="$4"
REMOTE_IMAGE_DIR="$5"
REMOTE_RESULTS_DIR="$6"
REMOTE_SUPER_GLUE_DIR="$7"
REMOTE_UNIK3D_DIR="$8"
remote_fisheye_mask_path="$9"
remote_transform_result_path="${10}"
DISTANCE_THRESHOLD="${11}"
SIZE_X="${12}"
SIZE_Y="${13}"
CALIB_PATH="${14}"    # JSON calibration for pinhole / dust3r-rect baselines

REMOTE_SCRIPT_DIR_pinhole="${REMOTE_SCRIPT_DIR_procrustes}"   # same remote src dir
REMOTE_SCRIPT_DIR_dust3r="${REMOTE_SCRIPT_DIR_procrustes}"

# ── Main pipeline (unik3d → superglue → procrustes) ───────────────────────────

JOB_UNIK3D=$(sbatch --parsable \
  "$REMOTE_SCRIPT_DIR_unik3d/sbatch_remote_pipeline_unik3d.sh" \
  "$REMOTE_LOG_DIR" "$REMOTE_SCRIPT_DIR_unik3d" "$REMOTE_SCRIPT_DIR_superglue" \
  "$REMOTE_SCRIPT_DIR_procrustes" "$REMOTE_IMAGE_DIR" "$REMOTE_RESULTS_DIR" \
  "$REMOTE_UNIK3D_DIR")
echo "Submitted unik3d: $JOB_UNIK3D"

JOB_SUPERGLUE=$(sbatch --parsable --dependency=afterok:$JOB_UNIK3D \
  "$REMOTE_SCRIPT_DIR_superglue/sbatch_remote_pipeline_superglue.sh" \
  "$REMOTE_LOG_DIR" "$REMOTE_SCRIPT_DIR_unik3d" "$REMOTE_SCRIPT_DIR_superglue" \
  "$REMOTE_SCRIPT_DIR_procrustes" "$REMOTE_IMAGE_DIR" "$REMOTE_RESULTS_DIR" \
  "$REMOTE_SUPER_GLUE_DIR" "$SIZE_X" "$SIZE_Y")
echo "Submitted superglue: $JOB_SUPERGLUE (after $JOB_UNIK3D)"

JOB_PROCRUSTES=$(sbatch --parsable --dependency=afterok:$JOB_SUPERGLUE \
  "$REMOTE_SCRIPT_DIR_procrustes/sbatch_remote_pipeline_procrustes.sh" \
  "$REMOTE_LOG_DIR" "$REMOTE_SCRIPT_DIR_unik3d" "$REMOTE_SCRIPT_DIR_superglue" \
  "$REMOTE_SCRIPT_DIR_procrustes" "$REMOTE_IMAGE_DIR" "$REMOTE_RESULTS_DIR" \
  "$remote_fisheye_mask_path" "$remote_transform_result_path" \
  "$DISTANCE_THRESHOLD" "$SIZE_X" "$SIZE_Y")
echo "Submitted procrustes: $JOB_PROCRUSTES (after $JOB_SUPERGLUE)"

# ── RANSAC ablation (same deps as procrustes, different flags) ─────────────────
# Reuses unik3d + superglue outputs, only reruns the procrustes step with RANSAC.

JOB_RANSAC=$(sbatch --parsable --dependency=afterok:$JOB_SUPERGLUE \
  "$REMOTE_SCRIPT_DIR_procrustes/sbatch_remote_pipeline_procrustes.sh" \
  "$REMOTE_LOG_DIR" "$REMOTE_SCRIPT_DIR_unik3d" "$REMOTE_SCRIPT_DIR_superglue" \
  "$REMOTE_SCRIPT_DIR_procrustes" "$REMOTE_IMAGE_DIR" "$REMOTE_RESULTS_DIR" \
  "$remote_fisheye_mask_path" \
  "${remote_transform_result_path%.json}_ransac.json" \
  "$DISTANCE_THRESHOLD" "$SIZE_X" "$SIZE_Y" \
  "--use-ransac")
echo "Submitted procrustes+RANSAC: $JOB_RANSAC (after $JOB_SUPERGLUE)"

# ── Pinhole baseline (independent — only needs images) ────────────────────────

JOB_PINHOLE=$(sbatch --parsable \
  "$REMOTE_SCRIPT_DIR_pinhole/sbatch_pinhole_baseline.sh" \
  "$REMOTE_LOG_DIR" "$REMOTE_IMAGE_DIR" "$REMOTE_RESULTS_DIR" \
  "$CALIB_PATH" "$REMOTE_SCRIPT_DIR_pinhole")
echo "Submitted pinhole baseline: $JOB_PINHOLE"

# ── DUSt3R baseline (independent — only needs images) ─────────────────────────

JOB_DUST3R=$(sbatch --parsable \
  "$REMOTE_SCRIPT_DIR_dust3r/sbatch_dust3r_baseline.sh" \
  "$REMOTE_LOG_DIR" "$REMOTE_IMAGE_DIR" "$REMOTE_RESULTS_DIR" \
  "$REMOTE_SCRIPT_DIR_dust3r" "true" "$CALIB_PATH")
echo "Submitted dust3r baseline: $JOB_DUST3R"

echo ""
echo "All jobs submitted."
echo "  unik3d:            $JOB_UNIK3D"
echo "  superglue:         $JOB_SUPERGLUE"
echo "  procrustes (ours): $JOB_PROCRUSTES"
echo "  procrustes+RANSAC: $JOB_RANSAC"
echo "  pinhole:           $JOB_PINHOLE"
echo "  dust3r:            $JOB_DUST3R"

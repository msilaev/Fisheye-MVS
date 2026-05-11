#!/bin/bash
# Regenerate result_gt.json for all pairs under PAIRS_DIR in a single batch job.
#
# Use this to fix up result_gt.json after the frame-convention correction was
# added to generate_result_gt.py (ADT images rotated 90 deg CW before UniK3D,
# so GT must be converted to the rotated-image frame with --rotate).
#
# Args:
#   $1  PAIRS_DIR    directory whose immediate subdirs are pair dirs
#                    (each must contain image0.txt and image1.txt)
#   $2  RESULTS_DIR  root results dir; overwrites $RESULTS_DIR/<pair>/result_gt.json
#   $3  PROJ_DIR     Fisheye-MVS project root on remote
#   $4  CALIB_TYPE   aria     -> apply 90-deg rotation correction (ADT)
#                    kitti360 -> no correction
#                    (default: aria)
#
# Usage examples (run on remote login node):
#   # ADT seq133 standard pairs
#   sbatch src_eval/sbatch_regen_gt.sh \
#       /scratch/work/<remote-user>/3d/experiments/ADT_seq133/eval_full/test_pairs \
#       /scratch/work/<remote-user>/3d/experiments/ADT_seq133/eval_full/results \
#       /scratch/work/<remote-user>/3d/Fisheye-MVS aria
#
#   # ADT seq133 custom pairs
#   sbatch src_eval/sbatch_regen_gt.sh \
#       /scratch/work/<remote-user>/3d/experiments/ADT_seq133/eval_full/test_pairs_custom \
#       /scratch/work/<remote-user>/3d/experiments/ADT_seq133/eval_full/results \
#       /scratch/work/<remote-user>/3d/Fisheye-MVS aria
#
#   # KITTI-360
#   sbatch src_eval/sbatch_regen_gt.sh \
#       /scratch/work/<remote-user>/3d/experiments/KITTI-360/eval_full/test_pairs \
#       /scratch/work/<remote-user>/3d/experiments/KITTI-360/eval_full/results \
#       /scratch/work/<remote-user>/3d/Fisheye-MVS kitti360

#SBATCH --job-name=regen_gt
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=batch-skl

set -eo pipefail
set -x

PAIRS_DIR="$1"
RESULTS_DIR="$2"
PROJ_DIR="$3"
CALIB_TYPE="${4:-aria}"

ROTATE_FLAG=""
[ "$CALIB_TYPE" = "aria" ] && ROTATE_FLAG="--rotate"

module load mamba
eval "$(mamba shell hook --shell bash)"
mamba activate 3d_pose_env

n_ok=0
n_skip=0
n_fail=0

for PAIR_DIR in "${PAIRS_DIR}"/*/; do
    PAIR_NAME="$(basename "$PAIR_DIR")"
    POSE0="${PAIR_DIR}image0.txt"
    POSE1="${PAIR_DIR}image1.txt"
    OUT="${RESULTS_DIR}/${PAIR_NAME}/result_gt.json"

    if [ ! -f "$POSE0" ] || [ ! -f "$POSE1" ]; then
        echo "[SKIP] $PAIR_NAME — poses not found"
        n_skip=$((n_skip + 1))
        continue
    fi

    mkdir -p "${RESULTS_DIR}/${PAIR_NAME}"

    if python "${PROJ_DIR}/src_eval/generate_result_gt.py" \
            --cam2w-0 "$POSE0" \
            --cam2w-1 "$POSE1" \
            --output  "$OUT" \
            $ROTATE_FLAG; then
        n_ok=$((n_ok + 1))
    else
        echo "[WARN] Failed: $PAIR_NAME"
        n_fail=$((n_fail + 1))
    fi
done

echo "[DONE] result_gt.json: ${n_ok} regenerated, ${n_skip} skipped (no poses), ${n_fail} failed"

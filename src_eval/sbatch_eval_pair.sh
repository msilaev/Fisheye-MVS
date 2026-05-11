#!/bin/bash
# Run all evaluation methods on a single pair directory.
# Methods: Procrustes (ours), Pinhole, DUSt3R raw, DUSt3R+rect
#
# Args:
#   $1  PAIR_DIR        path to the pair directory (contains image0.jpg, image1.jpg, *.txt)
#   $2  RESULTS_DIR     where to write per-method result JSONs
#   $3  LOG_DIR         where to write logs
#   $4  PROJ_DIR        Fisheye-MVS project root on remote
#   $5  UNIK3D_DIR      UniK3D repo root
#   $6  SUPERGLUE_DIR   SuperGluePretrainedNetwork repo root
#   $7  DUST3R_DIR      DUSt3R repo root
#   $8  FISHEYE_MASK    path to MaskADT_rot.png
#   $9  DISTANCE_THRESH distance threshold for procrustes (default 1000)
#   $10 SIZE_X          image width (default 1408)
#   $11 SIZE_Y          image height (default 1408)

#SBATCH --job-name=eval_pair
#SBATCH --output=eval_pair-%j.log
#SBATCH --error=eval_pair-%j.err
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1

set -e
set -x

PAIR_DIR="$1"
RESULTS_DIR="$2"
LOG_DIR="$3"
PROJ_DIR="$4"
UNIK3D_DIR="$5"
SUPERGLUE_DIR="$6"
DUST3R_DIR="$7"
FISHEYE_MASK="$8"
DISTANCE_THRESH="${9:-1000}"
SIZE_X="${10:-1408}"
SIZE_Y="${11:-1408}"

PAIR_NAME=$(basename "$PAIR_DIR")
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

if [ -f "${PAIR_DIR}/image0.jpg" ]; then
    IMAGE0="${PAIR_DIR}/image0.jpg"; IMAGE1="${PAIR_DIR}/image1.jpg"
else
    IMAGE0="${PAIR_DIR}/image0.png"; IMAGE1="${PAIR_DIR}/image1.png"
fi

# ── 1. UniK3D depth inference ─────────────────────────────────────────────────
module load mamba
eval "$(mamba shell hook --shell bash)"
source activate mvf-unik3d

CONFIG="${UNIK3D_DIR}/configs/train/vitb.json"
cp "${PROJ_DIR}/src_unik3d/custom_unik3d_inference.py" "${UNIK3D_DIR}/scripts/"
cd "${UNIK3D_DIR}/scripts"
python custom_unik3d_inference.py \
    --config-file "$CONFIG" \
    --data-source-dir "$PAIR_DIR" \
    --output-dir "$RESULTS_DIR" \
    --images-output-dir "$RESULTS_DIR" \
    --camera-path "${UNIK3D_DIR}/assets/demo/scannet.json" \
    > "${LOG_DIR}/${PAIR_NAME}_unik3d.log" 2>&1

# ── 2. SuperGlue matching ─────────────────────────────────────────────────────
source activate superglue38

# Step 2a: run match_pairs.py from the SuperGluePretrainedNetwork repo
cd "$SUPERGLUE_DIR"
python match_pairs.py \
    --resize 1600 \
    --superglue outdoor \
    --max_keypoints 2048 \
    --nms_radius 3 \
    --resize_float \
    --input_dir   "$PAIR_DIR" \
    --input_pairs "${PAIR_DIR}/image_pairs.txt" \
    --output_dir  "$RESULTS_DIR" \
    > "${LOG_DIR}/${PAIR_NAME}_superglue.log" 2>&1

# Step 2b: extract mkpts from the .npz into .npy files
cd "${PROJ_DIR}/src_superglue"
python superglue_extract_pairs.py \
    --matches-file-npz "${RESULTS_DIR}/image0_image1_matches.npz" \
    --size_x "$SIZE_X" \
    --size_y "$SIZE_Y" \
    --mkpts1-file "${RESULTS_DIR}/mkpts1.npy" \
    --mkpts2-file "${RESULTS_DIR}/mkpts2.npy" \
    >> "${LOG_DIR}/${PAIR_NAME}_superglue.log" 2>&1

# ── 3. Procrustes pose estimation (ours) ─────────────────────────────────────
source activate 3d_pose_env

cd "${PROJ_DIR}/src_procrustes"
python pose_estimation_procrustes.py \
    --point1  "${RESULTS_DIR}/image0_points.npy" \
    --point2  "${RESULTS_DIR}/image1_points.npy" \
    --mkpts1  "${RESULTS_DIR}/mkpts1.npy" \
    --mkpts2  "${RESULTS_DIR}/mkpts2.npy" \
    --remote_fisheye_mask_path "$FISHEYE_MASK" \
    --remote_transform_result_path "${RESULTS_DIR}/result_procrustes.json" \
    --distance_threshold "$DISTANCE_THRESH" \
    --size_x "$SIZE_X" --size_y "$SIZE_Y" \
    > "${LOG_DIR}/${PAIR_NAME}_procrustes.log" 2>&1

# ── 4. Pinhole baseline (projectaria_tools rectification + SIFT + E-matrix) ──
python "${PROJ_DIR}/src_baseline_pinhole/run_pinhole_baseline.py" \
    --image0 "$IMAGE0" \
    --image1 "$IMAGE1" \
    --output "${RESULTS_DIR}/result_pinhole.json" \
    > "${LOG_DIR}/${PAIR_NAME}_pinhole.log" 2>&1

# ── 5. DUSt3R raw fisheye ─────────────────────────────────────────────────────
source activate dust3r_env
export PYTHONPATH="${DUST3R_DIR}:${DUST3R_DIR}/croco:${PYTHONPATH:-}"

python "${PROJ_DIR}/src_baseline_dust3r/run_dust3r_baseline.py" \
    --image0 "$IMAGE0" \
    --image1 "$IMAGE1" \
    --output "${RESULTS_DIR}/result_dust3r.json" \
    --device cuda \
    > "${LOG_DIR}/${PAIR_NAME}_dust3r.log" 2>&1

# ── 6. DUSt3R + rectification ────────────────────────────────────────────────
python "${PROJ_DIR}/src_baseline_dust3r/run_dust3r_baseline.py" \
    --image0 "$IMAGE0" \
    --image1 "$IMAGE1" \
    --output "${RESULTS_DIR}/result_dust3r_rect.json" \
    --device cuda \
    --rectify \
    > "${LOG_DIR}/${PAIR_NAME}_dust3r_rect.log" 2>&1

echo "[DONE] $PAIR_NAME"

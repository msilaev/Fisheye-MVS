#!/bin/bash
# Local batch renderer for already-downloaded rendered-example pair folders.
#
# Produces comparison PNGs with:
#   GT | Ours+RANSAC | MADPose | MADPose+rect
#
# It uses the camera pose saved after manual rotation from:
#   - run_visualise_check_clouds_adt.sh
#   - run_visualise_check_clouds_kitti.sh
#
# Usage:
#   bash src_visualise_local/local_render_examples.sh adt
#   bash src_visualise_local/local_render_examples.sh kitti
#   PAIRS="pair_rot_40-50_trans_1.0-1.5_3" \
#     bash src_visualise_local/local_render_examples.sh adt
#   bash src_visualise_local/local_render_examples.sh kitti \
#     pair_rot_00-10_trans_10.0-20.0_2 pair_rot_10-20_trans_5.0-10.0_4
set -euo pipefail
set -x

DATASET="${1:-adt}"
if [ "$#" -gt 0 ]; then
    shift
fi

source "$(dirname "$0")/../.env"
source "$(dirname "$0")/../path_config.sh"

LOCAL_SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDER_SCRIPT="${LOCAL_SRC_DIR}/../src_visualise/render_merged_clouds.py"
PYTHON="${PYTHON:-python}"
METHODS="Ours+RANSAC:result_procrustes_ransac.json,MADPose:result_madpose.json"
RENDER_WIDTH="${RENDER_WIDTH:-1920}"
RENDER_HEIGHT="${RENDER_HEIGHT:-1080}"
IMAGE_ROTATION_DEG="${IMAGE_ROTATION_DEG:-}"
LABEL_FONT_SCALE="${LABEL_FONT_SCALE:-2.4}"

if [ "$DATASET" = "kitti" ]; then
    source "$(dirname "$0")/../experiment_config_kitti.sh"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-KITTI-360}"
    DIST_THR="${DISTANCE_THRESHOLD_PLT:-50}"
    MASK_PATH="${LOCAL_SRC_DIR}/../assets/fisheye_masks/MaskKitti360.png"
    CAMERA_BASENAME="camera_kitti.json"
    DEFAULT_IMAGE_ROTATION_DEG=0
else
    source "$(dirname "$0")/../experiment_config_adt.sh"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-ADT_seq133}"
    DIST_THR="${DISTANCE_THRESHOLD_PLT:-1000}"
    MASK_PATH="${LOCAL_SRC_DIR}/../assets/fisheye_masks/MaskADT_rot.png"
    CAMERA_BASENAME="camera_adt.json"
    DEFAULT_IMAGE_ROTATION_DEG=90
fi

if [ -z "$IMAGE_ROTATION_DEG" ]; then
    IMAGE_ROTATION_DEG="$DEFAULT_IMAGE_ROTATION_DEG"
fi

PREFERRED_DIR="${LOCAL_ROOT}/experiments/${EXPERIMENT_NAME}/rendered_examples"
LEGACY_DIR="${LOCAL_ROOT}/experiments/${EXPERIMENT_NAME}/rendered_example"

if [ -d "$PREFERRED_DIR" ]; then
    LOCAL_RENDER_DIR="$PREFERRED_DIR"
elif [ -d "$LEGACY_DIR" ]; then
    LOCAL_RENDER_DIR="$LEGACY_DIR"
else
    mkdir -p "$PREFERRED_DIR"
    LOCAL_RENDER_DIR="$PREFERRED_DIR"
fi

if [ ! -f "$RENDER_SCRIPT" ]; then
    echo "[ERROR] render_merged_clouds.py not found at: $RENDER_SCRIPT"
    exit 1
fi

if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] Python executable not found or not executable: $PYTHON"
    exit 1
fi

PAIR_NAMES=()
if [ -n "${PAIRS:-}" ]; then
    IFS=' ' read -r -a PAIR_NAMES <<< "$PAIRS"
elif [ "$#" -gt 0 ]; then
    PAIR_NAMES=("$@")
else
    while IFS= read -r pair_path; do
        PAIR_NAMES+=("$(basename "$pair_path")")
    done < <(find "$LOCAL_RENDER_DIR" -mindepth 1 -maxdepth 1 -type d -name 'pair_*' | sort)
fi

if [ ${#PAIR_NAMES[@]} -eq 0 ]; then
    echo "[ERROR] No pair folders found under: $LOCAL_RENDER_DIR"
    exit 1
fi

echo "[LOCAL] Rendering dataset=$DATASET from $LOCAL_RENDER_DIR"
echo "[LOCAL] Render size=${RENDER_WIDTH}x${RENDER_HEIGHT}, image rotation=${IMAGE_ROTATION_DEG}°, label scale=${LABEL_FONT_SCALE}"

auto_detect_image_ext() {
    local pair_dir="$1"
    if [ -f "${pair_dir}/image0.jpg" ]; then
        echo "jpg"
    else
        echo "png"
    fi
}

for PAIR_NAME in "${PAIR_NAMES[@]}"; do
    PAIR_DIR="${LOCAL_RENDER_DIR}/${PAIR_NAME}"
    OUT_PNG="${LOCAL_RENDER_DIR}/${PAIR_NAME}.png"

    if [ ! -d "$PAIR_DIR" ]; then
        echo "[WARN] Pair directory not found, skipping: $PAIR_DIR"
        continue
    fi

    IMG_EXT="$(auto_detect_image_ext "$PAIR_DIR")"
    CAMERA_JSON="${PAIR_DIR}/${CAMERA_BASENAME}"
    if [ ! -f "$CAMERA_JSON" ]; then
        CAMERA_JSON="${LOCAL_SRC_DIR}/${CAMERA_BASENAME}"
    fi

    REQUIRED_FILES=(
        "${PAIR_DIR}/image0_points.npy"
        "${PAIR_DIR}/image1_points.npy"
        "${PAIR_DIR}/image0.txt"
        "${PAIR_DIR}/image1.txt"
        "${PAIR_DIR}/image0.${IMG_EXT}"
        "${PAIR_DIR}/image1.${IMG_EXT}"
        "${PAIR_DIR}/result_procrustes_ransac.json"
    )

    MISSING_REQUIRED=0
    for f in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$f" ]; then
            echo "[WARN] Missing required file for ${PAIR_NAME}: $f"
            MISSING_REQUIRED=1
        fi
    done
    if [ "$MISSING_REQUIRED" -ne 0 ]; then
        echo "[WARN] Skipping ${PAIR_NAME} due to missing inputs"
        continue
    fi

    CMD=(
        "$PYTHON" "$RENDER_SCRIPT"
        --point0 "${PAIR_DIR}/image0_points.npy"
        --point1 "${PAIR_DIR}/image1_points.npy"
        --img0 "${PAIR_DIR}/image0.${IMG_EXT}"
        --img1 "${PAIR_DIR}/image1.${IMG_EXT}"
        --cam2w0 "${PAIR_DIR}/image0.txt"
        --cam2w1 "${PAIR_DIR}/image1.txt"
        --methods "$METHODS"
        --results-dir "$PAIR_DIR"
        --output "$OUT_PNG"
        --dataset "$DATASET"
        --distance-threshold "$DIST_THR"
        --point-size 2.0
        --width "$RENDER_WIDTH"
        --height "$RENDER_HEIGHT"
        --image-rotation "$IMAGE_ROTATION_DEG"
        --label-font-scale "$LABEL_FONT_SCALE"
    )

    # Custom ADT pairs: images are physically rotated 90° CW on disk.
    # → Apply GT frame correction so GT matches the rotated cloud frame.
    # → Skip display rotation (images are already upright after rotation).
    # → Mask convention: MaskADT_rot.png already matches rotated images.
    if [[ "$DATASET" == "adt" && "$PAIR_NAME" == pair_custom_* ]]; then
        CMD+=(--rotate-gt)
        # Replace --image-rotation value: custom images are already rotated
        for i in "${!CMD[@]}"; do
            if [[ "${CMD[$i]}" == "--image-rotation" ]]; then
                CMD[$((i+1))]="0"
                break
            fi
        done
    fi

    if [ -f "$CAMERA_JSON" ]; then
        CMD+=(--camera-json "$CAMERA_JSON")
        echo "[LOCAL] Using camera pose: $CAMERA_JSON"
    else
        echo "[LOCAL] No saved camera pose for ${PAIR_NAME}; using auto-fit view"
    fi

    if [ -f "$MASK_PATH" ]; then
        CMD+=(--mask "$MASK_PATH")
    else
        echo "[LOCAL] No mask found at $MASK_PATH; rendering without fisheye mask"
    fi

    "${CMD[@]}"
    echo "[DONE] ${PAIR_NAME} -> ${OUT_PNG}"
done

echo "[BATCH DONE] All requested local renders processed."

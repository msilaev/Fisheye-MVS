#!/bin/bash
set -e
set -x

###############################################
# 1. Load environment variables
###############################################
if [ -f ".env" ]; then
    source .env
fi

###############################################
# 2. Local configuration
###############################################

# Custom pair names (one per row). Keep only one PAIR_NAME active.
PAIR_NAME="pair_custom_0000006360__0000006367"
PAIR_NAME="pair_custom_0000008603__0000008587"

# Regular sampled pairs (uncomment one instead of a custom pair above)
# PAIR_NAME="pair_rot_10-20_trans_5.0-10.0_4"

# Keep all pair-related local data in one folder under rendered_examples.
# This folder should contain image*.npy/.jpg/.txt and result_*.json.
LOCAL_PAIR_DIR="C:/Users/mikes/Documents/experiments/KITTI-360/rendered_examples/${PAIR_NAME}"

LOCAL_SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASK_DIR="C:/Users/mikes/Documents/Fisheye-MVS/assets/fisheye_masks"
CAMERA_PARAM="${CAMERA_PARAM:-${LOCAL_PAIR_DIR}/camera_kitti.json}"
METHOD="${METHOD:-procrustes_ransac}"

if [ -z "${RESULT_JSON:-}" ]; then
    case "$METHOD" in
        procrustes|ours)
            RESULT_JSON="${LOCAL_PAIR_DIR}/result_procrustes.json"
            ;;
        procrustes_ransac|ours_ransac|ransac)
            RESULT_JSON="${LOCAL_PAIR_DIR}/result_procrustes_ransac.json"
            ;;
        madpose)
            RESULT_JSON="${LOCAL_PAIR_DIR}/result_madpose.json"
            ;;
        madpose_rect)
            RESULT_JSON="${LOCAL_PAIR_DIR}/result_madpose_rect.json"
            ;;
        dust3r)
            RESULT_JSON="${LOCAL_PAIR_DIR}/result_dust3r.json"
            ;;
        dust3r_rect)
            RESULT_JSON="${LOCAL_PAIR_DIR}/result_dust3r_rect.json"
            ;;
        pinhole)
            RESULT_JSON="${LOCAL_PAIR_DIR}/result_pinhole.json"
            ;;
        gt)
            RESULT_JSON="${LOCAL_PAIR_DIR}/result_gt.json"
            ;;
        *)
            echo "[ERROR] Unknown METHOD=$METHOD"
            exit 1
            ;;
    esac
fi

if [ ! -d "$LOCAL_PAIR_DIR" ]; then
    echo "[ERROR] LOCAL_PAIR_DIR not found: $LOCAL_PAIR_DIR"
    echo "[INFO] Available pair dirs in rendered_examples:"
    ls -1 "C:/Users/mikes/Documents/experiments/KITTI-360/rendered_examples" 2>/dev/null | grep '^pair_' || true
    echo "[INFO] Note: cloud visualisation needs image*_points.npy inside the selected pair folder."
    exit 1
fi

if [ ! -f "$RESULT_JSON" ]; then
    echo "[ERROR] RESULT_JSON not found: $RESULT_JSON"
    echo "[INFO] Available JSON files in $LOCAL_PAIR_DIR:"
    ls -1 "$LOCAL_PAIR_DIR"/result_*.json 2>/dev/null || true
    exit 1
fi

echo "[LOCAL] LOCAL_PAIR_DIR=$LOCAL_PAIR_DIR"
echo "[LOCAL] METHOD=$METHOD"
echo "[LOCAL] RESULT_JSON=$RESULT_JSON"
echo "[LOCAL] CAMERA_PARAM=$CAMERA_PARAM"

DISTANCE_THRESHOLD=${DISTANCE_THRESHOLD:-50}

###############################################
# 3. Python from conda env
###############################################
PYTHON="python"

###############################################
# 4. Run visualisation
###############################################
# Auto-detect image extension
IMG_EXT="png"
[ -f "${LOCAL_PAIR_DIR}/image0.jpg" ] && IMG_EXT="jpg"

cd "$LOCAL_SRC_DIR"
"$PYTHON" visualise_check_clouds_kitti.py \
    --point1              "${LOCAL_PAIR_DIR}/image0_points.npy" \
    --point2              "${LOCAL_PAIR_DIR}/image1_points.npy" \
    --img1                "${LOCAL_PAIR_DIR}/image0.${IMG_EXT}" \
    --img2                "${LOCAL_PAIR_DIR}/image1.${IMG_EXT}" \
    --cam2w_1             "${LOCAL_PAIR_DIR}/image0.txt" \
    --cam2w_2             "${LOCAL_PAIR_DIR}/image1.txt" \
    --transform_json      "${RESULT_JSON}" \
    --mask_fisheye_kitti  "${MASK_DIR}/MaskKitti360.png" \
    --distance_threshold  $DISTANCE_THRESHOLD \
    --camera_param        "${CAMERA_PARAM}"
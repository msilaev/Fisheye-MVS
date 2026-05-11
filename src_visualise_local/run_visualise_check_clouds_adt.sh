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
PAIR_NAME="pair_custom_Apartment_release_clean_seq136_M1292_frame000900__Apartment_release_clean_seq136_M1292_frame000840"
PAIR_NAME="pair_custom_Apartment_release_clean_seq136_M1292_frame002563__Apartment_release_clean_seq136_M1292_frame002537"
#PAIR_NAME="pair_custom_Apartment_release_decoration_seq136_M1292_frame002000__Apartment_release_decoration_seq136_M1292_frame001965"
#PAIR_NAME="pair_custom_Apartment_release_multiuser_cook_seq141_M1292_frame002033__Apartment_release_multiuser_cook_seq141_M1292_frame002090"

# Keep all pair-related local data in one folder under rendered_examples.
# This folder should contain image*.npy/.jpg/.txt and result_*.json.
LOCAL_PAIR_DIR="C:/Users/mikes/Documents/experiments/ADT_seq133/rendered_examples/${PAIR_NAME}"

LOCAL_SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASK_DIR="C:/Users/mikes/Documents/Fisheye-MVS/assets/fisheye_masks"
CAMERA_PARAM="${CAMERA_PARAM:-${LOCAL_PAIR_DIR}/camera_adt.json}"
METHOD="${METHOD:-procrustes}"  # Default to procrustes if not set

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
    echo "[INFO] Available custom pair dirs in rendered_examples:"
    ls -1 "C:/Users/mikes/Documents/experiments/ADT_seq133/rendered_examples" | grep '^pair_custom_.*$' || true
    echo "[INFO] Note: cloud visualisation needs image*_points.npy inside the selected pair folder."
    exit 1
fi

if [ ! -f "$RESULT_JSON" ]; then
    echo "[ERROR] RESULT_JSON not found: $RESULT_JSON"
    echo "[INFO] Available JSON files in $LOCAL_PAIR_DIR:"
    ls -1 "$LOCAL_PAIR_DIR"/result_*.json 2>/dev/null || true
    exit 1
fi

###############################################
# 2b. Auto-fetch npy point clouds if missing
###############################################
if [ ! -f "${LOCAL_PAIR_DIR}/image0_points.npy" ] || [ ! -f "${LOCAL_PAIR_DIR}/image1_points.npy" ]; then
    echo "[INFO] Point cloud .npy files missing — attempting to fetch from remote..."

    PROJ_ROOT="$(cd "${LOCAL_SRC_DIR}/.." && pwd)"
    # Source remote paths (path_config.sh) and SSH credentials (.env at project root)
    if [ -f "${PROJ_ROOT}/.env" ]; then source "${PROJ_ROOT}/.env"; fi
    if [ -f "${PROJ_ROOT}/path_config.sh" ]; then source "${PROJ_ROOT}/path_config.sh"; fi

    if [ -z "${REMOTE_USER:-}" ] || [ -z "${REMOTE_HOST:-}" ] || [ -z "${REMOTE_EXPERIMENTS_DIR:-}" ]; then
        echo "[ERROR] Remote credentials not set. Ensure .env + path_config.sh define"
        echo "        REMOTE_USER, REMOTE_HOST, REMOTE_EXPERIMENTS_DIR."
        exit 1
    fi

    # Extract experiment name from LOCAL_PAIR_DIR (e.g. ADT_seq133)
    EXPERIMENT_NAME_FETCH="$(echo "$LOCAL_PAIR_DIR" | sed 's|.*/experiments/\([^/]*\)/.*|\1|')"
    REMOTE_EVAL="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME_FETCH}/eval_full"

    NPY_FETCHED=0
    for REMOTE_PAIR in \
        "${REMOTE_EVAL}/rendered_example/${PAIR_NAME}" \
        "${REMOTE_EVAL}/rendered_examples/${PAIR_NAME}" \
        "${REMOTE_EVAL}/results/${PAIR_NAME}"; do
        if ssh "${REMOTE_USER}@${REMOTE_HOST}" \
               "[ -f '${REMOTE_PAIR}/image0_points.npy' ]" 2>/dev/null; then
            echo "[INFO] Fetching npy from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PAIR}/"
            scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PAIR}/image0_points.npy" \
                "${LOCAL_PAIR_DIR}/image0_points.npy"
            scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PAIR}/image1_points.npy" \
                "${LOCAL_PAIR_DIR}/image1_points.npy"
            NPY_FETCHED=1
            echo "[INFO] npy files fetched successfully."
            break
        fi
    done

    if [ "$NPY_FETCHED" -eq 0 ]; then
        echo "[ERROR] Could not find npy files on remote in any of:"
        echo "        ${REMOTE_EVAL}/rendered_example/${PAIR_NAME}"
        echo "        ${REMOTE_EVAL}/results/${PAIR_NAME}"
        echo "[INFO]  Re-run the remote render job first:"
        echo "        bash src_visualise/run_visualise_frames_adt.sh"
        exit 1
    fi
fi

echo "[LOCAL] LOCAL_PAIR_DIR=$LOCAL_PAIR_DIR"
echo "[LOCAL] METHOD=$METHOD"
echo "[LOCAL] RESULT_JSON=$RESULT_JSON"
echo "[LOCAL] CAMERA_PARAM=$CAMERA_PARAM"


###############################################
# 3. Python from conda env
###############################################
PYTHON="python"

###############################################
# 4. Run visualisation
###############################################
# Auto-detect image extension
IMG_EXT="jpg"
[ -f "${LOCAL_PAIR_DIR}/image0.png" ] && IMG_EXT="png"

cd "$LOCAL_SRC_DIR"
"$PYTHON" visualise_check_clouds_adt.py \
    --point1         "${LOCAL_PAIR_DIR}/image0_points.npy" \
    --point2         "${LOCAL_PAIR_DIR}/image1_points.npy" \
    --img1           "${LOCAL_PAIR_DIR}/image0.${IMG_EXT}" \
    --img2           "${LOCAL_PAIR_DIR}/image1.${IMG_EXT}" \
    --cam2w_1        "${LOCAL_PAIR_DIR}/image0.txt" \
    --cam2w_2        "${LOCAL_PAIR_DIR}/image1.txt" \
    --transform_json "${RESULT_JSON}" \
    --mask_fisheye_adt       "${MASK_DIR}/MaskADT_rot.png" \
    --camera_param           "${CAMERA_PARAM}"
#!/bin/bash
# GPU job: re-run UniK3D on selected pairs, render merged-cloud comparison figures.
#
# Args:
#   $1  PAIRS_LIST    text file with one pair name per line (e.g. pair_rot_20-30_5)
#   $2  PAIRS_DIR     dir with pair subdirs (images + GT poses)
#   $3  RESULTS_DIR   dir with per-pair result JSON subdirs
#   $4  OUTPUT_DIR    where rendered PNGs go
#   $5  PROJ_DIR      Fisheye-MVS root
#   $6  UNIK3D_DIR    UniK3D repo root
#   $7  DATASET       adt or kitti
#   $8  DIST_THR      distance threshold (metres); default 50 for ADT, 100 for KITTI
#   $9  CAMERA_JSON   optional fixed-viewpoint camera JSON (pass "" to skip)
#   $10 MASK          optional fisheye mask PNG (pass "" to skip)

#SBATCH --job-name=render_clouds
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1

set -x

export HF_HOME=/scratch/work/<remote-user>/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/work/<remote-user>/.cache/huggingface/hub
export TORCH_HOME=/scratch/work/<remote-user>/.cache/torch
export XDG_CACHE_HOME=/scratch/work/<remote-user>/.cache

PAIRS_LIST="$1"
PAIRS_DIR="$2"
RESULTS_DIR="$3"
OUTPUT_DIR="$4"
PROJ_DIR="$5"
UNIK3D_DIR="$6"
DATASET="${7:-adt}"
DIST_THR="${8:-50}"
CAMERA_JSON="${9:-}"
MASK="${10:-}"

METHODS="Ours+RANSAC:result_procrustes_ransac.json,MADPose:result_madpose.json,MADPose+rect:result_madpose_rect.json"

module load mamba
eval "$(mamba shell hook --shell bash)"
source "${PROJ_DIR}/path_config.sh"

PREPARE_SCRIPT="${PROJ_DIR}/src_visualise/prepare_render_pairs.py"
PREPARED_PAIRS_DIR="$(dirname "$PAIRS_DIR")/test_pairs_custom"

mkdir -p "$OUTPUT_DIR" "$PREPARED_PAIRS_DIR"

# Copy render script to UniK3D scripts dir so it runs in correct env
cp "${PROJ_DIR}/src_visualise/render_merged_clouds.py" "${UNIK3D_DIR}/scripts/"
cp "${PROJ_DIR}/src_unik3d/custom_unik3d_inference.py" "${UNIK3D_DIR}/scripts/"

CONFIG="${UNIK3D_DIR}/configs/train/vitb.json"

resolve_pair_spec() {
    local pair_spec="$1"

    if [[ "$pair_spec" == frames:* ]]; then
        mamba activate 3d_pose_env

        local rotate_flag=""
        local img_ext="png"
        if [ "$DATASET" = "adt" ]; then
            rotate_flag="--rotate"
            img_ext="jpg"
        fi

        python "$PREPARE_SCRIPT" \
            --dataset "$DATASET" \
            --pair-spec "$pair_spec" \
            --default-pairs-dir "$PAIRS_DIR" \
            --default-results-dir "$RESULTS_DIR" \
            --output-root "$PREPARED_PAIRS_DIR" \
            --experiments-root "$REMOTE_EXPERIMENTS_DIR" \
            --shared-datasets-root "$REMOTE_DATASETS_DIR" \
            --image-ext "$img_ext" \
            $rotate_flag
    else
        printf '%s\t%s\t%s\n' "$pair_spec" "${PAIRS_DIR}/${pair_spec}" "${RESULTS_DIR}/${pair_spec}"
    fi
}

while IFS= read -r PAIR_SPEC; do
    [ -z "$PAIR_SPEC" ] && continue

    META="$(resolve_pair_spec "$PAIR_SPEC")" || {
        echo "[WARN] Failed to resolve pair spec: $PAIR_SPEC"
        continue
    }
    IFS=$'\t' read -r PAIR_NAME PAIR_DIR RES_DIR <<< "$META"

    OUT_PNG="${OUTPUT_DIR}/${PAIR_NAME}.png"

    # Per-pair output dir (persisted for local download / local visualisation)
    PAIR_OUT_DIR="${OUTPUT_DIR}/${PAIR_NAME}"
    mkdir -p "$PAIR_OUT_DIR"

    # Skip only when both the summary PNG and the per-pair artifacts already exist.
    if [ -f "$OUT_PNG" ] && \
       [ -f "${PAIR_OUT_DIR}/image0_points.npy" ] && \
       [ -f "${PAIR_OUT_DIR}/image1_points.npy" ] && \
       [ -f "${PAIR_OUT_DIR}/image0.txt" ] && \
       [ -f "${PAIR_OUT_DIR}/image1.txt" ] && \
       [ -f "${PAIR_OUT_DIR}/result_procrustes_ransac.json" ]; then
        echo "[SKIP] $PAIR_NAME already rendered and pair folder is populated"
        continue
    fi

    # Determine image extension
    if [ -f "${PAIR_DIR}/image0.jpg" ]; then
        IMG0="${PAIR_DIR}/image0.jpg"
        IMG1="${PAIR_DIR}/image1.jpg"
    else
        IMG0="${PAIR_DIR}/image0.png"
        IMG1="${PAIR_DIR}/image1.png"
    fi

    # ── UniK3D inference ────────────────────────────────────────────────────
    mamba activate mvf-unik3d
    cd "${UNIK3D_DIR}/scripts"

    # Copy images into pair output dir for UniK3D
    if cp "$IMG0" "${PAIR_OUT_DIR}/image0.jpg" 2>/dev/null; then
        POUT_IMG0="${PAIR_OUT_DIR}/image0.jpg"
    else
        cp "$IMG0" "${PAIR_OUT_DIR}/image0.png"
        POUT_IMG0="${PAIR_OUT_DIR}/image0.png"
    fi
    if cp "$IMG1" "${PAIR_OUT_DIR}/image1.jpg" 2>/dev/null; then
        POUT_IMG1="${PAIR_OUT_DIR}/image1.jpg"
    else
        cp "$IMG1" "${PAIR_OUT_DIR}/image1.png"
        POUT_IMG1="${PAIR_OUT_DIR}/image1.png"
    fi

    python custom_unik3d_inference.py \
        --config-file "$CONFIG" \
        --data-source-dir "$PAIR_OUT_DIR" \
        --output-dir "$PAIR_OUT_DIR" \
        --images-output-dir "$PAIR_OUT_DIR" \
        --camera-path "${UNIK3D_DIR}/assets/demo/scannet.json"

    # Copy GT pose files
    cp "${PAIR_DIR}/image0.txt" "${PAIR_OUT_DIR}/image0.txt"
    cp "${PAIR_DIR}/image1.txt" "${PAIR_OUT_DIR}/image1.txt"

    # Copy result JSONs from results/ into pair output dir
    if [ -d "$RES_DIR" ]; then
        cp "$RES_DIR"/*.json "${PAIR_OUT_DIR}/" 2>/dev/null || true
    fi

    # ── Render comparison (open3d lives in 3d_pose_env) ────────────────────
    mamba activate 3d_pose_env

    CAM_ARG=""
    [ -n "$CAMERA_JSON" ] && [ -f "$CAMERA_JSON" ] && CAM_ARG="--camera-json $CAMERA_JSON"

    MASK_ARG=""
    [ -n "$MASK" ] && [ -f "$MASK" ] && MASK_ARG="--mask $MASK"

    python "${PROJ_DIR}/src_visualise/render_merged_clouds.py" \
        --point0  "${PAIR_OUT_DIR}/image0_points.npy" \
        --point1  "${PAIR_OUT_DIR}/image1_points.npy" \
        --img0    "$POUT_IMG0" \
        --img1    "$POUT_IMG1" \
        --cam2w0  "${PAIR_OUT_DIR}/image0.txt" \
        --cam2w1  "${PAIR_OUT_DIR}/image1.txt" \
        --methods "$METHODS" \
        --results-dir "$PAIR_OUT_DIR" \
        --output  "$OUT_PNG" \
        --dataset "$DATASET" \
        --distance-threshold "$DIST_THR" \
        --point-size 2.0 \
        --width 1400 \
        --height 1000 \
        $CAM_ARG \
        $MASK_ARG

    echo "[DONE] $PAIR_NAME → $OUT_PNG"
done < "$PAIRS_LIST"

echo "[BATCH DONE]"

#!/bin/bash
# Download KITTI-360 fisheye camera 02 images and calibration for one sequence,
# then extract poses and reorganise into the layout used by select_pairs_by_rotation.py:
#
#   $REMOTE_DATASETS_DIR/KITTI-360/<sequence>/
#       images/   0000000000.png  ...   (fisheye cam02)
#       poses/    0000000000.txt  ...   (cam02-to-world 4x4)
#
# Images/calibration downloaded via public S3 URLs (no auth required).
# Poses extracted from existing KITTI_ROOT/data_poses/ using kitti360scripts.
#
# Usage (run on remote cluster from project root):
#   bash src_datasets/setup_kitti360.sh [sequence_name]
#   e.g. bash src_datasets/setup_kitti360.sh 2013_05_28_drive_0000_sync

set -e
set -x

SEQUENCE="${1:-2013_05_28_drive_0000_sync}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${PROJECT_DIR}/path_config.sh"

# ── Paths ─────────────────────────────────────────────────────────────────────
# Standard KITTI-360 layout expected by kitti360scripts:
#   KITTI_ROOT/data_2d_raw/<sequence>/image_02/...
#   KITTI_ROOT/data_poses/<sequence>/cam0_to_world.txt  (must already exist)
#   KITTI_ROOT/calibration/calib_cam_to_pose.txt  ...
KITTI_ROOT="${REMOTE_DATASETS_DIR}/KITTI-360"
SEQ_DIR="${KITTI_ROOT}/${SEQUENCE}"
DL_DIR="${KITTI_ROOT}/.downloads"   # zip cache

mkdir -p "${SEQ_DIR}/images" "${SEQ_DIR}/poses" "${DL_DIR}"

# ── Skip if already set up ────────────────────────────────────────────────────
if [ "$(ls -A "${SEQ_DIR}/images" 2>/dev/null | wc -l)" -gt 0 ] && \
   [ "$(ls -A "${SEQ_DIR}/poses"  2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[SKIP] ${SEQUENCE} already set up ($(ls "${SEQ_DIR}/images" | wc -l) images, $(ls "${SEQ_DIR}/poses" | wc -l) poses)"
    exit 0
fi

S3="https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360"

# ── Download fisheye images (cam02) ──────────────────────────────────────────
IMAGE_ZIP="${SEQUENCE}_image_02.zip"
if [ ! -f "${DL_DIR}/${IMAGE_ZIP}" ]; then
    wget -c -P "${DL_DIR}" "${S3}/data_2d_raw/${IMAGE_ZIP}"
fi
if [ ! -d "${KITTI_ROOT}/data_2d_raw/${SEQUENCE}" ]; then
    cd "${KITTI_ROOT}"
    unzip -q "${DL_DIR}/${IMAGE_ZIP}" -d data_2d_raw
fi

# ── Download poses (requires registration — uses cvlibs.net authenticated URL) ─
# Set KITTI360_USER and KITTI360_PASS in .env (register at cvlibs.net/datasets/kitti-360)
source "${PROJECT_DIR}/.env" 2>/dev/null || true
POSE_ZIP="data_poses_${SEQUENCE}.zip"
if [ ! -d "${KITTI_ROOT}/data_poses/${SEQUENCE}" ]; then
    if [ -z "${KITTI360_USER:-}" ] || [ -z "${KITTI360_PASS:-}" ]; then
        echo "ERROR: KITTI360_USER/KITTI360_PASS not set — needed for poses download"
        exit 1
    fi
    if [ ! -f "${DL_DIR}/${POSE_ZIP}" ]; then
        wget -c --user="$KITTI360_USER" --password="$KITTI360_PASS" \
            -P "${DL_DIR}" \
            "http://www.cvlibs.net/download.php?file=data_poses_${SEQUENCE}.zip"
        # cvlibs.net download.php saves as index.html — rename
        mv "${DL_DIR}/index.html" "${DL_DIR}/${POSE_ZIP}" 2>/dev/null || true
    fi
    cd "${KITTI_ROOT}"
    unzip -q "${DL_DIR}/${POSE_ZIP}"
fi

# ── Load Python environment (needed for pose extraction and cleanup) ──────────
module load mamba
eval "$(mamba shell hook --shell bash)"
source activate 3d_pose_env

# ── Extract cam02-to-world poses using kitti360scripts ────────────────────────
if [ "$(ls -A "${SEQ_DIR}/poses" 2>/dev/null | wc -l)" -eq 0 ]; then
    python "${PROJECT_DIR}/KITTI360/src_frames_poses/extract_kitti_poses.py" \
        --kitti_root    "${KITTI_ROOT}" \
        --sequence_name "${SEQUENCE}" \
        --output_dir    "${SEQ_DIR}/poses"
else
    echo "[SKIP] Poses already extracted: $(ls "${SEQ_DIR}/poses" | wc -l) files"
fi
echo "Poses:  $(ls "${SEQ_DIR}/poses" | wc -l) files"

# ── Download calibration (shared across sequences, download once) ─────────────
CALIB_ZIP="calibration.zip"
if [ ! -f "${KITTI_ROOT}/calibration/calib_cam_to_pose.txt" ]; then
    wget -c -P "${DL_DIR}" "${S3}/calibration/${CALIB_ZIP}"
    cd "${KITTI_ROOT}"
    unzip -q "${DL_DIR}/${CALIB_ZIP}"
    # zip may extract to a calibration/ subfolder — flatten if needed
    if [ -d "${KITTI_ROOT}/calibration/calibration" ]; then
        mv "${KITTI_ROOT}/calibration/calibration/"* "${KITTI_ROOT}/calibration/"
        rmdir "${KITTI_ROOT}/calibration/calibration"
    fi
fi

# ── Copy fisheye images to SEQ_DIR/images/ ───────────────────────────────────
RAW_IMG_DIR="${KITTI_ROOT}/data_2d_raw/${SEQUENCE}/image_02/data_rgb"
if [ "$(ls -A "${SEQ_DIR}/images" 2>/dev/null | wc -l)" -eq 0 ]; then
    cp "${RAW_IMG_DIR}/"*.png "${SEQ_DIR}/images/"
fi
echo "Images: $(ls "${SEQ_DIR}/images" | wc -l) files"

# ── Remove images without a matching pose ─────────────────────────────────────
export SEQ_DIR
python - <<'PYEOF'
import os
from pathlib import Path
seq_dir = Path(os.environ["SEQ_DIR"])
pose_ids = {int(f.stem) for f in (seq_dir / "poses").glob("*.txt")}
removed = 0
for f in list((seq_dir / "images").glob("*.png")):
    if int(f.stem) not in pose_ids:
        f.unlink()
        removed += 1
if removed:
    print(f"Removed {removed} images without a matching pose.")
print(f"Final: {len(pose_ids)} matched pairs.")
PYEOF

echo ""
echo "Done. Dataset ready at: ${SEQ_DIR}"

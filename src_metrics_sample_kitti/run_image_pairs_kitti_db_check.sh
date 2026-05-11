#!/bin/bash
set -e
set -x
# Script to visualize ADT (Aria Digital Twin) frames

# --- Load environment variables ---
if [ -f ".env" ]; then
  source .env
fi

SEQUENCE_NAME="2013_05_28_drive_0000_sync"
CAM_NAME="02"
LOCAL_DIR_ROOT="/worktmp/THESES/GAUSSIAN-SPLATTING/KITTI-360"
root_dir=$LOCAL_DIR_ROOT #KITTI-360
data_2d_dir=data_2d_raw
poses_dir=poses_dir

POSES_DIR=$root_dir/$poses_dir/$SEQUENCE_NAME/image_${CAM_NAME}
#IMAGE_DIR=$root_dir/$data_2d_dir/$SEQUENCE_NAME/image_${CAM_NAME}
IMAGE_DIR="${root_dir}/${data_2d_dir}/${SEQUENCE_NAME}/image_${CAM_NAME}/data_rgb"

#POSES_DIR=root_dir/$poses_dir/$SEQUENCE_NAME/"image_${CAM_NAME}"
#POSES_DIR=$root_dir/$poses_dir/$SEQUENCE_NAME/image_${CAM_NAME}

#DATA_DIR="${LOCAL_DIR_EXP}/${SEQUENCE_NAME}/adt_dataset/${SEQUENCE_NAME}"
OUT_DIR="${POSES_DIR}/pair_dataset"


# --- Prepare output directory ---
#rm -rf "$OUT_DIR"
#mkdir -p "$OUT_DIR"

# Initialize Conda for bash
eval "$(conda shell.bash hook)"
conda activate unik3d

python image_pairs_kitti_db_check.py \
  --sequence_name "${SEQUENCE_NAME}" \
  --pose_data_dir "${DATA_DIR}/poses" \
  --image_data_dir "${DATA_DIR}/images" \
  --pair_database_path "${OUT_DIR}/pair_dataset.json"
  #\
  #--rgb_stream "214-1" #\
  #--rectify

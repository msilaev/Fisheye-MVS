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
LOCAL_DIR_KITTI_ROOT="/worktmp/THESES/GAUSSIAN-SPLATTING/KITTI-360"

#root_dir=$LOCAL_DIR_ROOT #KITTI-360
data_2d_dir=data_2d_raw
poses_dir=poses_dir

POSES_DIR=$LOCAL_DIR_KITTI_ROOT/$poses_dir/$SEQUENCE_NAME/image_${CAM_NAME}

#POSES_DIR="${root_dir}/${poses_dir}/${SEQUENCE_NAME}/image_${CAM_NAME}/data_rgb"

IMAGE_DIR="${LOCAL_DIR_KITTI_ROOT}/${data_2d_dir}/${SEQUENCE_NAME}/image_${CAM_NAME}/data_rgb"

SAMPLING_DIR="${LOCAL_DIR_KITTI_ROOT}/sampling_dir"
SAMPLING_DIR_POSES="${LOCAL_DIR_KITTI_ROOT}/sampling_dir/poses"
SAMPLING_DIR_IMGS="${LOCAL_DIR_KITTI_ROOT}/sampling_dir/images"

rm -rf $SAMPLING_DIR_POSES
rm -rf $SAMPLING_DIR_IMGS

mkdir -p $SAMPLING_DIR_IMGS
mkdir -p $SAMPLING_DIR_POSES



#POSES_DIR=root_dir/$poses_dir/$SEQUENCE_NAME/"image_${CAM_NAME}"
#POSES_DIR=$root_dir/$poses_dir/$SEQUENCE_NAME/image_${CAM_NAME}

#DATA_DIR="${LOCAL_DIR_EXP}/${SEQUENCE_NAME}/adt_dataset/${SEQUENCE_NAME}"
OUT_DIR="${POSES_DIR}/pair_dataset"

# Initialize Conda for bash
eval "$(conda shell.bash hook)"
conda activate unik3d

sample_t_R_test_db_path="${SAMPLING_DIR}/sample_t_R_pair_dataset.json"

# --- Get list length using jq ---
#num_samples=$(jq length "$sample_t_R_test_db_path")

#echo "Found $num_samples entries in sample list"

# --- Loop over each index ---

python sample_t_R_test_kitti_gen.py \
    --sequence_name "${SEQUENCE_NAME}" \
    --pose_data_dir "${SAMPLING_DIR_POSES}" \
    --image_data_dir "${SAMPLING_DIR_IMGS}" \
    --pair_database_path "${OUT_DIR}/pair_dataset.json" \
    --sample_t_R_test_db_path "${sample_t_R_test_db_path}"


#!/bin/bash
#SBATCH --job-name=unik3d_inference
#SBATCH --output=unik3d_inference-%j.log
#SBATCH --error=unik3d_inference-%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1

set -e
set -x

REMOTE_LOG_DIR="$1"
REMOTE_SCRIPT_DIR_unik3d="$2"
REMOTE_SCRIPT_DIR_superglue="$3"
REMOTE_SCRIPT_DIR_procrustes="$4"
REMOTE_IMAGE_DIR="$5"
REMOTE_RESULTS_DIR="$6"
REMOTE_UNIK3D_DIR="${7}"

CONFIG="${REMOTE_UNIK3D_DIR}/configs/train/vitb.json"
module load mamba
eval "$(mamba shell hook --shell bash)"
source activate mvf-unik3d

cp "${REMOTE_SCRIPT_DIR_unik3d}/custom_unik3d_inference.py" "$REMOTE_UNIK3D_DIR/scripts/"
cd "$REMOTE_UNIK3D_DIR/scripts"

python custom_unik3d_inference.py \
    --config-file "$CONFIG" \
    --data-source-dir "$REMOTE_IMAGE_DIR" \
    --output-dir "$REMOTE_RESULTS_DIR" \
    --images-output-dir "$REMOTE_RESULTS_DIR" \
    --camera-path "$REMOTE_UNIK3D_DIR/assets/demo/scannet.json" \
    > "$REMOTE_LOG_DIR/unik3d_metrics.log" 2>&1

echo "[REMOTE] Inference finished successfully."





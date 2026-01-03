#!/bin/bash
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
CONDA_SETUP="/home/mikhail/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SETUP" ]; then
    source "$CONDA_SETUP"
else
  eval "$(conda shell.bash hook)"
fi

conda deactivate || true
conda activate mvf-unik3d

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





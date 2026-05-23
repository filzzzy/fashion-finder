#!/usr/bin/env bash
#
# Convert ONNX graphs to TensorRT plan files. Requires NVIDIA TensorRT
# installed locally (typically inside the NGC tritonserver image).
#
# Usage:
#   ./scripts/export_tensorrt.sh path/to/checkpoints/onnx path/to/checkpoints/tensorrt fp16

set -euo pipefail

ONNX_DIR="${1:-checkpoints/onnx}"
TRT_DIR="${2:-checkpoints/tensorrt}"
PRECISION="${3:-fp16}"
MIN_BATCH="${MIN_BATCH:-1}"
OPT_BATCH="${OPT_BATCH:-16}"
MAX_BATCH="${MAX_BATCH:-64}"
WORKSPACE_MB="${WORKSPACE_MB:-4096}"

mkdir -p "${TRT_DIR}"

PRECISION_FLAG=""
if [[ "${PRECISION}" == "fp16" ]]; then
    PRECISION_FLAG="--fp16"
elif [[ "${PRECISION}" == "int8" ]]; then
    PRECISION_FLAG="--int8"
fi

echo "[*] Building TensorRT plan for vision encoder..."
trtexec \
    --onnx="${ONNX_DIR}/fashion_finder_vision.onnx" \
    --saveEngine="${TRT_DIR}/fashion_finder_vision.plan" \
    --memPoolSize=workspace:${WORKSPACE_MB} \
    --minShapes=images:${MIN_BATCH}x3x224x224 \
    --optShapes=images:${OPT_BATCH}x3x224x224 \
    --maxShapes=images:${MAX_BATCH}x3x224x224 \
    ${PRECISION_FLAG}

echo "[*] Building TensorRT plan for composer..."
trtexec \
    --onnx="${ONNX_DIR}/fashion_finder_composer.onnx" \
    --saveEngine="${TRT_DIR}/fashion_finder_composer.plan" \
    --memPoolSize=workspace:${WORKSPACE_MB} \
    --minShapes=images:${MIN_BATCH}x3x224x224,input_ids:${MIN_BATCH}x64,attention_mask:${MIN_BATCH}x64 \
    --optShapes=images:${OPT_BATCH}x3x224x224,input_ids:${OPT_BATCH}x64,attention_mask:${OPT_BATCH}x64 \
    --maxShapes=images:${MAX_BATCH}x3x224x224,input_ids:${MAX_BATCH}x64,attention_mask:${MAX_BATCH}x64 \
    ${PRECISION_FLAG}

echo "[+] TensorRT plans written to ${TRT_DIR}"

#!/usr/bin/env bash
set -euo pipefail

ONNX_DIR="${1:-/workspace/models/supertonic-2/onnx}"
OUT_DIR="${2:-/workspace/models/supertonic-2/coreml}"

python /workspace/scripts/convert_onnx_coreml.py \
  --onnx-dir "$ONNX_DIR" \
  --out-dir "$OUT_DIR" \
  --max-text-len 300 \
  --max-seconds 20 \
  --rewrite-layernorm \
  --rewrite-erf \
  --rewrite-sin \
  --rewrite-pad \
  --rewrite-axes \
  --simplify

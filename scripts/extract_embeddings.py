#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def extract_weight(onnx_path: str) -> np.ndarray:
    model = onnx.load(onnx_path)
    for init in model.graph.initializer:
        if "char_embedder.weight" in init.name and len(init.dims) == 2:
            return numpy_helper.to_array(init).astype(np.float32)
    raise RuntimeError(f"char_embedder.weight not found in {onnx_path}")


def save_embedding(out_dir: str, name: str, weight: np.ndarray) -> None:
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, f"{name}.fp32.bin")
    shape_path = os.path.join(out_dir, f"{name}.shape.json")
    weight.tofile(bin_path)
    with open(shape_path, "w", encoding="utf-8") as f:
        json.dump({"shape": list(weight.shape)}, f)
    print(f"wrote {bin_path} ({weight.shape})")
    print(f"wrote {shape_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract embedding weights from ONNX.")
    parser.add_argument("--onnx-dir", default="models/supertonic-2/onnx")
    parser.add_argument("--out-dir", default="models/supertonic-2/embeddings")
    args = parser.parse_args()

    dp_weight = extract_weight(os.path.join(args.onnx_dir, "duration_predictor.onnx"))
    te_weight = extract_weight(os.path.join(args.onnx_dir, "text_encoder.onnx"))

    save_embedding(args.out_dir, "char_embedder_dp", dp_weight)
    save_embedding(args.out_dir, "char_embedder_te", te_weight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

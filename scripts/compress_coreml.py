#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List

import coremltools as ct
from coremltools.optimize import coreml as copt


MODEL_BASES = [
    "duration_predictor",
    "text_encoder",
    "vector_estimator",
    "vocoder",
]


def weight_bin_size(path: str) -> int:
    bin_path = os.path.join(path, "Data", "com.apple.CoreML", "weights", "weight.bin")
    if os.path.exists(bin_path):
        return os.path.getsize(bin_path)
    return 0


def human_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def parse_models(models_arg: str) -> List[str]:
    items = [m.strip() for m in models_arg.split(",") if m.strip()]
    for m in items:
        if m not in MODEL_BASES:
            raise ValueError(f"Unknown model '{m}'. Options: {', '.join(MODEL_BASES)}")
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Compress CoreML ML Program weights.")
    parser.add_argument("--model-dir", default="models/supertonic-2/coreml")
    parser.add_argument("--out-dir", default="models/supertonic-2/coreml_compressed")
    parser.add_argument("--models", default="vector_estimator,vocoder")
    parser.add_argument("--method", choices=("linear", "palettize"), default="linear")
    parser.add_argument("--bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--mode", default="linear_symmetric")
    parser.add_argument("--granularity", default="per_channel")
    parser.add_argument("--weight-threshold", type=int, default=2048)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    models = parse_models(args.models)

    for base in models:
        in_path = os.path.join(args.model_dir, f"{base}_mlprogram.mlpackage")
        if not os.path.exists(in_path):
            raise FileNotFoundError(in_path)

        print(f"\n=== {base} ===")
        before = weight_bin_size(in_path)
        print(f"- input weights: {human_mb(before)}")

        mlmodel = ct.models.MLModel(in_path)

        if args.method == "linear":
            config = copt.OptimizationConfig(
                global_config=copt.OpLinearQuantizerConfig(
                    mode=args.mode,
                    dtype="int8" if args.bits == 8 else "int4",
                    granularity=args.granularity,
                    weight_threshold=args.weight_threshold,
                )
            )
            compressed = copt.linear_quantize_weights(mlmodel, config)
        else:
            config = copt.OptimizationConfig(
                global_config=copt.OpPalettizerConfig(
                    mode="kmeans",
                    nbits=args.bits,
                    granularity="per_tensor",
                    weight_threshold=args.weight_threshold,
                )
            )
            compressed = copt.palettize_weights(mlmodel, config)

        out_path = os.path.join(args.out_dir, f"{base}_mlprogram_{args.method}{args.bits}.mlpackage")
        compressed.save(out_path)
        after = weight_bin_size(out_path)
        print(f"- output weights: {human_mb(after)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

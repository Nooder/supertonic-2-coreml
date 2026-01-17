#!/usr/bin/env python3
import argparse
import os

import coremltools as ct


OUTPUT_NAMES = {
    "duration_predictor": "duration",
    "text_encoder": "text_emb",
    "vector_estimator": "denoised_latent",
    "vocoder": "wav_tts",
}


def rename_output(model_path: str, new_name: str, dry_run: bool) -> None:
    spec = ct.models.utils.load_spec(model_path)
    if len(spec.description.output) != 1:
        print(f"- skip {model_path} (unexpected output count)")
        return
    old_name = spec.description.output[0].name
    if old_name == new_name:
        print(f"- {model_path}: already {new_name}")
        return
    if dry_run:
        print(f"- {model_path}: {old_name} -> {new_name}")
        return
    ct.models.utils.rename_feature(spec, old_name, new_name, rename_inputs=False, rename_outputs=True)
    if model_path.endswith(".mlpackage"):
        model_spec_path = os.path.join(model_path, "Data", "com.apple.CoreML", "model.mlmodel")
        ct.models.utils.save_spec(spec, model_spec_path)
    else:
        ct.models.utils.save_spec(spec, model_path)
    print(f"- {model_path}: {old_name} -> {new_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rename CoreML model outputs to stable names.")
    parser.add_argument("--model-dir", default="models/supertonic-2/coreml")
    parser.add_argument("--format", choices=("mlprogram", "nn", "both"), default="mlprogram")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for base, new_name in OUTPUT_NAMES.items():
        paths = []
        if args.format in ("mlprogram", "both"):
            paths.append(os.path.join(args.model_dir, f"{base}_mlprogram.mlpackage"))
        if args.format in ("nn", "both"):
            paths.append(os.path.join(args.model_dir, f"{base}_fixed.mlmodel"))

        for path in paths:
            if not os.path.exists(path):
                print(f"- missing {path}")
                continue
            rename_output(path, new_name, args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

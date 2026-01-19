#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from unicodedata import normalize

import numpy as np
import coremltools as ct
from coremltools.models import MLModel
import onnx
from onnx import numpy_helper


# Default paths are resolved at runtime to support both repo and HF bundle layouts.
DEFAULT_MAX_TEXT_LEN = 300
LATENT_CHANNELS = 144
LATENT_LEN = 288
EXPECTED_OUTPUTS = {
    "duration_predictor": "duration",
    "text_encoder": "text_emb",
    "vector_estimator": "denoised_latent",
    "vocoder": "wav_tts",
}


def preprocess_text(text: str, lang: str) -> str:
    # Mirror the Swift-side text normalization so token IDs match.
    text = normalize("NFKD", text)
    emoji_pattern = re.compile(
        "[\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f700-\U0001f77f"
        "\U0001f780-\U0001f7ff"
        "\U0001f800-\U0001f8ff"
        "\U0001f900-\U0001f9ff"
        "\U0001fa00-\U0001fa6f"
        "\U0001fa70-\U0001faff"
        "\u2600-\u26ff"
        "\u2700-\u27bf"
        "\U0001f1e6-\U0001f1ff]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    replacements = {
        "–": "-",
        "‑": "-",
        "—": "-",
        "_": " ",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "´": "'",
        "`": "'",
        "[": " ",
        "]": " ",
        "|": " ",
        "/": " ",
        "#": " ",
        "→": " ",
        "←": " ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = re.sub(r"[♥☆♡©\\]", "", text)
    expr_replacements = {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}
    for k, v in expr_replacements.items():
        text = text.replace(k, v)
    text = re.sub(r" ,", ",", text)
    text = re.sub(r" \.", ".", text)
    text = re.sub(r" !", "!", text)
    text = re.sub(r" \?", "?", text)
    text = re.sub(r" ;", ";", text)
    text = re.sub(r" :", ":", text)
    text = re.sub(r" '", "'", text)
    while '""' in text:
        text = text.replace('""', '"')
    while "''" in text:
        text = text.replace("''", "'")
    while "``" in text:
        text = text.replace("``", "`")
    text = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
        text += "."
    return f"<{lang}>" + text + f"</{lang}>"


def build_text_inputs(text: str, lang: str, cfg_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(cfg_dir / "unicode_indexer.json", "r", encoding="utf-8") as f:
        indexer = json.load(f)
    text = preprocess_text(text, lang)
    ids = [indexer[ord(ch)] for ch in text]
    if any(i < 0 for i in ids):
        raise ValueError("Text contains unsupported characters.")
    text_ids = np.zeros((1, DEFAULT_MAX_TEXT_LEN), dtype=np.int32)
    text_ids[0, : len(ids)] = np.array(ids, dtype=np.int32)
    text_mask = np.zeros((1, 1, DEFAULT_MAX_TEXT_LEN), dtype=np.float32)
    text_mask[0, 0, : len(ids)] = 1.0
    return text_ids, text_mask


def resolve_model_paths(format_hint: str, model_dir: Path) -> dict[str, str]:
    # Decide between mlprogram vs. neural network format.
    bases = ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]
    mlprogram_paths = {b: str(model_dir / f"{b}_mlprogram.mlpackage") for b in bases}
    nn_paths = {b: str(model_dir / f"{b}_fixed.mlmodel") for b in bases}

    fmt = format_hint
    if fmt == "auto":
        if all(os.path.exists(p) for p in mlprogram_paths.values()):
            fmt = "mlprogram"
        else:
            fmt = "nn"

    if fmt == "mlprogram":
        paths = mlprogram_paths
    else:
        paths = nn_paths

    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing model files for format '{fmt}': {missing}")

    return paths


def get_model_input_names(model_path: str) -> set[str]:
    spec = ct.models.utils.load_spec(model_path)
    return {i.name for i in spec.description.input}


def load_embedding_weight(onnx_path: Path) -> np.ndarray:
    # We only need the embedding table to synthesize text embeddings for tests.
    model = onnx.load(onnx_path)
    for init in model.graph.initializer:
        if "char_embedder.weight" in init.name and len(init.dims) == 2:
            return numpy_helper.to_array(init).astype(np.float32)
    raise RuntimeError(f"char_embedder.weight not found in {onnx_path}")


def build_text_embed(text_ids: np.ndarray, onnx_path: Path) -> np.ndarray:
    # Produce the same [B, C, T] embedding layout as the CoreML models expect.
    weight = load_embedding_weight(onnx_path)
    embed = weight[text_ids]  # (B, T, C)
    return np.transpose(embed, (0, 2, 1)).astype(np.float32)


def run_model(name: str, inputs: dict) -> dict:
    # Keep tests on CPU for determinism and portability.
    model = MLModel(name, compute_units=ct.ComputeUnit.CPU_ONLY)
    return model.predict(inputs)


def get_output_value(outputs: dict, model_name: str) -> np.ndarray:
    expected = EXPECTED_OUTPUTS.get(model_name)
    if expected and expected in outputs:
        return outputs[expected]
    if expected:
        print(f"warning: {model_name} output '{expected}' not found; using first output")
    return next(iter(outputs.values()))


def resolve_default_paths(bundle_root: Path) -> tuple[Path, Path, Path]:
    """
    Resolve model, config, and onnx paths for either:
    - repo layout: models/supertonic-2/...
    - HF bundle layout: models/ and resources/
    """
    repo_models = bundle_root / "models" / "supertonic-2"
    if repo_models.exists():
        model_root = repo_models
        cfg_dir = repo_models / "onnx"
        onnx_dir = repo_models / "onnx"
    else:
        model_root = bundle_root / "models"
        cfg_dir = bundle_root / "resources" / "onnx"
        onnx_dir = bundle_root / "resources" / "onnx"

    model_dir = model_root / "coreml"
    if not model_dir.exists():
        for candidate in sorted(model_root.iterdir()):
            if candidate.is_dir() and candidate.name.startswith("coreml"):
                model_dir = candidate
                break
    return model_dir, cfg_dir, onnx_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test CoreML models.")
    parser.add_argument(
        "--format",
        choices=("auto", "mlprogram", "nn"),
        default="auto",
        help="Choose CoreML model format. 'auto' prefers mlprogram if available.",
    )
    parser.add_argument(
        "--bundle-dir",
        default=None,
        help="Root folder containing models/ and resources/ (auto-detected by default).",
    )
    parser.add_argument("--model-dir", default=None, help="Override model variant directory.")
    parser.add_argument("--config-dir", default=None, help="Override config directory.")
    parser.add_argument("--onnx-dir", default=None, help="Override onnx directory.")
    args = parser.parse_args()

    bundle_root = (
        Path(args.bundle_dir).resolve()
        if args.bundle_dir
        else Path(__file__).resolve().parents[1]
    )
    model_dir, cfg_dir, onnx_dir = resolve_default_paths(bundle_root)
    if args.model_dir:
        model_dir = Path(args.model_dir).resolve()
    if args.config_dir:
        cfg_dir = Path(args.config_dir).resolve()
    if args.onnx_dir:
        onnx_dir = Path(args.onnx_dir).resolve()

    # CoreML writes temporary compilation artifacts; keep them local and disposable.
    os.makedirs(".coremltmp", exist_ok=True)
    os.environ["TMPDIR"] = os.path.abspath(".coremltmp")

    failures = []
    model_paths = resolve_model_paths(args.format, model_dir)
    text_ids, text_mask = build_text_inputs("Hello world", "en", cfg_dir)
    # Some models take text_ids directly, others expect text_embed.
    dp_inputs = get_model_input_names(model_paths["duration_predictor"])
    te_inputs = get_model_input_names(model_paths["text_encoder"])
    text_emb_out = None
    ve_out = None
    try:
        # Duration predictor expects style + text embeddings (or ids) and returns total duration.
        dp_payload = {
            "style_dp": np.zeros((1, 8, 16), dtype=np.float32),
            "text_mask": text_mask,
        }
        if "text_embed" in dp_inputs:
            dp_payload["text_embed"] = build_text_embed(
                text_ids, onnx_dir / "duration_predictor.onnx"
            )
        else:
            dp_payload["text_ids"] = text_ids
        dp_out = run_model(model_paths["duration_predictor"], dp_payload)
        print("duration_predictor ok", {k: v.shape for k, v in dp_out.items()})
    except Exception as exc:
        failures.append(("duration_predictor", str(exc)))

    try:
        # Text encoder produces the conditioning embeddings for the denoiser.
        te_payload = {
            "style_ttl": np.zeros((1, 50, 256), dtype=np.float32),
            "text_mask": text_mask,
        }
        if "text_embed" in te_inputs:
            te_payload["text_embed"] = build_text_embed(
                text_ids, onnx_dir / "text_encoder.onnx"
            )
        else:
            te_payload["text_ids"] = text_ids
        te_out = run_model(model_paths["text_encoder"], te_payload)
        text_emb_out = get_output_value(te_out, "text_encoder")
        print("text_encoder ok", {k: v.shape for k, v in te_out.items()})
    except Exception as exc:
        failures.append(("text_encoder", str(exc)))

    try:
        # Vector estimator performs iterative denoising over the latent.
        if text_emb_out is None:
            text_emb_out = np.zeros((1, 256, DEFAULT_MAX_TEXT_LEN), dtype=np.float32)
        ve_out = run_model(
            model_paths["vector_estimator"],
            {
                "noisy_latent": np.zeros((1, LATENT_CHANNELS, LATENT_LEN), dtype=np.float32),
                "text_emb": text_emb_out.astype(np.float32),
                "style_ttl": np.zeros((1, 50, 256), dtype=np.float32),
                "latent_mask": np.ones((1, 1, LATENT_LEN), dtype=np.float32),
                "text_mask": text_mask.astype(np.float32),
                "current_step": np.array([0], dtype=np.float32),
                "total_step": np.array([2], dtype=np.float32),
            },
        )
        print("vector_estimator ok", {k: v.shape for k, v in ve_out.items()})
    except Exception as exc:
        failures.append(("vector_estimator", str(exc)))

    try:
        # Vocoder converts denoised latents to waveform samples.
        if ve_out is None:
            ve_latent = np.zeros((1, LATENT_CHANNELS, LATENT_LEN), dtype=np.float32)
        else:
            ve_latent = get_output_value(ve_out, "vector_estimator")
        vocoder_out = run_model(model_paths["vocoder"], {"latent": ve_latent.astype(np.float32)})
        print("vocoder ok", {k: v.shape for k, v in vocoder_out.items()})
    except Exception as exc:
        failures.append(("vocoder", str(exc)))

    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f"- {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

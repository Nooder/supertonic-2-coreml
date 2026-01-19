# Upstream Provenance

This repository is a CoreML-focused derivative of the original Supertonic 2
release. The goal is to provide iOS/macOS-ready artifacts plus a Swift demo app.

## Upstream sources

- Hugging Face: `Supertone/supertonic-2`
- GitHub: `supertone-inc/supertonic`

## Upstream license summary

- **Model weights:** OpenRAIL-M (see `models/supertonic-2/LICENSE`)
- **Sample code:** MIT (see `external/supertonic/LICENSE`)

## Local modifications

- Converted ONNX models to CoreML ML Program packages.
- Produced >=8-bit quantized variants for iOS/macOS.
- Added Swift demo app and CoreML inference pipeline.
- Added CoreML smoke tests and packaging scripts.

## Provenance checklist (fill before release)

- Upstream model commit or revision: Unknown (local snapshot under `models/supertonic-2/` without VCS metadata). Upstream release announced **2026-01-06** in `external/supertonic/README.md`.
- Conversion tool versions: Python 3.12.11 (uv venv); numpy 2.4.1; coremltools 9.0; onnx 1.20.1.
- Quantization recipe: `scripts/compress_coreml.py` using coremltools `linear_quantize_weights` (default: int8, linear_symmetric, per_channel, weight_threshold=2048) on selected stages.
- Release tag: Unreleased (set when publishing).

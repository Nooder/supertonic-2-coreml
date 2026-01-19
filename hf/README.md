---
license: openrail
language:
  - en
  - ko
  - es
  - pt
  - fr
pipeline_tag: text-to-speech
tags:
  - coreml
  - ios
  - macos
  - tts
  - supertonic
  - mlprogram
---

# Supertonic-2 CoreML

This repository provides CoreML exports of **Supertonic 2** for macOS and iOS.
It focuses on on-device inference with multiple >=8-bit quantization variants.

## What is included

- `models/`: CoreML model packages by variant (>=8-bit only)
- `resources/`: voice styles, embeddings, and text normalization assets
- `manifest.json`: list of artifacts with checksums and sizes
- `SHA256SUMS`: sha256 checksums for all files
- `tests/`: smoke tests for CoreML model loading

## Quickstart (iOS / macOS)

1. Pick a variant from `models/` (see the quant matrix in `docs/quant-matrix.md`).
2. Bundle the corresponding CoreML packages and `resources/` into your app.
3. Use the Swift demo app in the GitHub repo `supertonic-2-coreml` as the
   reference implementation.

## Choosing a variant

Use the folder naming to select the right artifact:

- `coreml_int8`: faster, lower fidelity
- `coreml_compressed`: smaller memory (linear8)
- `coreml_ios18_*`: for iOS 18 CoreML runtime (>=8-bit only)

4-bit variants are intentionally excluded due to quality.

## Tests

The `tests/test_coreml_models.py` script runs a simple smoke test that loads
all stages (duration predictor, text encoder, vector estimator, vocoder) with
dummy inputs.

## Attribution and license

This CoreML export is derived from **Supertone/supertonic-2**.
Model weights are licensed under **OpenRAIL-M** (see `LICENSE`).
Sample code is MIT-licensed (see `NOTICE` and `UPSTREAM.md`).

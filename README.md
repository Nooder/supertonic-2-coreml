# Supertonic-2 CoreML

CoreML‑optimized exports of **Supertonic 2** for iOS and macOS, plus a Swift
demo app, smoke tests, and packaging scripts for Hugging Face distribution.

## Repository structure

- `supertonic2-coreml-ios-test/`: Swift demo app (CoreML pipeline + UI).
- `models/supertonic-2/`: source models, CoreML artifacts, and resources.
- `scripts/`: conversion, benchmarking, smoke tests, and HF packaging.
- `docs/`: compatibility and quantization guidance.
- `hf/`: staging bundle for Hugging Face publishing.

## Quickstart (iOS/macOS demo app)

1. Open `supertonic2-coreml-ios-test.xcodeproj`.
2. Ensure `SupertonicResources/` (or `models/supertonic-2/` resources) are
   bundled in the app target.
3. Build and run on device or simulator.

The demo app uses the CoreML pipeline in:
`supertonic2-coreml-ios-test/TTSService.swift`.

## Model variants and quantization

See:
- `docs/compatibility-matrix.md` for OS/runtime expectations.
- `docs/quant-matrix.md` for quantization tradeoffs.

4‑bit variants are intentionally excluded due to quality.

## Hugging Face bundle

Build a HF‑ready bundle (>=8‑bit only):

```
python3 scripts/build_hf_bundle.py --clean
```

Outputs land in `hf/` with:
- `models/` (CoreML packages)
- `resources/` (voice styles, embeddings, indexers)
- `manifest.json` + `SHA256SUMS`
- `tests/` (CoreML smoke test)

## Smoke tests

Run a quick CoreML load/predict test:

```
python3 scripts/test_coreml_models.py
```

To point at a HF bundle layout:

```
python3 scripts/test_coreml_models.py --bundle-dir hf
```

## Attribution

This repository is derived from:
- Hugging Face: `Supertone/supertonic-2`
- GitHub: `supertone-inc/supertonic`

See `UPSTREAM.md` for provenance details.

## License

- Model weights: OpenRAIL‑M (see `models/supertonic-2/LICENSE`).
- Sample code: MIT (see `LICENSE`).

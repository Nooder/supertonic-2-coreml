# CoreML Compatibility Matrix

This matrix describes the CoreML artifact families and how to pick the right
folder for your device target. It is based on the folder names under
`models/supertonic-2/`. Always validate with your deployment target in Xcode.

Legend:
- **mlprogram**: Core ML ML Program packaged as `.mlpackage` (modern CoreML).
- **nn**: legacy neural network `.mlmodel` (older CoreML format).
- **ios18**: name indicates a build tuned for iOS 18 runtime features.

## Variant overview

| Variant folder | Format | Intended OS target | Notes |
| --- | --- | --- | --- |
| `coreml` | mixed (nn + mlprogram) | general | Includes legacy `.mlmodel` and mlprogram artifacts. |
| `coreml_int8` | mlprogram | general | Full int8 pipeline, faster but lower fidelity. |
| `coreml_compressed` | mlprogram | general | Linear8 compressed weights for smaller memory. |
| `coreml_ios18` | mlprogram | iOS 18+ | Uses iOS 18 CoreML runtime. |
| `coreml_ios18_int8_vocoder_only` | mlprogram | iOS 18+ | Only the vocoder is int8 (per naming). |
| `coreml_ios18_int8_both` | mlprogram | iOS 18+ | Multiple stages int8 (per naming). |
| `coreml_compressed_ios18` | mlprogram | iOS 18+ | Linear8-only subset (linear4 excluded). |

## Excluded variants (not published)

These are intentionally omitted due to quality concerns:
- `coreml_ios18_int4_only`
- `coreml_ios18_int4_int8`
- any package with `int4` or `linear4` in its filename

## Validation checklist

1. Open the target `.mlpackage` in Xcode and confirm it compiles for your
   deployment target.
2. Run the iOS/macOS demo app with your target device and compute units.
3. Compare audio quality and latency across variants using identical input
   text and step counts.

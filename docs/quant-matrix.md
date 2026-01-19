# Quantization Matrix (CoreML)

This repository publishes only >=8-bit CoreML artifacts. 4-bit variants are
excluded due to quality.

## Naming rules

The folder name encodes the intended runtime and quantization approach:

- `coreml_*`: generic CoreML export.
- `coreml_ios18_*`: tuned for iOS 18 CoreML runtime.
- `int8`: int8 weights for one or more stages.
- `vocoder_only`: only the vocoder is quantized (per naming).
- `both`: multiple stages are quantized (per naming).
- `compressed` / `linear8`: linear 8-bit compression for smaller memory.

## Variant table

| Variant folder | Quantization (by name) | Expected tradeoff | When to use |
| --- | --- | --- | --- |
| `coreml` | full precision (mixed) | best quality, larger | baseline quality checks |
| `coreml_int8` | int8 (all stages) | faster, smaller | general fast inference |
| `coreml_compressed` | linear8 | smallest memory | low-memory devices |
| `coreml_ios18` | full precision (mlprogram) | best quality on iOS 18 | iOS 18+ devices |
| `coreml_ios18_int8_vocoder_only` | int8 (vocoder only) | balanced | iOS 18+ with minimal quality loss |
| `coreml_ios18_int8_both` | int8 (multiple stages) | faster, more loss | iOS 18+ when latency matters |
| `coreml_compressed_ios18` | linear8 (subset) | smallest memory | iOS 18+ with tight memory |

## Steps vs. quality

The `steps` parameter controls the denoiser iterations:
- Fewer steps = faster, lower fidelity.
- More steps = slower, higher fidelity.

Recommended starting points:
- **Fast preview:** 10 steps
- **Balanced:** 20 steps
- **Higher quality:** 30 steps

## Excluded variants

The following are intentionally not published:
- `coreml_ios18_int4_only`
- `coreml_ios18_int4_int8`
- any package with `int4` or `linear4` in its filename

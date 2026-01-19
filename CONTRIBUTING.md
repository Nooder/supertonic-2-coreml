# Contributing

Thanks for helping improve Supertonic‑2 CoreML.

## Quick guidelines

- Keep CoreML artifacts >=8‑bit (no int4/linear4).
- Update `docs/` when you change model variants or naming.
- Add or update smoke tests for new model variants.

## Development workflow

1. Create a feature branch.
2. Make changes with clear commit messages.
3. Run the smoke test:

```
python3 scripts/test_coreml_models.py
```

4. Regenerate the HF bundle if you changed artifacts:

```
python3 scripts/build_hf_bundle.py --clean
```

## Large files

CoreML and model binaries should be tracked with Git LFS where possible.
See `.gitattributes` for the tracked patterns.

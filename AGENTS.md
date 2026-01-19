# Agent Notes (HF + GitHub)

This file captures the current publishing setup for future agents.

## Repositories

- **GitHub (code/docs):** https://github.com/Nooder/supertonic-2-coreml  
  - Remote `origin` is set in this repo.
  - Tag `v0.1.0` exists and was pushed.
- **Hugging Face (artifacts):** https://huggingface.co/Nooder/supertonic-2-coreml  
  - Model card is `hf/README.md` (copied into HF bundle root).

## HF publishing workflow

We publish artifacts to HF from a **local bundle repo**:

1. Build bundle:
   ```
   python3 scripts/build_hf_bundle.py --clean --output hf_publish
   ```
2. `hf_publish/` is a standalone git repo (ignored by GitHub).
   - It contains CoreML artifacts + `README.md` + `NOTICE` + `UPSTREAM.md`.
3. Commit/push from `hf_publish/`:
   ```
   cd hf_publish
   git lfs install
   git add -A
   git commit -m "Update CoreML bundle"
   git push
   ```

Alternative: use `scripts/sync_hf_repo.py` to sync into a separate HF repo checkout.

## GitHub publishing workflow

Normal repo (this one):

```
git add -A
git commit -m "Release vX.Y.Z"
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin main
git push origin vX.Y.Z
```

## Model card + docs

- HF model card source: `hf/README.md`
- Release checklist: `docs/release-checklist.md`
- Publishing guide: `PUBLISHING.md`

## LFS

- `.gitattributes` tracks CoreML + model binaries.
- LFS already enabled and used on GitHub and HF.

## Local test environment

CoreML smoke test requires Python 3.12 + coremltools:

```
uv venv --python 3.12 .venv
uv pip install -p .venv coremltools onnx
.venv/bin/python scripts/test_coreml_models.py
```

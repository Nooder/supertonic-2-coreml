# Release Checklist (GitHub + Hugging Face)

## Content sanity

- [ ] `README.md` updated (overview + quickstart)
- [ ] `hf/README.md` (model card) concise and accurate
- [ ] `LICENSE`, `NOTICE`, `CITATION.cff`, `UPSTREAM.md` present and accurate
- [ ] `docs/compatibility-matrix.md` + `docs/quant-matrix.md` reflect current artifacts
- [ ] `CHANGELOG.md` has a versioned entry

## Artifacts

- [ ] 4-bit variants excluded (no `int4`/`linear4`)
- [ ] All intended CoreML variants present (mlprogram packages)
- [ ] Resources present (voice styles, embeddings, indexers)
- [ ] `manifest.json` + `SHA256SUMS` regenerated

## Tests

- [ ] Smoke test passes:
  - `python3 scripts/test_coreml_models.py` (local repo)
  - `python3 scripts/test_coreml_models.py --bundle-dir hf_publish` (HF bundle)

## GitHub release

- [ ] Tag created (e.g., `v0.1.0`)
- [ ] Tag pushed to GitHub

## Hugging Face

- [ ] `hf_publish/` regenerated
- [ ] HF repo sync complete
- [ ] `git lfs ls-files` shows large files tracked

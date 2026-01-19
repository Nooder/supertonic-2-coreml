# Publishing Guide

This repository targets two destinations:
- **GitHub** for source code, docs, and tooling.
- **Hugging Face** for CoreML artifacts and model card.

## 1) GitHub publish

1. Ensure the working tree is clean:
   ```
   git status -sb
   ```
2. Update `CHANGELOG.md` and `UPSTREAM.md` (version + provenance).
3. Commit changes:
   ```
   git add -A
   git commit -m "Release vX.Y.Z"
   ```
4. Tag the release:
   ```
   git tag -a vX.Y.Z -m "vX.Y.Z"
   ```
5. Push to GitHub:
   ```
   git push origin main
   git push origin vX.Y.Z
   ```

## 2) Hugging Face publish

### Build the HF bundle

```
python3 scripts/build_hf_bundle.py --clean --output hf_publish
```

### Sync into an HF repo checkout

```
python3 scripts/sync_hf_repo.py --source hf_publish --target /path/to/hf-repo --clean
```

### Commit and push from the HF repo

```
cd /path/to/hf-repo
git lfs install
git add -A
git commit -m "Update CoreML bundle"
git push
```

## Auth notes

- Use `hf auth whoami` to confirm you’re logged in.
- If needed: `hf auth login` (token required).

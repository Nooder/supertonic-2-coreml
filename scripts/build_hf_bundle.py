#!/usr/bin/env python3
"""
Build a Hugging Face-friendly bundle for CoreML artifacts.

This script copies >=8-bit CoreML packages plus required resources into a
staging directory (default: hf/), generates a manifest, and writes SHA256SUMS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

EXCLUDE_SUBSTRINGS = ("int4", "linear4")
RESOURCE_DIRS = ("voice_styles", "embeddings")
ONNX_FILES = ("unicode_indexer.json", "tts.json", "duration_predictor.onnx", "text_encoder.onnx")
GENERATED_ITEMS = ("models", "resources", "tests", "docs", "manifest.json", "SHA256SUMS", "config.json", "LICENSE")


def should_exclude(path: Path) -> bool:
    lower = path.as_posix().lower()
    return any(token in lower for token in EXCLUDE_SUBSTRINGS)


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def copy_models(source: Path, dest_models: Path) -> list[Path]:
    copied: list[Path] = []
    model_roots = sorted(p for p in source.iterdir() if p.is_dir() and p.name.startswith("coreml"))
    for root in model_roots:
        # Copy mlpackage directories (mlprogram).
        for package in sorted(root.rglob("*.mlpackage")):
            if should_exclude(package):
                continue
            rel = package.relative_to(source)
            target = dest_models / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(package, target, dirs_exist_ok=True)
            copied.append(target)

        # Copy top-level mlmodel files that are not inside mlpackage bundles.
        for model_file in sorted(root.rglob("*.mlmodel")):
            if should_exclude(model_file):
                continue
            if any(part.endswith(".mlpackage") for part in model_file.parts):
                continue
            rel = model_file.relative_to(source)
            target = dest_models / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_file, target)
            copied.append(target)

    return copied


def copy_resources(source: Path, dest_resources: Path) -> list[Path]:
    copied: list[Path] = []
    for name in RESOURCE_DIRS:
        src_dir = source / name
        if not src_dir.exists():
            continue
        dst_dir = dest_resources / name
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        copied.append(dst_dir)

    onnx_src = source / "onnx"
    if onnx_src.exists():
        onnx_dst = dest_resources / "onnx"
        onnx_dst.mkdir(parents=True, exist_ok=True)
        for filename in ONNX_FILES:
            src_file = onnx_src / filename
            if src_file.exists():
                shutil.copy2(src_file, onnx_dst / filename)
                copied.append(onnx_dst / filename)

    return copied


def copy_tests(repo_root: Path, dest_tests: Path) -> None:
    test_source = repo_root / "scripts" / "test_coreml_models.py"
    if not test_source.exists():
        return
    dest_tests.mkdir(parents=True, exist_ok=True)
    shutil.copy2(test_source, dest_tests / "test_coreml_models.py")


def copy_docs(repo_root: Path, dest_docs: Path) -> None:
    docs_source = repo_root / "docs"
    if not docs_source.exists():
        return
    shutil.copytree(docs_source, dest_docs, dirs_exist_ok=True)


def copy_config_and_license(source: Path, output: Path) -> None:
    config = source / "config.json"
    if config.exists():
        shutil.copy2(config, output / "config.json")
    license_file = source / "LICENSE"
    if license_file.exists():
        shutil.copy2(license_file, output / "LICENSE")


def build_manifest(output: Path) -> dict:
    files = [
        p
        for p in output.rglob("*")
        if p.is_file() and p.name not in {"manifest.json", "SHA256SUMS"}
    ]
    files.sort(key=lambda p: p.as_posix())

    items = []
    for path in files:
        rel = path.relative_to(output).as_posix()
        kind = "meta"
        variant = None
        if rel.startswith("models/"):
            kind = "model"
            parts = rel.split("/")
            if len(parts) > 1:
                variant = parts[1]
        elif rel.startswith("resources/"):
            kind = "resource"
        elif rel.startswith("tests/"):
            kind = "test"
        elif rel.startswith("docs/"):
            kind = "doc"

        items.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "kind": kind,
                "variant": variant,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "excluded_patterns": list(EXCLUDE_SUBSTRINGS),
        "items": items,
        "summary": {
            "file_count": len(items),
            "total_bytes": sum(item["bytes"] for item in items),
        },
    }


def write_sha256sums(output: Path, manifest: dict) -> None:
    lines = [f"{item['sha256']}  {item['path']}" for item in manifest["items"]]
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build HF bundle for CoreML artifacts.")
    parser.add_argument("--source", default="models/supertonic-2", help="Source model directory.")
    parser.add_argument("--output", default="hf", help="Output staging directory.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove generated directories/files in the output before copying.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / args.source).resolve()
    output = (repo_root / args.output).resolve()

    output.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for name in GENERATED_ITEMS:
            remove_path(output / name)

    dest_models = output / "models"
    dest_resources = output / "resources"
    dest_tests = output / "tests"
    dest_docs = output / "docs"

    copy_models(source, dest_models)
    copy_resources(source, dest_resources)
    copy_tests(repo_root, dest_tests)
    copy_docs(repo_root, dest_docs)
    copy_config_and_license(source, output)

    manifest = build_manifest(output)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_sha256sums(output, manifest)

    print(f"HF bundle written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

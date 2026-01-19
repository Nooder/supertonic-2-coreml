#!/usr/bin/env python3
"""
Sync a staged HF bundle into a separate Hugging Face repo checkout.

Example:
  python3 scripts/sync_hf_repo.py --source hf_publish --target ../supertonic-2-coreml-hf --clean
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def sync_tree(source: Path, target: Path, clean: bool) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if clean:
        for item in target.iterdir():
            if item.name == ".git":
                continue
            remove_path(item)

    for item in source.iterdir():
        if item.name == ".git":
            continue
        dest = target / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync HF bundle into an HF repo checkout.")
    parser.add_argument("--source", default="hf_publish", help="HF bundle source folder.")
    parser.add_argument("--target", required=True, help="Path to HF repo checkout.")
    parser.add_argument("--clean", action="store_true", help="Remove target contents before copying.")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    target = Path(args.target).resolve()
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    sync_tree(source, target, clean=args.clean)
    print(f"Synced {source} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

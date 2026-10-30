#!/usr/bin/env python3
"""Verify frozen M0 baseline artifacts without executing legacy physics.

Usage (from anywhere): python scripts/verify_baseline.py
                       python scripts/verify_baseline.py --root /path/to/repo

The expected hashes live in tests/fixtures/baseline_manifest.json. Any changes
must be reviewed: these files are *historical outputs*, not physics truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = "tests/fixtures/baseline_manifest.json"


def verify(root: Path, manifest_path: Path) -> list[str]:
    errors: list[str] = []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data["artifacts"]
    if not entries:
        return ["Baseline manifest has no artifacts"]
    for item in entries:
        relative = Path(item["path"])
        # Reject unsafe paths even though this file is normally maintained in Git.
        if relative.is_absolute() or ".." in relative.parts:
            errors.append("Unsafe manifest path: {}".format(relative))
            continue
        target = root / relative
        if not target.is_file():
            errors.append("Missing: {}".format(relative))
            continue
        content = target.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != item["sha256"]:
            errors.append("SHA-256 mismatch: {}".format(relative))
        if len(content) != item["size_bytes"]:
            errors.append("Size mismatch: {}".format(relative))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST),
                        help="Manifest path relative to --root, or absolute path")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    if not manifest.is_file():
        print("Baseline manifest not found: {}".format(manifest))
        return 1
    errors = verify(root, manifest)
    if errors:
        for error in errors:
            print("FAIL: {}".format(error))
        return 1
    n = len(json.loads(manifest.read_text(encoding="utf-8"))["artifacts"])
    print("PASS: {} archived M0 baseline artifacts match their SHA-256 checksums".format(n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

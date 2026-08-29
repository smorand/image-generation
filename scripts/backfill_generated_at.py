#!/usr/bin/env python3
"""Backfill the `generated_at` metadata field on images that predate it.

`generated_at` was added to GenerationMetadata after a lot of images had
already been generated, so those older files carry no generation date (see
AGENTS.md-adjacent commit "Add generation date to metadata"). This script
finds every image-gen PNG under a target directory that has valid generation
metadata but no `generated_at`, and adds one derived from the file's mtime
(not perfect -- copies/backups can change mtime -- but good enough per the
project owner's own call).

Safety, in order of how much it matters:
1. Dry-run by default. Nothing is written unless --apply is passed.
2. Never touches a file whose metadata can't be read (no EXIF, corrupted,
   or simply not an image-gen output). Those are reported, never guessed at.
3. Never touches a file that already has `generated_at`.
4. Writes to a temp file in the same directory, decodes it back and compares
   pixel data byte-for-byte against the original before ever replacing the
   original (os.replace, atomic on the same filesystem). If pixels don't
   match bit-for-bit, the file is left untouched and reported as a failure.
5. --limit lets you dry-run/apply a small batch first to sanity-check output
   before running against the whole tree.

Usage:
    python scripts/backfill_generated_at.py ~/.data/image-gen/out           # dry run
    python scripts/backfill_generated_at.py ~/.data/image-gen/out --limit 20 --apply  # smoke test
    python scripts/backfill_generated_at.py ~/.data/image-gen/out --apply   # full run
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import fields
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image  # noqa: E402

from image_gen.metadata import (  # noqa: E402  # type: ignore[import-untyped]  # local src/, not an installed stub
    GenerationMetadata,
    _build_exif_bytes,
    load_metadata,
)


def _fill_missing_fields(raw: dict, mtime_iso: str) -> dict:
    """Fill dataclass fields absent from an older record's JSON.

    `to_json()` has always dropped None fields, so any Optional field that
    happened to be None at save time (vae, lora, embedding, hires_scale, ...)
    is simply missing from older records. GenerationMetadata declares those
    without a `None` default (only the newer fields at the end of the
    dataclass do), so constructing it from a partial dict would raise
    TypeError. Fill every missing field with None except `generated_at`,
    which gets the file's mtime.
    """
    known_fields = {f.name for f in fields(GenerationMetadata)}
    filled = dict(raw)
    for name in known_fields - filled.keys():
        filled[name] = None
    # Drop any key load_metadata returned that isn't a known field (forward
    # compatibility safety net; shouldn't happen for our own output).
    filled = {k: v for k, v in filled.items() if k in known_fields}
    filled["generated_at"] = mtime_iso
    return filled


def _process_one(path: Path, apply: bool) -> str:
    """Return one of: 'skipped-has-date', 'skipped-no-metadata', 'would-update',
    'updated', or 'failed:<reason>'."""
    try:
        raw = load_metadata(path)
    except ValueError:
        return "skipped-no-metadata"

    if "generated_at" in raw:
        return "skipped-has-date"

    mtime_iso = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    filled = _fill_missing_fields(raw, mtime_iso)
    try:
        metadata = GenerationMetadata(**filled)
    except TypeError as exc:
        return f"failed:unexpected metadata shape ({exc})"

    if not apply:
        return "would-update"

    try:
        with Image.open(path) as original_img:
            original_img.load()
            original_pixels = original_img.tobytes()
            original_mode = original_img.mode
            exif_bytes = _build_exif_bytes(metadata)

            fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=path.suffix)
            os.close(fd)
            tmp_path = Path(tmp_name)
            try:
                original_img.save(tmp_path, "PNG", exif=exif_bytes)

                with Image.open(tmp_path) as check_img:
                    check_img.load()
                    if check_img.mode != original_mode or check_img.tobytes() != original_pixels:
                        return "failed:pixel mismatch after rewrite, original left untouched"

                os.replace(tmp_path, path)
            finally:
                tmp_path.unlink(missing_ok=True)
    except OSError as exc:
        return f"failed:{exc}"

    return "updated"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="Directory to scan recursively for *.png")
    parser.add_argument("--apply", action="store_true", help="Actually write changes (default: dry-run)")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N eligible files (for smoke-testing)")
    args = parser.parse_args()

    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        print(f"Error: not a directory: {directory}", file=sys.stderr)
        return 1

    counts: dict[str, int] = {}
    processed = 0
    failures: list[str] = []

    for path in sorted(directory.rglob("*.png")):
        if args.limit is not None and processed >= args.limit:
            break
        outcome = _process_one(path, apply=args.apply)
        bucket = outcome.split(":", 1)[0]
        counts[bucket] = counts.get(bucket, 0) + 1
        if bucket in ("would-update", "updated", "failed"):
            processed += 1
        if bucket == "failed":
            failures.append(f"{path}: {outcome}")

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"\n[{mode}] Scanned under {directory}")
    for bucket, n in sorted(counts.items()):
        print(f"  {bucket}: {n}")
    if failures:
        print("\nFailures (left untouched):")
        for line in failures:
            print(f"  - {line}")
    if not args.apply and counts.get("would-update", 0):
        print(f"\nRe-run with --apply to write {counts['would-update']} file(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

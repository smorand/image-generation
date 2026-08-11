"""Append-only JSONL logging of every generated image.

One line per generated image, written to ``<log_dir>/generations-YYYY-MM-DD.jsonl``
(daily rotation). Each line carries the full ``GenerationMetadata`` (every
sampling parameter plus the resolved generate-var ``variables``) enriched with a
timestamp, the output path, and the originating command. Nothing is lost even if
the image file is later moved or deleted, so downstream stats stay exhaustive.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .metadata import GenerationMetadata


def log_file_for(log_dir: str | Path, when: datetime | None = None) -> Path:
    """Return the daily log file path inside ``log_dir`` for ``when`` (now)."""
    when = when or datetime.now()
    return Path(log_dir) / f"generations-{when:%Y-%m-%d}.jsonl"


def append_generation_log(
    log_dir: str | Path | None,
    metadata: GenerationMetadata,
    output_path: str | Path,
    command: str,
    number: str | None = None,
) -> Path | None:
    """Append one JSONL record describing a generated image.

    Args:
        log_dir: Directory for daily log files. ``None`` disables logging (returns
            ``None`` without touching the filesystem).
        metadata: Full generation metadata (embedded in the image EXIF too).
        output_path: Where the image was written.
        command: Originating command ("generate" or "generate-var").
        number: Optional zero-padded counter (generate-var only).

    Returns:
        The log file path written to, or ``None`` when logging is disabled.
    """
    if log_dir is None:
        return None

    now = datetime.now()
    record: dict = {
        "timestamp": now.isoformat(timespec="seconds"),
        "command": command,
        "output": str(output_path),
    }
    if number is not None:
        record["number"] = number
    # Merge the full metadata (prompt, negative, all params, variables, ...).
    # to_json() already drops None fields; reuse the same shape here.
    record.update(json.loads(metadata.to_json()))

    path = log_file_for(log_dir, now)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path

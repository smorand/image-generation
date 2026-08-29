"""Video generation metadata: MP4 comment tag plus a JSON sidecar.

MP4 has no EXIF, so provenance is carried two ways: embedded in the container
``comment`` tag (see :mod:`video_gen.encode`) and written next to the clip as a
one-line ``<name>.json`` sidecar, mirroring image-gen's single-line JSON so the
same tooling habits apply.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class VideoMetadata:
    """Provenance for one generated clip."""

    prompt: str
    negative_prompt: str
    backend: str
    model: str
    source_image: str
    seed: int
    width: int
    height: int
    num_frames: int
    fps: int
    duration_s: float
    steps: int
    guidance: float
    # Resolved generate-var placeholders: {variable_name: chosen_value}.
    variables: dict[str, str] | None = None
    # Chained generation: number of segments and the per-segment prompt series.
    segments: int | None = None
    prompt_series: list[str] | None = None
    # Local timestamp of generation, ISO 8601 ("2026-08-29T14:30:05"); stamped
    # automatically at construction time, right after the clip is produced.
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_json(self) -> str:
        """Single-line JSON string (kept one line, like image-gen metadata)."""
        data = {k: v for k, v in asdict(self).items() if v is not None}
        return json.dumps(data, ensure_ascii=False)


def write_sidecar(output_path: Path, metadata: VideoMetadata) -> Path:
    """Write ``<output>.json`` next to the clip and return the sidecar path."""
    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(metadata.to_json() + "\n", encoding="utf-8")
    return sidecar

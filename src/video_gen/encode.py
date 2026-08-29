"""Frame list to MP4 encoding via imageio-ffmpeg (bundled ffmpeg, no system dep).

Writes an H.264 MP4 with ``yuv420p`` for broad player compatibility and embeds
the generation metadata JSON as the container ``comment`` tag, so a clip carries
its own provenance the way image-gen embeds EXIF UserComment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def encode_frames(
    frames: Sequence[Image.Image | np.ndarray],
    output_path: str | Path,
    fps: int,
    comment: str | None = None,
    quality: int = 8,
) -> Path:
    """Encode frames to an MP4 file and return its path.

    Args:
        frames: PIL images or HxWx3 uint8 arrays, all the same size.
        output_path: Target ``.mp4`` path (suffix forced to ``.mp4``).
        fps: Output frame rate.
        comment: Optional metadata string stored as the container comment tag.
        quality: imageio quality (0-10, higher is better).
    """
    import imageio.v2 as imageio

    output_path = Path(output_path).with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_params: list[str] = []
    if comment:
        # ffmpeg splits on the first '=' only, so newlines/quotes must go.
        safe = comment.replace("\n", " ").replace("\r", " ")
        output_params += ["-metadata", f"comment={safe}"]

    writer = imageio.get_writer(
        str(output_path),
        format="ffmpeg",  # type: ignore[arg-type]  # imageio resolves the plugin by name at runtime
        fps=fps,
        codec="libx264",
        quality=quality,
        macro_block_size=16,
        pixelformat="yuv420p",  # broad player compatibility
        output_params=output_params or None,
    )
    try:
        for frame in frames:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = (arr.clip(0, 1) * 255).astype(np.uint8) if arr.max() <= 1 else arr.astype(np.uint8)
            writer.append_data(arr)
    finally:
        writer.close()

    return output_path

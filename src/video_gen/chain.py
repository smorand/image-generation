"""Chained (autoregressive) long-video generation.

A single image-to-video call denoises the whole clip in one latent tensor, so
its memory grows with the frame count. On Apple Silicon a ~10 s LTX clip (241
frames) tries to allocate a ~27 GB Metal buffer and dies. To make clips longer
than the backend's native ceiling, generate a **series of short segments**: each
segment is seeded by the *last frame* of the previous one and gets its own
prompt, then all segments are concatenated into one clip.

Because each segment is small, peak memory stays flat no matter the total
length. The trade-off is inherent to image-conditioned i2v: the model sees only
one frame between segments, so motion velocity resets at each seam (a slight
hitch is possible). Keeping segments a few seconds long minimises it.
"""

from __future__ import annotations

from typing import Callable, Sequence

from PIL import Image

# (prompt, conditioning_image, seed) -> generated frames for one segment.
SegmentGenerator = Callable[[str, Image.Image, int], list[Image.Image]]
# (index, total, prompt, seed) -> None. Progress hook.
SegmentCallback = Callable[[int, int, str, int], None]


def build_prompt_series(prompts: Sequence[str], segments: int | None) -> list[str]:
    """Return the per-segment prompt list.

    - ``segments`` is ``None``: one segment per given prompt.
    - ``segments`` > len(prompts): pad by repeating the last prompt.
    - ``segments`` <= len(prompts): truncate to the first ``segments`` prompts.
    """
    if not prompts:
        raise ValueError("at least one prompt is required")
    if segments is None:
        return list(prompts)
    if segments < 1:
        raise ValueError(f"segments must be >= 1, got {segments}")
    if segments <= len(prompts):
        return list(prompts[:segments])
    return list(prompts) + [prompts[-1]] * (segments - len(prompts))


def stitch_segments(segments: Sequence[Sequence[Image.Image]]) -> list[Image.Image]:
    """Concatenate segment frame lists into one, dropping duplicated seams.

    Every segment after the first opens on its conditioning image (the previous
    segment's last frame), so that first frame is dropped to avoid a stutter.
    """
    out: list[Image.Image] = []
    for i, seg in enumerate(segments):
        out.extend(seg if i == 0 else list(seg)[1:])
    return out


def generate_chain(
    generate_segment: SegmentGenerator,
    prompts: Sequence[str],
    start_image: Image.Image,
    *,
    base_seed: int,
    on_segment: SegmentCallback | None = None,
    free_memory: Callable[[], None] | None = None,
) -> list[Image.Image]:
    """Generate a long clip as a chain of prompt-driven segments.

    Each segment ``i`` is generated from ``prompts[i]``, conditioned on the last
    frame of segment ``i-1`` (or ``start_image`` for the first), with seed
    ``base_seed + i`` for reproducible-but-varied motion. ``free_memory`` (if
    given) is called after each segment to release transient buffers.

    Returns the stitched list of frames across all segments.
    """
    total = len(prompts)
    if total == 0:
        raise ValueError("at least one prompt is required")

    segments: list[list[Image.Image]] = []
    current = start_image
    for i, prompt in enumerate(prompts):
        seed = base_seed + i
        if on_segment is not None:
            on_segment(i, total, prompt, seed)
        frames = generate_segment(prompt, current, seed)
        if not frames:
            raise RuntimeError(f"segment {i} produced no frames")
        segments.append(list(frames))
        current = frames[-1]
        if free_memory is not None:
            free_memory()
    return stitch_segments(segments)

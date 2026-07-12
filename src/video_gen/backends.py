"""Video backend registry.

Two switchable image-to-video backends, both diffusers pipelines that run on
Apple Silicon (MPS) in bfloat16, no FP8 (Metal has no float8 support):

* ``wan``  Wan 2.2 TI2V-5B, higher quality, slower. Frame count is 4k+1.
* ``ltx``  LTX-Video 0.9, lighter and faster, lower coherence. Frame count is 8k+1.

Each backend declares its native frame rate, the frame-count arithmetic its VAE
requires, and sensible Mac-friendly defaults. ``resolve_frames`` snaps a desired
duration to a valid frame count for the chosen backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BackendSpec:
    """Static description of one video backend."""

    name: str
    repo: str
    pipeline_cls: str  # diffusers class name
    frame_multiple: int  # num_frames must equal frame_multiple * k + 1
    native_fps: int
    max_frames: int
    default_width: int
    default_height: int
    default_steps: int
    default_guidance: float
    dim_multiple: int  # width/height must be a multiple of this
    vae_fp32: bool = False  # load VAE in float32 to avoid MPS NaNs
    call_extra: dict[str, Any] = field(default_factory=dict)  # extra __call__ kwargs
    description: str = ""


BACKENDS: dict[str, BackendSpec] = {
    "wan": BackendSpec(
        name="wan",
        repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        pipeline_cls="WanImageToVideoPipeline",
        frame_multiple=4,
        native_fps=24,
        max_frames=121,  # ~5 s at 24 fps, native ceiling
        default_width=704,
        default_height=704,
        default_steps=30,
        default_guidance=5.0,
        dim_multiple=32,
        vae_fp32=True,
        description="Wan 2.2 TI2V-5B, quality, slower",
    ),
    "ltx": BackendSpec(
        name="ltx",
        repo="Lightricks/LTX-Video",
        pipeline_cls="LTXImageToVideoPipeline",
        frame_multiple=8,
        native_fps=24,
        max_frames=257,
        default_width=704,
        default_height=480,
        default_steps=40,
        default_guidance=3.0,
        dim_multiple=32,
        vae_fp32=False,
        call_extra={"decode_timestep": 0.03, "decode_noise_scale": 0.025},
        description="LTX-Video 0.9, fast, lighter",
    ),
}

DEFAULT_BACKEND = "ltx"

# Accepted --precision values (dtype override). Kept torch-free for light imports.
PRECISIONS = frozenset({"fp16", "float16", "bf16", "bfloat16", "fp32", "float32"})

# A generic, video-oriented negative prompt (both backends accept one).
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "low quality, deformed, watermark, text"
)


def get_backend(name: str) -> BackendSpec:
    """Return the backend spec or raise ValueError with the valid names."""
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown backend '{name}'. Supported: {', '.join(BACKENDS)}"
        ) from None


def snap_dim(value: int, multiple: int) -> int:
    """Round ``value`` down to the nearest positive multiple of ``multiple``."""
    v = (value // multiple) * multiple
    return max(multiple, v)


def resolve_frames(backend: BackendSpec, duration_s: float, fps: int) -> int:
    """Convert a duration in seconds to a valid frame count for the backend.

    Snaps ``round(duration * fps)`` to the nearest ``frame_multiple * k + 1``
    value, clamped to at least one step and to the backend ceiling.
    """
    target = max(1, round(duration_s * fps))
    m = backend.frame_multiple
    k = max(1, round((target - 1) / m))
    frames = m * k + 1
    if frames > backend.max_frames:
        frames = backend.max_frames
    return frames


def frames_to_duration(num_frames: int, fps: int) -> float:
    """Return clip duration in seconds for a frame count and fps."""
    return round(num_frames / fps, 2)

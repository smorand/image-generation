"""Image-to-video pipeline wrapper for the Wan and LTX backends.

Loads a diffusers video pipeline for the selected backend, keeps device/dtype
handling in one place (CUDA > MPS > CPU, bfloat16 on GPU), and exposes a single
``generate`` returning a list of PIL frames.

Memory notes (Apple Silicon, 32 GB unified):
* bfloat16 only. Metal has no float8, so FP8 checkpoints are unusable.
* Wan's text encoder (UMT5) is large; ``offload=True`` uses sequential CPU
  offload to fit. Default keeps the pipeline resident on the device with VAE
  tiling, which is faster when it fits.
"""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .backends import BackendSpec, DEFAULT_NEGATIVE_PROMPT, get_backend, snap_dim


@dataclass
class VideoGenConfig:
    """Parameters for one image-to-video generation."""

    prompt: str
    image: Image.Image
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    width: int = 704
    height: int = 480
    num_frames: int = 49
    steps: int = 40
    guidance: float = 3.0
    seed: int | None = None
    call_extra: dict = field(default_factory=dict)


def _autodetect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class VideoPipeline:
    """Wraps a diffusers image-to-video pipeline for one backend."""

    _PRECISION = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }

    def __init__(
        self,
        backend: str,
        repo_override: str | None = None,
        device: str | None = None,
        offload: bool = False,
        precision: str | None = None,
    ):
        self.spec: BackendSpec = get_backend(backend)
        self.repo = repo_override or self.spec.repo
        self.device = device or _autodetect_device()
        self.offload = offload
        if precision:
            self.dtype = self._PRECISION[precision]
        elif self.device in ("cuda", "mps"):
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        self._pipe: Any = None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        if self._pipe is not None:
            return
        if self.spec.pipeline_cls == "WanImageToVideoPipeline":
            self._load_wan()
        elif self.spec.pipeline_cls == "LTXImageToVideoPipeline":
            self._load_ltx()
        else:  # pragma: no cover - registry guards this
            raise ValueError(f"unsupported pipeline class {self.spec.pipeline_cls}")
        self._place()

    def _load_wan(self) -> None:
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

        if self.device == "mps":
            # Empirically broken on Metal: the Wan transformer produces scrambled
            # (grid-noise) frames after the conditioning frame, identically in
            # bf16/fp16/fp32, so it is an op-level MPS incompatibility, not a
            # precision issue. Use `ltx` on Apple Silicon. Kept loadable for
            # experimentation and for CUDA hosts, where Wan works.
            warnings.warn(
                "Wan on MPS/Metal produces broken output (known limitation); use --backend ltx on Apple Silicon.",
                stacklevel=2,
            )

        vae = AutoencoderKLWan.from_pretrained(self.repo, subfolder="vae", torch_dtype=torch.float32)
        self._pipe = WanImageToVideoPipeline.from_pretrained(self.repo, vae=vae, torch_dtype=self.dtype)

    def _load_ltx(self) -> None:
        from diffusers import LTXImageToVideoPipeline

        self._pipe = LTXImageToVideoPipeline.from_pretrained(self.repo, torch_dtype=self.dtype)

    def _place(self) -> None:
        pipe: Any = self._pipe
        pipe.set_progress_bar_config(disable=True)
        # VAE tiling/slicing keeps the decode step within budget on Mac.
        if hasattr(pipe, "vae"):
            try:
                pipe.vae.enable_tiling()
                pipe.vae.enable_slicing()
            except (AttributeError, ValueError):
                pass
        if self.offload:
            # Sequential offload streams weights per-submodule; slowest but
            # smallest footprint. Needed when the model does not fit resident.
            pipe.enable_sequential_cpu_offload(device=self.device)
        else:
            pipe.to(self.device)

    @property
    def pipe(self):
        if self._pipe is None:
            raise RuntimeError("pipeline not loaded; call load() first")
        return self._pipe

    def empty_cache(self) -> None:
        """Free transient activation memory, keeping the weights resident.

        Called between chained segments: the big per-clip latent/attention
        buffers are released so the next segment starts from a clean allocation,
        while the loaded pipeline stays on the device (no reload cost).
        """
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    def unload(self) -> None:
        """Release the pipeline and free device memory (for backend switches)."""
        self._pipe = None
        self.empty_cache()

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def prepare_image(self, image: Image.Image, width: int, height: int) -> Image.Image:
        """Center-crop-resize the source image to the target frame size."""
        image = image.convert("RGB")
        tw, th = width, height
        sw, sh = image.size
        ratio = max(tw / sw, th / sh)
        rw, rh = round(sw * ratio), round(sh * ratio)
        image = image.resize((rw, rh), Image.Resampling.LANCZOS)
        left = (rw - tw) // 2
        top = (rh - th) // 2
        return image.crop((left, top, left + tw, top + th))

    def generate(self, config: VideoGenConfig) -> list[Image.Image]:
        """Run image-to-video and return a list of PIL frames."""
        self.load()

        width = snap_dim(config.width, self.spec.dim_multiple)
        height = snap_dim(config.height, self.spec.dim_multiple)
        image = self.prepare_image(config.image, width, height)

        # CPU generator works across cuda/mps/cpu; MPS lacks a native generator.
        generator = None
        if config.seed is not None:
            generator = torch.Generator().manual_seed(config.seed)

        call_kwargs = dict(
            image=image,
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            width=width,
            height=height,
            num_frames=config.num_frames,
            num_inference_steps=config.steps,
            guidance_scale=config.guidance,
            generator=generator,
            output_type="pil",
        )
        call_kwargs.update(self.spec.call_extra)
        call_kwargs.update(config.call_extra)

        result = self.pipe(**call_kwargs)
        return result.frames[0]


def load_source_image(path: str | Path) -> Image.Image:
    """Load a source image from disk as RGB."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"source image not found: {p}")
    return Image.open(p).convert("RGB")

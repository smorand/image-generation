"""SDXL pipeline loader and image generation."""

import gc
from dataclasses import dataclass, field
from pathlib import Path

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from PIL import Image

from .embeddings import load_embeddings
from .ip_adapter import build_ip_adapter_image
from .ip_adapter import load_ip_adapter as _load_ip_adapter
from .lora import load_loras
from .prompt_encoding import SDXLPromptEncoder
from .schedulers import get_scheduler

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, bad anatomy, worst quality, low resolution, "
    "jpeg artifacts, ugly, duplicate, morbid, mutilated, deformed"
)


@dataclass
class GenerationConfig:
    """Configuration for image generation."""

    prompt: str
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg_scale: float = 7.0
    seed: int | None = None
    clip_skip: int = 1
    batch_size: int = 1
    # Hi-res fix
    hires_fix: bool = False
    hires_scale: float = 1.5
    hires_steps: int = 15
    hires_denoising: float = 0.5
    # IP-Adapter (identity/style conditioning from reference images)
    ip_adapter_images: list[Image.Image] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.negative_prompt is None:
            self.negative_prompt = DEFAULT_NEGATIVE_PROMPT


class SDXLPipeline:
    """SDXL pipeline wrapper for safetensors models."""

    def __init__(
        self,
        model_path: str | Path,
        vae_path: str | Path | None = None,
        scheduler_name: str = "dpm++_2m_karras",
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize SDXL pipeline from safetensors checkpoint.

        Args:
            model_path: Path to safetensors model file
            vae_path: Optional path to custom VAE
            scheduler_name: Scheduler to use
            device: Device to use (auto-detected if None)
            dtype: Data type (auto-detected if None)
        """
        self.model_path = Path(model_path)
        self.vae_path = Path(vae_path) if vae_path else None
        self.scheduler_name = scheduler_name

        # Auto-detect device and dtype
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if dtype is None:
            if self.device == "cuda":
                self.dtype = torch.float16
            elif self.device == "mps":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = dtype

        self._pipeline: StableDiffusionXLPipeline | None = None
        self._prompt_encoder: SDXLPromptEncoder | None = None
        self._loaded_loras: list[str] = []
        self._loaded_embeddings: list[str] = []
        self._ip_adapter_preset: str | None = None
        self._ip_adapter_scale: float | None = None

    def load(self) -> None:
        """Load the pipeline into memory."""
        if self._pipeline is not None:
            return

        # Load main pipeline from safetensors
        self._pipeline = StableDiffusionXLPipeline.from_single_file(
            str(self.model_path),
            torch_dtype=self.dtype,
            use_safetensors=True,
        )

        # Load custom VAE if specified
        if self.vae_path and self.vae_path.exists():
            vae = AutoencoderKL.from_single_file(
                str(self.vae_path),
                torch_dtype=self.dtype,
            )
            self._pipeline.vae = vae

        # Set scheduler
        self._pipeline.scheduler = get_scheduler(
            self.scheduler_name,
            self._pipeline.scheduler.config,
        )

        # Move to device
        self._pipeline = self._pipeline.to(self.device)

        # Enable optimizations
        if self.device == "cuda":
            self._pipeline.enable_model_cpu_offload()
        self._pipeline.vae.enable_slicing()
        self._pipeline.vae.enable_tiling()

        # Initialize prompt encoder for long prompt support
        self._prompt_encoder = SDXLPromptEncoder(self._pipeline)

    def free_cache(self) -> None:
        """Release cached device memory without unloading the pipeline.

        SDXL leaves intermediate activations (latents, VAE decode buffers,
        img2img wrappers) in the MPS/CUDA allocator cache after each call.
        In the ``generate-var`` loop these accumulate across iterations and
        starve the shared unified-memory pool, slowing every subsequent run.
        Call this after each generation to hand the memory back.
        """
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    @property
    def pipeline(self) -> StableDiffusionXLPipeline:
        """Get the loaded pipeline."""
        if self._pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")
        return self._pipeline

    @property
    def prompt_encoder(self) -> SDXLPromptEncoder:
        """Get the prompt encoder."""
        if self._prompt_encoder is None:
            raise RuntimeError("Prompt encoder not initialized. Call load() first.")
        return self._prompt_encoder

    def load_loras(self, lora_specs: list[str]) -> None:
        """Load LoRA weights into the pipeline."""
        self.load()
        load_loras(self.pipeline, lora_specs)
        self._loaded_loras = lora_specs

    def load_embeddings(self, embedding_paths: list[str]) -> list[str]:
        """Load textual inversion embeddings."""
        self.load()
        tokens = load_embeddings(self.pipeline, embedding_paths)
        self._loaded_embeddings = embedding_paths
        return tokens

    def load_ip_adapter(self, preset: str, scale: float | None = None) -> float:
        """
        Load an IP-Adapter for reference-image conditioning.

        Args:
            preset: IP-Adapter preset name (e.g. "face").
            scale: Conditioning strength (0-1). None uses the preset default.

        Returns:
            The effective scale applied.
        """
        self.load()
        effective_scale = _load_ip_adapter(
            self.pipeline, preset, scale, self.device, self.dtype
        )
        self._ip_adapter_preset = preset
        self._ip_adapter_scale = effective_scale
        return effective_scale

    def set_clip_skip(self, clip_skip: int) -> None:
        """
        Set CLIP skip layers.

        Args:
            clip_skip: Number of layers to skip (1 = no skip, 2+ = skip layers)
        """
        if clip_skip < 1:
            clip_skip = 1

        # For SDXL, we modify the text encoder outputs
        # clip_skip=1 means use all layers, clip_skip=2 means skip last layer, etc.
        # This is handled during generation via clip_skip parameter
        self._clip_skip = clip_skip

    def generate(self, config: GenerationConfig) -> list[Image.Image]:
        """
        Generate images from the configuration.

        Args:
            config: Generation configuration

        Returns:
            List of generated PIL images
        """
        self.load()

        # Set up generator for reproducibility
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(config.seed)

        # Encode prompts using compel (supports long prompts and weighting)
        prompt_kwargs = self.prompt_encoder.get_embeddings_for_pipeline(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
        )

        # Shape IP-Adapter reference image(s) if any were loaded
        ip_adapter_kwargs = {}
        if config.ip_adapter_images:
            ip_adapter_kwargs["ip_adapter_image"] = build_ip_adapter_image(
                config.ip_adapter_images
            )

        # Generate at base resolution
        result = self.pipeline(
            **prompt_kwargs,
            **ip_adapter_kwargs,
            width=config.width,
            height=config.height,
            num_inference_steps=config.steps,
            guidance_scale=config.cfg_scale,
            num_images_per_prompt=config.batch_size,
            generator=generator,
            clip_skip=config.clip_skip - 1 if config.clip_skip > 1 else None,
        )

        images = result.images

        # Apply hi-res fix if enabled
        if config.hires_fix:
            images = self._apply_hires_fix(
                images=images,
                config=config,
                generator=generator,
                prompt_kwargs=prompt_kwargs,
                ip_adapter_kwargs=ip_adapter_kwargs,
            )

        # Hand cached activations back to the shared memory pool so a looped
        # caller (generate-var) does not accumulate them across iterations.
        del result
        self.free_cache()

        return images

    def _apply_hires_fix(
        self,
        images: list[Image.Image],
        config: GenerationConfig,
        generator: torch.Generator | None,
        prompt_kwargs: dict,
        ip_adapter_kwargs: dict | None = None,
    ) -> list[Image.Image]:
        """
        Apply hi-res fix (img2img upscaling pass).

        Args:
            images: Base resolution images
            config: Generation config
            generator: Random generator for reproducibility
            prompt_kwargs: Pre-encoded prompt embeddings
            ip_adapter_kwargs: IP-Adapter image kwargs to carry into the pass

        Returns:
            Upscaled images
        """
        from diffusers import StableDiffusionXLImg2ImgPipeline

        ip_adapter_kwargs = ip_adapter_kwargs or {}

        # Create img2img pipeline from loaded pipeline. Reuse the IP-Adapter
        # image encoder/feature extractor so the identity carries into the pass.
        img2img_components = dict(
            vae=self.pipeline.vae,
            text_encoder=self.pipeline.text_encoder,
            text_encoder_2=self.pipeline.text_encoder_2,
            tokenizer=self.pipeline.tokenizer,
            tokenizer_2=self.pipeline.tokenizer_2,
            unet=self.pipeline.unet,
            scheduler=self.pipeline.scheduler,
        )
        if self._ip_adapter_preset is not None:
            img2img_components["image_encoder"] = self.pipeline.image_encoder
            img2img_components["feature_extractor"] = self.pipeline.feature_extractor
        img2img = StableDiffusionXLImg2ImgPipeline(**img2img_components)

        upscaled_images = []
        target_width = int(config.width * config.hires_scale)
        target_height = int(config.height * config.hires_scale)

        for img in images:
            # Resize image to target resolution
            resized = img.resize((target_width, target_height), Image.LANCZOS)

            # Run img2img pass with pre-encoded embeddings
            result = img2img(
                **prompt_kwargs,
                **ip_adapter_kwargs,
                image=resized,
                num_inference_steps=config.hires_steps,
                strength=config.hires_denoising,
                guidance_scale=config.cfg_scale,
                generator=generator,
            )

            upscaled_images.append(result.images[0])

        # Drop the per-call img2img wrapper before returning; its components are
        # shared with the base pipeline, but the wrapper and its scratch buffers
        # are not, and would otherwise linger in the allocator cache.
        del img2img
        self.free_cache()

        return upscaled_images

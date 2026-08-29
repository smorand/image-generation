"""IP-Adapter loading for identity/style conditioning on SDXL.

IP-Adapter injects a reference image (typically a face) into the cross-attention
of the UNet, so generated images keep the same identity across seeds and prompts.
It is orthogonal to LoRA: LoRA bakes a concept into the weights, IP-Adapter
conditions a single generation on a reference image with no training.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    import torch
    from diffusers import StableDiffusionXLPipeline

# HuggingFace repo hosting all SDXL IP-Adapter weights and image encoders.
IP_ADAPTER_REPO = "h94/IP-Adapter"

# Presets map a friendly name to the weight file and its matching image encoder.
# "face" is the default for consistent-model generation (portrait/identity).
IP_ADAPTER_PRESETS: dict[str, dict] = {
    "face": {
        "weight_subfolder": "sdxl_models",
        "weight_name": "ip-adapter-plus-face_sdxl_vit-h.safetensors",
        "encoder_subfolder": "models/image_encoder",  # ViT-H
        "default_scale": 0.6,
    },
    "plus": {
        "weight_subfolder": "sdxl_models",
        "weight_name": "ip-adapter-plus_sdxl_vit-h.safetensors",
        "encoder_subfolder": "models/image_encoder",  # ViT-H
        "default_scale": 0.6,
    },
    "standard": {
        "weight_subfolder": "sdxl_models",
        "weight_name": "ip-adapter_sdxl.bin",
        "encoder_subfolder": "sdxl_models/image_encoder",  # ViT-bigG
        "default_scale": 0.6,
    },
}

SUPPORTED_IP_ADAPTERS: list[str] = list(IP_ADAPTER_PRESETS.keys())


def load_reference_images(paths: list[str]) -> list[Image.Image]:
    """
    Load IP-Adapter reference images from disk.

    Args:
        paths: List of image file paths.

    Returns:
        List of RGB PIL images.
    """
    images: list[Image.Image] = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"IP-Adapter reference image not found: {path}")
        images.append(Image.open(path).convert("RGB"))
    return images


def build_ip_adapter_image(images: list[Image.Image]) -> Image.Image | list:
    """
    Shape reference images for a single IP-Adapter as diffusers expects.

    diffusers treats a flat list as "one image per adapter". With a single
    adapter we pass either one image, or a nested list holding all references
    for that adapter.

    Args:
        images: Reference images.

    Returns:
        A single image (one reference) or a nested list (multiple references).
    """
    if len(images) == 1:
        return images[0]
    return [images]


def load_ip_adapter(
    pipeline: "StableDiffusionXLPipeline",
    preset: str,
    scale: float | None,
    device: str,
    dtype: "torch.dtype",
) -> float:
    """
    Load an IP-Adapter preset into the pipeline and set its scale.

    Args:
        pipeline: Loaded SDXL pipeline.
        preset: One of SUPPORTED_IP_ADAPTERS.
        scale: Conditioning strength (0-1). None uses the preset default.
        device: Device to place the image encoder on.
        dtype: Data type for the image encoder.

    Returns:
        The effective scale applied.
    """
    from transformers import CLIPVisionModelWithProjection

    if preset not in IP_ADAPTER_PRESETS:
        raise ValueError(f"Unknown IP-Adapter '{preset}'. Supported: {', '.join(SUPPORTED_IP_ADAPTERS)}")

    cfg = IP_ADAPTER_PRESETS[preset]

    # Load the matching image encoder explicitly (the face/plus presets use a
    # ViT-H encoder that does not sit next to the weights, so autodetection
    # would pick the wrong one).
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        IP_ADAPTER_REPO,
        subfolder=cfg["encoder_subfolder"],
        torch_dtype=dtype,
    ).to(device)
    pipeline.image_encoder = image_encoder

    # image_encoder_folder=None -> reuse the encoder we just attached.
    pipeline.load_ip_adapter(
        IP_ADAPTER_REPO,
        subfolder=cfg["weight_subfolder"],
        weight_name=cfg["weight_name"],
        image_encoder_folder=None,
    )

    effective_scale = scale if scale is not None else cfg["default_scale"]
    pipeline.set_ip_adapter_scale(effective_scale)
    return effective_scale

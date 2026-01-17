"""EXIF metadata handling for generated images."""

import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import piexif
from PIL import Image


@dataclass
class GenerationMetadata:
    """Metadata for image generation."""

    prompt: str
    negative_prompt: str
    model: str
    vae: str | None
    seed: int
    width: int
    height: int
    steps: int
    cfg_scale: float
    scheduler: str
    clip_skip: int
    lora: list[str] | None
    embedding: list[str] | None
    hires_fix: bool
    hires_scale: float | None
    hires_steps: int | None
    hires_denoising: float | None

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = asdict(self)
        # Clean up None values for cleaner JSON
        data = {k: v for k, v in data.items() if v is not None}
        return json.dumps(data, ensure_ascii=False)


def save_image_with_metadata(
    image: Image.Image,
    output_path: Path,
    metadata: GenerationMetadata,
    quality: int = 95,
) -> None:
    """
    Save image as JPEG with EXIF metadata.

    Args:
        image: PIL Image to save
        output_path: Output file path
        metadata: Generation metadata to embed
        quality: JPEG quality (1-100)
    """
    # Ensure output path has .jpg extension
    output_path = output_path.with_suffix(".jpg")

    # Convert to RGB if necessary (JPEG doesn't support RGBA)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    # Create EXIF data with UserComment containing JSON metadata
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    # Encode metadata as UserComment (EXIF tag 37510)
    # UserComment requires a charset prefix
    json_str = metadata.to_json()
    user_comment = b"ASCII\x00\x00\x00" + json_str.encode("utf-8")
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment

    # Add software tag
    exif_dict["0th"][piexif.ImageIFD.Software] = "image-gen (SDXL)"

    # Dump EXIF to bytes
    exif_bytes = piexif.dump(exif_dict)

    # Save with EXIF
    image.save(
        output_path,
        "JPEG",
        quality=quality,
        exif=exif_bytes,
    )

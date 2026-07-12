"""EXIF metadata handling for generated images."""

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
    ip_adapter: str | None = None
    ip_adapter_images: list[str] | None = None
    ip_adapter_scale: float | None = None
    # Resolved generate-var placeholders: {variable_name: chosen_value}.
    variables: dict[str, str] | None = None

    def to_json(self) -> str:
        """Convert to a single-line JSON string (kept one line for get_info.sh)."""
        data = asdict(self)
        # Clean up None values for cleaner JSON
        data = {k: v for k, v in data.items() if v is not None}
        return json.dumps(data, ensure_ascii=False)


def _build_exif_bytes(metadata: GenerationMetadata) -> bytes:
    """Encode metadata JSON into EXIF UserComment (tag 37510) bytes."""
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    # UserComment requires a charset prefix.
    user_comment = b"ASCII\x00\x00\x00" + metadata.to_json().encode("utf-8")
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
    exif_dict["0th"][piexif.ImageIFD.Software] = "image-gen (SDXL)"
    return piexif.dump(exif_dict)


def save_image_with_metadata(
    image: Image.Image,
    output_path: Path,
    metadata: GenerationMetadata,
    quality: int = 95,
) -> Path:
    """
    Save an image with generation metadata in an EXIF UserComment.

    The format follows the path suffix: ``.png`` writes a PNG eXIf chunk,
    anything else is coerced to JPEG. Both carry the same JSON UserComment, so
    ``exiftool`` reports it as "User Comment" and ~/.local/bin/get_info.sh reads
    either one identically.

    Args:
        image: PIL Image to save
        output_path: Output file path (suffix selects the format)
        metadata: Generation metadata to embed
        quality: JPEG quality (1-100), ignored for PNG

    Returns:
        The actual path the image was written to.
    """
    exif_bytes = _build_exif_bytes(metadata)

    if output_path.suffix.lower() == ".png":
        image.save(output_path, "PNG", exif=exif_bytes)
        return output_path

    output_path = output_path.with_suffix(".jpg")
    # JPEG doesn't support RGBA/palette modes.
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(output_path, "JPEG", quality=quality, exif=exif_bytes)
    return output_path

"""EXIF metadata handling for generated images."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import piexif  # type: ignore[import-untyped]  # piexif ships no py.typed/stubs
from PIL import ExifTags, Image


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
    # Filename (not full path) of the source image, set only by generate-similar.
    source_image: str | None = None
    # --vary instruction text, set only by generate-similar --vary (LLM mode).
    llm_request: str | None = None
    # Local timestamp of generation, ISO 8601 ("2026-08-29T14:30:05"); stamped
    # automatically at construction time, right after the image is produced.
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_json(self) -> str:
        """Convert to a single-line JSON string (kept one line for get_info.sh)."""
        data = asdict(self)
        # Clean up None values for cleaner JSON
        data = {k: v for k, v in data.items() if v is not None}
        return json.dumps(data, ensure_ascii=False)


def _build_exif_bytes(metadata: GenerationMetadata) -> bytes:
    """Encode metadata JSON into EXIF UserComment (tag 37510) bytes."""
    exif_dict: dict[str, dict[int, object] | None] = {
        "0th": {},
        "Exif": {},
        "GPS": {},
        "1st": {},
        "thumbnail": None,
    }
    # UserComment requires a charset prefix.
    user_comment = b"ASCII\x00\x00\x00" + metadata.to_json().encode("utf-8")
    exif_ifd = exif_dict["Exif"]
    if exif_ifd is None:
        raise RuntimeError("unreachable: exif_dict['Exif'] is always a dict, never None")
    exif_ifd[piexif.ExifIFD.UserComment] = user_comment
    # Standard EXIF date tags, so Finder/Preview/exiftool/photo managers show
    # the actual generation time instead of falling back to the file's mtime
    # (which changes on every copy/move and says nothing about generation).
    # EXIF dates are ASCII "YYYY:MM:DD HH:MM:SS", no timezone, no "T" separator.
    exif_date = metadata.generated_at.replace("-", ":").replace("T", " ")
    exif_ifd[piexif.ExifIFD.DateTimeOriginal] = exif_date
    exif_ifd[piexif.ExifIFD.DateTimeDigitized] = exif_date
    zeroth_ifd = exif_dict["0th"]
    if zeroth_ifd is None:
        raise RuntimeError("unreachable: exif_dict['0th'] is always a dict, never None")
    zeroth_ifd[piexif.ImageIFD.Software] = "image-gen (SDXL)"
    zeroth_ifd[piexif.ImageIFD.DateTime] = exif_date
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


def load_metadata(path: Path) -> dict:
    """
    Read back the generation metadata embedded in an image's EXIF UserComment.

    Works on both PNG (eXIf chunk) and JPEG, mirroring the read side of
    ``save_image_with_metadata``: PNG exposes its EXIF bytes via
    ``Image.info["exif"]``, while JPEG is read directly by piexif.

    Args:
        path: Path to an image previously saved by ``save_image_with_metadata``.

    Returns:
        The raw metadata dict as it was written (JSON-decoded UserComment).
        Optional fields that were ``None`` at save time are simply absent.

    Raises:
        ValueError: If the image has no EXIF, no UserComment, the UserComment
            isn't valid JSON, or the JSON doesn't look like generation
            metadata (missing the mandatory ``prompt`` key).
    """
    # Pillow's own Image.getexif() already handles both containers we care
    # about: a real PNG eXIf chunk / JPEG APP1 segment, AND the legacy
    # ImageMagick zTXt/tEXt/iTXt "Raw profile type exif" convention (some
    # EXIF writers, e.g. the little_exif Rust crate used by
    # image-sec-gallery's `download`, only ever write the latter for PNG).
    # piexif has no PNG support at all, so it can't read either PNG case.
    # If more than one "Raw profile type exif" text chunk exists in a file
    # (same or mixed zTXt/tEXt/iTXt), Pillow's PNG chunk reader resolves it
    # for us: chunks are parsed in file order into a plain dict keyed by
    # chunk keyword, so the last chunk in file order silently wins. No
    # merge/error/warning logic is added here since no writer we control
    # (or little_exif) ever produces duplicates in the first place.
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    except (OSError, FileNotFoundError) as exc:
        raise ValueError(f"Cannot open image {path}: {exc}") from exc

    raw = exif_ifd.get(ExifTags.Base.UserComment)
    if not raw:
        raise ValueError(
            f"No generation metadata found in {path} (no EXIF UserComment). "
            "This image was likely not generated by image-gen, or its metadata "
            "was stripped."
        )

    # Strip the 8-byte charset prefix (b"ASCII\x00\x00\x00").
    payload = bytes(raw)[8:]
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid generation metadata JSON in {path}: {exc}") from exc

    if not isinstance(data, dict) or "prompt" not in data:
        raise ValueError(f"Generation metadata in {path} is missing the 'prompt' field.")

    return data

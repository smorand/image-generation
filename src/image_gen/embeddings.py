"""Textual inversion embedding loader for SDXL pipeline."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffusers import StableDiffusionXLPipeline


def load_embeddings(
    pipeline: "StableDiffusionXLPipeline",
    embedding_paths: list[str],
) -> list[str]:
    """
    Load textual inversion embeddings into the pipeline.

    Args:
        pipeline: SDXL pipeline instance
        embedding_paths: List of paths to embedding files (.pt, .safetensors, .bin)

    Returns:
        List of token names that can be used in prompts
    """
    if not embedding_paths:
        return []

    loaded_tokens = []

    for path_str in embedding_paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")

        # Use filename (without extension) as the token name
        token_name = path.stem

        # Load into both text encoders for SDXL
        pipeline.load_textual_inversion(
            str(path),
            token=token_name,
        )

        loaded_tokens.append(token_name)

    return loaded_tokens

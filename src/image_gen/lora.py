"""LoRA loading utilities for SDXL pipeline."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffusers import StableDiffusionXLPipeline


def parse_lora_arg(lora_arg: str) -> tuple[Path, float]:
    """
    Parse LoRA argument in format 'path:weight' or 'path'.

    Args:
        lora_arg: LoRA specification string

    Returns:
        Tuple of (path, weight)
    """
    if ":" in lora_arg:
        path_str, weight_str = lora_arg.rsplit(":", 1)
        try:
            weight = float(weight_str)
        except ValueError:
            # If weight parsing fails, treat the whole thing as a path
            path_str = lora_arg
            weight = 0.8
    else:
        path_str = lora_arg
        weight = 0.8

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"LoRA file not found: {path}")

    return path, weight


def load_loras(
    pipeline: "StableDiffusionXLPipeline",
    lora_specs: list[str],
) -> None:
    """
    Load multiple LoRAs into the pipeline.

    Args:
        pipeline: SDXL pipeline instance
        lora_specs: List of LoRA specifications ('path:weight' or 'path')
    """
    if not lora_specs:
        return

    adapter_names = []
    adapter_weights = []

    for i, spec in enumerate(lora_specs):
        path, weight = parse_lora_arg(spec)
        adapter_name = f"lora_{i}"

        pipeline.load_lora_weights(
            str(path.parent),
            weight_name=path.name,
            adapter_name=adapter_name,
        )

        adapter_names.append(adapter_name)
        adapter_weights.append(weight)

    # Set all adapters with their weights
    if adapter_names:
        pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)

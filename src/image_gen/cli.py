"""Typer CLI for SDXL image generation."""

import logging
import os
import random
import warnings
from pathlib import Path
from typing import Annotated, Optional

# Suppress warnings before any imports
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress transformers/compel logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("compel").setLevel(logging.ERROR)

import typer

from .ip_adapter import SUPPORTED_IP_ADAPTERS, load_reference_images
from .metadata import GenerationMetadata, save_image_with_metadata
from .pipeline import DEFAULT_NEGATIVE_PROMPT, GenerationConfig, SDXLPipeline
from .schedulers import SUPPORTED_SCHEDULERS

app = typer.Typer(
    name="image-gen",
    help="SDXL image generation CLI with safetensors, LoRA, and embedding support.",
    no_args_is_help=True,
)


@app.command()
def generate(
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            "-m",
            help="Path to safetensors SDXL model checkpoint",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    prompt: Annotated[
        str,
        typer.Option(
            "--prompt",
            "-p",
            help="Positive prompt for image generation",
        ),
    ],
    negative_prompt: Annotated[
        Optional[str],
        typer.Option(
            "--negative-prompt",
            "-n",
            help="Negative prompt (defaults to quality-focused)",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output image path (will be saved as .jpg)",
        ),
    ] = Path("./output.jpg"),
    width: Annotated[
        int,
        typer.Option(
            "--width",
            "-W",
            help="Image width (SDXL native: 1024)",
            min=512,
            max=2048,
        ),
    ] = 1024,
    height: Annotated[
        int,
        typer.Option(
            "--height",
            "-H",
            help="Image height (SDXL native: 1024)",
            min=512,
            max=2048,
        ),
    ] = 1024,
    steps: Annotated[
        int,
        typer.Option(
            "--steps",
            "-s",
            help="Number of sampling steps",
            min=1,
            max=150,
        ),
    ] = 30,
    cfg_scale: Annotated[
        float,
        typer.Option(
            "--cfg-scale",
            "-c",
            help="Classifier-free guidance scale",
            min=1.0,
            max=30.0,
        ),
    ] = 4.0,
    scheduler: Annotated[
        str,
        typer.Option(
            "--scheduler",
            help=f"Sampling scheduler. Supported: {', '.join(SUPPORTED_SCHEDULERS)}",
        ),
    ] = "euler_a",
    seed: Annotated[
        Optional[int],
        typer.Option(
            "--seed",
            help="Random seed for reproducibility",
        ),
    ] = None,
    clip_skip: Annotated[
        int,
        typer.Option(
            "--clip-skip",
            help="CLIP layers to skip from end (1=none, 2+=skip)",
            min=1,
            max=4,
        ),
    ] = 2,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Number of images to generate",
            min=1,
            max=8,
        ),
    ] = 1,
    vae: Annotated[
        Optional[Path],
        typer.Option(
            "--vae",
            help="Path to custom VAE safetensors",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    lora: Annotated[
        Optional[list[str]],
        typer.Option(
            "--lora",
            help="LoRA path:weight (repeatable). Example: /path/lora.safetensors:0.8",
        ),
    ] = None,
    embedding: Annotated[
        Optional[list[str]],
        typer.Option(
            "--embedding",
            help="Textual inversion embedding path (repeatable)",
        ),
    ] = None,
    # IP-Adapter options (reference-image identity/style conditioning)
    ip_adapter_image: Annotated[
        Optional[list[str]],
        typer.Option(
            "--ip-adapter-image",
            help="Reference image for IP-Adapter identity/style (repeatable)",
        ),
    ] = None,
    ip_adapter: Annotated[
        str,
        typer.Option(
            "--ip-adapter",
            help=f"IP-Adapter preset. Supported: {', '.join(SUPPORTED_IP_ADAPTERS)}",
        ),
    ] = "face",
    ip_adapter_scale: Annotated[
        Optional[float],
        typer.Option(
            "--ip-adapter-scale",
            help="IP-Adapter conditioning strength (0-1, default 0.6)",
            min=0.0,
            max=1.0,
        ),
    ] = None,
    # Hi-res fix options
    hires_fix: Annotated[
        bool,
        typer.Option(
            "--hires-fix/--no-hires-fix",
            help="Enable hi-res fix (2-pass upscaling)",
        ),
    ] = False,
    hires_scale: Annotated[
        float,
        typer.Option(
            "--hires-scale",
            help="Hi-res fix upscale factor",
            min=1.0,
            max=4.0,
        ),
    ] = 1.5,
    hires_steps: Annotated[
        int,
        typer.Option(
            "--hires-steps",
            help="Hi-res fix sampling steps",
            min=1,
            max=100,
        ),
    ] = 15,
    hires_denoising: Annotated[
        float,
        typer.Option(
            "--hires-denoising",
            help="Hi-res fix denoising strength",
            min=0.0,
            max=1.0,
        ),
    ] = 0.5,
) -> None:
    """Generate images using SDXL safetensors model."""
    # Validate scheduler
    if scheduler not in SUPPORTED_SCHEDULERS:
        typer.echo(
            f"Error: Unknown scheduler '{scheduler}'. " f"Supported: {', '.join(SUPPORTED_SCHEDULERS)}",
            err=True,
        )
        raise typer.Exit(1)

    # Validate IP-Adapter preset
    if ip_adapter not in SUPPORTED_IP_ADAPTERS:
        typer.echo(
            f"Error: Unknown IP-Adapter '{ip_adapter}'. "
            f"Supported: {', '.join(SUPPORTED_IP_ADAPTERS)}",
            err=True,
        )
        raise typer.Exit(1)

    # Generate random seed if not provided
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        typer.echo(f"Using seed: {seed}")

    # Initialize pipeline
    typer.echo(f"Loading model: {model}")
    pipeline = SDXLPipeline(
        model_path=model,
        vae_path=vae,
        scheduler_name=scheduler,
    )

    # Load LoRAs
    if lora:
        typer.echo(f"Loading {len(lora)} LoRA(s)...")
        pipeline.load_loras(lora)

    # Load embeddings
    if embedding:
        typer.echo(f"Loading {len(embedding)} embedding(s)...")
        tokens = pipeline.load_embeddings(embedding)
        typer.echo(f"Available tokens: {', '.join(tokens)}")

    # Load IP-Adapter (identity/style conditioning from reference images)
    ip_adapter_ref_images = []
    effective_ip_scale: Optional[float] = None
    if ip_adapter_image:
        typer.echo(f"Loading IP-Adapter ({ip_adapter})...")
        effective_ip_scale = pipeline.load_ip_adapter(ip_adapter, ip_adapter_scale)
        ip_adapter_ref_images = load_reference_images(ip_adapter_image)
        typer.echo(
            f"  {len(ip_adapter_ref_images)} reference image(s), scale {effective_ip_scale}"
        )

    # Create generation config
    config = GenerationConfig(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else DEFAULT_NEGATIVE_PROMPT,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        clip_skip=clip_skip,
        batch_size=batch_size,
        hires_fix=hires_fix,
        hires_scale=hires_scale,
        hires_steps=hires_steps,
        hires_denoising=hires_denoising,
        ip_adapter_images=ip_adapter_ref_images,
    )

    # Generate images
    typer.echo(f"Generating {batch_size} image(s)...")
    typer.echo(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    typer.echo(f"  Size: {width}x{height}, Steps: {steps}, CFG: {cfg_scale}")
    if hires_fix:
        typer.echo(f"  Hi-res fix: {hires_scale}x, {hires_steps} steps, " f"denoising {hires_denoising}")

    images = pipeline.generate(config)

    # Create metadata for EXIF
    actual_negative = negative_prompt if negative_prompt else DEFAULT_NEGATIVE_PROMPT
    metadata = GenerationMetadata(
        prompt=prompt,
        negative_prompt=actual_negative,
        model=model.name,
        vae=vae.name if vae else None,
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        scheduler=scheduler,
        clip_skip=clip_skip,
        lora=lora,
        embedding=embedding,
        hires_fix=hires_fix,
        hires_scale=hires_scale if hires_fix else None,
        hires_steps=hires_steps if hires_fix else None,
        hires_denoising=hires_denoising if hires_fix else None,
        ip_adapter=ip_adapter if ip_adapter_image else None,
        ip_adapter_images=list(ip_adapter_image) if ip_adapter_image else None,
        ip_adapter_scale=effective_ip_scale,
    )

    # Save images as JPG with EXIF metadata
    output_path = output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if batch_size == 1:
        saved = save_image_with_metadata(images[0], output_path, metadata)
        typer.echo(f"Saved: {saved}")
    else:
        # Save multiple images with numbered suffixes
        stem = output_path.stem
        parent = output_path.parent
        suffix = output_path.suffix or ".jpg"

        for i, img in enumerate(images):
            path = parent / f"{stem}_{i:02d}{suffix}"
            saved = save_image_with_metadata(img, path, metadata)
            typer.echo(f"Saved: {saved}")

    typer.echo("Done!")


@app.command(name="generate-var")
def generate_var(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-C",
            help="YAML spec file (template_prompt, template_output, variables, ...)",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print resolved prompts without loading the model or generating",
        ),
    ] = False,
    dry_run_count: Annotated[
        int,
        typer.Option(
            "--dry-run-count",
            help="How many sample prompts to print in --dry-run",
            min=1,
            max=200,
        ),
    ] = 10,
    poll: Annotated[
        float,
        typer.Option(
            "--poll",
            help="Seconds between config-file reload checks",
            min=1.0,
            max=60.0,
        ),
    ] = 5.0,
    var_seed: Annotated[
        Optional[int],
        typer.Option(
            "--var-seed",
            help="Seed for the variable RNG (reproducible prompt draws)",
        ),
    ] = None,
    # Overrides for the YAML `defaults:` block (only applied when set).
    model: Annotated[
        Optional[Path],
        typer.Option("--model", "-m", help="Override defaults.model", exists=True, dir_okay=False),
    ] = None,
    negative_prompt: Annotated[
        Optional[str],
        typer.Option("--negative-prompt", "-n", help="Override negative_prompt"),
    ] = None,
    steps: Annotated[Optional[int], typer.Option("--steps", "-s", min=1, max=150)] = None,
    cfg_scale: Annotated[Optional[float], typer.Option("--cfg-scale", "-c", min=1.0, max=30.0)] = None,
    width: Annotated[Optional[int], typer.Option("--width", "-W", min=512, max=2048)] = None,
    height: Annotated[Optional[int], typer.Option("--height", "-H", min=512, max=2048)] = None,
    scheduler: Annotated[Optional[str], typer.Option("--scheduler")] = None,
    clip_skip: Annotated[Optional[int], typer.Option("--clip-skip", min=1, max=4)] = None,
    lora: Annotated[Optional[list[str]], typer.Option("--lora", help="Override defaults.lora (repeatable)")] = None,
    vae: Annotated[
        Optional[Path],
        typer.Option("--vae", help="Override defaults.vae", exists=True, dir_okay=False),
    ] = None,
) -> None:
    """Run continuous, variable-driven generation from a YAML spec.

    The spec drives a hot-reloadable loop: edit the file while it runs to change
    prompts, variables, loop count, or set status to pause/stop.
    """
    from .runner import dry_run as _dry_run
    from .runner import run as _run
    from .variables import load_spec

    if scheduler is not None and scheduler not in SUPPORTED_SCHEDULERS:
        typer.echo(
            f"Error: Unknown scheduler '{scheduler}'. Supported: {', '.join(SUPPORTED_SCHEDULERS)}",
            err=True,
        )
        raise typer.Exit(1)

    overrides = {
        "model": str(model) if model else None,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "scheduler": scheduler,
        "clip_skip": clip_skip,
        "lora": lora if lora else None,
        "vae": str(vae) if vae else None,
    }

    try:
        if dry_run:
            spec = load_spec(config)
            _dry_run(spec, random.Random(var_seed), dry_run_count, typer.echo)
            return
        made = _run(
            config_path=config,
            overrides=overrides,
            poll=poll,
            var_rng=random.Random(var_seed) if var_seed is not None else None,
            echo=typer.echo,
        )
        typer.echo(f"Done. Generated {made} image(s).")
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        typer.echo("\nInterrupted.")


@app.command()
def info() -> None:
    """Show available schedulers and default settings."""
    typer.echo("Image Generation CLI - SDXL Pipeline\n")

    typer.echo("Supported Schedulers:")
    for s in SUPPORTED_SCHEDULERS:
        default = " (default)" if s == "euler_a" else ""
        typer.echo(f"  - {s}{default}")

    typer.echo("\nDefault Negative Prompt:")
    typer.echo(f"  {DEFAULT_NEGATIVE_PROMPT}")

    typer.echo("\nDefault Settings:")
    typer.echo("  Width: 1024, Height: 1024")
    typer.echo("  Steps: 30, CFG Scale: 4.0")
    typer.echo("  CLIP Skip: 2, Batch Size: 1")

    typer.echo("\nIP-Adapter Presets (reference-image conditioning):")
    for name in SUPPORTED_IP_ADAPTERS:
        default = " (default)" if name == "face" else ""
        typer.echo(f"  - {name}{default}")
    typer.echo("  Usage: --ip-adapter-image face.jpg [--ip-adapter face] [--ip-adapter-scale 0.6]")

    typer.echo("\nPrompt Weighting (compel syntax):")
    typer.echo("  (word:1.2)  - Increase weight to 1.2x")
    typer.echo("  (word:0.8)  - Decrease weight to 0.8x")
    typer.echo("  word++      - Increase weight (each + is 1.1x)")
    typer.echo("  word--      - Decrease weight (each - is 0.9x)")
    typer.echo('  "prompt A" AND "prompt B"  - Blend prompts')
    typer.echo("\n  Long prompts (>77 tokens) are supported automatically.")


if __name__ == "__main__":
    app()

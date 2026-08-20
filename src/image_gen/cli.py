"""Typer CLI for SDXL image generation."""

import logging
import os
import random
import uuid
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

from . import llm_variation
from .ip_adapter import SUPPORTED_IP_ADAPTERS, load_reference_images
from .logging_jsonl import append_generation_log
from .metadata import GenerationMetadata, load_metadata, save_image_with_metadata
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
    log_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--log-dir",
            help="Directory for the JSONL generation log (one line per image, "
            "daily-rotated). Omit to disable logging.",
        ),
    ] = None,
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
        log_path = append_generation_log(log_dir, metadata, saved, command="generate")
        if log_path is not None:
            typer.echo(f"Logged: {log_path}")
    else:
        # Save multiple images with numbered suffixes
        stem = output_path.stem
        parent = output_path.parent
        suffix = output_path.suffix or ".jpg"

        for i, img in enumerate(images):
            path = parent / f"{stem}_{i:02d}{suffix}"
            saved = save_image_with_metadata(img, path, metadata)
            typer.echo(f"Saved: {saved}")
            log_path = append_generation_log(log_dir, metadata, saved, command="generate")
            if log_path is not None:
                typer.echo(f"Logged: {log_path}")

    typer.echo("Done!")


@app.command(name="generate-similar")
def generate_similar(
    sources: Annotated[
        Optional[list[Path]],
        typer.Argument(
            help="Source image(s) previously generated by image-gen (reads their EXIF "
            "metadata). Zero images is a no-op. Multiple images are processed in order, "
            "each producing --count images with the same options. Missing files are "
            "skipped with a warning instead of aborting the whole run.",
        ),
    ] = None,
    count: Annotated[
        int,
        typer.Option(
            "--count",
            help="Number of new images to generate (each with a fresh random seed)",
            min=1,
            max=50,
        ),
    ] = 1,
    model: Annotated[
        Optional[Path],
        typer.Option(
            "--model",
            "-m",
            help="Override the source's model (defaults to looking up the source's "
            "model filename inside --model-dir)",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    model_dir: Annotated[
        Path,
        typer.Option(
            "--model-dir",
            help="Directory to search for the source's model/VAE filename when "
            "--model/--vae are not given (non-recursive lookup by exact filename)",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = Path.home() / ".cache" / "models",
    prompt: Annotated[
        Optional[str],
        typer.Option("--prompt", "-p", help="Override the source's prompt"),
    ] = None,
    negative_prompt: Annotated[
        Optional[str],
        typer.Option("--negative-prompt", "-n", help="Override the source's negative prompt"),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output path. Defaults to '<source>_similar[.ext]' next to the source "
            "(numbered '_00', '_01', ... when count > 1)",
        ),
    ] = None,
    width: Annotated[
        Optional[int],
        typer.Option("--width", "-W", help="Override the source's width", min=512, max=2048),
    ] = None,
    height: Annotated[
        Optional[int],
        typer.Option("--height", "-H", help="Override the source's height", min=512, max=2048),
    ] = None,
    steps: Annotated[
        Optional[int],
        typer.Option("--steps", "-s", help="Override the source's sampling steps", min=1, max=150),
    ] = None,
    cfg_scale: Annotated[
        Optional[float],
        typer.Option("--cfg-scale", "-c", help="Override the source's CFG scale", min=1.0, max=30.0),
    ] = None,
    scheduler: Annotated[
        Optional[str],
        typer.Option(
            "--scheduler",
            help=f"Override the source's scheduler. Supported: {', '.join(SUPPORTED_SCHEDULERS)}",
        ),
    ] = None,
    clip_skip: Annotated[
        Optional[int],
        typer.Option("--clip-skip", help="Override the source's CLIP skip", min=1, max=4),
    ] = None,
    vae: Annotated[
        Optional[Path],
        typer.Option(
            "--vae",
            help="Override the source's VAE (defaults to looking it up in --model-dir)",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    lora: Annotated[
        Optional[list[str]],
        typer.Option(
            "--lora",
            help="Override the source's LoRA list (repeatable, replaces it entirely). "
            "path:weight, e.g. /path/lora.safetensors:0.8",
        ),
    ] = None,
    embedding: Annotated[
        Optional[list[str]],
        typer.Option(
            "--embedding",
            help="Override the source's embedding list (repeatable, replaces it entirely)",
        ),
    ] = None,
    ip_adapter_image: Annotated[
        Optional[list[str]],
        typer.Option(
            "--ip-adapter-image",
            help="Override the source's IP-Adapter reference image(s) (repeatable, "
            "replaces the source's list entirely)",
        ),
    ] = None,
    ip_adapter: Annotated[
        Optional[str],
        typer.Option(
            "--ip-adapter",
            help=f"Override the source's IP-Adapter preset. Supported: {', '.join(SUPPORTED_IP_ADAPTERS)}",
        ),
    ] = None,
    ip_adapter_scale: Annotated[
        Optional[float],
        typer.Option("--ip-adapter-scale", help="Override the source's IP-Adapter scale", min=0.0, max=1.0),
    ] = None,
    hires_fix: Annotated[
        Optional[bool],
        typer.Option(
            "--hires-fix/--no-hires-fix",
            help="Override the source's hi-res fix flag (unset keeps the source's value)",
        ),
    ] = None,
    hires_scale: Annotated[
        Optional[float],
        typer.Option("--hires-scale", help="Override the source's hi-res fix scale", min=1.0, max=4.0),
    ] = None,
    hires_steps: Annotated[
        Optional[int],
        typer.Option("--hires-steps", help="Override the source's hi-res fix steps", min=1, max=100),
    ] = None,
    hires_denoising: Annotated[
        Optional[float],
        typer.Option(
            "--hires-denoising", help="Override the source's hi-res fix denoising", min=0.0, max=1.0
        ),
    ] = None,
    log_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--log-dir",
            help="Directory for the JSONL generation log (one line per image). Omit to disable.",
        ),
    ] = None,
    keep_seed: Annotated[
        bool,
        typer.Option(
            "--keep-seed",
            help="Reuse the source's exact seed for every generated image, instead of "
            "a fresh random seed each time. Independent of --vary.",
        ),
    ] = False,
    vary: Annotated[
        Optional[str],
        typer.Option(
            "--vary",
            help="Enable LLM-driven prompt variation: explicit instruction on the kind "
            "of variation wanted (e.g. 'more casual outfit, same pose'). Each image gets "
            "its own LLM-generated prompt/negative-prompt variation. Requires an LLM "
            "endpoint (see --llm-* options and IMAGEGEN_MODEL_* env vars).",
        ),
    ] = None,
    vocab: Annotated[
        Optional[Path],
        typer.Option(
            "--vocab",
            help="Vocabulary file (generate-var spec format) injected as inspiration for "
            "--vary. Requires --vary.",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    llm_base_url: Annotated[
        Optional[str],
        typer.Option("--llm-base-url", help=f"Override ${llm_variation.ENV_BASE_URL}"),
    ] = None,
    llm_api_key: Annotated[
        Optional[str],
        typer.Option("--llm-api-key", help=f"Override ${llm_variation.ENV_API_KEY}"),
    ] = None,
    llm_model: Annotated[
        Optional[str],
        typer.Option("--llm-model", help=f"Override ${llm_variation.ENV_MODEL_NAME}"),
    ] = None,
    llm_temperature: Annotated[
        float,
        typer.Option("--llm-temperature", help="LLM sampling temperature (variation diversity)", min=0.0, max=2.0),
    ] = 1.0,
) -> None:
    """Re-generate new images from one or more source images' parameters.

    Reads the generation metadata embedded in each SOURCES's EXIF (written by
    `generate` or `generate-var`) and reuses every parameter as-is. By default
    the seed is re-randomized per image; pass --keep-seed to reuse each
    source's own exact seed instead. Any parameter can be overridden with the
    same flags as `generate`, applied identically to every source. With
    multiple sources, --count images are generated per source, in order (3
    sources x --count 4 = 12 images total). No sources given is a no-op.

    Pass --vary "instruction" to have an LLM produce a fresh prompt (and/or
    negative prompt) variation for each image instead of reusing the source's
    prompt verbatim; see --vocab and the --llm-* options.
    """
    if not sources:
        typer.echo("No source images given, nothing to do.")
        return

    if vocab is not None and vary is None:
        typer.echo("Error: --vocab requires --vary.", err=True)
        raise typer.Exit(1)

    if output is not None and len(sources) > 1:
        typer.echo(
            "Error: --output cannot be used with multiple source images (their outputs "
            "would collide). Omit --output to use the default '<source>_similar' naming.",
            err=True,
        )
        raise typer.Exit(1)

    # Validate overridden scheduler / IP-Adapter preset up front (shared by every source).
    if scheduler is not None and scheduler not in SUPPORTED_SCHEDULERS:
        typer.echo(
            f"Error: Unknown scheduler '{scheduler}'. Supported: {', '.join(SUPPORTED_SCHEDULERS)}",
            err=True,
        )
        raise typer.Exit(1)
    if ip_adapter is not None and ip_adapter not in SUPPORTED_IP_ADAPTERS:
        typer.echo(
            f"Error: Unknown IP-Adapter '{ip_adapter}'. Supported: {', '.join(SUPPORTED_IP_ADAPTERS)}",
            err=True,
        )
        raise typer.Exit(1)

    llm_config = None
    vocabulary_text = None
    if vary is not None:
        try:
            llm_config = llm_variation.resolve_llm_config(llm_base_url, llm_api_key, llm_model)
        except ValueError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        if vocab is not None:
            vocabulary_text = llm_variation.format_vocabulary(vocab)

    pipeline: Optional[SDXLPipeline] = None
    pipeline_signature = None
    effective_ip_scale: Optional[float] = None
    ip_adapter_ref_images: list = []
    total_saved = 0

    for source_index, source in enumerate(sources):
        if len(sources) > 1:
            typer.echo(f"\n=== Source {source_index + 1}/{len(sources)}: {source.name} ===")

        if not source.is_file():
            typer.echo(f"Warning: '{source}' does not exist, skipping.", err=True)
            continue

        try:
            meta = load_metadata(source)
        except ValueError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc

        # Resolve model: explicit --model wins, else look up the source's filename in --model-dir.
        if model is not None:
            model_path = model
        else:
            source_model_name = meta.get("model")
            candidate = model_dir / source_model_name if source_model_name else None
            if not source_model_name or not candidate.exists():
                typer.echo(
                    f"Error: Model '{source_model_name}' not found in {model_dir}. "
                    "Pass --model to specify it explicitly.",
                    err=True,
                )
                raise typer.Exit(1)
            model_path = candidate

        # Resolve VAE the same way: explicit --vae wins, else look up in --model-dir if the
        # source used one. No VAE is used if neither the source nor an override specifies one.
        if vae is not None:
            vae_path: Optional[Path] = vae
        else:
            source_vae_name = meta.get("vae")
            if source_vae_name:
                candidate = model_dir / source_vae_name
                if not candidate.exists():
                    typer.echo(
                        f"Error: VAE '{source_vae_name}' not found in {model_dir}. "
                        "Pass --vae to specify it explicitly.",
                        err=True,
                    )
                    raise typer.Exit(1)
                vae_path = candidate
            else:
                vae_path = None

        # Resolve every other parameter: override if given, else this source's value.
        resolved_prompt = prompt if prompt is not None else meta["prompt"]
        resolved_negative = negative_prompt if negative_prompt is not None else meta.get(
            "negative_prompt", DEFAULT_NEGATIVE_PROMPT
        )
        resolved_width = width if width is not None else meta.get("width", 1024)
        resolved_height = height if height is not None else meta.get("height", 1024)
        resolved_steps = steps if steps is not None else meta.get("steps", 30)
        resolved_cfg = cfg_scale if cfg_scale is not None else meta.get("cfg_scale", 4.0)
        resolved_scheduler = scheduler if scheduler is not None else meta.get("scheduler", "euler_a")
        resolved_clip_skip = clip_skip if clip_skip is not None else meta.get("clip_skip", 2)
        resolved_lora = lora if lora is not None else meta.get("lora")
        resolved_embedding = embedding if embedding is not None else meta.get("embedding")

        resolved_hires_fix = hires_fix if hires_fix is not None else meta.get("hires_fix", False)
        resolved_hires_scale = hires_scale if hires_scale is not None else meta.get("hires_scale", 1.5)
        resolved_hires_steps = hires_steps if hires_steps is not None else meta.get("hires_steps", 15)
        resolved_hires_denoising = (
            hires_denoising if hires_denoising is not None else meta.get("hires_denoising", 0.5)
        )

        # IP-Adapter: overriding the reference image(s) resets preset/scale to `generate`
        # defaults unless also explicitly overridden; otherwise reuse the source as-is.
        if ip_adapter_image is not None:
            resolved_ip_adapter_preset = ip_adapter if ip_adapter is not None else "face"
            resolved_ip_adapter_images_paths = list(ip_adapter_image)
            resolved_ip_adapter_scale_arg = ip_adapter_scale
        else:
            resolved_ip_adapter_preset = ip_adapter if ip_adapter is not None else meta.get("ip_adapter")
            resolved_ip_adapter_images_paths = meta.get("ip_adapter_images") or []
            resolved_ip_adapter_scale_arg = (
                ip_adapter_scale if ip_adapter_scale is not None else meta.get("ip_adapter_scale")
            )

        # (Re)initialize the pipeline only when the resolved config actually changed from the
        # previous source, so sequential sources sharing the same model/vae/scheduler/LoRAs
        # don't pay a reload cost for nothing.
        signature = (
            model_path,
            vae_path,
            resolved_scheduler,
            tuple(resolved_lora) if resolved_lora else None,
            tuple(resolved_embedding) if resolved_embedding else None,
            resolved_ip_adapter_preset,
            tuple(resolved_ip_adapter_images_paths),
            resolved_ip_adapter_scale_arg,
        )
        if pipeline is None or signature != pipeline_signature:
            typer.echo(f"Loading model: {model_path}")
            pipeline = SDXLPipeline(
                model_path=model_path,
                vae_path=vae_path,
                scheduler_name=resolved_scheduler,
            )

            if resolved_lora:
                typer.echo(f"Loading {len(resolved_lora)} LoRA(s)...")
                pipeline.load_loras(resolved_lora)

            if resolved_embedding:
                typer.echo(f"Loading {len(resolved_embedding)} embedding(s)...")
                pipeline.load_embeddings(resolved_embedding)

            effective_ip_scale = None
            ip_adapter_ref_images = []
            if resolved_ip_adapter_images_paths:
                preset_name = resolved_ip_adapter_preset or "face"
                typer.echo(f"Loading IP-Adapter ({preset_name})...")
                effective_ip_scale = pipeline.load_ip_adapter(preset_name, resolved_ip_adapter_scale_arg)
                ip_adapter_ref_images = load_reference_images(resolved_ip_adapter_images_paths)
                typer.echo(f"  {len(ip_adapter_ref_images)} reference image(s), scale {effective_ip_scale}")

            pipeline_signature = signature
        else:
            typer.echo("Reusing already-loaded model/VAE/LoRA/embedding/IP-Adapter.")

        # Resolve output path template.
        if output is not None:
            output_path = output.resolve()
            stem = output_path.stem
            parent = output_path.parent
            suffix = output_path.suffix or ".jpg"
        else:
            default_suffix = ".png" if source.suffix.lower() == ".png" else ".jpg"
            parent = source.resolve().parent
            # A short random tag per source/run, so repeated generate-similar
            # invocations on the same source don't overwrite each other's files.
            stem = f"{source.stem}_s{uuid.uuid4().hex[:8]}"
            suffix = default_suffix
        parent.mkdir(parents=True, exist_ok=True)

        typer.echo(f"Generating {count} image(s) similar to {source.name}...")
        typer.echo(f"  Prompt: {resolved_prompt[:80]}{'...' if len(resolved_prompt) > 80 else ''}")
        typer.echo(f"  Size: {resolved_width}x{resolved_height}, Steps: {resolved_steps}, CFG: {resolved_cfg}")
        if resolved_hires_fix:
            typer.echo(
                f"  Hi-res fix: {resolved_hires_scale}x, {resolved_hires_steps} steps, "
                f"denoising {resolved_hires_denoising}"
            )

        previous_variations: list[str] = []

        for i in range(count):
            iter_prompt, iter_negative = resolved_prompt, resolved_negative
            if vary is not None:
                try:
                    variation = llm_variation.generate_variation(
                        llm_config,
                        resolved_prompt,
                        resolved_negative,
                        vary,
                        vocabulary_text,
                        previous_variations,
                        temperature=llm_temperature,
                    )
                except (ValueError, RuntimeError) as exc:
                    typer.echo(f"Error: LLM variation failed: {exc}", err=True)
                    raise typer.Exit(1) from exc
                iter_prompt = variation.prompt
                iter_negative = (
                    variation.negative_prompt if variation.negative_prompt is not None else resolved_negative
                )
                previous_variations.append(iter_prompt)
                typer.echo(f"  [{i}] prompt: {iter_prompt[:100]}{'...' if len(iter_prompt) > 100 else ''}")

            seed = meta["seed"] if keep_seed else random.randint(0, 2**32 - 1)

            config = GenerationConfig(
                prompt=iter_prompt,
                negative_prompt=iter_negative,
                width=resolved_width,
                height=resolved_height,
                steps=resolved_steps,
                cfg_scale=resolved_cfg,
                seed=seed,
                clip_skip=resolved_clip_skip,
                batch_size=1,
                hires_fix=resolved_hires_fix,
                hires_scale=resolved_hires_scale,
                hires_steps=resolved_hires_steps,
                hires_denoising=resolved_hires_denoising,
                ip_adapter_images=ip_adapter_ref_images,
            )

            images = pipeline.generate(config)

            out_metadata = GenerationMetadata(
                prompt=iter_prompt,
                negative_prompt=iter_negative,
                model=model_path.name,
                vae=vae_path.name if vae_path else None,
                seed=seed,
                width=resolved_width,
                height=resolved_height,
                steps=resolved_steps,
                cfg_scale=resolved_cfg,
                scheduler=resolved_scheduler,
                clip_skip=resolved_clip_skip,
                lora=resolved_lora,
                embedding=resolved_embedding,
                hires_fix=resolved_hires_fix,
                hires_scale=resolved_hires_scale if resolved_hires_fix else None,
                hires_steps=resolved_hires_steps if resolved_hires_fix else None,
                hires_denoising=resolved_hires_denoising if resolved_hires_fix else None,
                ip_adapter=resolved_ip_adapter_preset if resolved_ip_adapter_images_paths else None,
                ip_adapter_images=resolved_ip_adapter_images_paths or None,
                ip_adapter_scale=effective_ip_scale,
                source_image=source.name,
                llm_request=vary,
            )

            if count == 1:
                path = parent / f"{stem}{suffix}"
            else:
                path = parent / f"{stem}_{i:02d}{suffix}"

            saved = save_image_with_metadata(images[0], path, out_metadata)
            typer.echo(f"Saved: {saved} (seed {seed})")
            log_path = append_generation_log(log_dir, out_metadata, saved, command="generate-similar")
            if log_path is not None:
                typer.echo(f"Logged: {log_path}")
            total_saved += 1

    typer.echo(f"Done! Generated {total_saved} image(s) across {len(sources)} source(s).")


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
    set_pause: Annotated[
        bool,
        typer.Option("--pause", help="Set config status to 'pause' and exit (no generation)"),
    ] = False,
    set_live: Annotated[
        bool,
        typer.Option("--live", help="Set config status to 'live' and exit (no generation)"),
    ] = False,
    set_stop: Annotated[
        bool,
        typer.Option("--stop", help="Set config status to 'stop' and exit (no generation)"),
    ] = False,
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
    log_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--log-dir",
            help="Override log_dir: directory for the JSONL generation log "
            "(one line per image, daily-rotated). Defaults to the spec's log_dir, "
            "else the output directory.",
        ),
    ] = None,
) -> None:
    """Run continuous, variable-driven generation from a YAML spec.

    The spec drives a hot-reloadable loop: edit the file while it runs to change
    prompts, variables, loop count, or set status to pause/stop.
    """
    from .runner import dry_run as _dry_run
    from .runner import run as _run
    from .runner import set_status
    from .variables import load_spec

    # Status flags are exclusive and short-circuit: edit the file and exit.
    status_flags = [("live", set_live), ("pause", set_pause), ("stop", set_stop)]
    chosen = [name for name, on in status_flags if on]
    if chosen:
        if len(chosen) > 1:
            typer.echo(
                f"Error: --live/--pause/--stop are mutually exclusive (got {', '.join(chosen)})",
                err=True,
            )
            raise typer.Exit(1)
        set_status(config, chosen[0])
        typer.echo(f"{config}: status set to '{chosen[0]}'.")
        return

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
        "log_dir": str(log_dir) if log_dir else None,
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

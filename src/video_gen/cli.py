"""Typer CLI for local image-to-video generation."""

import logging
import os
import random
import warnings
from pathlib import Path
from typing import Annotated, Optional

# Let unsupported MPS ops fall back to CPU instead of erroring.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

import typer

from .backends import (
    BACKENDS,
    DEFAULT_BACKEND,
    DEFAULT_NEGATIVE_PROMPT,
    PRECISIONS,
    frames_to_duration,
    get_backend,
    resolve_frames,
)

app = typer.Typer(
    name="video-gen",
    help="Local image-to-video CLI (Wan 2.2 / LTX-Video) with variable-driven batches.",
    no_args_is_help=True,
)


@app.command()
def generate(
    image: Annotated[
        Path,
        typer.Option("--image", "-i", help="Source image to animate", exists=True, dir_okay=False),
    ],
    prompt: Annotated[str, typer.Option("--prompt", "-p", help="Motion/scene prompt")],
    backend: Annotated[
        str,
        typer.Option("--backend", "-b", help=f"Backend: {', '.join(BACKENDS)}"),
    ] = DEFAULT_BACKEND,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output .mp4 path")
    ] = Path("./out/video.mp4"),
    negative_prompt: Annotated[
        Optional[str], typer.Option("--negative-prompt", "-n", help="Negative prompt")
    ] = None,
    duration: Annotated[
        float, typer.Option("--duration", "-d", help="Clip length in seconds", min=0.5, max=10.0)
    ] = 3.0,
    num_frames: Annotated[
        Optional[int],
        typer.Option("--num-frames", help="Explicit frame count (overrides --duration)", min=5),
    ] = None,
    fps: Annotated[
        Optional[int], typer.Option("--fps", help="Frames per second (default: backend native)", min=6, max=30)
    ] = None,
    width: Annotated[Optional[int], typer.Option("--width", "-W", min=128, max=1280)] = None,
    height: Annotated[Optional[int], typer.Option("--height", "-H", min=128, max=1280)] = None,
    steps: Annotated[Optional[int], typer.Option("--steps", "-s", min=1, max=100)] = None,
    guidance: Annotated[Optional[float], typer.Option("--guidance", "-g", min=1.0, max=15.0)] = None,
    seed: Annotated[Optional[int], typer.Option("--seed", help="Random seed")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Override backend HF repo/path")] = None,
    offload: Annotated[
        bool, typer.Option("--offload/--no-offload", help="Sequential CPU offload (smaller footprint, slower)")
    ] = False,
    precision: Annotated[
        Optional[str],
        typer.Option("--precision", help="Override dtype: fp16 | bf16 | fp32 (default: bf16 on GPU)"),
    ] = None,
) -> None:
    """Animate a single image into an MP4 clip."""
    import time

    from .encode import encode_frames
    from .metadata import VideoMetadata, write_sidecar
    from .pipeline import VideoGenConfig, VideoPipeline, load_source_image

    if backend not in BACKENDS:
        typer.echo(f"Error: unknown backend '{backend}'. Supported: {', '.join(BACKENDS)}", err=True)
        raise typer.Exit(1)
    if precision is not None and precision not in PRECISIONS:
        typer.echo(f"Error: unknown precision '{precision}'. Use fp16 | bf16 | fp32.", err=True)
        raise typer.Exit(1)

    spec = get_backend(backend)
    eff_fps = fps if fps is not None else spec.native_fps
    frames = num_frames if num_frames is not None else resolve_frames(spec, duration, eff_fps)
    eff_width = width if width is not None else spec.default_width
    eff_height = height if height is not None else spec.default_height
    eff_steps = steps if steps is not None else spec.default_steps
    eff_guidance = guidance if guidance is not None else spec.default_guidance
    eff_negative = negative_prompt if negative_prompt else DEFAULT_NEGATIVE_PROMPT
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    typer.echo(f"Backend: {backend} ({spec.description})")
    typer.echo(f"Loading pipeline{' (offload)' if offload else ''}...")
    pipe = VideoPipeline(backend=backend, repo_override=model, offload=offload, precision=precision)
    pipe.load()

    dur = frames_to_duration(frames, eff_fps)
    typer.echo(
        f"Generating {dur}s ({frames} frames @ {eff_fps} fps), "
        f"{eff_width}x{eff_height}, {eff_steps} steps, seed {seed}"
    )
    typer.echo(f"  device={pipe.device} dtype={pipe.dtype}")
    typer.echo(f"  prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    config = VideoGenConfig(
        prompt=prompt,
        image=load_source_image(image),
        negative_prompt=eff_negative,
        width=eff_width,
        height=eff_height,
        num_frames=frames,
        steps=eff_steps,
        guidance=eff_guidance,
        seed=seed,
    )

    t0 = time.time()
    result = pipe.generate(config)
    gen_s = time.time() - t0

    metadata = VideoMetadata(
        prompt=prompt,
        negative_prompt=eff_negative,
        backend=backend,
        model=pipe.repo,
        source_image=str(image),
        seed=seed,
        width=eff_width,
        height=eff_height,
        num_frames=frames,
        fps=eff_fps,
        duration_s=dur,
        steps=eff_steps,
        guidance=eff_guidance,
    )
    saved = encode_frames(result, output, fps=eff_fps, comment=metadata.to_json())
    write_sidecar(saved, metadata)
    typer.echo(f"Saved: {saved}  ({gen_s:.0f}s, {gen_s / dur:.0f}s per output-second)")


@app.command(name="generate-var")
def generate_var(
    config: Annotated[
        Path,
        typer.Option("--config", "-C", help="Video YAML spec", exists=True, dir_okay=False),
    ],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print resolved prompts/inputs only")] = False,
    dry_run_count: Annotated[int, typer.Option("--dry-run-count", min=1, max=200)] = 5,
    poll: Annotated[float, typer.Option("--poll", min=1.0, max=60.0)] = 5.0,
    var_seed: Annotated[Optional[int], typer.Option("--var-seed", help="Seed for variable/input RNG")] = None,
    backend: Annotated[Optional[str], typer.Option("--backend", "-b", help="Override defaults.backend")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Override backend repo/path")] = None,
    duration: Annotated[Optional[float], typer.Option("--duration", "-d", min=0.5, max=10.0)] = None,
    num_frames: Annotated[Optional[int], typer.Option("--num-frames", min=5)] = None,
    fps: Annotated[Optional[int], typer.Option("--fps", min=6, max=30)] = None,
    width: Annotated[Optional[int], typer.Option("--width", "-W", min=128, max=1280)] = None,
    height: Annotated[Optional[int], typer.Option("--height", "-H", min=128, max=1280)] = None,
    steps: Annotated[Optional[int], typer.Option("--steps", "-s", min=1, max=100)] = None,
    guidance: Annotated[Optional[float], typer.Option("--guidance", "-g", min=1.0, max=15.0)] = None,
    offload: Annotated[Optional[bool], typer.Option("--offload/--no-offload")] = None,
    precision: Annotated[Optional[str], typer.Option("--precision", help="fp16 | bf16 | fp32")] = None,
) -> None:
    """Run continuous, variable-driven video generation from a YAML spec.

    Edit the spec while it runs to change prompts, variables, backend, loop, or
    set status to pause/stop. Same control model as image-gen generate-var.
    """
    from .runner import dry_run as _dry_run
    from .runner import run as _run
    from .variables import load_video_spec

    if backend is not None and backend not in BACKENDS:
        typer.echo(f"Error: unknown backend '{backend}'. Supported: {', '.join(BACKENDS)}", err=True)
        raise typer.Exit(1)

    if precision is not None and precision not in PRECISIONS:
        typer.echo(f"Error: unknown precision '{precision}'. Use fp16 | bf16 | fp32.", err=True)
        raise typer.Exit(1)

    overrides = {
        "backend": backend,
        "model": model,
        "duration": duration,
        "num_frames": num_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": guidance,
        "offload": offload,
        "precision": precision,
    }

    try:
        if dry_run:
            spec = load_video_spec(config)
            _dry_run(spec, random.Random(var_seed), dry_run_count, typer.echo)
            return
        made = _run(
            config_path=config,
            overrides=overrides,
            poll=poll,
            var_rng=random.Random(var_seed) if var_seed is not None else None,
            echo=typer.echo,
        )
        typer.echo(f"Done. Generated {made} clip(s).")
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        typer.echo("\nInterrupted.")


@app.command()
def info() -> None:
    """Show available backends and their Mac-friendly defaults."""
    typer.echo("Local Image-to-Video CLI\n")
    typer.echo("Backends:")
    for name, spec in BACKENDS.items():
        default = " (default)" if name == DEFAULT_BACKEND else ""
        native_max = frames_to_duration(spec.max_frames, spec.native_fps)
        typer.echo(f"  - {name}{default}: {spec.description}")
        typer.echo(f"      repo: {spec.repo}")
        typer.echo(
            f"      default {spec.default_width}x{spec.default_height}, "
            f"{spec.default_steps} steps, guidance {spec.default_guidance}, "
            f"{spec.native_fps} fps, frames = {spec.frame_multiple}k+1, "
            f"native max ~{native_max}s"
        )
    typer.echo("\nMac notes:")
    typer.echo("  - bfloat16 only (Metal has no FP8). Clips are minutes, not seconds.")
    typer.echo("  - Longer than the native max = chain segments (concat after).")
    typer.echo("  - Use --offload if you hit memory pressure with Wan.")
    typer.echo(f"\nDefault negative prompt:\n  {DEFAULT_NEGATIVE_PROMPT}")


if __name__ == "__main__":
    app()

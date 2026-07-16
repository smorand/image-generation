"""Continuous generate-var runner.

Drives a hot-reloadable YAML spec: on each iteration it resolves the prompt
template, generates one image, saves it under a zero-padded counter, and logs a
line to ``manifest.jsonl``. The loop is controlled by ``status`` (live / pause /
stop) and ``loop`` (0 = infinite) read fresh from the file every poll interval.
"""

from __future__ import annotations

import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .ip_adapter import load_reference_images
from .metadata import GenerationMetadata, save_image_with_metadata
from .pipeline import DEFAULT_NEGATIVE_PROMPT, GenerationConfig, SDXLPipeline
from .variables import NUMBER_WIDTH, VarSpec, load_spec, render_output, resolve_prompt

# Keys whose change forces a full pipeline rebuild (expensive).
_PIPELINE_KEYS = (
    "model",
    "vae",
    "scheduler",
    "clip_skip",
    "lora",
    "embedding",
    "ip_adapter",
    "ip_adapter_image",
    "ip_adapter_scale",
)

Echo = Callable[[str], None]


def _noop(_: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# Parameter resolution: CLI overrides win over the YAML `defaults` block.
# --------------------------------------------------------------------------- #
def _param(spec: VarSpec, overrides: dict[str, Any], key: str, default: Any) -> Any:
    if overrides.get(key) is not None:
        return overrides[key]
    if spec.defaults.get(key) is not None:
        return spec.defaults[key]
    return default


def _require_model(spec: VarSpec, overrides: dict[str, Any]) -> str:
    model = _param(spec, overrides, "model", None)
    if not model:
        raise ValueError("no model set: add defaults.model to the spec or pass --model")
    return str(model)


def _pipeline_signature(spec: VarSpec, overrides: dict[str, Any]) -> tuple:
    return tuple(_norm(_param(spec, overrides, k, None)) for k in _PIPELINE_KEYS)


def _norm(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


# --------------------------------------------------------------------------- #
# status: force to "live" on startup, editing only that line.
# --------------------------------------------------------------------------- #
_STATUS_LINE_RE = re.compile(r"^(\s*status\s*:\s*).*$", re.MULTILINE)


def set_status(path: Path, status: str) -> None:
    """Rewrite only the ``status:`` line to ``status``, preserving the rest of the file."""
    text = path.read_text(encoding="utf-8")
    if _STATUS_LINE_RE.search(text):
        new_text = _STATUS_LINE_RE.sub(rf"\g<1>{status}", text, count=1)
    else:
        sep = "" if text.endswith("\n") or text == "" else "\n"
        new_text = f"{text}{sep}status: {status}\n"
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")


def force_status_live(path: Path) -> None:
    """Rewrite the ``status:`` line to ``live``, preserving the rest of the file."""
    set_status(path, "live")


# --------------------------------------------------------------------------- #
# Counter: continue after the highest existing <number> in the output dir.
# --------------------------------------------------------------------------- #
def scan_start_number(template_output: str) -> int:
    """Return (max existing counter + 1), or 0 if none found."""
    p = Path(template_output)
    directory = p.parent if str(p.parent) else Path(".")
    name_pattern = re.escape(p.name)
    name_pattern = name_pattern.replace(re.escape("<number>"), r"(\d{%d})" % NUMBER_WIDTH)
    name_pattern = name_pattern.replace(re.escape("<seed>"), r"\d+")
    rx = re.compile(f"^{name_pattern}$")

    highest = -1
    if directory.exists():
        for f in directory.iterdir():
            m = rx.match(f.name)
            if m:
                highest = max(highest, int(m.group(1)))
    return highest + 1


# --------------------------------------------------------------------------- #
# Pipeline building
# --------------------------------------------------------------------------- #
def _build_pipeline(spec: VarSpec, overrides: dict[str, Any], echo: Echo):
    model = _require_model(spec, overrides)
    vae = _param(spec, overrides, "vae", None)
    scheduler = _param(spec, overrides, "scheduler", "euler_a")

    echo(f"Loading model: {model}")
    pipeline = SDXLPipeline(model_path=model, vae_path=vae, scheduler_name=scheduler)
    pipeline.load()

    lora = _param(spec, overrides, "lora", None)
    if lora:
        echo(f"Loading {len(lora)} LoRA(s)...")
        pipeline.load_loras(list(lora))

    embedding = _param(spec, overrides, "embedding", None)
    if embedding:
        echo(f"Loading {len(embedding)} embedding(s)...")
        pipeline.load_embeddings(list(embedding))

    ref_images: list = []
    effective_ip_scale = None
    ip_adapter_image = _param(spec, overrides, "ip_adapter_image", None)
    if ip_adapter_image:
        preset = _param(spec, overrides, "ip_adapter", "face")
        scale = _param(spec, overrides, "ip_adapter_scale", None)
        echo(f"Loading IP-Adapter ({preset})...")
        effective_ip_scale = pipeline.load_ip_adapter(preset, scale)
        ref_images = load_reference_images(list(ip_adapter_image))

    return pipeline, ref_images, effective_ip_scale


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def _append_manifest(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Dry run
# --------------------------------------------------------------------------- #
def dry_run(spec: VarSpec, rng: random.Random, samples: int, echo: Echo) -> None:
    """Print resolved prompts without loading the model."""
    start = scan_start_number(spec.template_output)
    for i in range(samples):
        prompt, chosen = resolve_prompt(spec, rng)
        out = render_output(spec.template_output, start + i, seed=0)
        echo(f"[{start + i:0{NUMBER_WIDTH}d}] -> {out}")
        echo(f"  prompt: {prompt}")
        echo(f"  vars:   {json.dumps(chosen, ensure_ascii=False)}")


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    poll: float = 5.0,
    seed_rng: random.Random | None = None,
    var_rng: random.Random | None = None,
    echo: Echo = _noop,
) -> int:
    """Run the continuous generation loop. Returns the number of images made."""
    config_path = Path(config_path)
    overrides = overrides or {}
    seed_rng = seed_rng or random.Random()
    var_rng = var_rng or random.Random()

    spec = load_spec(config_path)

    # Force status to live on startup, then reload so state matches the file.
    force_status_live(config_path)
    spec = load_spec(config_path)
    mtime = config_path.stat().st_mtime

    number = scan_start_number(spec.template_output)
    echo(f"Starting counter at {number:0{NUMBER_WIDTH}d}")

    pipeline, ref_images, ip_scale = _build_pipeline(spec, overrides, echo)
    pipeline_sig = _pipeline_signature(spec, overrides)

    manifest_path = Path(render_output(spec.template_output, 0, 0)).parent / "manifest.jsonl"

    made = 0
    while True:
        # Hot reload if the file changed on disk. Any failure (unreadable YAML,
        # invalid value, or a pipeline rebuild that blows up on a bad model /
        # scheduler) is caught: we report it and keep the previous configuration
        # running unchanged. Consume the edit (mtime) either way so a broken file
        # is retried only after the next save, not on every poll.
        current_mtime = config_path.stat().st_mtime
        if current_mtime != mtime:
            mtime = current_mtime
            try:
                new_spec = load_spec(config_path)
                new_sig = _pipeline_signature(new_spec, overrides)
                if new_sig != pipeline_sig:
                    echo("Pipeline settings changed, rebuilding...")
                    # Build into temporaries first: if this raises we keep the
                    # old pipeline AND old spec, so config stays fully coherent.
                    new_pipeline, new_ref_images, new_ip_scale = _build_pipeline(
                        new_spec, overrides, echo
                    )
                    pipeline, ref_images, ip_scale = new_pipeline, new_ref_images, new_ip_scale
                    pipeline_sig = new_sig
                spec = new_spec
                echo(f"Reloaded spec (status={spec.status}, loop={spec.loop})")
            except Exception as exc:  # noqa: BLE001 - keep the loop alive on any bad edit
                echo(f"Reload failed, keeping previous configuration unchanged: {exc}")

        if spec.status == "stop":
            echo("status=stop, exiting.")
            break

        if spec.status == "pause":
            time.sleep(poll)
            continue

        # status == live
        if spec.loop and made >= spec.loop:
            echo(f"Reached loop={spec.loop}, exiting.")
            break

        prompt, chosen = resolve_prompt(spec, var_rng)
        seed = seed_rng.randint(0, 2**32 - 1)
        out_path = Path(render_output(spec.template_output, number, seed))
        out_path.parent.mkdir(parents=True, exist_ok=True)

        steps = int(_param(spec, overrides, "steps", 30))
        cfg_scale = float(_param(spec, overrides, "cfg_scale", 4.0))
        width = int(_param(spec, overrides, "width", 1024))
        height = int(_param(spec, overrides, "height", 1024))
        clip_skip = int(_param(spec, overrides, "clip_skip", 2))
        negative = (
            _param(spec, overrides, "negative_prompt", None)
            or spec.negative_prompt
            or DEFAULT_NEGATIVE_PROMPT
        )
        hires_fix = bool(_param(spec, overrides, "hires_fix", False))

        echo(f"[{number:0{NUMBER_WIDTH}d}] seed={seed} {prompt[:70]}{'...' if len(prompt) > 70 else ''}")

        config = GenerationConfig(
            prompt=prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            clip_skip=clip_skip,
            batch_size=1,
            hires_fix=hires_fix,
            hires_scale=float(_param(spec, overrides, "hires_scale", 1.5)),
            hires_steps=int(_param(spec, overrides, "hires_steps", 15)),
            hires_denoising=float(_param(spec, overrides, "hires_denoising", 0.5)),
            ip_adapter_images=ref_images,
        )

        images = pipeline.generate(config)

        ip_adapter_image = _param(spec, overrides, "ip_adapter_image", None)
        metadata = GenerationMetadata(
            prompt=prompt,
            negative_prompt=negative,
            model=str(_require_model(spec, overrides)),
            vae=(str(_param(spec, overrides, "vae", None)) if _param(spec, overrides, "vae", None) else None),
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            scheduler=str(_param(spec, overrides, "scheduler", "euler_a")),
            clip_skip=clip_skip,
            lora=list(_param(spec, overrides, "lora", None)) if _param(spec, overrides, "lora", None) else None,
            embedding=(
                list(_param(spec, overrides, "embedding", None))
                if _param(spec, overrides, "embedding", None)
                else None
            ),
            hires_fix=hires_fix,
            hires_scale=float(_param(spec, overrides, "hires_scale", 1.5)) if hires_fix else None,
            hires_steps=int(_param(spec, overrides, "hires_steps", 15)) if hires_fix else None,
            hires_denoising=float(_param(spec, overrides, "hires_denoising", 0.5)) if hires_fix else None,
            ip_adapter=str(_param(spec, overrides, "ip_adapter", "face")) if ip_adapter_image else None,
            ip_adapter_images=list(ip_adapter_image) if ip_adapter_image else None,
            ip_adapter_scale=ip_scale,
            variables=chosen,
        )

        saved = save_image_with_metadata(images[0], out_path, metadata)
        _append_manifest(
            manifest_path,
            {
                "number": f"{number:0{NUMBER_WIDTH}d}",
                "output": str(saved),
                "seed": seed,
                "prompt": prompt,
                "variables": chosen,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        )
        echo(f"  saved: {saved}")

        number += 1
        made += 1

    return made

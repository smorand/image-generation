"""Continuous video generate-var runner.

Same control model as image-gen's runner: a hot-reloadable YAML spec drives a
loop that resolves the prompt, picks the source image, animates it, encodes an
MP4 under a zero-padded counter, and appends a manifest line. ``status`` (live /
pause / stop) and ``loop`` (0 = infinite) are re-read from the file every poll.
"""

from __future__ import annotations

import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .backends import DEFAULT_BACKEND, DEFAULT_NEGATIVE_PROMPT, frames_to_duration, get_backend, resolve_frames
from .encode import encode_frames
from .metadata import VideoMetadata, write_sidecar
from .pipeline import VideoGenConfig, VideoPipeline, load_source_image
from .variables import NUMBER_WIDTH, VideoVarSpec, load_video_spec, render_input, render_output, resolve_prompt

# Changing any of these forces an (expensive) pipeline rebuild.
_PIPELINE_KEYS = ("backend", "model", "offload", "precision")

Echo = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def _param(spec: VideoVarSpec, overrides: dict[str, Any], key: str, default: Any) -> Any:
    if overrides.get(key) is not None:
        return overrides[key]
    if spec.defaults.get(key) is not None:
        return spec.defaults[key]
    return default


def _pipeline_signature(spec: VideoVarSpec, overrides: dict[str, Any]) -> tuple:
    return (
        _param(spec, overrides, "backend", DEFAULT_BACKEND),
        _param(spec, overrides, "model", None),
        bool(_param(spec, overrides, "offload", False)),
        _param(spec, overrides, "precision", None),
    )


# --------------------------------------------------------------------------- #
# status: force to "live" on startup (edit only that line).
# --------------------------------------------------------------------------- #
_STATUS_LINE_RE = re.compile(r"^(\s*status\s*:\s*).*$", re.MULTILINE)


def force_status_live(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if _STATUS_LINE_RE.search(text):
        new_text = _STATUS_LINE_RE.sub(r"\1live", text, count=1)
    else:
        sep = "" if text.endswith("\n") or text == "" else "\n"
        new_text = f"{text}{sep}status: live\n"
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Counter: continue after the highest existing <number> in the output dir.
# --------------------------------------------------------------------------- #
def scan_start_number(template_output: str) -> int:
    p = Path(template_output)
    directory = p.parent if str(p.parent) else Path(".")
    name_pattern = re.escape(p.name)
    name_pattern = name_pattern.replace(re.escape("<number>"), r"(\d{%d})" % NUMBER_WIDTH)
    name_pattern = name_pattern.replace(re.escape("<seed>"), r"\d+")
    # template_output may carry an image suffix; match the .mp4 we actually write.
    name_pattern = re.sub(r"\\\.[A-Za-z0-9]+$", r"\\.mp4", name_pattern)
    rx = re.compile(f"^{name_pattern}$")

    highest = -1
    if directory.exists():
        for f in directory.iterdir():
            m = rx.match(f.name)
            if m:
                highest = max(highest, int(m.group(1)))
    return highest + 1


def _resolve_num_frames(spec: VideoVarSpec, overrides: dict[str, Any], backend) -> tuple[int, int]:
    """Return (num_frames, fps) from explicit frames or a duration in seconds."""
    fps = int(_param(spec, overrides, "fps", backend.native_fps))
    num_frames = _param(spec, overrides, "num_frames", None)
    if num_frames is not None:
        return int(num_frames), fps
    duration = float(_param(spec, overrides, "duration", 3.0))
    return resolve_frames(backend, duration, fps), fps


def _build_pipeline(spec: VideoVarSpec, overrides: dict[str, Any], echo: Echo) -> VideoPipeline:
    backend = _param(spec, overrides, "backend", DEFAULT_BACKEND)
    model = _param(spec, overrides, "model", None)
    offload = bool(_param(spec, overrides, "offload", False))
    precision = _param(spec, overrides, "precision", None)
    echo(f"Loading backend '{backend}'{' (offload)' if offload else ''}...")
    pipe = VideoPipeline(backend=backend, repo_override=model, offload=offload, precision=precision)
    pipe.load()
    echo(f"  device={pipe.device} dtype={pipe.dtype} repo={pipe.repo}")
    return pipe


def _append_manifest(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Dry run
# --------------------------------------------------------------------------- #
def dry_run(spec: VideoVarSpec, rng: random.Random, samples: int, echo: Echo) -> None:
    start = scan_start_number(spec.template_output)
    backend = get_backend(_param(spec, {}, "backend", DEFAULT_BACKEND))
    num_frames, fps = _resolve_num_frames(spec, {}, backend)
    echo(f"backend={backend.name} frames={num_frames} fps={fps} duration={frames_to_duration(num_frames, fps)}s")
    for i in range(samples):
        prompt, chosen = resolve_prompt(spec, rng)
        out = render_output(spec.template_output, start + i, seed=0)
        out = str(Path(out).with_suffix(".mp4"))
        try:
            src = str(render_input(spec.template_input, start + i, 0, rng))
        except FileNotFoundError as exc:
            src = f"<missing: {exc}>"
        echo(f"[{start + i:0{NUMBER_WIDTH}d}] {src} -> {out}")
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
    """Run the continuous video generation loop. Returns clips made."""
    config_path = Path(config_path)
    overrides = overrides or {}
    seed_rng = seed_rng or random.Random()
    var_rng = var_rng or random.Random()

    force_status_live(config_path)
    spec = load_video_spec(config_path)
    mtime = config_path.stat().st_mtime

    number = scan_start_number(spec.template_output)
    echo(f"Starting counter at {number:0{NUMBER_WIDTH}d}")

    pipeline = _build_pipeline(spec, overrides, echo)
    pipeline_sig = _pipeline_signature(spec, overrides)

    manifest_path = Path(render_output(spec.template_output, 0, 0)).parent / "manifest_video.jsonl"

    made = 0
    while True:
        current_mtime = config_path.stat().st_mtime
        if current_mtime != mtime:
            try:
                new_spec = load_video_spec(config_path)
            except (ValueError, OSError) as exc:
                echo(f"Reload failed, keeping previous spec: {exc}")
                mtime = current_mtime
            else:
                new_sig = _pipeline_signature(new_spec, overrides)
                if new_sig != pipeline_sig:
                    echo("Backend/model changed, rebuilding pipeline...")
                    pipeline.unload()
                    pipeline = _build_pipeline(new_spec, overrides, echo)
                    pipeline_sig = new_sig
                spec = new_spec
                mtime = current_mtime
                echo(f"Reloaded spec (status={spec.status}, loop={spec.loop})")

        if spec.status == "stop":
            echo("status=stop, exiting.")
            break
        if spec.status == "pause":
            time.sleep(poll)
            continue
        if spec.loop and made >= spec.loop:
            echo(f"Reached loop={spec.loop}, exiting.")
            break

        backend = pipeline.spec
        num_frames, fps = _resolve_num_frames(spec, overrides, backend)
        prompt, chosen = resolve_prompt(spec, var_rng)
        seed = seed_rng.randint(0, 2**32 - 1)

        try:
            src_path = render_input(spec.template_input, number, seed, var_rng)
        except FileNotFoundError as exc:
            echo(f"[{number:0{NUMBER_WIDTH}d}] skipped: {exc}")
            time.sleep(poll)
            continue

        out_path = Path(render_output(spec.template_output, number, seed)).with_suffix(".mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        steps = int(_param(spec, overrides, "steps", backend.default_steps))
        guidance = float(_param(spec, overrides, "guidance", backend.default_guidance))
        width = int(_param(spec, overrides, "width", backend.default_width))
        height = int(_param(spec, overrides, "height", backend.default_height))
        negative = _param(spec, overrides, "negative_prompt", None) or spec.negative_prompt or DEFAULT_NEGATIVE_PROMPT

        duration = frames_to_duration(num_frames, fps)
        echo(
            f"[{number:0{NUMBER_WIDTH}d}] seed={seed} {duration}s/{num_frames}f "
            f"{src_path.name} :: {prompt[:60]}{'...' if len(prompt) > 60 else ''}"
        )

        config = VideoGenConfig(
            prompt=prompt,
            image=load_source_image(src_path),
            negative_prompt=negative,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            guidance=guidance,
            seed=seed,
        )

        t0 = time.time()
        frames = pipeline.generate(config)
        gen_s = time.time() - t0

        metadata = VideoMetadata(
            prompt=prompt,
            negative_prompt=negative,
            backend=backend.name,
            model=pipeline.repo,
            source_image=str(src_path),
            seed=seed,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            duration_s=duration,
            steps=steps,
            guidance=guidance,
            variables=chosen,
        )
        saved = encode_frames(frames, out_path, fps=fps, comment=metadata.to_json())
        write_sidecar(saved, metadata)

        _append_manifest(
            manifest_path,
            {
                "number": f"{number:0{NUMBER_WIDTH}d}",
                "output": str(saved),
                "source_image": str(src_path),
                "backend": backend.name,
                "seed": seed,
                "prompt": prompt,
                "variables": chosen,
                "num_frames": num_frames,
                "fps": fps,
                "gen_seconds": round(gen_s, 1),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        )
        echo(f"  saved: {saved}  ({gen_s:.0f}s)")

        number += 1
        made += 1

    return made

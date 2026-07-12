"""YAML spec for video generate-var, reusing image-gen's variable engine.

The prompt-variable resolution (weighted, recursive ``<placeholder>`` draws) is
identical to image-gen and imported wholesale from :mod:`image_gen.variables`.
Video adds one concept: ``template_input``, the source image each clip animates.

``template_input`` supports three forms:
* a plain path (``seeds/cat.png``): the same image for every clip;
* a counter/seed template (``seeds/img_<number>.png``): pairs with the loop
  counter, e.g. to animate an image-gen output batch one by one;
* a glob or directory (``seeds/*.png`` or ``seeds/``): a random existing file is
  drawn per clip.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from image_gen.variables import (  # reuse the engine
    NUMBER_WIDTH,
    OUTPUT_BUILTINS,
    PLACEHOLDER_RE,
    VALID_STATUS,
    Option,
    _parse_variables,
    render_output,
    resolve_prompt,
)

__all__ = [
    "VideoVarSpec",
    "load_video_spec",
    "resolve_prompt",
    "render_output",
    "render_input",
    "NUMBER_WIDTH",
]


@dataclass
class VideoVarSpec:
    """A parsed video generate-var specification."""

    template_prompt: str
    template_output: str
    template_input: str
    variables: dict[str, list[Option]]
    negative_prompt: str | None = None
    loop: int = 0
    status: str = "live"
    defaults: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate(self)


def _validate(spec: VideoVarSpec) -> None:
    errors: list[str] = []
    top = set(spec.variables.keys())
    for name in PLACEHOLDER_RE.findall(spec.template_prompt):
        if name not in top:
            errors.append(f"template_prompt references undefined <{name}>")
    for name in PLACEHOLDER_RE.findall(spec.template_output):
        if name not in OUTPUT_BUILTINS:
            errors.append(
                f"template_output references <{name}> (only {sorted(OUTPUT_BUILTINS)} allowed)"
            )
    for name in PLACEHOLDER_RE.findall(spec.template_input):
        if name not in OUTPUT_BUILTINS:
            errors.append(
                f"template_input references <{name}> (only {sorted(OUTPUT_BUILTINS)} allowed)"
            )
    if errors:
        raise ValueError("invalid spec:\n  - " + "\n  - ".join(errors))


def load_video_spec(path: str | Path) -> VideoVarSpec:
    """Load and validate a video spec from a YAML file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: spec must be a YAML mapping")

    for key in ("template_prompt", "template_output", "template_input", "variables"):
        if key not in raw:
            raise ValueError(f"{path}: missing required key '{key}'")

    status = str(raw.get("status", "live")).strip().lower()
    if status not in VALID_STATUS:
        raise ValueError(f"{path}: status must be one of {sorted(VALID_STATUS)}, got '{status}'")

    loop = int(raw.get("loop", 0))
    if loop < 0:
        raise ValueError(f"{path}: loop must be >= 0, got {loop}")

    defaults = raw.get("defaults", {}) or {}
    if not isinstance(defaults, dict):
        raise ValueError(f"{path}: 'defaults' must be a mapping")

    negative = raw.get("negative_prompt")

    return VideoVarSpec(
        template_prompt=str(raw["template_prompt"]),
        template_output=str(raw["template_output"]),
        template_input=str(raw["template_input"]),
        variables=_parse_variables(raw["variables"]),
        negative_prompt=str(negative) if negative is not None else None,
        loop=loop,
        status=status,
        defaults=defaults,
    )


def render_input(
    template: str, number: int, seed: int, rng: random.Random
) -> Path:
    """Resolve ``template_input`` to a concrete existing image path.

    Counter/seed templates are rendered directly; globs and directories draw a
    random existing file; a plain path is returned as-is. Raises FileNotFoundError
    if the resolved path (or a non-empty match set) does not exist.
    """
    if "<number>" in template or "<seed>" in template:
        path = Path(render_output(template, number, seed))
        if not path.exists():
            raise FileNotFoundError(f"input image not found: {path}")
        return path

    if "*" in template or "?" in template or "[" in template:
        base = Path(template)
        matches = sorted(base.parent.glob(base.name))
        if not matches:
            raise FileNotFoundError(f"no input images match glob: {template}")
        return rng.choice(matches)

    path = Path(template)
    if path.is_dir():
        matches = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not matches:
            raise FileNotFoundError(f"no input images in directory: {template}")
        return rng.choice(matches)

    if not path.exists():
        raise FileNotFoundError(f"input image not found: {path}")
    return path

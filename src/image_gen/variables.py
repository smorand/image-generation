"""YAML-driven prompt variable expansion.

A spec file describes a prompt template with ``<placeholder>`` slots and a tree
of weighted, recursive variable options. Resolving the template draws one option
per variable (weighted random), reuses the same draw when a variable appears
more than once, and cleans up the resulting prompt (empty slots, double commas).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Placeholders look like <name>: ASCII letters, digits and underscore.
PLACEHOLDER_RE = re.compile(r"<([a-zA-Z0-9_]+)>")

# Builtins are only valid inside ``template_output`` (not the prompt).
OUTPUT_BUILTINS = frozenset({"number", "seed"})

# Zero-padded width of the <number> counter, for alphanumeric sorting.
NUMBER_WIDTH = 10


@dataclass
class Option:
    """One possible value for a variable.

    ``value`` may itself contain ``<sub>`` placeholders resolved against
    ``variables`` (this option's local scope) overlaid on the parent scope.
    """

    value: str
    weight: float = 1.0
    variables: dict[str, list["Option"]] = field(default_factory=dict)


@dataclass
class VarSpec:
    """A parsed generate-var specification."""

    template_prompt: str
    template_output: str
    variables: dict[str, list[Option]]
    negative_prompt: str | None = None
    loop: int = 0
    status: str = "live"
    defaults: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_spec(self)


VALID_STATUS = frozenset({"live", "pause", "stop"})


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _parse_option(raw: Any, var_name: str) -> Option:
    """Parse a single option (shorthand string or full mapping)."""
    if isinstance(raw, str):
        return Option(value=raw)
    if raw is None:
        return Option(value="")
    if isinstance(raw, (int, float)):
        return Option(value=str(raw))
    if isinstance(raw, dict):
        if "value" not in raw:
            raise ValueError(
                f"variable '{var_name}': option mapping must have a 'value' key: {raw!r}"
            )
        value = raw["value"]
        value = "" if value is None else str(value)
        weight = float(raw.get("weight", 1.0))
        if weight < 0:
            raise ValueError(f"variable '{var_name}': weight must be >= 0, got {weight}")
        sub = _parse_variables(raw.get("variables", {}) or {})
        return Option(value=value, weight=weight, variables=sub)
    raise ValueError(f"variable '{var_name}': unsupported option type {type(raw).__name__}")


def _parse_variables(raw: Any) -> dict[str, list[Option]]:
    """Parse a ``variables`` mapping into ``{name: [Option, ...]}``."""
    if not isinstance(raw, dict):
        raise ValueError(f"'variables' must be a mapping, got {type(raw).__name__}")
    out: dict[str, list[Option]] = {}
    for name, options in raw.items():
        if not isinstance(options, list):
            raise ValueError(f"variable '{name}': value must be a list of options")
        if not options:
            raise ValueError(f"variable '{name}': option list is empty")
        out[str(name)] = [_parse_option(o, str(name)) for o in options]
    return out


def load_spec(path: str | Path) -> VarSpec:
    """Load and validate a spec from a YAML file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: spec must be a YAML mapping")

    for key in ("template_prompt", "template_output", "variables"):
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

    return VarSpec(
        template_prompt=str(raw["template_prompt"]),
        template_output=str(raw["template_output"]),
        variables=_parse_variables(raw["variables"]),
        negative_prompt=str(negative) if negative is not None else None,
        loop=loop,
        status=status,
        defaults=defaults,
    )


# --------------------------------------------------------------------------- #
# Validation (catch typos in placeholder names early)
# --------------------------------------------------------------------------- #
def validate_spec(spec: VarSpec) -> None:
    """Raise ValueError if a placeholder references an undefined variable."""
    errors: list[str] = []

    top_names = set(spec.variables.keys())
    for name in PLACEHOLDER_RE.findall(spec.template_prompt):
        if name not in top_names:
            errors.append(f"template_prompt references undefined <{name}>")

    for name in PLACEHOLDER_RE.findall(spec.template_output):
        if name not in OUTPUT_BUILTINS:
            errors.append(
                f"template_output references <{name}> (only {sorted(OUTPUT_BUILTINS)} allowed)"
            )

    _walk_validate(spec.variables, top_names, errors)

    if errors:
        raise ValueError("invalid spec:\n  - " + "\n  - ".join(errors))


def _walk_validate(
    variables: dict[str, list[Option]],
    visible_parent: set[str],
    errors: list[str],
) -> None:
    visible = visible_parent | set(variables.keys())
    for vname, options in variables.items():
        for opt in options:
            opt_visible = visible | set(opt.variables.keys())
            for ref in PLACEHOLDER_RE.findall(opt.value):
                if ref not in opt_visible:
                    errors.append(f"variable '{vname}' option '{opt.value}' references undefined <{ref}>")
            _walk_validate(opt.variables, visible, errors)


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def _weighted_choice(options: list[Option], rng: random.Random) -> Option:
    total = sum(o.weight for o in options)
    if total <= 0:
        return rng.choice(options)
    r = rng.random() * total
    upto = 0.0
    for opt in options:
        upto += opt.weight
        if r < upto:
            return opt
    return options[-1]


def _resolve(
    template: str,
    scope: dict[str, list[Option]],
    rng: random.Random,
    memo: dict[str, str],
    chosen: dict[str, str],
    depth: int = 0,
) -> str:
    if depth > 50:
        raise ValueError("placeholder recursion too deep (>50); check for a variable cycle")

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in memo:
            return memo[name]
        options = scope.get(name)
        if options is None:
            # Undefined here (validation lets nested-only names through); leave as-is.
            return match.group(0)
        opt = _weighted_choice(options, rng)
        child_scope = {**scope, **opt.variables}
        resolved = _resolve(opt.value, child_scope, rng, memo, chosen, depth + 1)
        memo[name] = resolved
        # Collapse whitespace in the recorded value (empty sub-slots leave gaps)
        # without touching commas, so the metadata sub-dict stays grep-friendly.
        chosen[name] = re.sub(r"\s+", " ", resolved).strip()
        return resolved

    return PLACEHOLDER_RE.sub(repl, template)


def clean_prompt(text: str) -> str:
    """Normalise whitespace and commas after empty slots are removed.

    Collapses runs of whitespace, drops spaces before commas, merges repeated
    commas into one, ensures a single space after each comma, and strips
    leading/trailing commas and whitespace.
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",(?:\s*,)+", ",", text)
    text = re.sub(r",\s*", ", ", text)
    text = text.strip()
    text = re.sub(r"^(?:,\s*)+", "", text)
    text = re.sub(r"(?:\s*,)+$", "", text)
    return text.strip()


def resolve_prompt(spec: VarSpec, rng: random.Random) -> tuple[str, dict[str, str]]:
    """Resolve the prompt template once.

    Returns the cleaned prompt and the map of variable name -> chosen value.
    A variable is drawn once and reused wherever it appears again.
    """
    memo: dict[str, str] = {}
    chosen: dict[str, str] = {}
    raw = _resolve(spec.template_prompt, spec.variables, rng, memo, chosen)
    return clean_prompt(raw), chosen


def render_output(template: str, number: int, seed: int) -> str:
    """Render ``template_output`` with the counter and seed builtins."""
    out = template.replace("<number>", f"{number:0{NUMBER_WIDTH}d}")
    out = out.replace("<seed>", str(seed))
    return out

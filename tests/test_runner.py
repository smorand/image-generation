"""Tests for runner helpers and the loop (pipeline mocked, no model needed)."""

import json
import random

from PIL import Image

import image_gen.runner as runner
from image_gen.runner import (
    _param,
    _pipeline_signature,
    force_status_live,
    run,
    scan_start_number,
    set_status,
)
from image_gen.variables import Option, VarSpec


def _spec(**kw):
    base = dict(
        template_prompt="<eth>",
        template_output="out/img_<number>.png",
        variables={"eth": [Option("inuit")]},
    )
    base.update(kw)
    return VarSpec(**base)


def test_param_override_wins():
    spec = _spec(defaults={"steps": 20})
    assert _param(spec, {"steps": 40}, "steps", 30) == 40
    assert _param(spec, {"steps": None}, "steps", 30) == 20
    assert _param(spec, {}, "steps", 30) == 20
    assert _param(spec, {}, "missing", 30) == 30


def test_pipeline_signature_changes_on_model():
    spec = _spec(defaults={"model": "a.safetensors"})
    sig_a = _pipeline_signature(spec, {})
    sig_b = _pipeline_signature(spec, {"model": "b.safetensors"})
    assert sig_a != sig_b


def test_pipeline_signature_stable_for_prompt_only():
    spec1 = _spec(defaults={"model": "a"}, template_prompt="<eth>")
    spec2 = _spec(defaults={"model": "a"}, template_prompt="<eth> smiling")
    assert _pipeline_signature(spec1, {}) == _pipeline_signature(spec2, {})


def test_force_status_live_rewrites_line(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text("status: pause\nloop: 3\n# a comment\n", encoding="utf-8")
    force_status_live(p)
    text = p.read_text(encoding="utf-8")
    assert "status: live" in text
    assert "loop: 3" in text
    assert "# a comment" in text  # comment preserved


def test_force_status_live_appends_when_missing(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text("loop: 1\n", encoding="utf-8")
    force_status_live(p)
    text = p.read_text(encoding="utf-8")
    assert "loop: 1" in text
    assert "status: live" in text


def test_set_status_rewrites_only_status_line(tmp_path):
    p = tmp_path / "spec.yaml"
    original = "status: live\nloop: 3\n# keep me\n"
    p.write_text(original, encoding="utf-8")
    for target in ("pause", "stop", "live"):
        set_status(p, target)
        text = p.read_text(encoding="utf-8")
        assert f"status: {target}" in text
        assert "loop: 3" in text
        assert "# keep me" in text


def test_scan_start_number_empty_dir(tmp_path):
    template = str(tmp_path / "img_<number>.png")
    assert scan_start_number(template) == 0


def test_scan_start_number_continues_after_max(tmp_path):
    (tmp_path / "img_0000000000.png").write_bytes(b"x")
    (tmp_path / "img_0000000005.png").write_bytes(b"x")
    (tmp_path / "img_0000000002.png").write_bytes(b"x")
    (tmp_path / "unrelated.png").write_bytes(b"x")
    template = str(tmp_path / "img_<number>.png")
    assert scan_start_number(template) == 6


def test_scan_start_number_with_seed_placeholder(tmp_path):
    (tmp_path / "img_0000000003_777.png").write_bytes(b"x")
    template = str(tmp_path / "img_<number>_<seed>.png")
    assert scan_start_number(template) == 4


# --------------------------------------------------------------------------- #
# End-to-end loop with a fake pipeline (no model download / no diffusion).
# --------------------------------------------------------------------------- #
class _FakePipeline:
    def __init__(self):
        self.calls = 0

    def generate(self, config):
        self.calls += 1
        return [Image.new("RGB", (8, 8), (self.calls % 255, 0, 0))]


def _spec_yaml(out_dir, loop, status="live"):
    return (
        f"status: {status}\n"
        f"loop: {loop}\n"
        f'template_output: "{out_dir}/img_<number>.png"\n'
        'template_prompt: "solo, <eth> girl, <clothes>"\n'
        "defaults:\n"
        '  model: "fake.safetensors"\n'
        "variables:\n"
        "  eth: [inuit]\n"
        "  clothes:\n"
        '    - "wearing a dress"\n'
        '    - "wearing a <color> bikini"\n'
    )


def _spec_yaml_with_color(out_dir, loop):
    text = _spec_yaml(out_dir, loop)
    return text + ("  color:\n    - red\n    - blue\n")


def test_run_loop_generates_and_manifests(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(_spec_yaml_with_color(out_dir, loop=3), encoding="utf-8")

    fake = _FakePipeline()
    monkeypatch.setattr(runner, "_build_pipeline", lambda spec, ov, echo: (fake, [], None))

    made = run(spec_path, poll=0.01, var_rng=random.Random(0), seed_rng=random.Random(0))

    assert made == 3
    assert fake.calls == 3
    files = sorted(p.name for p in out_dir.glob("img_*.png"))
    assert files == ["img_0000000000.png", "img_0000000001.png", "img_0000000002.png"]

    # status forced to live on startup
    assert "status: live" in spec_path.read_text(encoding="utf-8")

    # JSONL log (daily-rotated, defaults to the output dir) has one full record
    # per image, merging the resolved variables and every sampling parameter.
    from image_gen.logging_jsonl import log_file_for

    log_path = log_file_for(out_dir)
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    rec = json.loads(lines[0])
    assert rec["number"] == "0000000000"
    assert rec["command"] == "generate-var"
    assert rec["variables"]["eth"] == "inuit"
    assert "prompt" in rec and "seed" in rec
    # Full params are present now (not just the old manifest subset).
    assert rec["cfg_scale"] and rec["scheduler"] and rec["steps"]


def test_run_counter_continues_after_existing(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "img_0000000004.png").write_bytes(b"x")
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(_spec_yaml_with_color(out_dir, loop=1), encoding="utf-8")

    fake = _FakePipeline()
    monkeypatch.setattr(runner, "_build_pipeline", lambda spec, ov, echo: (fake, [], None))

    run(spec_path, poll=0.01)
    assert (out_dir / "img_0000000005.png").exists()


def test_run_stop_status_exits_immediately(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    spec_path = tmp_path / "spec.yaml"
    # loop=0 (infinite) but status stop -> must exit with 0 images.
    # force_status_live rewrites to live, so simulate a stop by patching after start.
    spec_path.write_text(_spec_yaml_with_color(out_dir, loop=0), encoding="utf-8")

    fake = _FakePipeline()
    monkeypatch.setattr(runner, "_build_pipeline", lambda spec, ov, echo: (fake, [], None))

    # After the first generation, flip the file to status: stop so the loop ends.
    original_save = runner.save_image_with_metadata

    def _save_then_stop(image, path, metadata, quality=95):
        spec_path.write_text(
            _spec_yaml_with_color(out_dir, loop=0).replace("status: live", "status: stop"), encoding="utf-8"
        )
        return original_save(image, path, metadata, quality)

    monkeypatch.setattr(runner, "save_image_with_metadata", _save_then_stop)

    made = run(spec_path, poll=0.01)
    assert made == 1  # one image, then reload sees stop and exits


# --------------------------------------------------------------------------- #
# Resilient hot reload: a bad edit is reported and the previous config is kept.
# --------------------------------------------------------------------------- #
def test_run_keeps_previous_spec_on_unreadable_yaml(tmp_path, monkeypatch):
    """A syntactically broken YAML on reload must not crash: keep old spec."""
    out_dir = tmp_path / "out"
    spec_path = tmp_path / "spec.yaml"
    # loop=2 so that, keeping the OLD spec, the loop still terminates on its own.
    spec_path.write_text(_spec_yaml_with_color(out_dir, loop=2), encoding="utf-8")

    fake = _FakePipeline()
    monkeypatch.setattr(runner, "_build_pipeline", lambda spec, ov, echo: (fake, [], None))

    original_save = runner.save_image_with_metadata
    corrupted = {"done": False}

    def _save_then_break(image, path, metadata, quality=95):
        if not corrupted["done"]:
            # Tabs + dangling colon => yaml.YAMLError (not ValueError/OSError).
            spec_path.write_text("status: live\n\tbad: : :\n", encoding="utf-8")
            corrupted["done"] = True
        return original_save(image, path, metadata, quality)

    monkeypatch.setattr(runner, "save_image_with_metadata", _save_then_break)

    logs: list[str] = []
    made = run(spec_path, poll=0.01, echo=logs.append)

    assert made == 2  # old spec (loop=2) preserved, no crash
    assert fake.calls == 2
    assert any("keeping previous configuration" in m for m in logs)


def test_run_keeps_previous_pipeline_on_bad_model(tmp_path, monkeypatch):
    """A reload whose model breaks the pipeline rebuild keeps the old pipeline."""
    out_dir = tmp_path / "out"
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(_spec_yaml_with_color(out_dir, loop=2), encoding="utf-8")

    fake = _FakePipeline()

    def _build(spec, ov, echo):
        if runner._require_model(spec, ov) == "bad.safetensors":
            raise RuntimeError("model file not found: bad.safetensors")
        return fake, [], None

    monkeypatch.setattr(runner, "_build_pipeline", _build)

    original_save = runner.save_image_with_metadata
    switched = {"done": False}

    def _save_then_bad_model(image, path, metadata, quality=95):
        if not switched["done"]:
            new = _spec_yaml_with_color(out_dir, loop=2).replace('"fake.safetensors"', '"bad.safetensors"')
            spec_path.write_text(new, encoding="utf-8")
            switched["done"] = True
        return original_save(image, path, metadata, quality)

    monkeypatch.setattr(runner, "save_image_with_metadata", _save_then_bad_model)

    logs: list[str] = []
    made = run(spec_path, poll=0.01, echo=logs.append)

    assert made == 2  # old pipeline (fake) kept, loop finishes
    assert fake.calls == 2
    assert any("keeping previous configuration" in m for m in logs)

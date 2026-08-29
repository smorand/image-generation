"""Tests for the video generate-var runner (fake pipeline, real MP4 encode)."""

import json
import random

from PIL import Image

import video_gen.runner as runner
from video_gen.backends import BACKENDS
from video_gen.runner import (
    _param,
    _pipeline_signature,
    _resolve_num_frames,
    dry_run,
    force_status_live,
    run,
    scan_start_number,
)
from video_gen.variables import Option, VideoVarSpec


def _spec(**kw):
    base = dict(
        template_prompt="<subject>",
        template_output="out/clip_<number>.mp4",
        template_input="seeds/cat.png",
        variables={"subject": [Option("a cat")]},
    )
    base.update(kw)
    return VideoVarSpec(**base)


# --------------------------------------------------------------------------- #
# _param / _pipeline_signature
# --------------------------------------------------------------------------- #
def test_param_override_wins():
    spec = _spec(defaults={"fps": 12})
    assert _param(spec, {"fps": 30}, "fps", 24) == 30
    assert _param(spec, {"fps": None}, "fps", 24) == 12
    assert _param(spec, {}, "fps", 24) == 12
    assert _param(spec, {}, "missing", 24) == 24


def test_pipeline_signature_changes_on_backend():
    spec_ltx = _spec(defaults={"backend": "ltx"})
    spec_wan = _spec(defaults={"backend": "wan"})
    assert _pipeline_signature(spec_ltx, {}) != _pipeline_signature(spec_wan, {})


def test_pipeline_signature_stable_for_prompt_only():
    spec1 = _spec(defaults={"backend": "ltx"}, template_prompt="<subject>")
    spec2 = _spec(defaults={"backend": "ltx"}, template_prompt="<subject>, cinematic")
    assert _pipeline_signature(spec1, {}) == _pipeline_signature(spec2, {})


# --------------------------------------------------------------------------- #
# _resolve_num_frames
# --------------------------------------------------------------------------- #
def test_resolve_num_frames_explicit_wins():
    spec = _spec(defaults={"num_frames": 49, "fps": 24})
    num_frames, fps = _resolve_num_frames(spec, {}, BACKENDS["ltx"])
    assert num_frames == 49
    assert fps == 24


def test_resolve_num_frames_from_duration():
    spec = _spec(defaults={"duration": 2.0})
    num_frames, fps = _resolve_num_frames(spec, {}, BACKENDS["ltx"])
    assert fps == BACKENDS["ltx"].native_fps
    assert num_frames > 0


# --------------------------------------------------------------------------- #
# force_status_live / scan_start_number (same regex-based helpers as image-gen)
# --------------------------------------------------------------------------- #
def test_force_status_live_rewrites_line(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text("status: pause\nloop: 3\n# a comment\n", encoding="utf-8")
    force_status_live(p)
    text = p.read_text(encoding="utf-8")
    assert "status: live" in text
    assert "loop: 3" in text
    assert "# a comment" in text


def test_force_status_live_appends_when_missing(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text("loop: 1\n", encoding="utf-8")
    force_status_live(p)
    assert "status: live" in p.read_text(encoding="utf-8")


def test_scan_start_number_empty_dir(tmp_path):
    assert scan_start_number(str(tmp_path / "clip_<number>.mp4")) == 0


def test_scan_start_number_continues_after_max(tmp_path):
    (tmp_path / "clip_0000000002.mp4").write_bytes(b"")
    (tmp_path / "clip_0000000005.mp4").write_bytes(b"")
    assert scan_start_number(str(tmp_path / "clip_<number>.mp4")) == 6


# --------------------------------------------------------------------------- #
# dry_run
# --------------------------------------------------------------------------- #
def test_dry_run_prints_prompt_and_missing_source(tmp_path, capsys):
    seed_dir = tmp_path / "seeds"
    spec = _spec(
        template_output=str(tmp_path / "out" / "clip_<number>.mp4"),
        template_input=str(seed_dir / "cat.png"),  # does not exist
        defaults={"backend": "ltx"},
    )
    lines = []
    dry_run(spec, random.Random(0), samples=1, echo=lines.append)
    text = "\n".join(lines)
    assert "backend=ltx" in text
    assert "prompt: a cat" in text
    assert "missing:" in text


# --------------------------------------------------------------------------- #
# End-to-end loop with a fake pipeline (no model download, real MP4 encode).
# --------------------------------------------------------------------------- #
class _FakePipeline:
    def __init__(self):
        self.spec = BACKENDS["ltx"]
        self.device = "cpu"
        self.dtype = "bf16"
        self.repo = "fake/ltx"
        self.calls = 0
        self.unloaded = False

    def generate(self, config):
        self.calls += 1
        shade = (self.calls * 40) % 255
        return [Image.new("RGB", (config.width, config.height), (shade, 0, 0)) for _ in range(config.num_frames)]

    def unload(self):
        self.unloaded = True


def _video_spec_yaml(out_dir, seed_path, loop, status="live"):
    return (
        f"status: {status}\n"
        f"loop: {loop}\n"
        f'template_output: "{out_dir}/clip_<number>.mp4"\n'
        f'template_input: "{seed_path}"\n'
        'template_prompt: "<subject>, cinematic"\n'
        "defaults:\n"
        '  backend: "ltx"\n'
        "  duration: 0.2\n"
        "variables:\n"
        "  subject: [a cat, a dog]\n"
    )


def test_run_loop_generates_clips_and_manifests(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    seed_img = tmp_path / "seed.png"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(seed_img)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(_video_spec_yaml(out_dir, seed_img, loop=2), encoding="utf-8")

    fake = _FakePipeline()
    monkeypatch.setattr(runner, "_build_pipeline", lambda spec, ov, echo: fake)

    made = run(spec_path, poll=0.01, var_rng=random.Random(0), seed_rng=random.Random(0))

    assert made == 2
    assert fake.calls == 2
    files = sorted(p.name for p in out_dir.glob("clip_*.mp4"))
    assert files == ["clip_0000000000.mp4", "clip_0000000001.mp4"]

    # status forced to live on startup
    assert "status: live" in spec_path.read_text(encoding="utf-8")

    # Sidecar JSON metadata written next to each clip
    sidecar = json.loads((out_dir / "clip_0000000000.mp4").with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["backend"] == "ltx"
    assert "prompt" in sidecar and "seed" in sidecar

    # Manifest JSONL has one record per clip
    manifest = out_dir / "manifest_video.jsonl"
    lines = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["number"] == "0000000000"
    assert "prompt" in rec and "seed" in rec


def test_run_stop_status_exits_immediately(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    seed_img = tmp_path / "seed.png"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(seed_img)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(_video_spec_yaml(out_dir, seed_img, loop=5), encoding="utf-8")

    fake = _FakePipeline()
    monkeypatch.setattr(runner, "_build_pipeline", lambda spec, ov, echo: fake)

    def _generate_then_stop(config):
        spec_path.write_text(_video_spec_yaml(out_dir, seed_img, loop=5, status="stop"), encoding="utf-8")
        fake.calls += 1
        return [Image.new("RGB", (config.width, config.height), (0, 0, 0)) for _ in range(config.num_frames)]

    monkeypatch.setattr(fake, "generate", _generate_then_stop)

    made = run(spec_path, poll=0.01, var_rng=random.Random(0), seed_rng=random.Random(0))

    assert made == 1
    assert fake.calls == 1


def test_run_skips_when_source_image_missing(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    spec_path = tmp_path / "spec.yaml"
    # template_input points at a counter-templated path that never exists.
    spec_path.write_text(
        _video_spec_yaml(out_dir, tmp_path / "seeds" / "img_<number>.png", loop=1),
        encoding="utf-8",
    )

    fake = _FakePipeline()
    monkeypatch.setattr(runner, "_build_pipeline", lambda spec, ov, echo: fake)

    calls_at_stop_check = {"n": 0}

    def _fast_sleep(_seconds):
        calls_at_stop_check["n"] += 1
        if calls_at_stop_check["n"] > 3:
            spec_path.write_text(
                _video_spec_yaml(out_dir, tmp_path / "seeds" / "img_<number>.png", loop=1, status="stop"),
                encoding="utf-8",
            )

    monkeypatch.setattr(runner.time, "sleep", _fast_sleep)

    made = run(spec_path, poll=0.01, var_rng=random.Random(0), seed_rng=random.Random(0))

    assert made == 0
    assert fake.calls == 0

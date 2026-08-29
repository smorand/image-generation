"""Tests for LoRA loading helpers (no model needed, pipeline is a fake)."""

import pytest

from image_gen.lora import load_loras, parse_lora_arg


def test_parse_lora_arg_with_weight(tmp_path):
    lora_file = tmp_path / "style.safetensors"
    lora_file.write_bytes(b"fake")
    path, weight = parse_lora_arg(f"{lora_file}:0.6")
    assert path == lora_file
    assert weight == 0.6


def test_parse_lora_arg_without_weight_defaults_to_0_8(tmp_path):
    lora_file = tmp_path / "style.safetensors"
    lora_file.write_bytes(b"fake")
    path, weight = parse_lora_arg(str(lora_file))
    assert path == lora_file
    assert weight == 0.8


def test_parse_lora_arg_unparsable_weight_falls_back_to_whole_path(tmp_path):
    # A Windows-style path ("C:\\loras\\x.safetensors") contains a ':' that is
    # not a weight separator; the float() parse fails and the whole string is
    # treated as the path instead.
    lora_file = tmp_path / "c:oops.safetensors"
    lora_file.write_bytes(b"fake")
    path, weight = parse_lora_arg(str(lora_file))
    assert path == lora_file
    assert weight == 0.8


def test_parse_lora_arg_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="LoRA file not found"):
        parse_lora_arg(str(tmp_path / "missing.safetensors"))


class _FakePipeline:
    def __init__(self):
        self.lora_calls: list[dict] = []
        self.adapter_calls: list[tuple] = []

    def load_lora_weights(self, path, weight_name, adapter_name):
        self.lora_calls.append({"path": path, "weight_name": weight_name, "adapter_name": adapter_name})

    def set_adapters(self, names, adapter_weights):
        self.adapter_calls.append((names, adapter_weights))


def test_load_loras_noop_on_empty_list():
    pipeline = _FakePipeline()
    load_loras(pipeline, [])
    assert pipeline.lora_calls == []
    assert pipeline.adapter_calls == []


def test_load_loras_loads_each_and_sets_adapters(tmp_path):
    lora_a = tmp_path / "a.safetensors"
    lora_b = tmp_path / "b.safetensors"
    lora_a.write_bytes(b"fake")
    lora_b.write_bytes(b"fake")

    pipeline = _FakePipeline()
    load_loras(pipeline, [f"{lora_a}:0.5", str(lora_b)])

    assert len(pipeline.lora_calls) == 2
    assert pipeline.lora_calls[0]["adapter_name"] == "lora_0"
    assert pipeline.lora_calls[0]["weight_name"] == lora_a.name
    assert pipeline.lora_calls[0]["path"] == str(lora_a.parent)
    assert pipeline.lora_calls[1]["adapter_name"] == "lora_1"

    assert pipeline.adapter_calls == [(["lora_0", "lora_1"], [0.5, 0.8])]

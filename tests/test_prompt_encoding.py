"""Tests for SDXL prompt encoding (compel mocked, no real model needed)."""

import torch

from image_gen.prompt_encoding import SDXLPromptEncoder, suppress_output


class _FakePipeline:
    """Minimal stand-in: Compel is mocked, so these attrs are never really used."""

    tokenizer = object()
    tokenizer_2 = object()
    text_encoder = object()
    text_encoder_2 = object()


class _FakeCompel:
    """Stand-in for compel.Compel: returns deterministic embeddings per prompt."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls: list[str] = []

    def __call__(self, prompt: str):
        self.calls.append(prompt)
        seq_len = 5 if prompt else 2  # empty negative prompt -> shorter sequence
        conditioning = torch.ones(1, seq_len, 8) * (len(prompt) + 1)
        pooled = torch.ones(1, 8)
        return conditioning, pooled


def _fake_compel_module(monkeypatch):
    fake_module = type("FakeCompelModule", (), {})()
    fake_module.Compel = _FakeCompel
    fake_module.ReturnedEmbeddingsType = type(
        "FakeReturnedEmbeddingsType", (), {"PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED": "fake-enum"}
    )
    import sys

    monkeypatch.setitem(sys.modules, "compel", fake_module)


def test_suppress_output_hides_stdout_stderr(capsys):
    with suppress_output():
        print("should not appear")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_encoder_init_builds_compel_with_both_tokenizers_and_encoders(monkeypatch):
    _fake_compel_module(monkeypatch)
    encoder = SDXLPromptEncoder(_FakePipeline())

    assert encoder.compel.init_kwargs["tokenizer"] == [_FakePipeline.tokenizer, _FakePipeline.tokenizer_2]
    assert encoder.compel.init_kwargs["text_encoder"] == [_FakePipeline.text_encoder, _FakePipeline.text_encoder_2]
    assert encoder.compel.init_kwargs["requires_pooled"] == [False, True]
    assert encoder.compel.init_kwargs["truncate_long_prompts"] is False


def test_pad_embeddings_noop_when_already_equal_length(monkeypatch):
    _fake_compel_module(monkeypatch)
    encoder = SDXLPromptEncoder(_FakePipeline())

    a = torch.zeros(1, 4, 8)
    b = torch.ones(1, 4, 8)
    padded_a, padded_b = encoder._pad_embeddings(a, b)
    assert padded_a is a
    assert padded_b is b


def test_pad_embeddings_pads_shorter_prompt_side():
    encoder = SDXLPromptEncoder.__new__(SDXLPromptEncoder)  # skip __init__ (no compel needed here)
    prompt_embeds = torch.zeros(1, 3, 8)
    negative_embeds = torch.ones(1, 5, 8)

    padded_prompt, padded_negative = encoder._pad_embeddings(prompt_embeds, negative_embeds)

    assert padded_prompt.shape == (1, 5, 8)
    assert padded_negative.shape == (1, 5, 8)
    # Padding repeats the last real token, not zeros.
    assert torch.equal(padded_prompt[:, 3:, :], prompt_embeds[:, -1:, :].repeat(1, 2, 1))


def test_pad_embeddings_pads_shorter_negative_side():
    encoder = SDXLPromptEncoder.__new__(SDXLPromptEncoder)
    prompt_embeds = torch.zeros(1, 6, 8)
    negative_embeds = torch.ones(1, 2, 8)

    padded_prompt, padded_negative = encoder._pad_embeddings(prompt_embeds, negative_embeds)

    assert padded_prompt.shape == (1, 6, 8)
    assert padded_negative.shape == (1, 6, 8)
    assert torch.equal(padded_negative[:, 2:, :], negative_embeds[:, -1:, :].repeat(1, 4, 1))


def test_encode_uses_empty_string_when_no_negative_prompt(monkeypatch):
    _fake_compel_module(monkeypatch)
    encoder = SDXLPromptEncoder(_FakePipeline())

    encoder.encode("a cat")

    assert encoder.compel.calls == ["a cat", ""]


def test_encode_uses_given_negative_prompt(monkeypatch):
    _fake_compel_module(monkeypatch)
    encoder = SDXLPromptEncoder(_FakePipeline())

    encoder.encode("a cat", "blurry")

    assert encoder.compel.calls == ["a cat", "blurry"]


def test_encode_pads_and_returns_four_tensors(monkeypatch):
    _fake_compel_module(monkeypatch)
    encoder = SDXLPromptEncoder(_FakePipeline())

    prompt_embeds, negative_embeds, pooled, negative_pooled = encoder.encode("a cat", "b")

    # "a cat" (5 chars) -> seq_len 5; "b" (1 char) -> seq_len 5 too by the fake's
    # rule, so no padding is actually exercised here beyond equal-length passthrough.
    assert prompt_embeds.shape[1] == negative_embeds.shape[1]
    assert pooled.shape == (1, 8)
    assert negative_pooled.shape == (1, 8)


def test_get_embeddings_for_pipeline_returns_expected_keys(monkeypatch):
    _fake_compel_module(monkeypatch)
    encoder = SDXLPromptEncoder(_FakePipeline())

    result = encoder.get_embeddings_for_pipeline("a cat", "blurry")

    assert set(result) == {
        "prompt_embeds",
        "negative_prompt_embeds",
        "pooled_prompt_embeds",
        "negative_pooled_prompt_embeds",
    }
    assert result["prompt_embeds"].shape == result["negative_prompt_embeds"].shape

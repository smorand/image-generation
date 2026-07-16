"""Tests for chained long-video generation (no model, fake segment generator)."""

import pytest

from video_gen.chain import build_prompt_series, generate_chain, stitch_segments


# --------------------------------------------------------------------------- #
# build_prompt_series
# --------------------------------------------------------------------------- #
def test_prompt_series_one_per_prompt_by_default():
    assert build_prompt_series(["a", "b", "c"], None) == ["a", "b", "c"]


def test_prompt_series_pads_with_last_prompt():
    assert build_prompt_series(["a", "b"], 4) == ["a", "b", "b", "b"]


def test_prompt_series_truncates_when_segments_fewer():
    assert build_prompt_series(["a", "b", "c"], 2) == ["a", "b"]


def test_prompt_series_single_prompt_repeated():
    assert build_prompt_series(["only"], 3) == ["only", "only", "only"]


def test_prompt_series_rejects_empty():
    with pytest.raises(ValueError):
        build_prompt_series([], None)


def test_prompt_series_rejects_zero_segments():
    with pytest.raises(ValueError):
        build_prompt_series(["a"], 0)


# --------------------------------------------------------------------------- #
# stitch_segments
# --------------------------------------------------------------------------- #
def test_stitch_single_segment_unchanged():
    assert stitch_segments([[1, 2, 3]]) == [1, 2, 3]


def test_stitch_drops_duplicated_seam_frame():
    # Each later segment opens on the previous last frame (3, then 5) -> dropped.
    segs = [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
    assert stitch_segments(segs) == [1, 2, 3, 4, 5, 6, 7]


def test_stitch_empty():
    assert stitch_segments([]) == []


# --------------------------------------------------------------------------- #
# generate_chain
# --------------------------------------------------------------------------- #
def test_generate_chain_feeds_last_frame_forward_and_stitches():
    seen_inputs = []
    seen_seeds = []

    def fake_generate(prompt, image, seed):
        seen_inputs.append(image)
        seen_seeds.append(seed)
        # 3 frames per segment tagged with the prompt so we can trace them.
        return [f"{prompt}#0", f"{prompt}#1", f"{prompt}#2"]

    freed = {"n": 0}
    progress = []
    frames = generate_chain(
        fake_generate,
        ["A", "B", "C"],
        start_image="SEED_IMG",
        base_seed=100,
        on_segment=lambda i, total, p, s: progress.append((i, total, p, s)),
        free_memory=lambda: freed.__setitem__("n", freed["n"] + 1),
    )

    # Segment 0 conditioned on the start image; each next on the previous last frame.
    assert seen_inputs == ["SEED_IMG", "A#2", "B#2"]
    # Seeds are base + index.
    assert seen_seeds == [100, 101, 102]
    # Stitched: first segment whole, later segments drop their seam frame.
    assert frames == ["A#0", "A#1", "A#2", "B#1", "B#2", "C#1", "C#2"]
    # Memory freed once per segment; progress reported once per segment.
    assert freed["n"] == 3
    assert progress == [(0, 3, "A", 100), (1, 3, "B", 101), (2, 3, "C", 102)]


def test_generate_chain_raises_on_empty_segment():
    def empty_generate(prompt, image, seed):
        return []

    with pytest.raises(RuntimeError):
        generate_chain(empty_generate, ["A"], start_image="X", base_seed=0)


def test_generate_chain_works_without_optional_callbacks():
    def fake_generate(prompt, image, seed):
        return [f"{prompt}a", f"{prompt}b"]

    frames = generate_chain(fake_generate, ["P", "Q"], start_image="i", base_seed=7)
    assert frames == ["Pa", "Pb", "Qb"]

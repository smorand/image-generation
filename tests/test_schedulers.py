"""Tests for scheduler resolution (no model needed)."""

import pytest

from image_gen.schedulers import SCHEDULER_MAPPING, SUPPORTED_SCHEDULERS, get_scheduler


def test_supported_schedulers_matches_mapping():
    assert set(SUPPORTED_SCHEDULERS) == set(SCHEDULER_MAPPING)


def test_get_scheduler_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown scheduler"):
        get_scheduler("not-a-scheduler", {})


@pytest.mark.parametrize("name", SUPPORTED_SCHEDULERS)
def test_get_scheduler_builds_each_supported_scheduler(name):
    scheduler_class = SCHEDULER_MAPPING[name][0]
    # A real, minimal scheduler config (every diffusers scheduler accepts this).
    base_config = scheduler_class().config
    scheduler = get_scheduler(name, base_config)
    assert isinstance(scheduler, scheduler_class)

"""Config contracts."""

import dataclasses

import pytest

from camera.server import DEFAULT_OVERLAY_CONFIG
from config import AppConfig, load_config, validate_perception_colors


def test_config_loads_overrides_and_rejects_unknown_key(tmp_path):
    cfg = load_config(overrides=["fsm.time_budget_s=180", "robot.cameras={}"])
    assert cfg.fsm.time_budget_s == 180.0
    assert cfg.robot.cameras == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("fsm:\n  typo_key: 1\n")
    with pytest.raises(ValueError, match="typo_key"):
        load_config(bad)


def test_default_yaml_matches_dataclass_defaults():
    """The shipped YAML and the dataclasses duplicate every tuning value.

    Both are read at startup by different processes (camera overlay vs task
    runner), so silent drift between them makes the operator page disagree
    with what the robot actually sees.
    """
    assert dataclasses.asdict(
        load_config(DEFAULT_OVERLAY_CONFIG)
    ) == dataclasses.asdict(AppConfig())


def test_default_agent_uses_openai_luna_with_gemini_fallback():
    agent = AppConfig().agent
    assert agent.provider == "openai"
    assert agent.models["openai"] == "gpt-5.6-luna"
    assert agent.fallback_provider == "gemini"


def test_every_gated_colour_must_have_a_prototype(tmp_path):
    """Gates may overlap — they must, since no fixed box separates wood from
    yellow in every arrangement. What must not exist is a colour that can be
    gated but never named: its blobs would compete for other colours' slots."""
    bad = tmp_path / "unnamed.yaml"
    bad.write_text(
        "perception:\n  hsv_ranges:\n    teal: [[80, 60, 60, 95, 255, 255]]\n"
    )
    with pytest.raises(ValueError, match="color_prototypes is missing"):
        load_config(bad)

    # the shipped palette is complete, and its gates are deliberately overlapping
    load_config(DEFAULT_OVERLAY_CONFIG)
    cfg = AppConfig().perception
    validate_perception_colors(cfg)
    wood, yellow = cfg.hsv_ranges["wood"][0], cfg.hsv_ranges["yellow"][0]
    assert wood[3] > yellow[0], "wood and yellow gates are expected to overlap in hue"

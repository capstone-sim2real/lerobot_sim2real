"""Config contracts."""

import dataclasses

import pytest

from camera.server import DEFAULT_OVERLAY_CONFIG
from config import AppConfig, load_config, validate_ik, validate_perception_colors


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

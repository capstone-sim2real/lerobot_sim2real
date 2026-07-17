from pathlib import Path

import pytest

from pick_stack.config import AppConfig, apply_override, load_config

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "src" / "pick_stack" / "configs" / "default.yaml"


def test_defaults_without_yaml():
    cfg = load_config()
    assert cfg.fsm.num_blocks == 5
    assert cfg.fsm.time_budget_s == 300.0
    assert set(cfg.robot.cameras) == {"top", "wrist"}


def test_default_yaml_matches_dataclass_defaults():
    assert load_config(DEFAULT_YAML).to_dict() == AppConfig().to_dict()


def test_yaml_unknown_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("fsm:\n  num_blocks: 5\n  typo_key: 1\n")
    with pytest.raises(ValueError, match="typo_key"):
        load_config(bad)


def test_override_scalars():
    cfg = load_config(overrides=["fsm.time_budget_s=240", "fsm.num_blocks=3", "logging.save_transitions=false"])
    assert cfg.fsm.time_budget_s == 240.0
    assert isinstance(cfg.fsm.time_budget_s, float)
    assert cfg.fsm.num_blocks == 3
    assert cfg.logging.save_transitions is False


def test_override_string_and_dict():
    cfg = load_config(overrides=["robot.port=/dev/ttyACM0"])
    assert cfg.robot.port == "/dev/ttyACM0"
    cfg = load_config(overrides=['robot.cameras={"top": {"index_or_path": 4}}'])
    assert cfg.robot.cameras == {"top": {"index_or_path": 4}}


@pytest.mark.parametrize(
    "override",
    [
        "fsm.nope=1",  # unknown leaf
        "nope.time_budget_s=1",  # unknown group
        "fsm=1",  # cannot replace a group
        "fsm.num_blocks=hello",  # type mismatch
        "fsm.num_blocks",  # missing '='
        "logging.save_transitions=3",  # bool expected
    ],
)
def test_bad_overrides_rejected(override):
    with pytest.raises(ValueError):
        apply_override(AppConfig(), override)

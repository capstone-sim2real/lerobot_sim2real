"""Configuration tree for pick_stack.

Every tunable lives here as a dataclass field with a default, and can be
overridden by a YAML file (``configs/default.yaml``) and/or CLI ``--set``
key=value overrides. Later PRs add their own config groups (perception,
policy, motion, ...) as new dataclasses wired into ``AppConfig``.

Usage:
    cfg = load_config(Path("src/pick_stack/configs/default.yaml"),
                      overrides=["fsm.time_budget_s=240"])
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class RobotIOConfig:
    """Follower arm + cameras. Defaults match the team's Orin wiring."""

    port: str = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00"
    id: str = "my_follower"
    max_relative_target: float = 10.0
    disable_torque_on_disconnect: bool = True
    # name -> OpenCVCameraConfig kwargs; physical top camera must map to key
    # "top" and wrist camera to key "wrist" (dataset convention, EPISODE.md).
    cameras: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "top": {"index_or_path": "/dev/video0", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"},
            "wrist": {"index_or_path": "/dev/video2", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"},
        }
    )


@dataclass
class FsmConfig:
    num_blocks: int = 5
    time_budget_s: float = 300.0
    # attempts per selected target before it is skipped for the run
    max_retries_per_block: int = 2
    # safety margin: force DONE when remaining budget drops below this
    reserve_time_s: float = 10.0


@dataclass
class LoggingConfig:
    log_dir: str = "logs/pick_stack"
    # CSV of state transitions per run, named <run_id>_transitions.csv
    save_transitions: bool = True


@dataclass
class AppConfig:
    robot: RobotIOConfig = field(default_factory=RobotIOConfig)
    fsm: FsmConfig = field(default_factory=FsmConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build_dataclass(cls: type, data: dict[str, Any], path: str) -> Any:
    """Recursively build a dataclass from a dict, rejecting unknown keys."""
    hints = get_type_hints(cls)
    valid = {f.name: hints[f.name] for f in fields(cls)}
    unknown = set(data) - set(valid)
    if unknown:
        raise ValueError(f"Unknown config key(s) at '{path}': {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        ftype = valid[name]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _build_dataclass(ftype, value, f"{path}.{name}" if path else name)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(yaml_path: Path | str | None = None, overrides: list[str] | None = None) -> AppConfig:
    """Build AppConfig from defaults, then YAML, then ``key.path=value`` overrides."""
    data: dict[str, Any] = {}
    if yaml_path is not None:
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
    cfg = _build_dataclass(AppConfig, data, path="")
    for override in overrides or []:
        apply_override(cfg, override)
    return cfg


def apply_override(cfg: AppConfig, override: str) -> None:
    """Apply one ``a.b.c=value`` override in place.

    The value is parsed with YAML semantics (so ``true``, ``3.5``, ``[1,2]``
    work), then must match the existing field's container/scalar kind.
    """
    if "=" not in override:
        raise ValueError(f"Override must look like key.path=value, got: {override!r}")
    key_path, raw_value = override.split("=", 1)
    keys = key_path.strip().split(".")
    target: Any = cfg
    for key in keys[:-1]:
        if not hasattr(target, key):
            raise ValueError(f"Unknown config group '{key}' in override {override!r}")
        target = getattr(target, key)
    leaf = keys[-1]
    if not (is_dataclass(target) and hasattr(target, leaf)):
        raise ValueError(f"Unknown config key '{key_path}' in override {override!r}")
    current = getattr(target, leaf)
    if is_dataclass(current):
        raise ValueError(f"Cannot override config group '{key_path}' directly; set its leaf keys")
    value = yaml.safe_load(raw_value)
    if current is not None and value is not None:
        if isinstance(current, bool) != isinstance(value, bool):
            raise ValueError(f"Override {override!r}: expected bool, got {type(value).__name__}")
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Override {override!r}: expected number, got {type(value).__name__}")
            value = type(current)(value)
        elif not isinstance(value, type(current)):
            raise ValueError(
                f"Override {override!r}: expected {type(current).__name__}, got {type(value).__name__}"
            )
    setattr(target, leaf, value)

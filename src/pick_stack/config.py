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
class SensingConfig:
    """Grasp verification + contact detection thresholds.

    All load values are lerobot's decoded Present_Load (signed int, sign =
    direction). Defaults are placeholders — measure real distributions with
    tools/tune_gripper_load.py before trusting them.
    """

    # gripper commands (normalized RANGE_0_100). The open width must match
    # the teleop recording convention (EPISODE.md fixes it per dataset).
    gripper_open_pos: float = 50.0
    gripper_close_pos: float = 2.0
    # grasp check, primary signal: a held 20 mm block stops the gripper well
    # above the empty-hand closed position
    gripper_empty_closed_max: float = 6.0
    # grasp check, secondary signal: sustained |Present_Load| on the gripper
    gripper_load_min: float = 120.0
    # position_only | load_only | position_and_load | position_or_load
    grasp_check_mode: str = "position_and_load"
    # let the close settle before sampling
    grasp_settle_s: float = 0.4
    grasp_samples: int = 5
    sample_interval_s: float = 0.05
    # contact detection during stack descent: |load - baseline| spike on any
    # of these joints. Contact can *reduce* load (surface takes the gravity
    # torque), hence the absolute delta.
    contact_joints: list[str] = field(default_factory=lambda: ["shoulder_lift", "elbow_flex"])
    contact_load_delta: float = 80.0
    contact_baseline_samples: int = 5


@dataclass
class MotionConfig:
    """Scripted motion (TRANSPORT / PLACE / STACK). All joint values are in
    the robot's action units (normalized; gripper 0-100) — poses recorded
    with tools/record_pose.py are stored in the same units, so they become
    invalid after recalibration and must be re-recorded."""

    poses_path: str = "src/pick_stack/configs/poses.yaml"
    fps: float = 30.0
    # per-tick joint delta cap for interpolation (action units); the robot's
    # own max_relative_target clamp stays on as a second net
    max_step_per_tick: float = 2.0
    # slower cap while descending onto the tower (contact must be gentle)
    descent_step_per_tick: float = 0.6
    # a move counts as arrived when every joint is within this tolerance
    arrival_tol: float = 3.0
    move_timeout_s: float = 10.0
    # pause after open/close commands before moving on
    gripper_action_wait_s: float = 0.6
    # pose names (must exist in poses.yaml)
    home_pose: str = "home"
    retreat_pose: str = "retreat"
    transport_waypoints: list[str] = field(default_factory=lambda: ["zone_approach"])
    # Task 1: slot i is used for the (i+1)-th placed block
    slot_poses: list[str] = field(default_factory=lambda: ["slot_0", "slot_1", "slot_2", "slot_3", "slot_4"])
    # Task 2: approach above the tower, then descend along the ladder
    tower_approach_pose: str = "tower_approach"
    tower_ladder_prefix: str = "tower_descent"
    # ticks to reverse after contact before releasing (0 = release in place)
    contact_backoff_ticks: int = 1
    place_settle_s: float = 0.5


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
    sensing: SensingConfig = field(default_factory=SensingConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
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

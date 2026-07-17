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
class PerceptionConfig:
    """Top-down camera perception. Metric values are in the board frame (mm)
    defined by the venue calibration JSON (tools/calibrate_homography.py)."""

    # venue calibration produced by tools/calibrate_homography.py
    calibration_path: str = "src/pick_stack/configs/calib/venue_default.json"
    top_camera_key: str = "top"
    # chessboard square edge length on the physical board — measure it!
    square_mm: float = 25.0
    # minimal inner-corner grid to search for; CALIB_CB_LARGER extends it,
    # so partial board views (tight framing) still calibrate
    min_pattern: list[int] = field(default_factory=lambda: [5, 5])
    # rectified top-down view scale used by the detector
    rectified_mm_per_px: float = 1.0
    # colour -> list of HSV bands [h_lo, s_lo, v_lo, h_hi, s_hi, v_hi]
    # (OpenCV hue 0-179; red wraps around, hence two bands).
    # NOTE: tuned on synthetic fixtures only — re-tune on real frames with
    # tools/view_detect.py before trusting them.
    hsv_ranges: dict[str, list[list[int]]] = field(
        default_factory=lambda: {
            "red": [[0, 90, 60, 8, 255, 255], [172, 90, 60, 179, 255, 255]],
            "yellow": [[26, 90, 80, 34, 255, 255]],
            "green": [[40, 60, 50, 85, 255, 255]],
            "blue": [[95, 90, 50, 130, 255, 255]],
            "wood": [[10, 40, 80, 25, 180, 255]],
        }
    )
    # block top face is 40x40 mm = 1600 mm^2; allow perspective/mask slack
    area_mm2_min: float = 900.0
    area_mm2_max: float = 2600.0
    # geometry filters that reject tape: elongated / hollow / sparse shapes
    aspect_ratio_max: float = 1.6
    solidity_min: float = 0.85
    fill_min: float = 0.65
    morph_kernel_px: int = 5


@dataclass
class SelectConfig:
    # deterministic rule; must match the teleop demonstration convention
    rule: str = "nearest_first"
    # blocks within this margin of the zone polygon count as "already placed"
    zone_margin_mm: float = 20.0
    # quantization cell for stable target ids across re-detections
    target_cell_mm: float = 40.0


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
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
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

"""Configuration tree: dataclass defaults, then ``configs/*.yaml``, then ``--set`` overrides.

    cfg = load_config("src/configs/default.yaml", overrides=["fsm.time_budget_s=240"])

Unknown YAML keys are rejected at load time. Importing this package must not
require lerobot, placo or torch.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from config.agent import (
    WorkspaceBoundaryConfig,
    CameraOverlayConfig,
    CameraConfig,
    CalibrationCaptureConfig,
    SessionToolsConfig,
    ZoneSlotNamesConfig,
    RelativeMotionConfig,
    TableRegionsConfig,
    PlaceCorrectionConfig,
    BoardGridConfig,
    AgentCameraViewConfig,
    CalibrationClearanceConfig,
    PrimitiveConfig,
    AgentCollectionConfig,
    AgentConfig,
    YoloeConfig,
)
from config.robot import (
    RobotIOConfig,
    PerceptionConfig,
    SelectConfig,
    SensingConfig,
    MotionConfig,
    IkConfig,
)
from config.tasks import (
    FsmConfig,
    Task1Config,
    Task2Config,
    Task3Config,
    LoggingConfig,
)
from config.validate import (
    validate_yoloe,
    validate_perception_colors,
    validate_ik,
    validate_task1,
    validate_task2,
    validate_task3,
    AGENT_PROVIDERS,
    normalise_place_name,
    validate_agent,
)



@dataclass
class AppConfig:
    yoloe: YoloeConfig = field(default_factory=YoloeConfig)
    robot: RobotIOConfig = field(default_factory=RobotIOConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    sensing: SensingConfig = field(default_factory=SensingConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    ik: IkConfig = field(default_factory=IkConfig)
    fsm: FsmConfig = field(default_factory=FsmConfig)
    task1: Task1Config = field(default_factory=Task1Config)
    task2: Task2Config = field(default_factory=Task2Config)
    task3: Task3Config = field(default_factory=Task3Config)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    calibration_capture: CalibrationCaptureConfig = field(
        default_factory=CalibrationCaptureConfig
    )
    session_tools: SessionToolsConfig = field(default_factory=SessionToolsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

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
            kwargs[name] = _build_dataclass(
                ftype, value, f"{path}.{name}" if path else name
            )
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(
    yaml_path: Path | str | None = None, overrides: list[str] | None = None
) -> AppConfig:
    """Build AppConfig from defaults, then YAML, then ``key.path=value`` overrides."""
    data: dict[str, Any] = {}
    if yaml_path is not None:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    cfg = _build_dataclass(AppConfig, data, path="")
    for override in overrides or []:
        apply_override(cfg, override)
    validate_perception_colors(cfg.perception)
    validate_ik(cfg.ik)
    validate_task1(cfg)
    validate_task2(cfg)
    validate_task3(cfg)
    validate_agent(cfg)
    validate_yoloe(cfg)
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
        raise ValueError(
            f"Cannot override config group '{key_path}' directly; set its leaf keys"
        )
    value = yaml.safe_load(raw_value)
    if current is not None and value is not None:
        if isinstance(current, bool) != isinstance(value, bool):
            raise ValueError(
                f"Override {override!r}: expected bool, got {type(value).__name__}"
            )
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"Override {override!r}: expected number, got {type(value).__name__}"
                )
            value = type(current)(value)
        elif not isinstance(value, type(current)):
            raise ValueError(
                f"Override {override!r}: expected {type(current).__name__}, got {type(value).__name__}"
            )
    setattr(target, leaf, value)


__all__ = [
    "AGENT_PROVIDERS",
    "AgentCameraViewConfig",
    "AgentCollectionConfig",
    "AgentConfig",
    "AppConfig",
    "BoardGridConfig",
    "CalibrationCaptureConfig",
    "CalibrationClearanceConfig",
    "CameraConfig",
    "CameraOverlayConfig",
    "FsmConfig",
    "IkConfig",
    "LoggingConfig",
    "MotionConfig",
    "PerceptionConfig",
    "PlaceCorrectionConfig",
    "PrimitiveConfig",
    "RelativeMotionConfig",
    "RobotIOConfig",
    "SelectConfig",
    "SensingConfig",
    "SessionToolsConfig",
    "TableRegionsConfig",
    "Task1Config",
    "Task2Config",
    "Task3Config",
    "WorkspaceBoundaryConfig",
    "YoloeConfig",
    "ZoneSlotNamesConfig",
    "apply_override",
    "load_config",
    "normalise_place_name",
    "validate_agent",
    "validate_ik",
    "validate_perception_colors",
    "validate_task1",
    "validate_task2",
    "validate_task3",
    "validate_yoloe",
]

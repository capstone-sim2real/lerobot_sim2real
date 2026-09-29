"""Named joint poses (poses.yaml), recorded on the arm with tools.hardware.record_pose.

Values are in the robot's normalized action units, so a motor recalibration
invalidates every recorded pose.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from control.robot_io import JOINT_NAMES

Pose = dict[str, float]


class PoseRegistry:
    def __init__(self, poses: dict[str, Pose] | None = None, path: Path | str | None = None):
        self._poses: dict[str, Pose] = dict(poses or {})
        self._path = Path(path) if path is not None else None

    @classmethod
    def load(cls, path: Path | str) -> "PoseRegistry":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        poses = data.get("poses") or {}
        for name, pose in poses.items():
            missing = set(JOINT_NAMES) - set(pose)
            if missing:
                raise ValueError(f"Pose '{name}' is missing joint(s): {sorted(missing)}")
        return cls({name: {j: float(v) for j, v in pose.items()} for name, pose in poses.items()}, path)

    def save(self, path: Path | str | None = None) -> None:
        target = Path(path) if path is not None else self._path
        if target is None:
            raise ValueError("No path given and registry was not loaded from a file")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"poses": {name: {j: round(float(v), 3) for j, v in pose.items()} for name, pose in sorted(self._poses.items())}}
        with open(target, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    def get(self, name: str) -> Pose:
        if name not in self._poses:
            raise KeyError(
                f"Pose '{name}' not recorded (have: {sorted(self._poses)}). "
                f"Record it with: python -m tools.record_pose --name {name}"
            )
        return dict(self._poses[name])

    def set(self, name: str, pose: Pose) -> None:
        missing = set(JOINT_NAMES) - set(pose)
        if missing:
            raise ValueError(f"Pose '{name}' is missing joint(s): {sorted(missing)}")
        self._poses[name] = {j: float(pose[j]) for j in JOINT_NAMES}

    def names(self) -> list[str]:
        return sorted(self._poses)

    def __contains__(self, name: str) -> bool:
        return name in self._poses

    def require(self, names: list[str]) -> None:
        missing = [n for n in names if n not in self._poses]
        if missing:
            raise KeyError(
                f"Missing recorded pose(s): {missing}. Record them with tools.hardware.record_pose"
            )

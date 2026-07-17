"""Named joint pose registry, backed by a YAML file.

Poses are recorded on the physical arm with tools/record_pose.py and shared
by everything that needs a fixed pose: the FSM's scripted motion, the PICK
policy's retreat-detection, and the episode-recording convention (home /
retreat must be the *same numbers* during teleop recording and at runtime —
this file is the single source of truth, EPISODE.md §1).

Values are in the robot's action units (normalized; gripper 0-100), so a
recalibration invalidates every recorded pose — re-record after calibrating.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from pick_stack.control.robot_io import JOINT_NAMES

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
                f"Record it with: python -m pick_stack.tools.record_pose --name {name}"
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
                f"Missing recorded pose(s): {missing}. Record them with tools/record_pose.py"
            )

    def ladder(self, prefix: str) -> list[tuple[str, Pose]]:
        """Poses named ``<prefix>_<int>`` sorted by the numeric suffix
        (descent keyframes: _0 highest ... _N lowest)."""
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
        found = []
        for name in self._poses:
            m = pattern.match(name)
            if m:
                found.append((int(m.group(1)), name))
        return [(name, self.get(name)) for _, name in sorted(found)]

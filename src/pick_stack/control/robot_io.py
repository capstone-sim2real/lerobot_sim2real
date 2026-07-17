"""Robot IO abstraction.

The FSM (and every state handler) talks to the arm only through
``BaseRobotIO`` so that:
  - unit tests run against ``MockRobotIO`` with no hardware or lerobot install,
  - the PICK policy client can be handed the same robot the rule-based states
    use (single owner of the serial bus),
  - safety clamping (``max_relative_target``) stays inside lerobot's
    ``send_action`` for every caller.

``So101RobotIO`` imports lerobot lazily: importing pick_stack must never
require lerobot (CI runs without it).
"""

from __future__ import annotations

import abc
from typing import Any

from pick_stack.config import RobotIOConfig

JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class BaseRobotIO(abc.ABC):
    """Minimal interface the FSM needs from the arm."""

    joint_names: tuple[str, ...] = JOINT_NAMES

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool: ...

    @abc.abstractmethod
    def read_joints(self) -> dict[str, float]:
        """Present positions keyed by joint name (no cameras — fast path)."""

    @abc.abstractmethod
    def read_observation(self) -> dict[str, Any]:
        """Full observation: ``<joint>.pos`` floats + one array per camera key."""

    @abc.abstractmethod
    def send_joints(self, positions: dict[str, float]) -> dict[str, float]:
        """Command goal positions; returns what was actually sent (post-clamp)."""

    @abc.abstractmethod
    def read_loads(self) -> dict[str, int]:
        """Raw Present_Load per joint (unnormalized; sign encodes direction)."""


class So101RobotIO(BaseRobotIO):
    """Real SO-101 follower behind the BaseRobotIO interface."""

    def __init__(self, config: RobotIOConfig):
        self._config = config
        self._robot = None

    @property
    def robot(self):
        """Underlying lerobot SO101Follower (for the policy client). Connect first."""
        if self._robot is None:
            raise RuntimeError("Robot is not connected; call connect() first")
        return self._robot

    def connect(self) -> None:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        cameras = {
            name: OpenCVCameraConfig(**kwargs) for name, kwargs in self._config.cameras.items()
        }
        robot_config = SO101FollowerConfig(
            port=self._config.port,
            id=self._config.id,
            max_relative_target=self._config.max_relative_target,
            disable_torque_on_disconnect=self._config.disable_torque_on_disconnect,
            cameras=cameras,
        )
        self._robot = SO101Follower(robot_config)
        self._robot.connect()

    def disconnect(self) -> None:
        if self._robot is not None and self._robot.is_connected:
            self._robot.disconnect()
        self._robot = None

    @property
    def is_connected(self) -> bool:
        return self._robot is not None and self._robot.is_connected

    def read_joints(self) -> dict[str, float]:
        return self.robot.bus.sync_read("Present_Position")

    def read_observation(self) -> dict[str, Any]:
        return self.robot.get_observation()

    def send_joints(self, positions: dict[str, float]) -> dict[str, float]:
        action = {f"{name}.pos": pos for name, pos in positions.items()}
        sent = self.robot.send_action(action)
        return {key.removesuffix(".pos"): value for key, value in sent.items()}

    def read_loads(self) -> dict[str, int]:
        return self.robot.bus.sync_read("Present_Load", normalize=False)


class MockRobotIO(BaseRobotIO):
    """In-memory stand-in for tests: joints teleport to commanded positions."""

    def __init__(self, initial_joints: dict[str, float] | None = None):
        self.joints: dict[str, float] = {name: 0.0 for name in JOINT_NAMES}
        if initial_joints:
            self.joints.update(initial_joints)
        self.loads: dict[str, int] = {name: 0 for name in JOINT_NAMES}
        self.frames: dict[str, Any] = {}
        self.sent_actions: list[dict[str, float]] = []
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def read_joints(self) -> dict[str, float]:
        return dict(self.joints)

    def read_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = {f"{name}.pos": pos for name, pos in self.joints.items()}
        obs.update(self.frames)
        return obs

    def send_joints(self, positions: dict[str, float]) -> dict[str, float]:
        self.sent_actions.append(dict(positions))
        self.joints.update(positions)
        return dict(positions)

    def read_loads(self) -> dict[str, int]:
        return dict(self.loads)

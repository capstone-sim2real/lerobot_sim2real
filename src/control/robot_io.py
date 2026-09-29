"""Robot IO abstraction.

Everything talks to the arm through ``BaseRobotIO`` so that unit tests run
against ``MockRobotIO`` without hardware, and lerobot's ``max_relative_target``
clamp inside ``send_action`` applies to every caller. ``So101RobotIO`` imports
lerobot lazily.
"""

from __future__ import annotations

import abc

from config import RobotIOConfig

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
    def send_joints(self, positions: dict[str, float]) -> dict[str, float]:
        """Command goal positions; returns what was actually sent (post-clamp)."""

    @abc.abstractmethod
    def read_loads(self) -> dict[str, int]:
        """Raw Present_Load per joint (unnormalized; sign encodes direction)."""

    @abc.abstractmethod
    def set_torque(self, enabled: bool) -> None:
        """Enable/disable holding torque on all joints (pose recording uses off)."""


class So101RobotIO(BaseRobotIO):
    """Real SO-101 follower behind the BaseRobotIO interface."""

    def __init__(self, config: RobotIOConfig):
        self._config = config
        self._robot = None

    @property
    def robot(self):
        """Underlying lerobot SO101Follower. Connect first."""
        if self._robot is None:
            raise RuntimeError("Robot is not connected; call connect() first")
        return self._robot

    def connect(self) -> None:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        # No cameras on the robot: camera.server is the only /dev/video* owner.
        robot_config = SO101FollowerConfig(
            port=self._config.port,
            id=self._config.id,
            max_relative_target=self._config.max_relative_target,
            disable_torque_on_disconnect=self._config.disable_torque_on_disconnect,
            cameras={},
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

    def send_joints(self, positions: dict[str, float]) -> dict[str, float]:
        action = {f"{name}.pos": pos for name, pos in positions.items()}
        sent = self.robot.send_action(action)
        return {key.removesuffix(".pos"): value for key, value in sent.items()}

    def read_loads(self) -> dict[str, int]:
        return self.robot.bus.sync_read("Present_Load", normalize=False)

    def set_torque(self, enabled: bool) -> None:
        if enabled:
            self.robot.bus.enable_torque()
        else:
            self.robot.bus.disable_torque()


class MockRobotIO(BaseRobotIO):
    """In-memory stand-in for tests: joints teleport to commanded positions."""

    def __init__(self, initial_joints: dict[str, float] | None = None):
        self.joints: dict[str, float] = {name: 0.0 for name in JOINT_NAMES}
        if initial_joints:
            self.joints.update(initial_joints)
        self.loads: dict[str, int] = {name: 0 for name in JOINT_NAMES}
        self.sent_actions: list[dict[str, float]] = []
        self.torque_enabled = True
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

    def send_joints(self, positions: dict[str, float]) -> dict[str, float]:
        self.sent_actions.append(dict(positions))
        self.joints.update(positions)
        return dict(positions)

    def read_loads(self) -> dict[str, int]:
        return dict(self.loads)

    def set_torque(self, enabled: bool) -> None:
        self.torque_enabled = enabled

"""Named-pose motions shared by every flow: home and gripper open/close."""

from __future__ import annotations

from config import MotionConfig, SensingConfig
from control.poses import PoseRegistry
from control.robot_io import BaseRobotIO
from control.trajectory import TrajectoryPlayer


class MotionController:
    def __init__(
        self,
        robot: BaseRobotIO,
        poses: PoseRegistry,
        motion_cfg: MotionConfig,
        sensing_cfg: SensingConfig,
    ):
        self._robot = robot
        self._poses = poses
        self._cfg = motion_cfg
        self._sensing_cfg = sensing_cfg
        self._player = TrajectoryPlayer(robot, motion_cfg)

    def validate_poses(self, *, required: list[str]) -> None:
        """Fail fast at startup if a required pose was never recorded."""
        self._poses.require(required)

    def go_home(self, *, include_gripper: bool = True) -> None:
        """Return to the recorded home pose.

        The recorded home has a nearly-closed gripper. Callers that open the
        jaws themselves right after homing pass ``include_gripper=False`` to
        avoid a pointless close/open on every retry.
        """
        home = self._poses.get(self._cfg.home_pose)
        if not include_gripper:
            home = {joint: value for joint, value in home.items() if joint != "gripper"}
        self._player.move_to(home)

    def open_gripper(self) -> None:
        self._player.set_gripper(self._sensing_cfg.gripper_open_pos)

    def close_gripper(self) -> None:
        self._player.set_gripper(self._sensing_cfg.gripper_close_pos)

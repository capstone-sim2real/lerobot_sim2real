"""Rule-based motion primitives: TRANSPORT, Task-1 PLACE, Task-2 STACK.

The retreat pose is fixed by the episode convention, so one recorded
waypoint chain serves every block. Stack placement never dead-reckons the
tower height: it descends the keyframe ladder slowly and stops on the
ContactMonitor's load spike (AGENTS.md §5).
"""

from __future__ import annotations

import logging
import time

from config import MotionConfig, SensingConfig
from control.poses import Pose, PoseRegistry
from control.robot_io import BaseRobotIO
from control.sensing import ContactMonitor
from control.trajectory import TrajectoryPlayer, interpolate

logger = logging.getLogger(__name__)


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

    def validate_poses(self, *, task: int | None = None, required: list[str] | None = None) -> None:
        """Fail fast at startup if a required pose was never recorded."""
        if required is not None:
            self._poses.require(required)
            return
        if task is None:
            raise ValueError("task or required poses must be provided")
        required = [self._cfg.home_pose, self._cfg.retreat_pose, *self._cfg.transport_waypoints]
        if task == 1:
            required += self._cfg.slot_poses
        else:
            required += [self._cfg.tower_approach_pose]
            if not self._poses.ladder(self._cfg.tower_ladder_prefix):
                raise KeyError(
                    f"No '{self._cfg.tower_ladder_prefix}_<n>' descent keyframes recorded"
                )
        self._poses.require(required)

    def go_home(self) -> None:
        self._player.move_to(self._poses.get(self._cfg.home_pose))

    def open_gripper(self) -> None:
        self._player.set_gripper(self._sensing_cfg.gripper_open_pos)

    def transport_to_zone(self) -> None:
        """Retreat pose -> above the zone, along the recorded waypoints."""
        self._player.follow([self._poses.get(name) for name in self._cfg.transport_waypoints])

    def place_in_slot(self, slot_index: int) -> None:
        """Task 1: lower into the given slot, release gently, lift back out."""
        if not 0 <= slot_index < len(self._cfg.slot_poses):
            raise IndexError(f"slot_index {slot_index} out of range (have {len(self._cfg.slot_poses)} slots)")
        lift_pose = self._poses.get(self._cfg.transport_waypoints[-1])
        slot_pose = self._poses.get(self._cfg.slot_poses[slot_index])
        self._player.move_to(slot_pose, max_step=self._cfg.descent_step_per_tick)
        if self._cfg.place_settle_s > 0:
            time.sleep(self._cfg.place_settle_s)
        self.open_gripper()
        self._player.move_to(lift_pose)

    def stack_place(self) -> bool:
        """Task 2: descend the keyframe ladder until the held block touches
        the tower top, back off, release. Returns True if contact was seen.

        Falls back to releasing at the ladder bottom when no contact fires
        (better a low drop than crushing into the base plate)."""
        approach = self._poses.get(self._cfg.tower_approach_pose)
        self._player.move_to(approach)

        monitor = ContactMonitor(self._robot, self._sensing_cfg)
        monitor.start()

        ladder = [pose for _, pose in self._poses.ladder(self._cfg.tower_ladder_prefix)]
        contact = False
        executed: list[Pose] = []
        current = self._robot.read_joints()
        for keyframe in ladder:
            for step in interpolate(current, keyframe, self._cfg.descent_step_per_tick):
                self._robot.send_joints(step)
                executed.append(step)
                if self._cfg.fps > 0:
                    time.sleep(1.0 / self._cfg.fps)
                reading = monitor.check()
                if reading.contact:
                    logger.info("Contact at deltas=%s", reading.deltas)
                    contact = True
                    break
            current = self._robot.read_joints()
            if contact:
                break
        if not contact:
            logger.warning("Stack descent hit ladder bottom without a contact spike; releasing anyway")

        if contact and self._cfg.contact_backoff_ticks > 0 and executed:
            backoff_index = max(0, len(executed) - 1 - self._cfg.contact_backoff_ticks)
            self._robot.send_joints(executed[backoff_index])
            if self._cfg.fps > 0:
                time.sleep(1.0 / self._cfg.fps)

        if self._cfg.place_settle_s > 0:
            time.sleep(self._cfg.place_settle_s)
        self.open_gripper()
        self._player.move_to(approach)
        return contact

"""Joint-space interpolation and bounded playback.

Two safety nets stack here: interpolation caps the per-tick delta
(``max_step_per_tick``), and the robot's own ``max_relative_target`` clamp
inside lerobot's send_action stays on. A trajectory always starts from the
*measured* current pose, so the NN-retreat -> scripted-motion handoff cannot
jump even if the policy stopped slightly off-pose (AGENTS.md §11).
"""

from __future__ import annotations

import time

from pick_stack.config import MotionConfig
from pick_stack.control.robot_io import BaseRobotIO
from pick_stack.control.poses import Pose


def interpolate(start: Pose, goal: Pose, max_step: float) -> list[Pose]:
    """Linear joint-space path from start to goal, per-tick delta <= max_step.

    Returns the intermediate ticks including the goal (empty if already there).
    Only joints present in ``goal`` are interpolated; other joints are left
    uncommanded (e.g. gripper stays where it is unless the goal names it).
    """
    if max_step <= 0:
        raise ValueError(f"max_step must be positive, got {max_step}")
    deltas = {j: goal[j] - start[j] for j in goal}
    largest = max(abs(d) for d in deltas.values()) if deltas else 0.0
    if largest == 0.0:
        return []
    n_steps = max(1, int(-(-largest // max_step)))  # ceil
    return [
        {j: start[j] + deltas[j] * (i / n_steps) for j in goal} for i in range(1, n_steps + 1)
    ]


class TrajectoryPlayer:
    """Plays interpolated moves on the robot at a fixed tick rate."""

    def __init__(self, robot: BaseRobotIO, cfg: MotionConfig):
        self._robot = robot
        self._cfg = cfg

    def _tick_sleep(self) -> None:
        if self._cfg.fps > 0:
            time.sleep(1.0 / self._cfg.fps)

    def move_to(self, goal: Pose, *, max_step: float | None = None) -> Pose:
        """Move to goal from the measured current pose; returns the final
        measured pose. Raises TimeoutError if arrival_tol is not reached."""
        max_step = max_step if max_step is not None else self._cfg.max_step_per_tick
        start = self._robot.read_joints()
        deadline = time.monotonic() + self._cfg.move_timeout_s
        for step in interpolate(start, goal, max_step):
            if time.monotonic() > deadline:
                break
            self._robot.send_joints(step)
            self._tick_sleep()
        # settle until within tolerance (the arm lags the command stream)
        while True:
            current = self._robot.read_joints()
            err = max(abs(current[j] - goal[j]) for j in goal)
            if err <= self._cfg.arrival_tol:
                return current
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"move_to did not reach goal within {self._cfg.move_timeout_s}s (max joint error {err:.1f})"
                )
            self._robot.send_joints(goal)
            self._tick_sleep()

    def follow(self, waypoints: list[Pose], *, max_step: float | None = None) -> Pose:
        current: Pose = {}
        for pose in waypoints:
            current = self.move_to(pose, max_step=max_step)
        return current

    def set_gripper(self, position: float) -> None:
        self._robot.send_joints({"gripper": position})
        if self._cfg.gripper_action_wait_s > 0:
            time.sleep(self._cfg.gripper_action_wait_s)

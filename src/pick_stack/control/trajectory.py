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

    def move_to(self, goal: Pose, *, max_step: float | None = None, tol: float | None = None) -> Pose:
        """Move to goal from the measured current pose; returns the final
        measured pose. Raises TimeoutError if the tolerance is not reached.

        ``tol`` defaults to ``arrival_tol``. Carrying a block leaves a
        steady-state offset (gravity holds the joint short of its command),
        so transit moves should pass a looser tolerance than the grasp
        descent — waiting longer does not close that gap."""
        max_step = max_step if max_step is not None else self._cfg.max_step_per_tick
        tol = tol if tol is not None else self._cfg.arrival_tol
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
            if err <= tol:
                return current
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"move_to did not reach goal within {self._cfg.move_timeout_s}s "
                    f"(max joint error {err:.1f}, tol {tol:.1f})"
                )
            self._robot.send_joints(goal)
            self._tick_sleep()

    def descend(
        self,
        goal: Pose,
        *,
        max_step: float | None = None,
        tol: float | None = None,
        settle_s: float | None = None,
    ) -> tuple[Pose, bool]:
        """Descend toward ``goal``; report whether something stopped it short.

        The command stream is byte-for-byte what ``move_to`` sends — no
        sensor read inside the interpolation loop. A per-tick bus read at
        fps=30 is a serial round trip that slows the loop enough to eat the
        whole ``move_timeout_s`` budget, which strands the arm partway down.

        The two differences from ``move_to`` are that it settles against a
        short budget of its own rather than the full timeout, and that
        falling short is *returned* rather than raised: a grasp descent that
        lands on the block instead of beside it is a normal outcome for the
        caller to retry, not an error.

        Returns ``(measured_pose, blocked)``. ``blocked`` is a hint for
        ordering retries — it is never a reason to skip closing the jaws,
        since only closing them establishes whether the block is holdable.
        """
        max_step = max_step if max_step is not None else self._cfg.descent_step_per_tick
        tol = tol if tol is not None else self._cfg.arrival_tol
        settle_s = settle_s if settle_s is not None else self._cfg.descent_settle_s
        start = self._robot.read_joints()
        deadline = time.monotonic() + self._cfg.move_timeout_s
        for step in interpolate(start, goal, max_step):
            if time.monotonic() > deadline:
                break
            self._robot.send_joints(step)
            self._tick_sleep()
        # settle until within tolerance (the arm lags the command stream),
        # but give up quickly: if it is stuck on the block, holding the
        # command against it for the full timeout only leans on the servos.
        settle_deadline = min(time.monotonic() + settle_s, deadline)
        while time.monotonic() <= settle_deadline:
            current = self._robot.read_joints()
            if max(abs(current[j] - goal[j]) for j in goal) <= tol:
                break
            self._robot.send_joints(goal)
            self._tick_sleep()
        current = self._robot.read_joints()
        shortfall = max(abs(current[j] - goal[j]) for j in goal)
        return current, shortfall > self._cfg.descent_blocked_tol

    def follow(self, waypoints: list[Pose], *, max_step: float | None = None) -> Pose:
        current: Pose = {}
        for pose in waypoints:
            current = self.move_to(pose, max_step=max_step)
        return current

    def set_gripper(self, position: float, *, stall_ticks: int = 4, stall_eps: float = 0.3) -> float:
        """Drive the gripper to ``position``, stopping early if it stalls.

        Deliberately neither a single send nor ``move_to``:

        - a single send is capped by the robot's ``max_relative_target``
          clamp, so a full open (2 -> 95) would only move 10 units;
        - ``move_to`` treats not reaching the goal as a TimeoutError, but a
          gripper closing onto a block *cannot* reach the goal — stopping
          short is exactly how ``check_grasp`` recognises a held block
          (AGENTS.md §10).

        So: step toward the target like an interpolated move, and return as
        soon as the measured position stops changing. Returns the final
        measured gripper position.
        """
        current = self._robot.read_joints()["gripper"]
        stalled = 0
        for step in interpolate({"gripper": current}, {"gripper": position}, self._cfg.max_step_per_tick):
            self._robot.send_joints(step)
            self._tick_sleep()
            measured = self._robot.read_joints()["gripper"]
            stalled = stalled + 1 if abs(measured - current) < stall_eps else 0
            current = measured
            if stalled >= stall_ticks:
                break  # jaws are against something (or at a hard stop)
        if self._cfg.gripper_action_wait_s > 0:
            time.sleep(self._cfg.gripper_action_wait_s)
        return self._robot.read_joints()["gripper"]

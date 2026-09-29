"""Joint-space interpolation and bounded playback.

Two safety nets stack here: interpolation caps the per-tick delta
(``max_step_per_tick``), and the robot's own ``max_relative_target`` clamp
inside lerobot's send_action stays on. A trajectory always starts from the
*measured* current pose, so the NN-retreat -> scripted-motion handoff cannot
jump even if the policy stopped slightly off-pose.
"""

from __future__ import annotations

import math
import time

from config import MotionConfig
from control.robot_io import BaseRobotIO
from control.poses import Pose


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

    def move_through(self, waypoints: list[Pose], *, tol: float | None = None,
                     check_progress=None, timeout_s: float | None = None,
                     max_step: float | None = None) -> Pose:
        """Stream a preplanned path, settling only at its final goal.

        Start from feedback once; interpolate subsequent segments from their
        preceding command so waypoint boundaries do not restart the motion.
        All writes retain RobotIO cancellation, recording and target clamps.
        ``check_progress`` may reject measured path deviation after each tick.
        """
        current = self._robot.read_joints()
        if not waypoints:
            return current
        tol = self._cfg.arrival_tol if tol is None else tol
        deadline = time.monotonic() + (self._cfg.move_timeout_s if timeout_s is None else max(0.0, timeout_s))
        previous = current
        step_limit = self._cfg.max_step_per_tick if max_step is None else min(max_step,self._cfg.max_step_per_tick)
        for goal in waypoints:
            for step in interpolate(previous, goal, step_limit):
                if time.monotonic() > deadline:
                    raise TimeoutError("Continuous move deadline reached")
                self._robot.send_joints(step)
                self._tick_sleep()
                if check_progress is not None:
                    check_progress()
            previous = {**previous, **goal}
        goal = waypoints[-1]
        while True:
            current = self._robot.read_joints()
            if max(abs(current[j] - goal[j]) for j in goal) <= tol:
                return current
            if time.monotonic() > deadline:
                raise TimeoutError("Continuous move did not reach final goal")
            self._robot.send_joints(goal)
            self._tick_sleep()
            if check_progress is not None:
                check_progress()

    def settle(self, goal: Pose, *, tol: float, timeout_s: float, check_progress=None) -> tuple[float, bool]:
        """Hold ``goal`` for up to ``timeout_s``, trying to tighten onto ``tol``.

        Unlike ``move_to`` this neither interpolates nor raises: it is the
        "wait a moment longer" step for a pose the arm has already reached
        loosely. ``move_to`` would spend the whole ``move_timeout_s`` on a
        tolerance the servos may simply not have the resolution to hit, which
        is seconds of dead time before every grasp descent.

        Returns ``(residual_error, reached_tol)``.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            err = max(abs(self._robot.read_joints()[j] - goal[j]) for j in goal)
            if err <= tol:
                return err, True
            if time.monotonic() > deadline:
                return err, False
            self._robot.send_joints(goal)
            self._tick_sleep()
            if check_progress is not None:
                check_progress()

    def descend(self, goal: Pose) -> tuple[Pose, bool]:
        """Descend toward ``goal``, and stop the moment the arm stops following.

        Unlike ``move_to``, falling short is *returned* rather than raised: a
        grasp descent that lands on the block instead of beside it is a
        normal outcome for the caller to retry, not an error.

        The descent watches how far the measured pose trails the pose just
        commanded. Without that watch, a gripper that lands on top of a block
        a third of the way down still gets the remaining (deeper) commands,
        and then the settle loop re-sends an unreachable goal for the whole
        ``descent_settle_s`` — seconds of servos leaning on the block, which
        shoves it out of position and binds the arm against it.

        Trailing distance, not per-tick movement, is the signal: a servo
        accelerating at the start of a descent moves little per tick but its
        gap to the command stays bounded, whereas a jammed one's gap only
        grows. (An earlier attempt used a per-tick ``read_loads`` instead;
        that extra bus round trip slowed the loop enough to strand the arm
        partway down. ``read_joints`` is the same cost ``move_to``'s settle
        loop and ``set_gripper`` already pay every tick.)

        Returns ``(measured_pose, blocked)``. ``blocked`` is a hint for
        ordering retries — it is never a reason to skip closing the jaws,
        since only closing them establishes whether the block is holdable.
        """
        max_step = self._cfg.descent_step_per_tick
        tol = self._cfg.arrival_tol
        settle_s = self._cfg.descent_settle_s
        max_lag = self._cfg.descent_max_lag
        start = self._robot.read_joints()
        deadline = time.monotonic() + self._cfg.move_timeout_s
        jammed = False
        for step in interpolate(start, goal, max_step):
            if time.monotonic() > deadline:
                break
            self._robot.send_joints(step)
            self._tick_sleep()
            measured = self._robot.read_joints()
            if max(abs(measured[j] - step[j]) for j in goal) > max_lag:
                jammed = True  # the arm is no longer following: something is in the way
                break
        if not jammed:
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
        return current, jammed or shortfall > self._cfg.descent_blocked_tol

    def set_gripper(self, position: float, *, stall_ticks: int = 4, stall_eps: float = 0.3) -> float:
        """Drive the gripper to ``position``, stopping early if it stalls.

        Deliberately neither a single send nor ``move_to``:

        - a single send is capped by the robot's ``max_relative_target``
          clamp, so a full open (2 -> 95) would only move 10 units;
        - ``move_to`` treats not reaching the goal as a TimeoutError, but a
          gripper closing onto a block *cannot* reach the goal — stopping
          short is exactly how ``check_grasp`` recognises a held block.

        So: step toward the target like an interpolated move, and return as
        soon as the measured position stops changing. Returns the final
        measured gripper position.
        """
        current = self._robot.read_joints()["gripper"]
        opening = position > current
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
            if opening and self._cfg.fps > 0:
                # Keep the opening jaw on regular control ticks until it
                # settles. Cap the hold rate at 30 Hz: collection raises
                # motion.fps to 300 while its recorder paces actual ticks.
                ticks = max(1, math.ceil(self._cfg.gripper_action_wait_s * min(self._cfg.fps, 30.0)))
                for _ in range(ticks):
                    self._robot.send_joints({"gripper": position})
                    self._tick_sleep()
            else:
                time.sleep(self._cfg.gripper_action_wait_s)
        return self._robot.read_joints()["gripper"]

"""CV+IK PICK state: pre-solve grasp attempts for the selected block, run them,
and lift with the gripper closed. VERIFY decides whether the grasp held."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from config import AppConfig
from control.grasp import GraspAttempt, plan_grasp_attempts, run_grasp_attempts
from control.ik import TopDownIK
from control.motion import MotionController
from control.robot_io import BaseRobotIO
from control.trajectory import TrajectoryPlayer
from fsm.handlers import SelectState
from fsm.states import RunContext, State, StateName
from perception.select import SelectionResult

logger = logging.getLogger(__name__)


class CvIkPickState(State):
    """Top-down IK pick. ``grasp_z_mm`` is the calibrated block-top plane."""

    name = StateName.PICK

    def __init__(
        self,
        *,
        robot: BaseRobotIO,
        motion: MotionController,
        cfg: AppConfig,
        grasp_z_mm: float,
        retreat_after_grasp: bool = True,
        radial_tilt_extra_key: str | None = None,
        max_grasp_attempts: int | None = None,
        ik: TopDownIK | None = None,
        player: TrajectoryPlayer | None = None,
        project_root: Path | str = ".",
    ):
        self._robot = robot
        self._motion = motion
        self._cfg = cfg
        self._grasp_z_mm = grasp_z_mm
        self._retreat_after_grasp = retreat_after_grasp
        self._radial_tilt_extra_key = radial_tilt_extra_key
        # None keeps the one rotated retry. Task 3 pins this to 1 so a
        # recorded episode holds one clean grasp attempt or is discarded.
        self._max_grasp_attempts = max_grasp_attempts
        self._ik = ik or TopDownIK(cfg.ik, project_root=project_root)
        self._player = player or TrajectoryPlayer(robot, cfg.motion)

    def _retry_or_skip(self, ctx: RunContext, reason: str) -> StateName:
        self._motion.open_gripper()
        assert ctx.target_id is not None
        if ctx.should_skip(ctx.target_id):
            ctx.skip(ctx.target_id)
            ctx.last_note = f"cv_ik_{reason}_skip"
        else:
            ctx.last_note = f"cv_ik_{reason}_retry"
        return StateName.SELECT

    def _lift(self, held: GraspAttempt) -> None:
        # A successful grasp ends at the low pick pose; lift straight up.
        self._player.move_to(held.hover.joints, tol=self._cfg.motion.transit_arrival_tol)

    def step(self, ctx: RunContext) -> StateName | None:
        selection = ctx.extras.get("selection")
        if not isinstance(selection, SelectionResult) or selection.target is None or ctx.target_id is None:
            ctx.last_note = "cv_ik_missing_selection"
            return StateName.SELECT

        target = selection.target
        ctx.record_attempt(ctx.target_id)
        x_mm, y_mm = target.center_mm
        radial_tilt_deg = (
            float(ctx.extras.get(self._radial_tilt_extra_key, 0.0))
            if self._radial_tilt_extra_key is not None
            else 0.0
        )
        plan = plan_grasp_attempts(
            self._ik,
            self._cfg,
            x_mm,
            y_mm,
            self._grasp_z_mm,
            block_angle_deg=target.angle_deg,
            radial_tilt_deg=radial_tilt_deg,
            log=logger.info,
        )
        ctx.extras["grasp_plan"] = plan

        # run_grasp_attempts skips unreachable entries; give up only if none solve.
        if not any(attempt.reachable for attempt in plan.attempts):
            reach = math.hypot(x_mm, y_mm)
            worst = min(a.grasp.position_error_mm for a in plan.attempts)
            logger.warning(
                "block at x=%.0f y=%.0f is %.0fmm out — TOO FAR for the configured grasp "
                "posture (best IK still misses by %.0fmm). Move it closer to the base "
                "or tune the Task-1 far-reach correction and retry.",
                x_mm, y_mm, reach, worst,
            )
            return self._retry_or_skip(ctx, "unreachable")

        try:
            held = run_grasp_attempts(
                self._player,
                self._robot,
                self._cfg,
                plan,
                max_attempts=self._max_grasp_attempts,
                log=logger.info,
            )
            if held is None:
                return self._retry_or_skip(ctx, "empty")
            ctx.extras["ik_pick_attempt"] = held
            if self._retreat_after_grasp:
                self._lift(held)
        except TimeoutError as exc:
            logger.warning("CV+IK PICK motion timed out: %s", exc)
            return self._retry_or_skip(ctx, "motion_timeout")

        ctx.last_note = f"cv_ik_held_{held.label}"
        return StateName.VERIFY


class CvIkSelectState(SelectState):
    """SELECT that homes without touching the gripper; the pick opens it anyway."""

    def enter(self, ctx: RunContext) -> None:
        self._motion.go_home(include_gripper=False)

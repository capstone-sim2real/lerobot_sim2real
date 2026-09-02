"""CV+IK implementation of the FSM PICK state.

This adapter owns only the pick phase.  It turns the block selected by
``SelectState`` into a pre-solved grasp plan, runs the guarded retry loop,
and retreats while keeping the gripper closed.  VERIFY, TRANSPORT, and PLACE
remain the shared FSM handlers.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from config import AppConfig
from control.grasp import GraspAttempt, plan_grasp_attempts, run_grasp_attempts
from control.ik import TopDownIK
from control.motion import MotionController
from control.poses import Pose
from control.robot_io import BaseRobotIO
from control.trajectory import TrajectoryPlayer
from fsm.handlers import SelectState
from fsm.states import RunContext, State, StateName
from perception.select import SelectionResult

logger = logging.getLogger(__name__)


class CvIkPickState(State):
    """Pick the selected block through deterministic top-down IK.

    ``grasp_z_mm`` comes from the venue calibration metadata: the calibration
    plane is the top of a block, not the table.  The optional IK/player
    injections keep this state unit-testable without placo or hardware.
    """

    name = StateName.PICK

    def __init__(
        self,
        *,
        robot: BaseRobotIO,
        motion: MotionController,
        cfg: AppConfig,
        grasp_z_mm: float,
        retreat_pose: Pose | None,
        retreat_after_grasp: bool = True,
        ik: TopDownIK | None = None,
        player: TrajectoryPlayer | None = None,
        project_root: Path | str = ".",
    ):
        self._robot = robot
        self._motion = motion
        self._cfg = cfg
        self._grasp_z_mm = grasp_z_mm
        self._retreat_pose = retreat_pose
        self._retreat_after_grasp = retreat_after_grasp
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

    def _retreat_with_block(self, held: GraspAttempt) -> None:
        # ``attempt_grasp`` leaves a successful grasp at the low pick pose.
        # Lift vertically first, then follow the pre-recorded retreat using
        # arm joints only: a recorded pose may contain an open gripper value.
        self._player.move_to(held.hover.joints, tol=self._cfg.motion.transit_arrival_tol)
        if self._retreat_pose is None:
            return
        retreat_arm = {joint: value for joint, value in self._retreat_pose.items() if joint != "gripper"}
        if retreat_arm:
            self._player.move_to(
                retreat_arm,
                max_step=1.0,
                tol=self._cfg.motion.transit_arrival_tol,
            )

    def step(self, ctx: RunContext) -> StateName | None:
        selection = ctx.extras.get("selection")
        if not isinstance(selection, SelectionResult) or selection.target is None or ctx.target_id is None:
            ctx.last_note = "cv_ik_missing_selection"
            return StateName.SELECT

        target = selection.target
        ctx.record_attempt(ctx.target_id)
        x_mm, y_mm = target.center_mm
        plan = plan_grasp_attempts(
            self._ik,
            self._cfg,
            x_mm,
            y_mm,
            self._grasp_z_mm,
            block_angle_deg=target.angle_deg,
            log=logger.info,
        )
        ctx.extras["grasp_plan"] = plan

        # The centre point is preferred, but a calibration/IK miss at that
        # exact point must not discard nearby grasp points that are solvable.
        # ``run_grasp_attempts`` filters the unreachable entries itself.
        if not any(attempt.reachable for attempt in plan.attempts):
            reach = math.hypot(x_mm, y_mm)
            worst = min(a.grasp.position_error_mm for a in plan.attempts)
            logger.warning(
                "block at x=%.0f y=%.0f is %.0fmm out — TOO FAR to grasp top-down "
                "(best IK still misses by %.0fmm; the arm's top-down envelope runs "
                "out near 300mm reach). Move it closer to the base and retry.",
                x_mm, y_mm, reach, worst,
            )
            return self._retry_or_skip(ctx, "unreachable")

        try:
            held = run_grasp_attempts(self._player, self._robot, self._cfg, plan, log=logger.info)
            if held is None:
                return self._retry_or_skip(ctx, "empty")
            ctx.extras["ik_pick_attempt"] = held
            if self._retreat_after_grasp:
                self._retreat_with_block(held)
        except TimeoutError as exc:
            logger.warning("CV+IK PICK motion timed out: %s", exc)
            return self._retry_or_skip(ctx, "motion_timeout")

        ctx.last_note = f"cv_ik_held_{held.label}"
        return StateName.VERIFY


class CvIkSelectState(SelectState):
    """SELECT that homes the arm without commanding the jaws.

    The recorded home pose carries a nearly-closed gripper value, so the
    stock SELECT closes the jaws on the way home and the next pick attempt
    immediately reopens them — a visible open/close on every retry that
    achieves nothing. The CV+IK pick opens the jaws itself as its first
    action, so it does not need home to set them.
    """

    def enter(self, ctx: RunContext) -> None:
        self._motion.go_home(include_gripper=False)

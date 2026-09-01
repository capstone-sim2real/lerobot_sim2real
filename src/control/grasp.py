"""Grasp-point planning and multi-attempt execution for the CV+IK pick path.

Kept out of ``tools/demo_pick_and_place.py`` so that ``fsm/ik_handler.py``
can reuse it unchanged (AGENTS.md
§14.1 — CV+IK work is additive).

The accuracy this has to survive is measured, not assumed: RMS ~12mm and
worst ~29mm, and the error is *random per point* rather than a smooth
function of position — ``docs/report/CV_IK_전환_정리.md`` §4 rejected the
interpolation hypothesis explicitly. Recalibrating does not shrink it, so
the response is to try several grasp points inside the +-15mm that the 70mm
jaws tolerate around a 40mm block, rather than to aim harder at one.

Every offset here is expressed in the *gripper's* frame, not the board's
(``ik.gripper_frame_offset``): the arm always aims along base -> target, so
"further out" and "to the left" rotate with it. Near the middle of the board
they coincide with the board axes; toward either side they do not.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from config import AppConfig, MotionConfig
from control.ik import IkResult, TopDownIK, gripper_frame_offset
from control.robot_io import BaseRobotIO
from control.sensing import GraspCheck, check_grasp
from control.trajectory import TrajectoryPlayer

# Names for the default four retry offsets, in the (radial, tangential)
# order they are configured. Radial + is away from the base, tangential + is
# the gripper's left.
_RETRY_LABELS = ("front-left", "front-right", "back-left", "back-right")


class GraspOutcome(Enum):
    HELD = "held"
    EMPTY = "empty"
    BLOCKED = "blocked"


@dataclass
class GraspAttempt:
    """One grasp point, with its two waypoints already solved."""

    label: str
    offset_mm: tuple[float, float]  # (radial, tangential) from the biased centre
    xy_mm: tuple[float, float]
    hover: IkResult
    grasp: IkResult
    reachable: bool


@dataclass
class GraspPlan:
    detected_xy_mm: tuple[float, float]
    biased_xy_mm: tuple[float, float]
    grasp_z_mm: float
    hover_z_mm: float
    attempts: list[GraspAttempt]  # the biased centre, then the diagonals
    lateral: list[GraspAttempt]  # tangential-only, held back for a blocked descent


def highest_reachable_hover(
    ik: TopDownIK,
    x_mm: float,
    y_mm: float,
    base_z_mm: float,
    cfg: AppConfig,
) -> float:
    """Find the highest genuinely reachable top-down hover at this point."""
    z_mm = base_z_mm + cfg.motion.hover_clearance_mm
    floor = base_z_mm + cfg.motion.hover_min_clearance_mm
    while z_mm >= floor:
        # A broad IK gate accepts the calibration error budget.  Hover needs
        # a stricter check: reporting a pose that is 12mm short would make
        # the clearance fictional and can drag a held block.
        if ik.solve(x_mm, y_mm, z_mm).position_error_mm <= 3.0:
            return z_mm
        z_mm -= cfg.motion.hover_search_step_mm
    return floor


def biased_grasp_xy(cfg: MotionConfig, x_mm: float, y_mm: float) -> tuple[float, float]:
    """Where the arm should aim for a block detected at ``(x_mm, y_mm)``.

    The detector reports the block centre, but the jaws were measured closing
    on its near edge, so the aim point is pushed outward. Blocks on the left
    half of the workspace need more of the same plus a tangential nudge —
    which half a block is on is decided from the *detected* position, before
    any bias is applied.
    """
    radial = cfg.grasp_radial_offset_mm
    tangential = cfg.grasp_tangential_offset_mm
    if y_mm > cfg.left_half_y_mm:
        radial += cfg.left_half_radial_offset_mm
        tangential += cfg.left_half_tangential_offset_mm
    return gripper_frame_offset(x_mm, y_mm, radial, tangential)


def _solve_attempt(
    ik: TopDownIK,
    cfg: AppConfig,
    label: str,
    base_xy: tuple[float, float],
    offset: tuple[float, float],
    grasp_z: float,
    hover_z: float,
) -> GraspAttempt:
    x_mm, y_mm = gripper_frame_offset(base_xy[0], base_xy[1], offset[0], offset[1])
    hover = ik.solve(x_mm, y_mm, hover_z)
    grasp = ik.solve(x_mm, y_mm, grasp_z)
    reachable = all(
        r.position_error_mm <= cfg.ik.max_position_error_mm
        and r.tilt_error_deg <= cfg.ik.max_tilt_error_deg
        for r in (hover, grasp)
    )
    return GraspAttempt(
        label=label,
        offset_mm=(float(offset[0]), float(offset[1])),
        xy_mm=(x_mm, y_mm),
        hover=hover,
        grasp=grasp,
        reachable=reachable,
    )


def plan_grasp_attempts(
    ik: TopDownIK,
    cfg: AppConfig,
    x_mm: float,
    y_mm: float,
    grasp_z: float,
    hover_z: float,
) -> GraspPlan:
    """Solve every grasp point that might be tried, before the arm moves.

    All candidates share one ``hover_z``: the offsets are small next to the
    reach envelope, so re-running the hover search per candidate would cost
    many IK solves to land on the same height.
    """
    base_xy = biased_grasp_xy(cfg.motion, x_mm, y_mm)
    attempts = [_solve_attempt(ik, cfg, "centre", base_xy, (0.0, 0.0), grasp_z, hover_z)]
    for i, offset in enumerate(cfg.motion.grasp_retry_offsets_mm):
        label = _RETRY_LABELS[i] if i < len(_RETRY_LABELS) else f"retry-{i}"
        attempts.append(
            _solve_attempt(ik, cfg, label, base_xy, (float(offset[0]), float(offset[1])), grasp_z, hover_z)
        )
    lateral = [
        _solve_attempt(
            ik,
            cfg,
            f"lateral-{'left' if t > 0 else 'right'}",
            base_xy,
            (0.0, float(t)),
            grasp_z,
            hover_z,
        )
        for t in cfg.motion.blocked_descent_offsets_mm
    ]
    return GraspPlan(
        detected_xy_mm=(x_mm, y_mm),
        biased_xy_mm=base_xy,
        grasp_z_mm=grasp_z,
        hover_z_mm=hover_z,
        attempts=attempts,
        lateral=lateral,
    )


def attempt_grasp(
    player: TrajectoryPlayer,
    robot: BaseRobotIO,
    cfg: AppConfig,
    attempt: GraspAttempt,
) -> tuple[GraspOutcome, GraspCheck | None]:
    """One open-descend-close-check cycle, leaving the arm back at hover.

    BLOCKED and EMPTY are both failures that closed the jaws and found
    nothing; BLOCKED adds that the descent also stopped short, which points
    at the lateral position rather than the depth.
    """
    player.set_gripper(cfg.sensing.gripper_open_pos)
    player.move_to(attempt.hover.joints, max_step=1.0, tol=cfg.motion.transit_arrival_tol)
    _, blocked = player.descend(attempt.grasp.joints)

    # Always close, whatever the descent reported. Whether this position can
    # hold the block is only knowable by closing the jaws on it: a descent
    # that stopped a few mm short may still grasp, and check_grasp is the
    # authority (AGENTS.md §10). ``blocked`` only reorders what is tried next.
    player.set_gripper(cfg.sensing.gripper_close_pos)
    # check_grasp settles for grasp_settle_s itself; do not sleep again first.
    check = check_grasp(robot, cfg.sensing)
    if check.grasped:
        return GraspOutcome.HELD, check

    player.set_gripper(cfg.sensing.gripper_open_pos)
    player.move_to(attempt.hover.joints, max_step=1.0, tol=cfg.motion.transit_arrival_tol)
    return (GraspOutcome.BLOCKED if blocked else GraspOutcome.EMPTY), check


def run_grasp_attempts(
    player: TrajectoryPlayer,
    robot: BaseRobotIO,
    cfg: AppConfig,
    plan: GraspPlan,
    *,
    log: Callable[[str], None] = print,
) -> GraspAttempt | None:
    """Work through the planned grasp points until one holds.

    A blocked descent reorders what is left rather than just moving on: it
    says the depth was right, so the tangential-only nudges go to the front
    of the queue ahead of the diagonals. Unreachable candidates are dropped
    here instead of aborting the run — only the first attempt has to clear
    the IK gate for the pick to be worth starting.
    """
    queue = deque(a for a in plan.attempts if a.reachable)
    lateral = [a for a in plan.lateral if a.reachable]
    lateral_used = False
    while queue:
        attempt = queue.popleft()
        log(f"  attempt '{attempt.label}' at x={attempt.xy_mm[0]:.1f} y={attempt.xy_mm[1]:.1f}")
        outcome, check = attempt_grasp(player, robot, cfg, attempt)
        detail = (
            f" pos={check.gripper_pos:.1f} load={check.gripper_load_abs:.0f}" if check else ""
        )
        log(f"    -> {outcome.value.upper()}{detail}")
        if outcome is GraspOutcome.HELD:
            return attempt
        if outcome is GraspOutcome.BLOCKED and not lateral_used and lateral:
            lateral_used = True
            log("    descent was blocked — trying the sideways nudges next")
            queue.extendleft(reversed(lateral))
    return None

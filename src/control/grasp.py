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

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from config import AppConfig, MotionConfig
from control.ik import IkResult, TopDownIK, gripper_frame_offset
from control.robot_io import BaseRobotIO
from control.sensing import GraspCheck, check_grasp
from control.trajectory import TrajectoryPlayer


def _offset_label(radial: float, tangential: float) -> str:
    """Name a retry point from the signs of its offset.

    Derived rather than tabulated: a positional label list silently mislabels
    every point the moment the configured offsets change shape (cardinals
    wearing diagonal names), and the labels are what the log, the camera
    overlay and the dry-run table all report.
    """
    parts = []
    if radial:
        parts.append("front" if radial > 0 else "back")
    if tangential:
        parts.append("left" if tangential > 0 else "right")
    return "-".join(parts) or "centre"


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
    hover_z_mm: float = 0.0
    grasp_z_mm: float = 0.0
    yaw_deg: float | None = None


@dataclass
class GraspPlan:
    detected_xy_mm: tuple[float, float]
    biased_xy_mm: tuple[float, float]
    grasp_z_mm: float
    hover_z_mm: float
    attempts: list[GraspAttempt]  # the biased centre, then the configured retries
    # Radial-bias scale this plan settled on. Anything drawing the aim point
    # must use it, or it shows a point the arm never goes to.
    bias_scale: float = 1.0
    # Jaw yaw the whole plan holds, or None for the neutral (radial) yaw.
    yaw_deg: float | None = None


def highest_reachable_hover(
    ik: TopDownIK,
    x_mm: float,
    y_mm: float,
    base_z_mm: float,
    cfg: AppConfig,
    yaw_deg: float | None = None,
) -> float:
    """Find the highest genuinely reachable top-down hover at this point.

    Pass the same ``yaw_deg`` the grasp will use: the hover must hold the
    jaw plane the descent is about to keep, and an explicit yaw also skips
    the neutral probe solve that ``yaw_deg=None`` runs on every call.
    """
    z_mm = base_z_mm + cfg.motion.hover_clearance_mm
    floor = base_z_mm + cfg.motion.hover_min_clearance_mm
    while z_mm >= floor:
        # A broad IK gate accepts the calibration error budget.  Hover needs
        # a stricter check: reporting a pose that is 12mm short would make
        # the clearance fictional and can drag a held block.
        if ik.solve(x_mm, y_mm, z_mm, yaw_deg=yaw_deg).position_error_mm <= 3.0:
            return z_mm
        z_mm -= cfg.motion.hover_search_step_mm
    return floor


def biased_grasp_xy(
    cfg: MotionConfig, x_mm: float, y_mm: float, *, scale: float = 1.0
) -> tuple[float, float]:
    """Where the arm should aim for a block detected at ``(x_mm, y_mm)``.

    The detector reports the block centre, but the jaws were measured closing
    on its near edge, so the aim point is pushed outward. Blocks on the left
    half of the workspace need more of the same plus a tangential nudge —
    which half a block is on is decided from the *detected* position, before
    any bias is applied.

    ``scale`` shrinks the bias where it would cost reachability — but only
    the RADIAL half of it. Radial is what pushes the aim point past what the
    arm can hover over; a 10mm tangential nudge changes reach by
    ``hypot(300, 10) - 300 = 0.17mm``. Scaling it down buys nothing and
    throws away the whole left-half correction exactly where it is needed,
    since the left half hits the envelope ~10mm of reach sooner precisely
    because it carries the extra radial offset.
    """
    radial = cfg.grasp_radial_offset_mm
    tangential = cfg.grasp_tangential_offset_mm
    if y_mm > cfg.left_half_y_mm:
        radial += cfg.left_half_radial_offset_mm
        tangential += cfg.left_half_tangential_offset_mm
        # Optional ramp on top of the step, growing with distance from the
        # centre line. Default OFF: the 15 calibration points give a
        # tangential-residual/y correlation of +0.017, and a smooth
        # positional correction is hypothesis 4 of the pivot report, which
        # was tested and rejected (LOO 13.99 -> 14.45mm). This is a knob for
        # the hands-on observation that the left gets worse further out, not
        # a model this repo's data supports.
        from_centre = (y_mm - cfg.left_half_y_mm) / 100.0
        radial += cfg.left_ramp_radial_mm_per_100mm * from_centre
        tangential += cfg.left_ramp_tangential_mm_per_100mm * from_centre
    return gripper_frame_offset(x_mm, y_mm, radial * scale, tangential)


def grasp_candidate_points(
    cfg: MotionConfig, x_mm: float, y_mm: float, *, scale: float = 1.0
) -> list[tuple[str, tuple[float, float]]]:
    """Return the board-frame points shown and tried for one detected block.

    The first point is the biased centre; the rest are the configured retry
    offsets.  Keeping this independent of IK lets the camera overlay show
    the exact plan without solving kinematics for every browser refresh.
    """
    base_xy = biased_grasp_xy(cfg, x_mm, y_mm, scale=scale)
    points = [("centre", base_xy)]
    for offset in cfg.grasp_retry_offsets_mm:
        radial, tangential = float(offset[0]), float(offset[1])
        points.append(
            (
                _offset_label(radial, tangential),
                gripper_frame_offset(base_xy[0], base_xy[1], radial, tangential),
            )
        )
    return points


def _solve_attempt(
    ik: TopDownIK,
    cfg: AppConfig,
    label: str,
    base_xy: tuple[float, float],
    offset: tuple[float, float],
    grasp_z: float,
    hover_z: float,
    yaw_deg: float | None = None,
) -> GraspAttempt:
    x_mm, y_mm = gripper_frame_offset(base_xy[0], base_xy[1], offset[0], offset[1])
    hover = ik.solve(x_mm, y_mm, hover_z, yaw_deg=yaw_deg)
    grasp = ik.solve(x_mm, y_mm, grasp_z, yaw_deg=yaw_deg)
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
        hover_z_mm=hover_z,
        grasp_z_mm=grasp_z,
        yaw_deg=yaw_deg,
    )


def _plan_at_scale(
    ik: TopDownIK,
    cfg: AppConfig,
    x_mm: float,
    y_mm: float,
    grasp_z: float,
    scale: float,
    yaw_deg: float | None = None,
) -> GraspPlan:
    base_xy = biased_grasp_xy(cfg.motion, x_mm, y_mm, scale=scale)
    # Search the hover at the aim point, not at the detection: that is where
    # the arm actually holds station, and the envelope shrinks fast with
    # reach, so a height found 12mm further in can be unreachable here.
    hover_z = highest_reachable_hover(ik, *base_xy, grasp_z, cfg, yaw_deg)
    candidate_points = grasp_candidate_points(cfg.motion, x_mm, y_mm, scale=scale)
    attempts = [_solve_attempt(ik, cfg, "centre", base_xy, (0.0, 0.0), grasp_z, hover_z, yaw_deg)]
    for (label, _xy), offset in zip(candidate_points[1:], cfg.motion.grasp_retry_offsets_mm, strict=True):
        attempts.append(
            _solve_attempt(
                ik, cfg, label, base_xy, (float(offset[0]), float(offset[1])), grasp_z, hover_z, yaw_deg
            )
        )
    return GraspPlan(
        detected_xy_mm=(x_mm, y_mm),
        biased_xy_mm=base_xy,
        grasp_z_mm=grasp_z,
        hover_z_mm=hover_z,
        attempts=attempts,
        bias_scale=scale,
        yaw_deg=yaw_deg,
    )


def plan_grasp_attempts(
    ik: TopDownIK,
    cfg: AppConfig,
    x_mm: float,
    y_mm: float,
    grasp_z: float,
    *,
    block_angle_deg: float | None = None,
    log: Callable[[str], None] | None = None,
) -> GraspPlan:
    """Solve every grasp point that might be tried, before the arm moves.

    All candidates share one hover height, searched at the aim point: the
    offsets are small next to the reach envelope, so re-running the search
    per candidate would cost many IK solves to land on the same height.

    The grasp bias pushes the aim point *outward*, so near the edge of the
    workspace it can push a perfectly reachable block past the point where
    the arm can hover at all — measured: a block detected at 288mm reach had
    all five candidates solvable, and the 12mm bias moved it to 300mm where
    only one was. So the bias is backed off rather than surrendering the
    block: full bias first, then half, then none.

    ``block_angle_deg`` turns the jaws to grip two faces of a square block
    instead of two corners. It is routed through ``grasp_yaw_deg``, never
    used as an absolute yaw: the mod-90 fold keeps ``wrist_roll`` within
    +-45 deg of neutral, and it is holding a *fixed base-frame* yaw that
    overheated that servo on 2026-08-31 (AGENTS.md §7).
    """
    yaw_deg = None
    if block_angle_deg is not None:
        # Once, outside the scale loop: grasp_yaw_deg costs a probe solve.
        yaw_deg = ik.grasp_yaw_deg(x_mm, y_mm, grasp_z, block_angle_deg)

    for yaw in ([yaw_deg, None] if yaw_deg is not None else [None]):
        if yaw is None and yaw_deg is not None and log is not None:
            log("  rotated jaws unreachable here — falling back to the neutral yaw")
        plan = None
        for scale in (1.0, 0.5, 0.0):
            plan = _plan_at_scale(ik, cfg, x_mm, y_mm, grasp_z, scale, yaw)
            if plan.attempts[0].reachable:
                if scale < 1.0 and log is not None:
                    log(
                        f"  radial bias reduced to {scale:.0%}: the full bias put the aim "
                        f"point outside the reachable envelope (the sideways bias is kept)"
                    )
                if yaw is not None and log is not None:
                    log(f"  jaws turned to {yaw % 90.0:.0f} deg for the block's faces")
                return plan
    if log is not None:
        log("  no bias setting puts this block in reach")
    return plan


def attempt_grasp(
    player: TrajectoryPlayer,
    robot: BaseRobotIO,
    cfg: AppConfig,
    attempt: GraspAttempt,
    *,
    log: Callable[[str], None] = lambda _message: None,
) -> tuple[GraspOutcome, GraspCheck | None]:
    """One open-descend-close-check cycle, leaving the arm back at hover.

    BLOCKED and EMPTY are both failures that closed the jaws and found
    nothing; BLOCKED adds that the descent also stopped short, which points
    at the lateral position rather than the depth.
    """
    log(f"      open jaws, move to hover z={attempt.hover_z_mm:.0f}mm")
    player.set_gripper(cfg.sensing.gripper_open_pos)
    player.move_to(attempt.hover.joints, max_step=1.0, tol=cfg.motion.transit_arrival_tol)
    log(f"      descend to grasp z={attempt.grasp_z_mm:.0f}mm")
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

    log(f"      open jaws, lift back to hover z={attempt.hover_z_mm:.0f}mm")
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

    A blocked descent reorders what is left rather than just moving on: the
    descent stopping short means the depth was right and only the lateral
    position was off, so the sideways points are promoted ahead of the ones
    that only change reach. Unreachable candidates are dropped here instead
    of aborting the run — only the first attempt has to clear the IK gate
    for the pick to be worth starting.
    """
    usable = [a for a in plan.attempts if a.reachable]
    for dropped in (a for a in plan.attempts if not a.reachable):
        reach = math.hypot(*dropped.xy_mm)
        log(
            f"  skipping '{dropped.label}' at reach {reach:.0f}mm — "
            f"outside the top-down envelope "
            f"(hover misses by {dropped.hover.position_error_mm:.0f}mm, "
            f"grasp by {dropped.grasp.position_error_mm:.0f}mm)"
        )
    log(f"  {len(usable)} of {len(plan.attempts)} grasp points usable")
    queue = deque(usable)
    promoted = False
    while queue:
        attempt = queue.popleft()
        log(
            f"  attempt '{attempt.label}' at x={attempt.xy_mm[0]:.1f} "
            f"y={attempt.xy_mm[1]:.1f} (reach {math.hypot(*attempt.xy_mm):.0f}mm)"
        )
        outcome, check = attempt_grasp(player, robot, cfg, attempt, log=log)
        detail = (
            f" pos={check.gripper_pos:.1f} load={check.gripper_load_abs:.0f}" if check else ""
        )
        log(f"    -> {outcome.value.upper()}{detail}")
        if outcome is GraspOutcome.HELD:
            return attempt
        if outcome is GraspOutcome.BLOCKED and not promoted:
            promoted = True
            is_sideways = lambda a: a.offset_mm[0] == 0.0 and a.offset_mm[1] != 0.0
            sideways = [a for a in queue if is_sideways(a)]
            if sideways:
                log("    descent stopped short — trying the sideways points next")
                queue = deque(sideways + [a for a in queue if not is_sideways(a)])
    return None

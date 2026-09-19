"""Dynamic IK transport and non-stacking placement for Task 1."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from config import AppConfig
from control.grasp import GraspAttempt, highest_reachable_hover
from control.ik import IkResult, TopDownIK
from perception.homography import PlaneCalibration
from perception.zone import zone_slot_centres

if TYPE_CHECKING:
    from control.motion import MotionController
    from control.trajectory import TrajectoryPlayer


@dataclass(frozen=True)
class Task1SlotPlan:
    index: int
    xy_mm: tuple[float, float]
    drop_z_mm: float
    hover_z_mm: float
    radial_tilt_deg: float
    hover: IkResult
    drop: IkResult


@dataclass(frozen=True)
class Task1TransportPlan:
    slot: Task1SlotPlan
    carry: tuple[tuple[str, IkResult], ...]


def pull_in(x_mm: float, y_mm: float, radius_mm: float) -> tuple[float, float]:
    """Keep azimuth but retract to the high-clearance transport radius."""
    radius = math.hypot(x_mm, y_mm)
    if radius_mm <= 0.0 or radius <= radius_mm:
        return x_mm, y_mm
    scale = radius_mm / radius
    return x_mm * scale, y_mm * scale


def push_out_from_base(
    xy_mm: tuple[float, float],
    base_xy_mm: tuple[float, float],
    offset_mm: float,
) -> tuple[float, float]:
    """Move a point radially away from the base by an exact distance."""
    dx = xy_mm[0] - base_xy_mm[0]
    dy = xy_mm[1] - base_xy_mm[1]
    radius = math.hypot(dx, dy)
    if offset_mm == 0.0:
        return xy_mm
    if radius == 0.0:
        raise ValueError("Cannot radially offset a slot located at the robot base")
    scale = (radius + offset_mm) / radius
    return base_xy_mm[0] + dx * scale, base_xy_mm[1] + dy * scale


def over_ik_gate(result: IkResult, cfg: AppConfig) -> bool:
    """True when a solve missed by more than the configured reach gate."""
    return (
        result.position_error_mm > cfg.ik.max_position_error_mm
        or result.tilt_error_deg > cfg.ik.max_tilt_error_deg
    )


def transit_apex(
    ik: TopDownIK,
    cfg: AppConfig,
    name: str,
    xy: tuple[float, float],
    base_z: float,
    *,
    clearance_mm: float | None = None,
    min_clearance_mm: float | None = None,
) -> tuple[str, IkResult] | None:
    """Highest carry waypoint over ``xy``, folded in to the transit radius.

    Returns None when the point is already inside the radius or the folded
    pose misses the gate -- both mean "no apex worth flying through".
    """
    apex_xy = pull_in(*xy, cfg.motion.transit_apex_radius_mm)
    if apex_xy == xy:
        return None
    apex_z = highest_reachable_hover(
        ik,
        *apex_xy,
        base_z,
        cfg,
        clearance_mm=clearance_mm,
        min_clearance_mm=min_clearance_mm,
    )
    result = ik.solve(*apex_xy, apex_z)
    if over_ik_gate(result, cfg):
        return None
    return name, result


def place_tilt_deg(
    xy_mm: tuple[float, float], base_xy_mm: tuple[float, float], cfg: AppConfig
) -> float:
    """Tilt only genuinely far placements; the pick base tilt is pick-only.

    Deliberately NOT ``fsm.task1.far_reach_tilt_deg``: that one applies
    ``pick_tilt_base_deg`` at every reach, this one is zero inside the start
    radius. ``tests/test_task1.py`` pins the difference.
    """
    radius = math.dist(xy_mm, base_xy_mm)
    start = cfg.task1.pick_tilt_start_radius_mm
    end = cfg.task1.pick_tilt_max_radius_mm
    maximum = cfg.task1.pick_tilt_max_deg
    if radius <= start or maximum == 0.0:
        return 0.0
    fraction = min(1.0, (radius - start) / (end - start))
    return -maximum * fraction


def solve_place_point(
    ik: TopDownIK,
    cfg: AppConfig,
    xy_mm: tuple[float, float],
    drop_z_mm: float,
    *,
    base_xy_mm: tuple[float, float],
    index: int = 0,
    label: str | None = None,
) -> Task1SlotPlan:
    """Hover and drop poses for releasing a held block at ``xy_mm``.

    The zone slots and every agent free placement share this one solve.
    Raises ValueError when either pose misses the IK gate.
    """
    radial_tilt = place_tilt_deg(xy_mm, base_xy_mm, cfg)
    hover_z = highest_reachable_hover(
        ik,
        *xy_mm,
        drop_z_mm,
        cfg,
        radial_tilt_deg=radial_tilt,
    )
    hover = ik.solve(*xy_mm, hover_z, radial_tilt_deg=radial_tilt)
    drop = ik.solve(*xy_mm, drop_z_mm, radial_tilt_deg=radial_tilt)
    if over_ik_gate(hover, cfg) or over_ik_gate(drop, cfg):
        name = label if label is not None else f"Task-1 slot {index}"
        raise ValueError(
            f"{name} is outside the IK gate: "
            f"hover={hover.position_error_mm:.1f}mm drop={drop.position_error_mm:.1f}mm"
        )
    return Task1SlotPlan(index, tuple(xy_mm), drop_z_mm, hover_z, radial_tilt, hover, drop)


def carry_waypoints(
    ik: TopDownIK,
    cfg: AppConfig,
    held: GraspAttempt,
    slot: Task1SlotPlan,
) -> tuple[tuple[str, IkResult], ...]:
    """Transit apexes over the pick and the place point (either may be absent)."""
    carry = []
    for item in (
        transit_apex(ik, cfg, "apex_pick", held.xy_mm, held.grasp_z_mm),
        transit_apex(ik, cfg, "apex_place", slot.xy_mm, slot.drop_z_mm),
    ):
        if item is not None:
            carry.append(item)
    return tuple(carry)


def fly_carry(player: "TrajectoryPlayer", cfg: AppConfig, plan: Task1TransportPlan) -> None:
    """Carry a held block through the apexes to the release hover."""
    for _name, waypoint in plan.carry:
        player.move_to(waypoint.joints, tol=cfg.motion.transit_arrival_tol)
    player.move_to(plan.slot.hover.joints, tol=cfg.motion.transit_arrival_tol)


def release_at(
    player: "TrajectoryPlayer",
    motion: "MotionController",
    cfg: AppConfig,
    slot: Task1SlotPlan,
) -> None:
    """Lower to the drop pose, open the jaws, lift back to the hover."""
    player.move_to(
        slot.drop.joints,
        max_step=cfg.motion.descent_step_per_tick,
        tol=cfg.motion.arrival_tol,
    )
    if cfg.motion.place_settle_s > 0:
        time.sleep(cfg.motion.place_settle_s)
    motion.open_gripper()
    player.move_to(slot.hover.joints, tol=cfg.motion.transit_arrival_tol)


class Task1TransportPlanner:
    """Solve fixed zone slots once and carry waypoints per successful grasp."""

    def __init__(self, calib: PlaneCalibration, cfg: AppConfig, ik: TopDownIK):
        self._calib = calib
        self._cfg = cfg
        self._ik = ik
        try:
            grasp_z = float(calib.meta["grasp_z_mm_mean"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Calibration metadata is missing grasp_z_mm_mean") from exc
        self._drop_z = grasp_z + cfg.task1.release_clearance_mm
        self._slots = self._solve_slots()

    @property
    def slots(self) -> tuple[Task1SlotPlan, ...]:
        return self._slots

    def _over_gate(self, result: IkResult) -> bool:
        return over_ik_gate(result, self._cfg)

    @property
    def drop_z_mm(self) -> float:
        return self._drop_z

    def _far_reach_tilt(self, xy_mm: tuple[float, float]) -> float:
        """Tilt only genuinely far placement slots; pick base tilt is pick-only."""
        return place_tilt_deg(xy_mm, self._calib.base_xy_mm or (0.0, 0.0), self._cfg)

    def _solve_slots(self) -> tuple[Task1SlotPlan, ...]:
        plans = []
        base = self._calib.base_xy_mm or (0.0, 0.0)
        raw_slots = zone_slot_centres(self._calib, self._cfg.task1.slot_uv)
        for index, (raw_xy, radial_offset) in enumerate(
            zip(raw_slots, self._cfg.task1.slot_radial_offset_mm, strict=True)
        ):
            xy = push_out_from_base(raw_xy, base, radial_offset)
            plans.append(
                solve_place_point(
                    self._ik, self._cfg, xy, self._drop_z, base_xy_mm=base, index=index
                )
            )
        if not plans:
            raise ValueError("task1.slot_uv defines no placement slots")
        return tuple(plans)

    def _apex(self, name: str, xy: tuple[float, float], base_z: float) -> tuple[str, IkResult] | None:
        return transit_apex(self._ik, self._cfg, name, xy, base_z)

    def plan(self, held: GraspAttempt, slot_index: int) -> Task1TransportPlan:
        if not 0 <= slot_index < len(self._slots):
            raise IndexError(f"Task-1 slot {slot_index} is not defined")
        slot = self._slots[slot_index]
        return Task1TransportPlan(slot=slot, carry=carry_waypoints(self._ik, self._cfg, held, slot))

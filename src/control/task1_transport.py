"""Dynamic IK transport and non-stacking placement for Task 1."""

from __future__ import annotations

import math
from dataclasses import dataclass

from config import AppConfig
from control.grasp import GraspAttempt, highest_reachable_hover
from control.ik import IkResult, TopDownIK
from perception.homography import PlaneCalibration
from perception.zone import zone_slot_centres


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
        return (
            result.position_error_mm > self._cfg.ik.max_position_error_mm
            or result.tilt_error_deg > self._cfg.ik.max_tilt_error_deg
        )

    def _far_reach_tilt(self, xy_mm: tuple[float, float]) -> float:
        base = self._calib.base_xy_mm or (0.0, 0.0)
        radius = math.dist(xy_mm, base)
        start = self._cfg.task1.pick_tilt_start_radius_mm
        end = self._cfg.task1.pick_tilt_max_radius_mm
        maximum = self._cfg.task1.pick_tilt_max_deg
        if radius <= start or maximum == 0.0:
            return 0.0
        return -maximum * min(1.0, (radius - start) / (end - start))

    def _solve_slots(self) -> tuple[Task1SlotPlan, ...]:
        plans = []
        base = self._calib.base_xy_mm or (0.0, 0.0)
        raw_slots = zone_slot_centres(self._calib, self._cfg.task1.slot_uv)
        for index, (raw_xy, radial_offset) in enumerate(
            zip(raw_slots, self._cfg.task1.slot_radial_offset_mm, strict=True)
        ):
            xy = push_out_from_base(raw_xy, base, radial_offset)
            radial_tilt = self._far_reach_tilt(xy)
            hover_z = highest_reachable_hover(
                self._ik,
                *xy,
                self._drop_z,
                self._cfg,
                radial_tilt_deg=radial_tilt,
            )
            hover = self._ik.solve(*xy, hover_z, radial_tilt_deg=radial_tilt)
            drop = self._ik.solve(*xy, self._drop_z, radial_tilt_deg=radial_tilt)
            if self._over_gate(hover) or self._over_gate(drop):
                raise ValueError(
                    f"Task-1 slot {index} is outside the IK gate: "
                    f"hover={hover.position_error_mm:.1f}mm drop={drop.position_error_mm:.1f}mm"
                )
            plans.append(
                Task1SlotPlan(index, xy, self._drop_z, hover_z, radial_tilt, hover, drop)
            )
        if not plans:
            raise ValueError("task1.slot_uv defines no placement slots")
        return tuple(plans)

    def _apex(self, name: str, xy: tuple[float, float], base_z: float) -> tuple[str, IkResult] | None:
        apex_xy = pull_in(*xy, self._cfg.motion.transit_apex_radius_mm)
        if apex_xy == xy:
            return None
        apex_z = highest_reachable_hover(self._ik, *apex_xy, base_z, self._cfg)
        result = self._ik.solve(*apex_xy, apex_z)
        if self._over_gate(result):
            return None
        return name, result

    def plan(self, held: GraspAttempt, slot_index: int) -> Task1TransportPlan:
        if not 0 <= slot_index < len(self._slots):
            raise IndexError(f"Task-1 slot {slot_index} is not defined")
        slot = self._slots[slot_index]
        carry = []
        for item in (
            self._apex("apex_pick", held.xy_mm, held.grasp_z_mm),
            self._apex("apex_place", slot.xy_mm, slot.drop_z_mm),
        ):
            if item is not None:
                carry.append(item)
        return Task1TransportPlan(slot=slot, carry=tuple(carry))

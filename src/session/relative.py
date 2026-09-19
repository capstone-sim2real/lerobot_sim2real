"""Pure geometry for relative requests and named table points.

"Forward/left" come in two frames (``agent.relative.frame``):

- ``arm``: forward is radial from the base and left is tangential -- the
  gripper's own frame, ``control.ik.gripper_frame_offset``. Straight ahead it
  coincides with the base axes; toward the sides it turns with the arm.
- ``base``: forward is +x and left is +y.

Left/right agree with the camera page in both frames (+y is image-left).
"""

from __future__ import annotations

import math
from typing import Callable

from config import PerceptionConfig, TableRegionsConfig
from control.ik import gripper_frame_offset
from perception.detector import workspace_radius_at_angle

XY = tuple[float, float]


def offset_xy(
    xy_mm: XY,
    forward_mm: float,
    left_mm: float,
    *,
    frame: str,
    base_xy_mm: XY,
) -> XY:
    if frame == "base":
        return xy_mm[0] + forward_mm, xy_mm[1] + left_mm
    if frame != "arm":
        raise ValueError(f"unknown relative frame {frame!r}")
    # gripper_frame_offset assumes the base at the origin; translate around it.
    local = (xy_mm[0] - base_xy_mm[0], xy_mm[1] - base_xy_mm[1])
    if local == (0.0, 0.0):
        return xy_mm[0] + forward_mm, xy_mm[1] + left_mm
    x, y = gripper_frame_offset(local[0], local[1], forward_mm, left_mm)
    return x + base_xy_mm[0], y + base_xy_mm[1]


def decompose_xy(
    before_mm: XY,
    after_mm: XY,
    *,
    frame: str,
    base_xy_mm: XY,
) -> tuple[float, float]:
    """(forward, left) components of a measured displacement, in ``frame``
    evaluated at ``before_mm`` -- the inverse of :func:`offset_xy` for small moves."""
    dx, dy = after_mm[0] - before_mm[0], after_mm[1] - before_mm[1]
    if frame == "base":
        return dx, dy
    phi = math.atan2(before_mm[1] - base_xy_mm[1], before_mm[0] - base_xy_mm[0])
    c, s = math.cos(phi), math.sin(phi)
    return dx * c + dy * s, -dx * s + dy * c


def vector_norm(*components: float) -> float:
    return math.sqrt(sum(float(c) ** 2 for c in components))


def clamp_vector(components: tuple[float, ...], limit_mm: float) -> tuple[tuple[float, ...], bool]:
    """Scale a vector down to ``limit_mm`` if longer. Returns (vector, clamped)."""
    norm = vector_norm(*components)
    if norm <= limit_mm or norm == 0.0:
        return tuple(float(c) for c in components), False
    scale = limit_mm / norm
    return tuple(float(c) * scale for c in components), True


def azimuth_deg(xy_mm: XY, base_xy_mm: XY) -> float:
    return math.degrees(math.atan2(xy_mm[1] - base_xy_mm[1], xy_mm[0] - base_xy_mm[0]))


def table_region_xy(
    column: str,
    row: str,
    perception_cfg: PerceptionConfig,
    regions: TableRegionsConfig,
    base_xy_mm: XY,
) -> XY:
    """Nominal point of a named (column, row) cell of the workspace sector."""
    if column not in regions.columns_deg:
        raise KeyError(f"unknown table column {column!r}")
    if row not in regions.rows_fraction:
        raise KeyError(f"unknown table row {row!r}")
    azimuth = float(regions.columns_deg[column])
    r_max = workspace_radius_at_angle(perception_cfg, azimuth) - regions.edge_margin_mm
    r_min = min(regions.min_radius_mm, r_max)
    radius = r_min + float(regions.rows_fraction[row]) * (r_max - r_min)
    rad = math.radians(azimuth)
    return base_xy_mm[0] + radius * math.cos(rad), base_xy_mm[1] + radius * math.sin(rad)


def search_points(nominal_xy: XY, *, step_mm: float, max_mm: float) -> list[tuple[XY, float]]:
    """Candidate points in concentric rings around ``nominal_xy``, nearest first."""
    points: list[tuple[XY, float]] = [(nominal_xy, 0.0)]
    if step_mm <= 0 or max_mm <= 0:
        return points
    ring = 1
    while ring * step_mm <= max_mm + 1e-9:
        radius = ring * step_mm
        count = max(6, int(round(2 * math.pi * radius / step_mm)))
        for i in range(count):
            angle = 2 * math.pi * i / count
            points.append(
                (
                    (nominal_xy[0] + radius * math.cos(angle), nominal_xy[1] + radius * math.sin(angle)),
                    radius,
                )
            )
        ring += 1
    return points


def find_free_point(
    nominal_xy: XY,
    *,
    is_valid: Callable[[XY], str | None],
    step_mm: float,
    max_mm: float,
) -> tuple[XY, float, str | None]:
    """First point (nearest to nominal) that ``is_valid`` accepts.

    ``is_valid`` returns None for an acceptable point, else a reason code.
    Returns ``(point, displacement_mm, None)`` on success, or
    ``(nominal_xy, 0.0, reason_at_nominal)`` when nothing in range passes.
    """
    nominal_reason: str | None = None
    for point, displacement in search_points(nominal_xy, step_mm=step_mm, max_mm=max_mm):
        reason = is_valid(point)
        if reason is None:
            return point, displacement, None
        if displacement == 0.0:
            nominal_reason = reason
    return nominal_xy, 0.0, nominal_reason or "no_free_region"

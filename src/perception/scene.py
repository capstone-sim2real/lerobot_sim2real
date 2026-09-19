"""One camera frame as the LLM agent sees it: blocks inside and outside the zone.

The detector drops in-zone blocks on purpose (Task 1 completion depends on
it), so the agent runs two passes over the same frame instead of changing
that default: the normal pass for outside blocks, and an ``include_zone``
pass on a *copy* of the perception config whose results are kept only when
they fall inside the zone. Neither pass can let a placed block take the
colour slot of an outside one, and the caller's config is never mutated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from config import PerceptionConfig
from perception.detector import BlockDetection, detect_blocks
from perception.homography import PlaneCalibration
from perception.zone import point_in_zone


@dataclass(frozen=True)
class SceneBlock:
    color: str
    center_mm: tuple[float, float]
    reach_mm: float
    angle_deg: float
    in_zone: bool
    # nearest zone slot when in_zone and within the snap radius
    slot_index: int | None
    detection: BlockDetection = field(repr=False, compare=False)


@dataclass(frozen=True)
class Scene:
    outside: dict[str, SceneBlock]
    inside: dict[str, SceneBlock]
    slot_occupancy: dict[int, str | None]
    frame_seq: int = -1
    captured_at: float = 0.0

    def all(self) -> list[SceneBlock]:
        return [*self.outside.values(), *self.inside.values()]

    def find(self, color: str) -> SceneBlock | None:
        return self.outside.get(color) or self.inside.get(color)


def _nearest_slot(
    xy: tuple[float, float], slot_xy: list[tuple[float, float]], snap_radius_mm: float
) -> int | None:
    best: tuple[float, int] | None = None
    for index, centre in enumerate(slot_xy):
        distance = math.dist(xy, centre)
        if distance <= snap_radius_mm and (best is None or distance < best[0]):
            best = (distance, index)
    return None if best is None else best[1]


def build_scene(
    outside: list[BlockDetection],
    inside: list[BlockDetection],
    calib: PlaneCalibration,
    slot_xy: list[tuple[float, float]],
    *,
    snap_radius_mm: float,
    frame_seq: int = -1,
    captured_at: float = 0.0,
) -> Scene:
    """Pure: partition detections and derive which zone cells are occupied."""
    base = calib.base_xy_mm or (0.0, 0.0)

    def block(det: BlockDetection, in_zone: bool) -> SceneBlock:
        return SceneBlock(
            color=det.color,
            center_mm=(float(det.center_mm[0]), float(det.center_mm[1])),
            reach_mm=math.dist(det.center_mm, base),
            angle_deg=float(det.angle_deg),
            in_zone=in_zone,
            slot_index=_nearest_slot(det.center_mm, slot_xy, snap_radius_mm) if in_zone else None,
            detection=det,
        )

    outside_blocks = {
        d.color: block(d, False) for d in outside if not point_in_zone(d.center_mm, calib)
    }
    inside_blocks = {
        d.color: block(d, True)
        for d in inside
        if point_in_zone(d.center_mm, calib) and d.color not in outside_blocks
    }
    occupancy: dict[int, str | None] = {index: None for index in range(len(slot_xy))}
    for item in inside_blocks.values():
        if item.slot_index is not None and occupancy[item.slot_index] is None:
            occupancy[item.slot_index] = item.color
    return Scene(outside_blocks, inside_blocks, occupancy, frame_seq, captured_at)


def detect_scene(
    frame: np.ndarray,
    calib: PlaneCalibration,
    perception_cfg: PerceptionConfig,
    slot_xy: list[tuple[float, float]],
    *,
    zone_max_per_color: int,
    snap_radius_mm: float,
    is_rgb: bool = False,
    frame_seq: int = -1,
    captured_at: float = 0.0,
) -> Scene:
    outside = detect_blocks(frame, calib, perception_cfg, is_rgb=is_rgb)
    zone_cfg = replace(perception_cfg, max_per_color=zone_max_per_color)
    everything = detect_blocks(frame, calib, zone_cfg, is_rgb=is_rgb, include_zone=True)
    inside = [d for d in everything if point_in_zone(d.center_mm, calib)]
    return build_scene(
        outside,
        inside,
        calib,
        slot_xy,
        snap_radius_mm=snap_radius_mm,
        frame_seq=frame_seq,
        captured_at=captured_at,
    )

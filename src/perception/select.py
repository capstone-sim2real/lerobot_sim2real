"""Target selection: deterministic nearest-first, matching the teleop
demonstration convention (EPISODE.md) so the policy and the FSM agree on
which block is "next".

Blocks inside (or within ``zone_margin_mm`` of) the target zone polygon are
treated as already placed and never selected again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from config import SelectConfig
from perception.detector import BlockDetection
from perception.homography import PlaneCalibration


@dataclass
class SelectionResult:
    target: BlockDetection | None
    target_id: str | None
    # detections outside the zone that are still eligible (incl. the target)
    remaining: int
    detections: list[BlockDetection]


def target_id_for(det: BlockDetection, cell_mm: float) -> str:
    """Stable id for a physical block across re-detections: colour + grid cell."""
    cx = int(round(det.center_mm[0] / cell_mm))
    cy = int(round(det.center_mm[1] / cell_mm))
    return f"{det.color}:{cx},{cy}"


def _in_zone(det: BlockDetection, calib: PlaneCalibration, margin_mm: float) -> bool:
    if not calib.zone_polygon_mm:
        return False
    polygon = np.array(calib.zone_polygon_mm, dtype=np.float32)
    # signed distance: positive inside, negative outside
    dist = cv2.pointPolygonTest(polygon, det.center_mm, measureDist=True)
    return dist >= -margin_mm


def select_target(
    detections: list[BlockDetection],
    calib: PlaneCalibration,
    cfg: SelectConfig,
    skipped: set[str] | None = None,
) -> SelectionResult:
    if calib.base_xy_mm is None:
        raise ValueError(
            "Calibration has no robot base position; rerun tools/calibrate_homography.py with --base-px"
        )
    if cfg.rule != "nearest_first":
        raise ValueError(f"Unknown selection rule: {cfg.rule!r}")
    skipped = skipped or set()

    eligible = [
        d
        for d in detections
        if not _in_zone(d, calib, cfg.zone_margin_mm)
        and target_id_for(d, cfg.target_cell_mm) not in skipped
    ]
    if not eligible:
        return SelectionResult(target=None, target_id=None, remaining=0, detections=detections)

    bx, by = calib.base_xy_mm
    target = min(
        eligible,
        key=lambda d: (math.hypot(d.center_mm[0] - bx, d.center_mm[1] - by), d.center_mm[0], d.center_mm[1]),
    )
    return SelectionResult(
        target=target,
        target_id=target_id_for(target, cfg.target_cell_mm),
        remaining=len(eligible),
        detections=detections,
    )

"""Lightweight perception metadata for the camera web UI.

The operator overlay is deliberately display-only. It reuses the detector's
already-defined geometry, but never runs IK and never feeds a result back into
the task runner. JPEG composition happens in the browser, so this module only
decodes and analyses frames at the configured low rate.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import AppConfig, PerceptionConfig, WorkspaceBoundaryConfig, load_config
from control.grasp import biased_grasp_xy, grasp_candidate_points
from perception import (
    BlockDetection,
    PlaneCalibration,
    RejectedCandidate,
    detect_blocks_with_rejects,
)


def _point_list(
    values: np.ndarray | tuple[float, float] | list[tuple[float, float]],
) -> list[list[float]]:
    points = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in points]


def workspace_boundary_metadata(
    calibration: PlaneCalibration,
    cfg: WorkspaceBoundaryConfig,
    perception: PerceptionConfig,
) -> dict[str, Any] | None:
    """Project the detector's workspace sector into the calibrated image.

    The radius and angles come from ``perception`` rather than from the
    display config, so the outline on the page is exactly the region the
    detector reports blocks in — an arc that could drift away from the gate
    would be worse than no arc.
    """
    if not cfg.enabled:
        return None
    if perception.workspace_radius_mm <= 0:
        raise ValueError("perception.workspace_radius_mm must be positive to draw the arc")
    if cfg.sample_step_deg <= 0:
        raise ValueError("workspace sample_step_deg must be positive")

    lo, hi = perception.workspace_angle_min_deg, perception.workspace_angle_max_deg
    radius = perception.workspace_radius_mm
    count = max(2, int(math.ceil((hi - lo) / cfg.sample_step_deg)) + 1)
    angles_deg = np.linspace(lo, hi, count)
    angles_rad = np.radians(angles_deg)
    base_x, base_y = calibration.base_xy_mm or (0.0, 0.0)
    points_mm = np.column_stack(
        [
            base_x + radius * np.cos(angles_rad),
            base_y + radius * np.sin(angles_rad),
        ]
    )
    points_px = calibration.board_to_pixel(points_mm)
    base_px = calibration.board_to_pixel(np.asarray([[base_x, base_y]], dtype=np.float64))
    return {
        "kind": "nominal_topdown_outer",
        "radius_mm": float(radius),
        "angle_min_deg": float(lo),
        "angle_max_deg": float(hi),
        "points_px": _point_list(points_px),
        # the two radial legs back to the base close the arc into a sector
        "base_px": _point_list(base_px)[0],
    }


def target_zone_metadata(calibration: PlaneCalibration) -> dict[str, Any] | None:
    if not calibration.zone_polygon_mm:
        return None
    points_mm = np.asarray(calibration.zone_polygon_mm, dtype=np.float64)
    return {
        "kind": "excluded_target_zone",
        "points_mm": _point_list(points_mm),
        "points_px": _point_list(calibration.board_to_pixel(points_mm)),
    }


def detection_metadata(
    detection: BlockDetection,
    calibration: PlaneCalibration,
    cfg: AppConfig,
) -> dict[str, Any]:
    """Convert one detector result into browser-drawable raw-image geometry.

    Candidate points use the configured full display bias. The task runner may
    reduce only its radial bias after an IK reachability check; that exact
    decision is intentionally not duplicated by the operator display.
    """
    centre = tuple(float(value) for value in detection.center_mm)
    biased = biased_grasp_xy(cfg.motion, *centre)
    candidates = grasp_candidate_points(cfg.motion, *centre)
    box_mm = np.asarray(detection.box_mm, dtype=np.float64).reshape(-1, 2)

    centre_px = calibration.board_to_pixel(np.asarray([centre], dtype=np.float64))[0]
    biased_px = calibration.board_to_pixel(np.asarray([biased], dtype=np.float64))[0]
    candidate_mm = np.asarray([xy for _label, xy in candidates], dtype=np.float64)
    candidate_px = calibration.board_to_pixel(candidate_mm)
    box_px = calibration.board_to_pixel(box_mm) if len(box_mm) else np.empty((0, 2))

    # A detector edge axis is cheap and truthful. It is deliberately not
    # labelled as the IK-selected wrist yaw used by the robot.
    angle_rad = math.radians(detection.angle_deg)
    dx, dy = math.cos(angle_rad), math.sin(angle_rad)
    direction = np.asarray([dx, dy], dtype=np.float64)
    half_axis_mm = (
        float(np.max(np.abs((box_mm - np.asarray(centre)) @ direction)))
        if len(box_mm)
        else 0.0
    )
    axis_mm = np.asarray(
        [
            [centre[0] - dx * half_axis_mm, centre[1] - dy * half_axis_mm],
            [centre[0] + dx * half_axis_mm, centre[1] + dy * half_axis_mm],
        ],
        dtype=np.float64,
    )
    axis_px = calibration.board_to_pixel(axis_mm)

    return {
        "color": detection.color,
        "center_mm": list(centre),
        "center_px": [float(value) for value in centre_px],
        "box_mm": _point_list(box_mm),
        "box_px": _point_list(box_px),
        "block_angle_deg": float(detection.angle_deg),
        "block_axis_px": _point_list(axis_px),
        "biased_center_mm": list(biased),
        "biased_center_px": [float(value) for value in biased_px],
        "candidates_mm": [
            {"label": label, "xy": [float(xy[0]), float(xy[1])]}
            for label, xy in candidates
        ],
        "candidates_px": [
            {"label": label, "xy": [float(point[0]), float(point[1])]}
            for (label, _xy), point in zip(candidates, candidate_px, strict=True)
        ],
        "display_plan": "nominal_full_bias",
    }


def reject_metadata(
    reject: RejectedCandidate,
    calibration: PlaneCalibration,
) -> dict[str, Any]:
    """Convert one near-miss candidate into browser-drawable geometry.

    Carries the measured values as well as the gate name so the operator can
    read *how far off* a candidate was, not just that something was dropped.
    """
    centre = np.asarray([reject.center_mm], dtype=np.float64)
    box_mm = np.asarray(reject.box_mm, dtype=np.float64).reshape(-1, 2)
    return {
        "color": reject.color,
        "reason": reject.reason,
        "center_mm": [float(value) for value in reject.center_mm],
        "center_px": [float(value) for value in calibration.board_to_pixel(centre)[0]],
        "box_px": _point_list(calibration.board_to_pixel(box_mm)) if len(box_mm) else [],
        "area_mm2": round(reject.area_mm2, 1),
        "aspect": round(reject.aspect, 2),
        "fill": round(reject.fill, 2),
        "solidity": round(reject.solidity, 2),
    }


class OverlayAnalyzer:
    """Detect blocks and publish geometry, without IK or JPEG re-encoding."""

    def __init__(self, config_path: Path | str) -> None:
        config_path = Path(config_path).resolve()
        self.cfg = load_config(config_path)
        project_root = config_path.parents[2]
        calibration_path = Path(self.cfg.perception.calibration_path)
        if not calibration_path.is_absolute():
            calibration_path = project_root / calibration_path
        self.calibration = PlaneCalibration.load(calibration_path)

    def static_metadata(self) -> dict[str, Any]:
        return {
            "image_size": list(self.calibration.image_size),
            "workspace_boundary": workspace_boundary_metadata(
                self.calibration,
                self.cfg.camera.overlay.workspace_boundary,
                self.cfg.perception,
            ),
            "target_zone": target_zone_metadata(self.calibration),
            "display_only": True,
        }

    def analyse(
        self,
        jpeg: bytes,
        *,
        camera_name: str,
        frame_seq: int,
        captured_at: float,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Camera returned an invalid JPEG")
        report_rejects = self.cfg.camera.overlay.report_rejects
        detections, rejects = detect_blocks_with_rejects(
            frame,
            self.calibration,
            self.cfg.perception,
            is_rgb=False,
            collect_rejects=report_rejects,
        )
        return {
            "camera": camera_name,
            "ready": True,
            "frame_seq": int(frame_seq),
            "captured_at": float(captured_at),
            "analysed_at": time.time(),
            "analysis_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "image_size": [int(frame.shape[1]), int(frame.shape[0])],
            "display_only": True,
            "detections": [
                detection_metadata(item, self.calibration, self.cfg) for item in detections
            ],
            "rejects": [reject_metadata(item, self.calibration) for item in rejects],
        }

"""Suggest a calibration block centre without trusting the old homography.

Reuses the colour/shape detector with an identity pixel mapping. The marked
centre is a candidate: visible side faces, occlusion or block movement must
be rejected by the operator before it becomes a calibration pair.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from config import AppConfig
from tools.calibration_records import read_csv, write_csv
from perception.detector import detect_blocks
from perception.homography import PlaneCalibration


def detect_pixel_candidate(frame: np.ndarray, cfg: AppConfig):
    capture = cfg.calibration_capture
    if capture.color not in cfg.perception.color_prototypes:
        raise ValueError(f"Unknown calibration colour: {capture.color}")
    # These values express pixels, never robot millimetres. Do not apply the
    # stale workspace/zone or suppress a second same-colour candidate.
    pixel_cfg = replace(
        cfg.perception,
        hsv_ranges={capture.color: cfg.perception.hsv_ranges[capture.color]},
        rectified_mm_per_px=1.0,
        area_mm2_min=capture.area_px2_min,
        area_mm2_max=capture.area_px2_max,
        morph_kernel_px=capture.morph_kernel_px,
        workspace_radius_mm=0.0,
        max_per_color=0,
        min_color_separation_mm=0.0,
    )
    h, w = frame.shape[:2]
    identity = PlaneCalibration(np.eye(3), (w, h), 1.0, base_xy_mm=(0.0, 0.0))
    matches = [
        d
        for d in detect_blocks(frame, identity, pixel_cfg, is_rgb=False)
        if d.color == capture.color
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one unobscured {capture.color} block, found {len(matches)}; "
            "clear the view/remove other same-colour objects and retry. No pixel was saved."
        )
    return matches[0]


def make_pixel_preview(
    image_path: Path, cfg: AppConfig
) -> tuple[tuple[float, float], Path]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Cannot decode {image_path}")
    candidate = detect_pixel_candidate(frame, cfg)
    centre = candidate.center_mm  # identity mapping: these are raw pixels
    preview = frame.copy()
    cv2.polylines(
        preview, [np.rint(candidate.box_mm).astype(np.int32)], True, (0, 255, 0), 2
    )
    cv2.drawMarker(
        preview,
        tuple(int(round(v)) for v in centre),
        (255, 0, 255),
        cv2.MARKER_CROSS,
        20,
        2,
    )
    cv2.putText(
        preview,
        f"CANDIDATE u={centre[0]:.2f} v={centre[1]:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
    )
    path = image_path.with_name(image_path.stem + "_candidate.png")
    if not cv2.imwrite(str(path), preview):
        raise OSError(f"Could not save preview: {path}")
    return centre, path


def complete_pixel_pair(
    csv_path: Path, name: str, centre: tuple[float, float], image_name: str
) -> None:
    """Commit only an explicitly reviewed candidate to an incomplete FK row."""
    if not np.isfinite(centre).all():
        raise ValueError("Pixel coordinates must be finite")
    fields, rows = read_csv(csv_path)
    matches = [row for row in rows if row["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one FK row named {name}")
    row = matches[0]
    if row["u_px"] or row["v_px"]:
        raise ValueError(f"{name} already has pixels; use a new point name")
    row.update(u_px=f"{centre[0]:.3f}", v_px=f"{centre[1]:.3f}", image=image_name)
    row["notes"] = (
        "FK at gripper_frame_link; auto pixel candidate reviewed; block unchanged during arm withdrawal"
    )
    write_csv(csv_path, fields, rows)

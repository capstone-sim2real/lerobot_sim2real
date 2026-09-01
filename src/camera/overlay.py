"""Perception overlay for the camera web UI.

This is deliberately optional: the camera server remains the sole USB owner,
while an overlay request only reads its already-encoded latest JPEG.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import load_config
from control.grasp import biased_grasp_xy, grasp_candidate_points
from perception import PlaneCalibration, detect_blocks


class OverlayRenderer:
    """Draw detections and the non-IK grasp candidate points on a frame."""

    def __init__(self, config_path: Path | str) -> None:
        self._cfg = load_config(config_path)
        self._calib = PlaneCalibration.load(self._cfg.perception.calibration_path)

    def render(self, jpeg: bytes, *, color: str | None = None) -> tuple[bytes, list[dict[str, object]]]:
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Camera returned an invalid JPEG")
        detections = detect_blocks(frame, self._calib, self._cfg.perception, is_rgb=False)
        if color:
            detections = [detection for detection in detections if detection.color == color]
        payload: list[dict[str, object]] = []
        for detection in detections:
            centre = tuple(float(v) for v in detection.center_mm)
            biased = biased_grasp_xy(self._cfg.motion, *centre)
            candidates = grasp_candidate_points(self._cfg.motion, *centre)
            payload.append(
                {
                    "color": detection.color,
                    "center_mm": centre,
                    "biased_center_mm": biased,
                    "candidates_mm": [{"label": label, "xy": xy} for label, xy in candidates],
                }
            )
            self._draw_detection(frame, detection.color, centre, biased, candidates)
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("Could not encode perception overlay")
        return encoded.tobytes(), payload

    def _draw_detection(
        self,
        frame: np.ndarray,
        color: str,
        centre: tuple[float, float],
        biased: tuple[float, float],
        candidates: list[tuple[str, tuple[float, float]]],
    ) -> None:
        all_mm = np.array([centre, biased, *(xy for _, xy in candidates[1:])], dtype=np.float64)
        all_px = self._calib.board_to_pixel(all_mm).round().astype(int)
        centre_px, biased_px, *retry_px = all_px
        self._cross(frame, tuple(centre_px), (255, 255, 0), 10, 2)
        self._text(frame, f"{color} C ({centre[0]:.0f},{centre[1]:.0f})", tuple(centre_px + (10, -10)), (255, 255, 0))
        cv2.circle(frame, tuple(biased_px), 6, (0, 165, 255), 2)
        self._text(frame, f"B ({biased[0]:.0f},{biased[1]:.0f})", tuple(biased_px + (8, 16)), (0, 165, 255))
        for (label, xy), point_px in zip(candidates[1:], retry_px, strict=True):
            cv2.circle(frame, tuple(point_px), 5, (255, 0, 255), 2)
            short = {"front-left": "FL", "front-right": "FR", "back-left": "BL", "back-right": "BR"}.get(label, label)
            self._text(frame, f"{short} ({xy[0]:.0f},{xy[1]:.0f})", tuple(point_px + (7, -7)), (255, 0, 255))

    @staticmethod
    def _cross(frame: np.ndarray, point: tuple[int, int], color: tuple[int, int, int], size: int, thickness: int) -> None:
        x, y = point
        cv2.line(frame, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)

    @staticmethod
    def _text(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

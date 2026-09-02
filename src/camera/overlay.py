"""Perception overlay for the camera web UI.

This is deliberately optional: the camera server remains the sole USB owner,
while an overlay request only reads its already-encoded latest JPEG.
"""

from __future__ import annotations

import math
from pathlib import Path
import threading

import cv2
import numpy as np

from config import load_config
from control.grasp import biased_grasp_xy, grasp_candidate_points, plan_grasp_attempts
from control.ik import TopDownIK
from perception import PlaneCalibration, detect_blocks


class OverlayRenderer:
    """Draw detections, retry candidates, and the first IK-reachable goal."""

    def __init__(self, config_path: Path | str) -> None:
        config_path = Path(config_path).resolve()
        self._cfg = load_config(config_path)
        self._project_root = config_path.parents[2]
        calibration_path = Path(self._cfg.perception.calibration_path)
        if not calibration_path.is_absolute():
            calibration_path = self._project_root / calibration_path
        self._calib = PlaneCalibration.load(calibration_path)
        self._ik: TopDownIK | None = None
        # IK preview is substantially more expensive than detection. Cache
        # its selected label; candidate coordinates are still redrawn from
        # the current detection every frame.
        self._target_labels: dict[tuple[str, int, int, int], tuple[str | None, float, float | None]] = {}
        self._target_lock = threading.Lock()

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
            # The plan may have backed the radial bias off to stay in reach.
            # Draw what the arm will actually do, not the full-bias point.
            target_label, bias_scale, yaw_deg = self._plan_summary(
                detection.color, centre, detection.angle_deg
            )
            biased = biased_grasp_xy(self._cfg.motion, *centre, scale=bias_scale)
            candidates = grasp_candidate_points(self._cfg.motion, *centre, scale=bias_scale)
            payload.append(
                {
                    "color": detection.color,
                    "center_mm": centre,
                    "biased_center_mm": biased,
                    "candidates_mm": [{"label": label, "xy": xy} for label, xy in candidates],
                    "target_label": target_label,
                    "block_angle_deg": detection.angle_deg,
                    "jaw_yaw_deg": yaw_deg,
                    "bias_scale": bias_scale,
                }
            )
            self._draw_detection(
                frame, detection.color, centre, biased, candidates, target_label, yaw_deg
            )
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
        target_label: str | None,
        yaw_deg: float | None = None,
    ) -> None:
        all_mm = np.array([centre, biased, *(xy for _, xy in candidates[1:])], dtype=np.float64)
        all_px = self._calib.board_to_pixel(all_mm).round().astype(int)
        centre_px, biased_px, *retry_px = all_px
        self._cross_label(frame, centre_px, "C", (255, 255, 0), (10, -10))
        self._cross_label(frame, biased_px, "B", (0, 165, 255), (8, 16))
        for (label, xy), point_px in zip(candidates[1:], retry_px, strict=True):
            # initials of the direction words, so cardinals ("front" -> F) and
            # diagonals ("front-left" -> FL) both render without a lookup table
            short = "".join(word[0].upper() for word in label.split("-"))
            self._cross_label(frame, point_px, short, (255, 0, 255), (7, -7))
        if target_label is not None:
            target_xy = dict(candidates).get(target_label)
            if target_xy is not None:
                target_px = self._calib.board_to_pixel(np.array([target_xy])).round().astype(int)[0]
                self._cross_label(frame, target_px, "T", (0, 0, 255), (10, 26), size=9)
        if yaw_deg is not None:
            # The direction the jaws close along, so a glance says whether
            # they will meet two faces of the block or two of its corners.
            # _topdown_pose puts that axis on column 0 = (-sin yaw, cos yaw).
            half = 26.0
            dx, dy = -math.sin(math.radians(yaw_deg)), math.cos(math.radians(yaw_deg))
            ends_mm = np.array(
                [[biased[0] - dx * half, biased[1] - dy * half],
                 [biased[0] + dx * half, biased[1] + dy * half]]
            )
            (x0, y0), (x1, y1) = self._calib.board_to_pixel(ends_mm).round().astype(int)
            cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 255), 2, cv2.LINE_AA)

    def _plan_summary(
        self, color: str, centre: tuple[float, float], angle_deg: float
    ) -> tuple[str | None, float, float | None]:
        """What the FSM would do here: first usable point, bias scale, jaw yaw."""
        key = (color, round(centre[0] / 5.0), round(centre[1] / 5.0), round(angle_deg / 5.0))
        with self._target_lock:
            if key not in self._target_labels:
                grasp_z = self._calib.meta.get("grasp_z_mm_mean")
                if grasp_z is None:
                    self._target_labels[key] = (None, 1.0, None)
                else:
                    self._ik = self._ik or TopDownIK(self._cfg.ik, project_root=self._project_root)
                    plan = plan_grasp_attempts(
                        self._ik, self._cfg, *centre, float(grasp_z), block_angle_deg=angle_deg
                    )
                    label = next((a.label for a in plan.attempts if a.reachable), None)
                    self._target_labels[key] = (label, plan.bias_scale, plan.yaw_deg)
            return self._target_labels[key]

    @staticmethod
    def _cross_label(
        frame: np.ndarray,
        point: np.ndarray,
        label: str,
        color: tuple[int, int, int],
        text_offset: tuple[int, int],
        size: int = 5,
    ) -> None:
        x, y = point
        cv2.line(frame, (x - size, y), (x + size, y), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x, y - size), (x, y + size), color, 2, cv2.LINE_AA)
        OverlayRenderer._text(frame, label, tuple(point + text_offset), color)

    @staticmethod
    def _cross(frame: np.ndarray, point: tuple[int, int], color: tuple[int, int, int], size: int, thickness: int) -> None:
        x, y = point
        cv2.line(frame, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)

    @staticmethod
    def _text(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

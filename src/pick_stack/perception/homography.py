"""Pixel <-> board-plane (mm) mapping for the fixed top-down camera.

The camera is rigidly mounted to the robot base, so one homography per venue
is enough. The "board frame" origin/orientation is whatever the calibration
session produced (an arbitrary chessboard corner) — that is fine because the
robot base position and the target-zone polygon are recorded *in the same
frame* by the calibration tool, and only relative geometry is ever used
(nearest-first distances, zone membership). Nothing downstream assumes a
particular origin.

Calibration is per venue: rerun ``tools/calibrate_homography.py`` after any
camera-mount change or venue move.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# flags that let findChessboardCornersSB lock onto a *partial* board view:
# we ask for a small min_pattern and allow the detected grid to be larger.
_SB_FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_LARGER


@dataclass
class PlaneCalibration:
    """Homography H maps pixel (x, y) -> board-plane (x_mm, y_mm)."""

    H: np.ndarray
    image_size: tuple[int, int]  # (width, height) the calibration was made at
    square_mm: float
    base_xy_mm: tuple[float, float] | None = None
    zone_polygon_mm: list[tuple[float, float]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def pixel_to_board(self, points_px: np.ndarray) -> np.ndarray:
        """(N, 2) pixel coords -> (N, 2) board mm coords."""
        pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def board_to_pixel(self, points_mm: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(-1, 2)

    def rectify(self, frame: np.ndarray, mm_per_px: float) -> tuple[np.ndarray, tuple[float, float]]:
        """Warp the full frame to a metric top-down view.

        Returns (rectified image, origin_mm) where a rectified pixel (u, v)
        corresponds to board point (origin_x + u * mm_per_px,
        origin_y + v * mm_per_px).
        """
        h, w = frame.shape[:2]
        corners_px = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
        corners_mm = self.pixel_to_board(corners_px)
        x_min, y_min = corners_mm.min(axis=0)
        x_max, y_max = corners_mm.max(axis=0)
        out_w = int(np.ceil((x_max - x_min) / mm_per_px))
        out_h = int(np.ceil((y_max - y_min) / mm_per_px))
        # mm -> rectified px: translate to origin then scale
        A = np.array(
            [[1.0 / mm_per_px, 0.0, -x_min / mm_per_px], [0.0, 1.0 / mm_per_px, -y_min / mm_per_px], [0.0, 0.0, 1.0]]
        )
        rectified = cv2.warpPerspective(frame, A @ self.H, (out_w, out_h))
        return rectified, (float(x_min), float(y_min))

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "H": self.H.tolist(),
            "image_size": list(self.image_size),
            "square_mm": self.square_mm,
            "base_xy_mm": list(self.base_xy_mm) if self.base_xy_mm is not None else None,
            "zone_polygon_mm": [list(p) for p in self.zone_polygon_mm]
            if self.zone_polygon_mm is not None
            else None,
            "meta": {**self.meta, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "PlaneCalibration":
        with open(path) as f:
            payload = json.load(f)
        return cls(
            H=np.array(payload["H"], dtype=np.float64),
            image_size=tuple(payload["image_size"]),
            square_mm=float(payload["square_mm"]),
            base_xy_mm=tuple(payload["base_xy_mm"]) if payload.get("base_xy_mm") else None,
            zone_polygon_mm=[tuple(p) for p in payload["zone_polygon_mm"]]
            if payload.get("zone_polygon_mm")
            else None,
            meta=payload.get("meta", {}),
        )


def calibrate_from_chessboard(
    gray: np.ndarray, square_mm: float, min_pattern: tuple[int, int] = (5, 5)
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a pixel->mm homography from a (possibly partial) chessboard view.

    Returns (H, info) where info holds the detected grid size and the RMS
    residual in mm. Raises RuntimeError when no grid is found.
    """
    found, corners, meta = cv2.findChessboardCornersSBWithMeta(gray, tuple(min_pattern), _SB_FLAGS)
    if not found or corners is None or len(corners) < 4:
        raise RuntimeError(
            "Chessboard not found. Check focus/lighting, lower min_pattern, "
            "or fall back to manual point pairs (calibrate_from_pairs)."
        )
    rows, cols = meta.shape[:2]
    corners_px = corners.reshape(-1, 2).astype(np.float64)
    grid_v, grid_u = np.mgrid[0:rows, 0:cols]
    corners_mm = np.stack([grid_u.ravel() * square_mm, grid_v.ravel() * square_mm], axis=1).astype(np.float64)
    H, inliers = cv2.findHomography(corners_px, corners_mm, cv2.RANSAC, ransacReprojThreshold=square_mm * 0.2)
    if H is None:
        raise RuntimeError("Homography fit failed on detected chessboard corners.")
    projected = cv2.perspectiveTransform(corners_px.reshape(-1, 1, 2), H).reshape(-1, 2)
    rms_mm = float(np.sqrt(np.mean(np.sum((projected - corners_mm) ** 2, axis=1))))
    info = {
        "grid": [int(rows), int(cols)],
        "num_corners": int(len(corners_px)),
        "num_inliers": int(inliers.sum()) if inliers is not None else int(len(corners_px)),
        "rms_mm": rms_mm,
    }
    return H, info


def calibrate_from_pairs(pairs_px_mm: list[tuple[tuple[float, float], tuple[float, float]]]) -> np.ndarray:
    """Manual fallback: fit H from >= 4 (pixel, mm) point pairs."""
    if len(pairs_px_mm) < 4:
        raise ValueError(f"Need at least 4 point pairs, got {len(pairs_px_mm)}")
    src = np.array([p for p, _ in pairs_px_mm], dtype=np.float64)
    dst = np.array([m for _, m in pairs_px_mm], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        raise RuntimeError("Homography fit failed on manual point pairs.")
    return H

"""Fitting the chessboard lattice: exact on a synthetic board, sane on a real frame."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from perception.board import detect_corners
from perception.homography import PlaneCalibration, calibrate_from_chessboard
from tools.calibrate_board_grid import MIN_CORNERS, fit_lattice, orient, screen_axes_mm


def _lattice(angle_deg: float, pitch: float, offset, size: int = 9) -> np.ndarray:
    angle = math.radians(angle_deg)
    e1 = np.array([math.cos(angle), math.sin(angle)])
    e2 = np.array([-e1[1], e1[0]])
    return np.array(
        [offset + i * pitch * e1 + j * pitch * e2
         for i in range(-size, size + 1) for j in range(-size, size + 1)]
    )


def _top_down_calibration(mm_per_px: float = 0.8) -> PlaneCalibration:
    """Camera behind the robot: image right is -y mm, image up is +x mm."""
    width, height = 1280, 720
    H = np.array(
        [[0.0, -mm_per_px, mm_per_px * height], [-mm_per_px, 0.0, mm_per_px * width / 2], [0.0, 0.0, 1.0]]
    )
    return PlaneCalibration(H=H, image_size=(width, height), square_mm=25.0, base_xy_mm=(0.0, 0.0))


def test_fit_recovers_pitch_angle_and_phase():
    corners = _lattice(7.0, 25.0, np.array([13.0, -6.0]))
    fit = fit_lattice(corners)
    assert fit["pitch_mm"] == pytest.approx(25.0, abs=1e-6)
    assert fit["angle_deg"] == pytest.approx(7.0, abs=1e-6)
    assert fit["rms_mm"] == pytest.approx(0.0, abs=1e-9)
    # the origin is *a* lattice point, not necessarily the one we started from
    delta = np.asarray(fit["origin_mm"]) - np.array([13.0, -6.0])
    assert min(abs(delta @ fit["e1"] % 25.0), 25.0 - abs(delta @ fit["e1"] % 25.0)) < 1e-6

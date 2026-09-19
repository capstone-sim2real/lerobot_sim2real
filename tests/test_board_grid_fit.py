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


def test_fit_survives_corner_noise():
    rng = np.random.default_rng(0)
    corners = _lattice(-11.0, 24.0, np.array([5.0, 40.0]))
    noisy = corners + rng.normal(0.0, 0.6, corners.shape)
    fit = fit_lattice(noisy)
    assert fit["pitch_mm"] == pytest.approx(24.0, abs=0.3)
    assert fit["angle_deg"] == pytest.approx(-11.0, abs=0.5)
    assert fit["rms_mm"] < 1.0


def test_axes_are_chosen_for_the_operator_not_the_fit():
    calib = _top_down_calibration()
    right_mm, up_mm = screen_axes_mm(calib)
    assert right_mm[1] < 0 and abs(right_mm[0]) < 1e-9  # image right is -y
    assert up_mm[0] > 0 and abs(up_mm[1]) < 1e-9  # image up is +x (away)

    # a board turned by 93 degrees must still produce screen-facing axes:
    # the lattice's 90-degree symmetry is resolved by the camera, not the fit
    for angle in (7.0, 93.0, -84.0, 178.0):
        fit = fit_lattice(_lattice(angle, 25.0, np.array([0.0, 0.0])))
        u, v = orient(fit, calib)
        assert np.asarray(u) @ right_mm > 0
        assert np.asarray(v) @ up_mm > 0
        assert abs(np.asarray(u) @ np.asarray(v)) < 1e-6  # still perpendicular
        assert np.hypot(*u) == pytest.approx(25.0)


def test_fit_on_a_real_board_frame():
    """End to end on a camera frame: detect corners, rectify, recover 25mm.

    The fit must run on the *plane* (millimetres), never on pixels: the same
    corners fitted in pixel space miss by a third of a square, because
    perspective makes the far squares smaller.
    """
    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "p1_top.jpg"))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H, _info = calibrate_from_chessboard(gray, 25.0)
    calib = PlaneCalibration(H=H, image_size=(frame.shape[1], frame.shape[0]), square_mm=25.0,
                             base_xy_mm=(0.0, 0.0))
    corners_px = detect_corners(gray)
    assert len(corners_px) >= MIN_CORNERS

    fit = fit_lattice(calib.pixel_to_board(corners_px))
    assert fit["pitch_mm"] == pytest.approx(25.0, abs=0.5)
    assert fit["rms_mm"] < 0.1 * fit["pitch_mm"]
    assert fit_lattice(corners_px)["rms_mm"] > fit["rms_mm"]


def test_too_few_corners_is_refused():
    with pytest.raises(ValueError):
        fit_lattice(_lattice(0.0, 25.0, np.array([0.0, 0.0]), size=1))

"""Perception contracts."""

import math

import cv2
import numpy as np
import pytest

from camera.overlay import (
    reject_metadata,
)
from config import AppConfig, PerceptionConfig
from perception import detector
from perception import (
    BlockDetection,
    PlaneCalibration,
    calibrate_from_pairs,
    detect_blocks,
    detect_blocks_with_rejects,
)


from core_helpers import _block_bgr, _calibration


def test_calibration_round_trip():
    pixels = np.array([[40.0, 80.0], [500.0, 10.0]])
    calib = _calibration()
    assert calib.board_to_pixel(calib.pixel_to_board(pixels)) == pytest.approx(
        pixels, abs=1e-6
    )


def test_detector_accepts_blocks_and_rejects_same_colour_tape_in_rgb_and_bgr():
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 80), (120, 120), _block_bgr("green"), -1)
    cv2.rectangle(image, (380, 280), (420, 320), _block_bgr("red"), -1)
    cv2.rectangle(image, (180, 191), (400, 209), _block_bgr("red"), -1)
    calib = PlaneCalibration(H=np.eye(3), image_size=(600, 400), square_mm=25.0)
    perception = AppConfig().perception
    perception.workspace_radius_mm = 0.0  # synthetic frame, not a real board
    for frame, is_rgb in (
        (image, False),
        (cv2.cvtColor(image, cv2.COLOR_BGR2RGB), True),
    ):
        assert {
            d.color for d in detect_blocks(frame, calib, perception, is_rgb=is_rgb)
        } == {"green", "red"}


def test_one_block_is_merged_before_it_is_named():
    """A block trips several gates on purpose, so coincident blobs collapse to
    one candidate before a colour is chosen — merging afterwards would let one
    block hold two colour slots."""
    cfg = AppConfig().perception
    block = BlockDetection("yellow", (100.0, 100.0), 1700.0, 1.05, 0.95, 0.85, [], 0.0)
    same = BlockDetection("wood", (108.0, 100.0), 1700.0, 1.10, 0.90, 0.80, [], 0.0)
    other = BlockDetection("wood", (300.0, 100.0), 1500.0, 1.10, 0.90, 0.80, [], 0.0)

    kept, merged = detector._merge_coincident(
        [same, block, other], cfg.min_color_separation_mm
    )
    assert kept == [block, other] and merged == [same]

    kept, merged = detector._merge_coincident(
        [block, other], cfg.min_color_separation_mm
    )
    assert len(kept) == 2 and merged == []


def test_workspace_uses_angle_dependent_reach_envelope():
    cfg = PerceptionConfig(
        workspace_radius_mm=320.0,
        workspace_angle_min_deg=-90.0,
        workspace_angle_max_deg=90.0,
        workspace_radius_by_angle_mm=[[-90.0, 250.0], [0.0, 320.0], [90.0, 270.0]],
    )
    base = (0.0, 0.0)
    assert detector._in_workspace((300.0, 0.0), cfg, base)
    angle = math.radians(90.0)
    assert detector._in_workspace((260.0 * math.cos(angle), 260.0 * math.sin(angle)), cfg, base)
    assert not detector._in_workspace((280.0 * math.cos(angle), 280.0 * math.sin(angle)), cfg, base)


def test_detects_block_rotation_for_jaw_alignment():
    """A diamond-oriented block must report its edge angle, not 0.

    The gripper grips two faces only if it can be turned to the block's
    edges; without an angle it always meets two corners and slips.
    """
    pairs = [
        ((0.0, 0.0), (0.0, 0.0)),
        ((400.0, 0.0), (400.0, 0.0)),
        ((400.0, 300.0), (400.0, 300.0)),
        ((0.0, 300.0), (0.0, 300.0)),
    ]
    calib = PlaneCalibration(
        H=calibrate_from_pairs(pairs),
        image_size=(400, 300),
        square_mm=25.0,
        base_xy_mm=(200.0, 400.0),
    )
    cfg = AppConfig().perception
    cfg.rectified_mm_per_px = 1.0
    cfg.workspace_radius_mm = 0.0  # synthetic frame, not a real board
    for drawn in (0, 20, 45, 70):
        frame = np.full((300, 400, 3), 200, np.uint8)
        box = cv2.boxPoints(((200.0, 150.0), (40.0, 40.0), float(drawn)))
        cv2.fillPoly(frame, [box.astype(np.int32)], _block_bgr("green"))
        found = detect_blocks(frame, calib, cfg, is_rgb=False)
        assert found, drawn
        # folded to [0, 90): a square grasps identically every quarter turn
        off_by = ((found[0].angle_deg - drawn) + 45.0) % 90.0 - 45.0
        assert abs(off_by) < 3.0, (drawn, found[0].angle_deg)

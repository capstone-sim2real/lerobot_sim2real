"""Perception contracts."""

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


def test_reject_diagnostics_name_the_gate_without_changing_accepted_detections():
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 80), (120, 120), _block_bgr("green"), -1)
    cv2.rectangle(image, (180, 191), (400, 209), _block_bgr("red"), -1)
    calib = PlaneCalibration(H=np.eye(3), image_size=(600, 400), square_mm=25.0)
    perception = AppConfig().perception
    perception.workspace_radius_mm = 0.0  # synthetic frame, not a real board

    detections, rejects = detect_blocks_with_rejects(
        image, calib, perception, is_rgb=False
    )
    assert [d.color for d in detections] == ["green"]
    # 220x18 mm of tape is 3960 mm^2, past the area ceiling before shape matters
    assert [(r.color, r.reason) for r in rejects] == [("red", "area")]
    assert rejects[0].area_mm2 > perception.area_mm2_max

    # the plain entry point stays byte-identical for the robot control path
    plain = detect_blocks(image, calib, perception, is_rgb=False)
    assert plain == detections

    payload = reject_metadata(rejects[0], calib)
    assert payload["reason"] == "area"
    np.testing.assert_allclose(payload["center_px"], [290.0, 200.0], atol=1.0)
    assert len(payload["box_px"]) == 4


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


def test_colour_comes_from_the_nearest_prototype_not_the_gate():
    """The wood/yellow pair, in both arrangements that were measured on the
    real table. In one, hue separates them and saturation does not; in the
    other it is exactly the reverse. Nearest-prototype handles both; no fixed
    band can."""
    cfg = AppConfig().perception
    for label, wood_hs, yellow_hs in (
        ("dark corners: hue splits them", (13.0, 77.0), (22.0, 105.0)),
        ("bright board: saturation splits them", (21.0, 85.0), (24.0, 127.0)),
    ):
        # both arrive mislabelled by whichever gate happened to catch them
        wood = BlockDetection(
            "yellow", (100.0, 50.0), 1700.0, 1.0, 0.9, 0.8, [], 0.0, wood_hs
        )
        yellow = BlockDetection(
            "yellow", (200.0, 50.0), 1700.0, 1.0, 0.9, 0.8, [], 0.0, yellow_hs
        )
        named, unnamed = detector._assign_colors([wood, yellow], cfg)
        assert unnamed == []
        assert [d.color for d in named] == ["wood", "yellow"], label


def test_yellow_keeps_a_separate_point_per_lighting_regime():
    """The regression this guards: yellow's saturation swings 65->200 between
    a dark corner and full board light. A single averaged point sits so far
    from BOTH extremes that a real dark-corner yellow block ends up closer to
    wood's prototype than to its own — this was measured to actually happen
    (own-match 0.299 vs wood-match 0.221) before yellow got a second point.
    """
    cfg = AppConfig().perception
    assert len(cfg.color_prototypes["yellow"]) >= 2, (
        "yellow needs multiple reference points to span its lighting range "
        "without drifting into wood's saturation band"
    )

    dark_corner_yellow = (22.0, 105.0)  # measured: moderate light, near wood's band
    full_board_yellow = (27.0, 198.0)  # measured: bright board, today's failure case
    wood_sample = (13.0, 70.0)

    for label, hue_sat in (
        ("dark corner", dark_corner_yellow),
        ("full board", full_board_yellow),
    ):
        to_yellow = detector._nearest_prototype_distance(
            hue_sat, cfg.color_prototypes["yellow"], cfg
        )
        to_wood = detector._nearest_prototype_distance(
            hue_sat, cfg.color_prototypes["wood"], cfg
        )
        assert to_yellow < to_wood, label
        assert to_yellow <= cfg.prototype_max_distance, label

    # and a real wood sample must still prefer wood over yellow's wide net
    assert detector._nearest_prototype_distance(
        wood_sample, cfg.color_prototypes["wood"], cfg
    ) < detector._nearest_prototype_distance(
        wood_sample, cfg.color_prototypes["yellow"], cfg
    )


def test_blocks_outside_the_reach_sector_are_not_reported():
    """Past the arc the arm cannot pick anything, and the clutter out there
    (wooden floor, far wall) is what produced phantom warm candidates."""
    cfg = AppConfig().perception
    base = (0.0, 0.0)
    inside = (200.0, 100.0)  # r=224, az=+27deg
    too_far = (400.0, 0.0)  # r=400 > 320
    behind = (-200.0, -50.0)  # az=-166deg
    assert detector._in_workspace(inside, cfg, base)
    assert not detector._in_workspace(too_far, cfg, base)
    assert not detector._in_workspace(behind, cfg, base)

    # the gate is measured from the robot base, not the board origin
    shifted = PerceptionConfig(workspace_radius_mm=100.0)
    assert detector._in_workspace((250.0, 0.0), shifted, (200.0, 0.0))
    assert not detector._in_workspace((250.0, 0.0), shifted, (0.0, 0.0))


def test_only_one_block_of_each_colour_survives():
    """The arena holds one block of each colour, so a second surviving blob of
    that colour cannot be a block — and the loose gates the dark table edges
    need would otherwise turn that noise into a phantom target."""
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 80), (120, 120), _block_bgr("red"), -1)
    cv2.rectangle(image, (300, 300), (340, 340), _block_bgr("red"), -1)
    calib = PlaneCalibration(H=np.eye(3), image_size=(600, 400), square_mm=25.0)
    perception = AppConfig().perception
    perception.workspace_radius_mm = 0.0  # synthetic frame, not a real board
    assert perception.max_per_color == 1

    detections, rejects = detect_blocks_with_rejects(
        image, calib, perception, is_rgb=False
    )
    assert [d.color for d in detections] == ["red"]
    assert any(r.reason == "unassigned" for r in rejects)

    perception.max_per_color = 0
    unlimited, _ = detect_blocks_with_rejects(image, calib, perception, is_rgb=False)
    assert len(unlimited) == 2


def test_reject_reason_reports_the_shape_gate_for_in_range_clutter():
    # 100x15 mm = 1500 mm^2 sits inside the area window, so the elongation
    # itself has to be what rejects it.
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (200, 193), (300, 208), _block_bgr("red"), -1)
    calib = PlaneCalibration(H=np.eye(3), image_size=(600, 400), square_mm=25.0)
    perception = AppConfig().perception
    perception.workspace_radius_mm = 0.0  # synthetic frame, not a real board

    detections, rejects = detect_blocks_with_rejects(
        image, calib, perception, is_rgb=False
    )
    assert detections == []
    assert [(r.color, r.reason) for r in rejects] == [("red", "aspect")]
    assert perception.area_mm2_min <= rejects[0].area_mm2 <= perception.area_mm2_max
    assert rejects[0].aspect > perception.aspect_ratio_max


def test_reject_collection_is_opt_out_and_drops_mask_speckle():
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (180, 191), (400, 209), _block_bgr("red"), -1)
    cv2.rectangle(image, (500, 40), (505, 45), _block_bgr("red"), -1)
    calib = PlaneCalibration(H=np.eye(3), image_size=(600, 400), square_mm=25.0)
    perception = AppConfig().perception
    perception.workspace_radius_mm = 0.0  # synthetic frame, not a real board

    _, rejects = detect_blocks_with_rejects(image, calib, perception, is_rgb=False)
    assert [r.reason for r in rejects] == ["area"]  # the 6x6 speck is below the floor

    _, none_collected = detect_blocks_with_rejects(
        image, calib, perception, is_rgb=False, collect_rejects=False
    )
    assert none_collected == []


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

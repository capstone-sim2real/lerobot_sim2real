"""In-zone visibility for the agent without changing the Task 1 detector default."""

import cv2
import numpy as np
import pytest

from config import AppConfig
from perception import PlaneCalibration, detect_blocks
from perception.detector import BlockDetection
from perception.scene import build_scene, detect_scene

from core_helpers import _block_bgr


def _frame_and_calib():
    image = np.zeros((400, 500, 3), dtype=np.uint8)
    cv2.rectangle(image, (230, 30), (270, 70), _block_bgr("green"), -1)   # in zone
    cv2.rectangle(image, (80, 180), (120, 220), _block_bgr("red"), -1)    # outside
    calib = PlaneCalibration(
        H=np.eye(3), image_size=(500, 400), square_mm=1.0, base_xy_mm=(0.0, 0.0),
        zone_polygon_mm=[(200.0, 0.0), (300.0, 0.0), (300.0, 100.0), (200.0, 100.0)],
    )
    perception = AppConfig().perception
    perception.workspace_radius_mm = 0.0
    return image, calib, perception


def test_default_detection_still_hides_the_zone_and_the_flag_reveals_it():
    image, calib, perception = _frame_and_calib()
    assert [d.color for d in detect_blocks(image, calib, perception, is_rgb=False)] == ["red"]
    both = detect_blocks(image, calib, perception, is_rgb=False, include_zone=True)
    assert sorted(d.color for d in both) == ["green", "red"]


def test_detect_scene_partitions_and_snaps_without_mutating_config():
    image, calib, perception = _frame_and_calib()
    scene = detect_scene(image, calib, perception, [(250.0, 50.0), (220.0, 20.0)],
                         zone_max_per_color=1, snap_radius_mm=20.0)
    assert set(scene.outside) == {"red"} and set(scene.inside) == {"green"}
    assert scene.inside["green"].slot_index == 0
    assert scene.slot_occupancy == {0: "green", 1: None}
    assert perception.max_per_color == AppConfig().perception.max_per_color


def test_build_scene_snaps_only_within_the_radius():
    calib = PlaneCalibration(H=np.eye(3), image_size=(10, 10), square_mm=1.0, base_xy_mm=(0.0, 0.0),
                             zone_polygon_mm=[(200.0, 0.0), (300.0, 0.0), (300.0, 100.0), (200.0, 100.0)])
    inside = [BlockDetection("blue", (230.0, 50.0), 1600.0, 1.0, 1.0, 1.0, [])]
    scene = build_scene([], inside, calib, [(250.0, 50.0)], snap_radius_mm=10.0)
    assert scene.inside["blue"].slot_index is None and scene.slot_occupancy == {0: None}
    assert scene.find("blue").reach_mm == pytest.approx((230.0 ** 2 + 50.0 ** 2) ** 0.5)

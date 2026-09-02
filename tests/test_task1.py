"""Task 1 active-region, timeout, slot, and retry-sweep contracts."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from config import AppConfig
from control.ik import IkResult
from control.task1_transport import Task1TransportPlanner, push_out_from_base
from fsm.states import RunContext, StateName
from fsm.task1 import Task1Perception, Task1SelectState, corrected_pick_xy, far_reach_tilt_deg
from perception import BlockDetection, PlaneCalibration, detect_blocks, detect_zone_inner_polygon
from perception.zone import point_in_zone, zone_slot_centres


def _block(color: str, x: float, y: float) -> BlockDetection:
    return BlockDetection(color, (x, y), 1600.0, 1.0, 1.0, 1.0, [])


def _calibration() -> PlaneCalibration:
    return PlaneCalibration(
        H=np.eye(3),
        image_size=(500, 400),
        square_mm=1.0,
        base_xy_mm=(0.0, 0.0),
        zone_polygon_mm=[(300.0, 100.0), (300.0, -100.0), (200.0, -100.0), (200.0, 100.0)],
        meta={"grasp_z_mm_mean": 10.0},
    )


class _Motion:
    def __init__(self):
        self.home_calls = 0

    def go_home(self, *, include_gripper=True):
        assert include_gripper is False
        self.home_calls += 1


class _Samples:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def test_task1_requires_five_seconds_of_fresh_empty_frames(monkeypatch):
    cfg = AppConfig()
    cfg.task1.scan_interval_s = 0.0
    clock = {"wall": 1000.0, "mono": 10.0}
    monkeypatch.setattr("fsm.task1.time.time", lambda: clock["wall"])
    monkeypatch.setattr("fsm.task1.time.monotonic", lambda: clock["mono"])
    samples = _Samples([
        Task1Perception([], 1, 1000.0),
        Task1Perception([], 2, 1004.9),
        Task1Perception([], 3, 1005.0),
    ])
    state = Task1SelectState(_Motion(), samples, _calibration(), cfg)
    ctx = RunContext(cfg.fsm, placed_count=999)  # count must not terminate Task 1
    state.enter(ctx)

    assert state.step(ctx) is None
    clock.update(wall=1004.9, mono=14.9)
    assert state.step(ctx) is None
    clock.update(wall=1005.0, mono=15.0)
    assert state.step(ctx) is StateName.DONE
    assert ctx.extras["task1_complete"] is True


def test_stale_or_duplicate_frame_never_completes_empty_timeout(monkeypatch):
    cfg = AppConfig()
    cfg.task1.scan_interval_s = 0.0
    cfg.task1.max_frame_age_s = 1.0
    clock = {"wall": 1000.0, "mono": 10.0}
    monkeypatch.setattr("fsm.task1.time.time", lambda: clock["wall"])
    monkeypatch.setattr("fsm.task1.time.monotonic", lambda: clock["mono"])
    samples = _Samples([
        Task1Perception([], 7, 1000.0),
        Task1Perception([], 7, 1000.0),
        Task1Perception([], 8, 1010.0),
    ])
    state = Task1SelectState(_Motion(), samples, _calibration(), cfg)
    ctx = RunContext(cfg.fsm)
    state.enter(ctx)
    assert state.step(ctx) is None
    clock.update(wall=1002.0, mono=12.0)
    assert state.step(ctx) is None  # old duplicate resets the proof
    clock.update(wall=1010.0, mono=20.0)
    assert state.step(ctx) is None  # new frame starts a new five-second proof


def test_outside_detection_resets_timeout_and_uses_colour_as_identity(monkeypatch):
    cfg = AppConfig()
    cfg.task1.scan_interval_s = 0.0
    monkeypatch.setattr("fsm.task1.time.time", lambda: 1000.0)
    sample = Task1Perception([_block("blue", 150.0, 20.0)], 1, 1000.0)
    motion = _Motion()
    state = Task1SelectState(motion, lambda: sample, _calibration(), cfg)
    ctx = RunContext(cfg.fsm)
    state.enter(ctx)

    assert state.step(ctx) is StateName.PICK
    assert ctx.target_id == "blue"
    assert ctx.extras["task1_slot_by_color"] == {"blue": 0}
    assert motion.home_calls == 1


def test_task1_pick_correction_grows_radially_to_twenty_mm():
    cfg = AppConfig()
    base = (0.0, 0.0)

    assert corrected_pick_xy((180.0, 0.0), base, cfg) == (180.0, 0.0)
    assert corrected_pick_xy((260.0, 0.0), base, cfg) == pytest.approx((270.0, 0.0))
    corrected = corrected_pick_xy((0.0, 320.0), base, cfg)
    assert corrected == pytest.approx((0.0, 340.0))


def test_task1_far_reach_tilt_opens_only_at_long_reach():
    cfg = AppConfig()
    base = (0.0, 0.0)

    assert far_reach_tilt_deg((270.0, 0.0), base, cfg) == 0.0
    assert far_reach_tilt_deg((300.0, 0.0), base, cfg) == pytest.approx(-2.5)
    assert far_reach_tilt_deg((340.0, 0.0), base, cfg) == pytest.approx(-5.0)


def test_task1_selection_corrects_pick_only_not_active_detections(monkeypatch):
    cfg = AppConfig()
    cfg.task1.scan_interval_s = 0.0
    monkeypatch.setattr("fsm.task1.time.time", lambda: 1000.0)
    raw = _block("red", 320.0, 0.0)
    state = Task1SelectState(
        _Motion(), lambda: Task1Perception([raw], 1, 1000.0), _calibration(), cfg
    )
    ctx = RunContext(cfg.fsm)
    state.enter(ctx)

    assert state.step(ctx) is StateName.PICK
    selection = ctx.extras["selection"]
    assert selection.target.center_mm == pytest.approx((340.0, 0.0))
    assert selection.detections[0].center_mm == (320.0, 0.0)
    assert ctx.extras["task1_pick_radial_tilt_deg"] == pytest.approx(-5.0)


def test_skipped_colour_is_deferred_for_one_sweep_not_abandoned(monkeypatch):
    cfg = AppConfig()
    cfg.task1.scan_interval_s = 0.0
    monkeypatch.setattr("fsm.task1.time.time", lambda: 1000.0)
    sample = Task1Perception([_block("blue", 150.0, 20.0)], 1, 1000.0)
    state = Task1SelectState(_Motion(), lambda: sample, _calibration(), cfg)
    ctx = RunContext(cfg.fsm, attempts={"blue": 2}, skipped={"blue"})
    state.enter(ctx)

    assert state.step(ctx) is StateName.PICK
    assert "blue" not in ctx.skipped and "blue" not in ctx.attempts
    assert ctx.extras["task1_attempts_total"] == {"blue": 2}
    assert ctx.extras["task1_retry_rounds"] == 1


def test_zone_filter_runs_before_one_per_colour_assignment():
    image = np.zeros((400, 500, 3), dtype=np.uint8)
    cfg = AppConfig().perception
    cfg.workspace_radius_mm = 0.0
    hue, sat = cfg.color_prototypes["red"][0]
    bgr = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[hue, sat, 170]]]), cv2.COLOR_HSV2BGR)[0, 0])
    cv2.rectangle(image, (230, 30), (270, 70), bgr, -1)   # in zone
    cv2.rectangle(image, (80, 180), (120, 220), bgr, -1)  # active region
    calib = PlaneCalibration(
        H=np.eye(3), image_size=(500, 400), square_mm=1.0,
        base_xy_mm=(0.0, 0.0),
        zone_polygon_mm=[(200.0, 0.0), (300.0, 0.0), (300.0, 100.0), (200.0, 100.0)],
    )

    found = detect_blocks(image, calib, cfg, is_rgb=False)
    assert len(found) == 1 and found[0].color == "red"
    assert found[0].center_mm == pytest.approx((100.0, 200.0), abs=1.0)


def test_slots_are_inside_zone_and_fill_far_row_first():
    calib = _calibration()
    slots = zone_slot_centres(calib, AppConfig().task1.slot_uv)
    assert len(slots) == 5 and all(point_in_zone(slot, calib) for slot in slots)
    far_radii = [np.hypot(*slot) for slot in slots[:3]]
    near_radii = [np.hypot(*slot) for slot in slots[3:]]
    assert min(far_radii) > max(near_radii)
    assert all(pair[1] == 0.20 for pair in AppConfig().task1.slot_uv[:3])
    assert all(pair[1] == 0.76 for pair in AppConfig().task1.slot_uv[3:])


class _AlwaysReachableIk:
    def solve(self, x_mm, y_mm, z_mm, yaw_deg=None, radial_tilt_deg=0.0):
        return IkResult(
            {"wrist_flex": 0.0},
            position_error_mm=0.0,
            tilt_error_deg=abs(radial_tilt_deg),
        )


def test_task1_all_slots_are_commanded_twenty_mm_farther():
    cfg = AppConfig()
    calib = _calibration()
    raw = zone_slot_centres(calib, cfg.task1.slot_uv)
    planner = Task1TransportPlanner(calib, cfg, _AlwaysReachableIk())

    for index, slot in enumerate(planner.slots):
        radial_delta = np.hypot(*slot.xy_mm) - np.hypot(*raw[index])
        assert radial_delta == pytest.approx(20.0)
    assert planner.slots[0].xy_mm == pytest.approx(
        push_out_from_base(raw[0], calib.base_xy_mm, 20.0)
    )


def test_red_tape_inner_hole_can_be_registered_without_a_block_model():
    image = np.zeros((320, 420, 3), dtype=np.uint8)
    cfg = AppConfig().perception
    hue, sat = cfg.color_prototypes["red"][0]
    bgr = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[hue, sat, 170]]]), cv2.COLOR_HSV2BGR)[0, 0])
    cv2.rectangle(image, (80, 80), (340, 220), bgr, -1)
    cv2.rectangle(image, (100, 100), (320, 200), (0, 0, 0), -1)
    calib = PlaneCalibration(
        H=np.eye(3), image_size=(420, 320), square_mm=1.0, base_xy_mm=(210.0, 300.0)
    )

    polygon_mm, polygon_px = detect_zone_inner_polygon(image, calib, cfg)
    assert len(polygon_mm) == len(polygon_px) == 4
    lengths = sorted(
        [np.linalg.norm(np.asarray(polygon_mm[(i + 1) % 4]) - polygon_mm[i]) for i in range(4)]
    )
    assert lengths[:2] == pytest.approx([100.0, 100.0], abs=3.0)
    assert lengths[2:] == pytest.approx([220.0, 220.0], abs=3.0)

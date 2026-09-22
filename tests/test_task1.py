"""Task 1 active-region, timeout, slot, and retry-sweep contracts."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from config import AppConfig
from control.ik import IkResult
from control.grasp import GraspAttempt
from control.task1_transport import Task1TransportPlanner, push_out_from_base
from fsm.states import RunContext, StateName
from fsm.task1 import (
    Task1Perception,
    Task1SelectState,
    Task1TransportState,
    corrected_pick_xy,
    far_reach_tilt_deg,
)
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


def test_task1_assigns_slots_in_verified_grasp_order():
    cfg = AppConfig()
    result = IkResult({"wrist_flex": 0.0}, position_error_mm=0.0, tilt_error_deg=0.0)
    held = GraspAttempt("centre", (0.0, 0.0), (100.0, 0.0), result, result, True)

    class Planner:
        def __init__(self):
            self.indices = []

        def plan(self, _held, slot_index):
            self.indices.append(slot_index)
            slot = type("Slot", (), {"index": slot_index, "hover": result})()
            return type("Plan", (), {"slot": slot, "carry": ()})()

    class Player:
        def move_to(self, *_args, **_kwargs):
            pass

    planner = Planner()
    state = Task1TransportState(planner, Player(), cfg)
    ctx = RunContext(cfg.fsm)
    ctx.extras["ik_pick_attempt"] = held

    # A failed selection never reaches TRANSPORT, so green consumes no slot.
    ctx.target_id = "wood"
    assert state.step(ctx) is StateName.PLACE
    ctx.target_id = "blue"
    assert state.step(ctx) is StateName.PLACE
    ctx.target_id = "green"
    assert state.step(ctx) is StateName.PLACE

    assert planner.indices == [0, 1, 2]
    assert ctx.extras["task1_slot_by_color"] == {"wood": 0, "blue": 1, "green": 2}


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


class _AlwaysReachableIk:
    def solve(self, x_mm, y_mm, z_mm, yaw_deg=None, radial_tilt_deg=0.0):
        return IkResult(
            {"wrist_flex": 0.0},
            position_error_mm=0.0,
            tilt_error_deg=abs(radial_tilt_deg),
        )

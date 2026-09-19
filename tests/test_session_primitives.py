"""Cancellation, bus lock, relative geometry, and the session lifecycle."""

import copy
import json
import threading

import pytest

from config import AppConfig
from control.robot_io import MockRobotIO
from control.task1_transport import place_tilt_deg
from fsm.task1 import far_reach_tilt_deg
from session.arm_session import ArmSession
from session.cancel import CancellableRobotIO, Cancelled, CancelToken, guard
from session.lock import RobotBusBusy, RobotBusLock
from session.relative import (
    clamp_vector,
    decompose_xy,
    find_free_point,
    offset_xy,
    table_region_xy,
)
from perception.detector import point_in_workspace

from agent_helpers import HOME, calibration


def test_cancelled_is_not_swallowed_by_the_task1_perceive_guard():
    # fsm/task1.py catches exactly these around perceive()
    assert not isinstance(Cancelled(), (RuntimeError, OSError, ValueError))


def test_cancellable_robot_raises_on_the_next_write_but_still_reads():
    inner, token = MockRobotIO(dict(HOME)), CancelToken()
    robot = CancellableRobotIO(inner, token)
    robot.send_joints({"gripper": 10.0})
    token.set()
    assert robot.read_joints()["gripper"] == 10.0
    with pytest.raises(Cancelled):
        robot.send_joints({"gripper": 20.0})
    assert inner.sent_actions == [{"gripper": 10.0}]
    with pytest.raises(Cancelled):
        guard(token, lambda: 1)()


def test_bus_lock_names_the_holder_and_releases(tmp_path):
    path = tmp_path / "robot.lock"
    first = RobotBusLock(path)
    first.acquire()
    with pytest.raises(RobotBusBusy, match="pid"):
        RobotBusLock(path).acquire()
    assert json.loads(first.holder())["pid"]
    first.release()
    with RobotBusLock(path):
        pass


def test_offset_frames_agree_straight_ahead_and_differ_to_the_side():
    base = (0.0, 0.0)
    assert offset_xy((200.0, 0.0), 5.0, 10.0, frame="arm", base_xy_mm=base) == pytest.approx((205.0, 10.0))
    assert offset_xy((200.0, 0.0), 5.0, 10.0, frame="base", base_xy_mm=base) == pytest.approx((205.0, 10.0))
    side = (100.0, 173.2)  # 60 degrees left
    moved = offset_xy(side, 0.0, 10.0, frame="arm", base_xy_mm=base)
    assert moved[0] < side[0]  # tangential left turns with the arm
    # a pure tangential nudge keeps the radius to first order
    assert abs((moved[0] ** 2 + moved[1] ** 2) ** 0.5 - 200.0) < 0.3
    shifted_base = offset_xy((250.0, 50.0), 0.0, 10.0, frame="arm", base_xy_mm=(50.0, 50.0))
    assert shifted_base == pytest.approx((250.0, 60.0))
    f, l = decompose_xy(side, moved, frame="arm", base_xy_mm=base)
    assert (f, l) == pytest.approx((0.0, 10.0), abs=1e-6)


def test_clamp_vector_scales_and_reports():
    assert clamp_vector((3.0, 4.0), 10.0) == ((3.0, 4.0), False)
    vector, clamped = clamp_vector((30.0, 40.0), 10.0)
    assert clamped and vector == pytest.approx((6.0, 8.0))


def test_table_regions_sit_inside_the_sector_on_the_named_side():
    cfg = AppConfig()
    regions = cfg.agent.table_regions
    for column in regions.columns_deg:
        for row in regions.rows_fraction:
            xy = table_region_xy(column, row, cfg.perception, regions, (0.0, 0.0))
            assert point_in_workspace(xy, cfg.perception, (0.0, 0.0))
    right_near = table_region_xy("rightmost", "near", cfg.perception, regions, (0.0, 0.0))
    left_far = table_region_xy("leftmost", "far", cfg.perception, regions, (0.0, 0.0))
    assert right_near[1] < 0 < left_far[1]
    assert (right_near[0] ** 2 + right_near[1] ** 2) < (left_far[0] ** 2 + left_far[1] ** 2)


def test_find_free_point_prefers_nominal_then_nearest_ring():
    point, moved, reason = find_free_point((0.0, 0.0), is_valid=lambda p: None, step_mm=10, max_mm=20)
    assert (point, moved, reason) == ((0.0, 0.0), 0.0, None)
    point, moved, reason = find_free_point(
        (0.0, 0.0), is_valid=lambda p: None if p[0] > 5 else "destination_blocked", step_mm=10, max_mm=20
    )
    assert reason is None and moved == 10 and point[0] > 5
    _, _, reason = find_free_point((0.0, 0.0), is_valid=lambda p: "out_of_workspace", step_mm=10, max_mm=20)
    assert reason == "out_of_workspace"


def test_pick_and_place_tilt_ramps_differ_below_start_radius():
    """Two ramps on the same keys, deliberately different: do not unify them."""
    cfg = AppConfig()
    near = (cfg.task1.pick_tilt_start_radius_mm - 50.0, 0.0)
    assert place_tilt_deg(near, (0.0, 0.0), cfg) == 0.0
    assert far_reach_tilt_deg(near, (0.0, 0.0), cfg) == -cfg.task1.pick_tilt_base_deg


def test_session_open_uses_a_private_config_and_closes_in_order(tmp_path, monkeypatch):
    calib_path = tmp_path / "calib.json"
    calibration().save(calib_path)
    poses_path = tmp_path / "poses.yaml"
    poses_path.write_text("poses:\n  home: " + json.dumps(HOME) + "\n")
    cfg = AppConfig()
    cfg.perception.calibration_path = str(calib_path)
    cfg.motion.poses_path = str(poses_path)
    cfg.motion.fps = 0.0
    cfg.agent.lock_path = str(tmp_path / "robot.lock")
    before = copy.deepcopy(cfg)

    class Robot(MockRobotIO):
        events: list = []

        def disconnect(self):
            self.events.append("disconnect")
            super().disconnect()

    robot = Robot(dict(HOME))
    session = ArmSession.open(cfg, robot=robot, overrides=["motion.fps=200"])
    assert session.cfg is not cfg and session.cfg.motion.fps == 200.0
    assert cfg == before
    with pytest.raises(RobotBusBusy):
        RobotBusLock(cfg.agent.lock_path).acquire()

    session.cancel.set()
    session.close()  # clears the flag it would otherwise trip over
    assert robot.events == ["disconnect"] and not robot.is_connected
    RobotBusLock(cfg.agent.lock_path).acquire()


def test_session_open_releases_the_lock_when_connect_fails(tmp_path):
    calib_path = tmp_path / "calib.json"
    calibration().save(calib_path)
    poses_path = tmp_path / "poses.yaml"
    poses_path.write_text("poses:\n  home: " + json.dumps(HOME) + "\n")
    cfg = AppConfig()
    cfg.perception.calibration_path = str(calib_path)
    cfg.motion.poses_path = str(poses_path)
    cfg.agent.lock_path = str(tmp_path / "robot.lock")

    class Broken(MockRobotIO):
        def connect(self):
            raise ConnectionError("no bus")

    with pytest.raises(ConnectionError):
        ArmSession.open(cfg, robot=Broken())
    RobotBusLock(cfg.agent.lock_path).acquire()

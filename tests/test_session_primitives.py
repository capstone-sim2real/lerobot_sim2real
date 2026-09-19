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

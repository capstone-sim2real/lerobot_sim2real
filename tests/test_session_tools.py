"""Session utilities must be discoverable without connecting to hardware."""

import importlib
import json
import subprocess
import sys
from concurrent.futures import Future
from pathlib import Path

import pytest
from tools._live_capture import LiveCapture


class FakeBus:
    def __init__(self, torque):
        from types import SimpleNamespace

        self.torque = torque
        self.is_connected = False
        self.is_calibrated = True
        self.motors = {
            "shoulder_pan": SimpleNamespace(id=1),
            "gripper": SimpleNamespace(id=6),
        }
        self.writes = []
        self.disconnected = []

    def connect(self):
        self.is_connected = True

    def disconnect(self, disable_torque):
        self.disconnected.append(disable_torque)
        self.is_connected = False

    def read_calibration(self):
        return {"identity": "fake"}

    def ping(self, _):
        return 777

    def sync_read(self, field, **kwargs):
        return {
            "shoulder_pan": {
                "Torque_Enable": self.torque,
                "Operating_Mode": 0,
                "Present_Position": 0.0,
                "Present_Temperature": 30,
            }[field],
            "gripper": {
                "Torque_Enable": self.torque,
                "Operating_Mode": 0,
                "Present_Position": 0.0,
                "Present_Temperature": 30,
            }[field],
        }

    def sync_write(self, *args, **kwargs):
        self.writes.append((args, kwargs))


def teleop_arguments(tmp_path, extra=()):
    from tools.teleop_session import parse_args

    return parse_args(
        [
            "--follower-port",
            "fake-follower",
            "--leader-port",
            "fake-leader",
            "--runtime-dir",
            str(tmp_path),
            "--fps",
            "60",
            "--step",
            "5",
            "--max-relative-target",
            "10",
            "--seconds",
            "0",
            "--capture-max-delta",
            "1",
            "--startup-limit",
            "30",
            "--stable-seconds",
            "0",
            "--stable-delta",
            "1",
            *extra,
        ]
    )


def test_teleop_dry_run_reads_without_writing_or_disabling_torque(tmp_path):
    import threading
    from types import SimpleNamespace
    from tools.teleop_session import run_session

    follower = SimpleNamespace(bus=FakeBus(1))
    leader = SimpleNamespace(bus=FakeBus(0))
    io = SimpleNamespace(
        robot=follower, read_joints=lambda: follower.bus.sync_read("Present_Position")
    )
    run_session(
        teleop_arguments(tmp_path, ["--dry-run"]), io, leader, threading.Event()
    )
    assert not follower.bus.writes and not leader.bus.writes
    assert follower.bus.disconnected == [False]
    assert leader.bus.disconnected == [False]


def test_teleop_invalid_preflight_never_writes(tmp_path):
    import threading
    from types import SimpleNamespace
    from tools.teleop_session import run_session

    follower = SimpleNamespace(bus=FakeBus(1))
    leader = SimpleNamespace(bus=FakeBus(1))
    io = SimpleNamespace(robot=follower)
    with pytest.raises(AssertionError, match="Leader must be torque off"):
        run_session(
            teleop_arguments(tmp_path, ["--dry-run"]), io, leader, threading.Event()
        )
    assert not follower.bus.writes and not leader.bus.writes
    assert not follower.bus.is_connected and not leader.bus.is_connected

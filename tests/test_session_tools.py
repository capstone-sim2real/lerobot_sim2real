"""Session utilities must be discoverable without connecting to hardware."""

import importlib
import json
import subprocess
import sys
from concurrent.futures import Future
from pathlib import Path

import pytest
from tools._live_capture import LiveCapture


@pytest.mark.parametrize(
    "name",
    [
        "teleop_session",
        "capture_teleop_points",
        "finalize_teleop_capture",
        "live_fk_overlay",
    ],
)
def test_session_cli_is_import_safe_and_has_help(name):
    assert callable(importlib.import_module("tools." + name).main)
    result = subprocess.run(
        [sys.executable, "-m", "tools." + name, "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("moved", [False, True])
def test_capture_rejects_motion_and_preserves_stationary_joint_record(tmp_path, moved):
    worker = LiveCapture(tmp_path / "request.json", "http://unused", 1.0)
    future = Future()
    future.set_result(b"\xff\xd8test")
    prefix = tmp_path / "p1_live"
    worker.pending = dict(
        future=future,
        before={"wrist_roll": 0.0},
        max_delta=0.0,
        prefix=prefix,
        name="P1",
    )
    try:
        worker.poll({"wrist_roll": 2.0 if moved else 0.0})
        result = json.loads((tmp_path / "p1_live_result.json").read_text())
        assert result["ok"] is (not moved)
        assert prefix.with_suffix(".jpg").exists() is (not moved)
        if not moved:
            assert json.loads(prefix.with_suffix(".json").read_text())["joints"] == {
                "wrist_roll": 0.0
            }
    finally:
        worker.close()


def test_telemetry_reader_ignores_partial_tail(tmp_path):
    from tools.session_io import read_telemetry

    path = tmp_path / "observations.jsonl"
    path.write_bytes(b'{"time":10,"follower":{"wrist_roll":2}}\n{"time":11,')
    assert read_telemetry(path, 4096)["time"] == 10


def test_session_yaml_defaults_and_cli_precedence(tmp_path):
    from tools.live_fk_overlay import parse_args

    cfg = tmp_path / "session.yaml"
    cfg.write_text(
        "session_tools:\n  runtime_dir: custom/session\n  telemetry_stale_s: 7\n"
    )
    args = parse_args(
        [
            "--config",
            str(cfg),
            "--set",
            "session_tools.poll_interval_s=0.2",
            "--stale-after",
            "9",
        ]
    )
    assert args.telemetry == Path("custom/session/observations.jsonl")
    assert args.output == Path("custom/session/gripper-reference.json")
    assert args.stale_after == 9
    assert args.interval == 0.2


def test_fk_payload_marks_stale_measurement_without_retimestamping():
    import numpy as np
    from types import SimpleNamespace
    from tools.live_fk_overlay import reference_payload

    k = SimpleNamespace(forward_kinematics=lambda _: np.eye(4))
    cal = SimpleNamespace(board_to_pixel=lambda xy: xy)
    row = dict(time=10.0, follower={"wrist_roll": 2.0})
    data = reference_payload(row, k, cal, ["wrist_roll"], 3.0, now=14.0)
    assert data["available"] is False
    assert data["measured_at"] == 10.0


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


def test_teleop_tracking_keeps_tick_limit_and_publishes_measured_pose(tmp_path):
    import threading
    from types import SimpleNamespace
    from tools.teleop_session import track

    stop = threading.Event()
    commands = []
    current = {"shoulder_pan": 0.0, "gripper": 0.0}

    def send(command):
        commands.append(command)
        stop.set()
        return command

    io = SimpleNamespace(
        robot=SimpleNamespace(bus=FakeBus(1)),
        read_joints=lambda: current.copy(),
        send_joints=send,
    )
    leader = SimpleNamespace(
        bus=SimpleNamespace(sync_read=lambda _: {"shoulder_pan": 40.0, "gripper": 0.0})
    )
    capture = SimpleNamespace(poll=lambda _: None)
    track(teleop_arguments(tmp_path), io, leader, capture, current.copy(), stop)
    assert 0 < commands[0]["shoulder_pan"] <= 5
    row = json.loads((tmp_path / "observations.jsonl").read_text())
    assert row["follower"]["shoulder_pan"] == 0
    assert row["command"]["shoulder_pan"] == commands[0]["shoulder_pan"]

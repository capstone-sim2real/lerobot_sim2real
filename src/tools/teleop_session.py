"""Guarded leader/follower session with file-based calibration capture."""

import argparse
import json
import logging
import signal
import threading
import time
from pathlib import Path

from config import RobotIOConfig
from control.trajectory import interpolate
from tools._live_capture import LiveCapture
from tools.session_io import atomic_json, parse_session_args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--capture-request", type=Path)
    parser.add_argument("--snapshot-url")
    parser.add_argument("--capture-max-delta", type=float, required=True)
    parser.add_argument("--offsets-json", type=Path)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--startup-step", type=float)
    parser.add_argument("--startup-ramp-seconds", type=float)
    parser.add_argument("--step", type=float, required=True)
    parser.add_argument("--max-relative-target", type=float, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--temperature-limit", type=float)
    parser.add_argument("--startup-limit", type=float, required=True)
    parser.add_argument("--stable-seconds", type=float, required=True)
    parser.add_argument("--stable-delta", type=float, required=True)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--follower-port", required=True)
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--follower-id", default="my_follower")
    parser.add_argument("--leader-id", default="my_leader")
    args = parse_session_args(
        parser,
        argv,
        {
            "runtime_dir": "runtime_dir",
            "snapshot_url": "snapshot_url",
            "temperature_limit": "temperature_limit_c",
            "startup_ramp_seconds": "startup_ramp_s",
        },
    )
    if min(args.fps, args.step, args.max_relative_target) <= 0 or args.seconds < 0:
        parser.error(
            "fps/step/max-relative-target must be positive; seconds must be non-negative"
        )
    if (
        min(
            args.stable_seconds,
            args.stable_delta,
            args.startup_limit,
            args.startup_ramp_seconds,
            args.capture_max_delta,
        )
        < 0
    ):
        parser.error("startup and capture limits must be non-negative")
    if args.startup_step is not None and args.startup_step <= 0:
        parser.error("startup-step must be positive")
    args.capture_request = (
        args.capture_request or args.runtime_dir / "capture_request.json"
    )
    return args


def create_devices(args):
    """Optional hardware dependencies are loaded only for an explicit run."""
    from control.robot_io import So101RobotIO
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.teleoperators.so_leader.so_leader import SOLeader

    follower = SOFollower(
        SOFollowerRobotConfig(
            id=args.follower_id,
            port=args.follower_port,
            use_degrees=True,
            max_relative_target=args.max_relative_target,
            disable_torque_on_disconnect=False,
        )
    )
    leader = SOLeader(
        SOLeaderTeleopConfig(id=args.leader_id, port=args.leader_port, use_degrees=True)
    )
    io = So101RobotIO(RobotIOConfig())
    io._robot = follower
    return io, leader


def preflight(follower, leader, expected_model):
    """Read identity, mode and torque; never write calibration or motor goals."""
    assert follower.bus.is_calibrated, "Follower calibration mismatch"
    leader.bus.calibration = leader.bus.read_calibration()
    for device in (follower, leader):
        torque = device.bus.sync_read("Torque_Enable")
        assert len(set(torque.values())) == 1, "Mixed torque state"
        if device is leader:
            assert all(value == 0 for value in torque.values()), (
                "Leader must be torque off"
            )
        assert all(
            value == 0 for value in device.bus.sync_read("Operating_Mode").values()
        ), "Expected position mode"
        assert all(
            device.bus.ping(m.id) == expected_model for m in device.bus.motors.values()
        ), "Unexpected model"


def wait_for_alignment(args, io, leader, capture, offsets, stop):
    startup_log = 0
    stable_start = None
    anchor = None
    deadline = time.monotonic() + args.seconds if args.seconds else float("inf")
    print("WAITING: align leader with follower, then hold still", flush=True)
    while not stop.is_set():
        if time.monotonic() > deadline:
            raise RuntimeError("Startup alignment timed out; no motion sent")
        follower_pose = io.read_joints()
        leader_pose = leader.bus.sync_read("Present_Position")
        capture.poll(follower_pose)
        delta = {
            j: leader_pose[j] + (offsets[j] if offsets else 0) - follower_pose[j]
            for j in follower_pose
            if j != "gripper"
        }
        aligned = max(abs(value) for value in delta.values()) <= args.startup_limit
        if time.monotonic() - startup_log >= args.settings.telemetry_interval_s:
            atomic_json(
                args.runtime_dir / "startup_status.json",
                dict(
                    time=time.time(),
                    follower=follower_pose,
                    leader=leader_pose,
                    delta=delta,
                ),
            )
            startup_log = time.monotonic()
        if not aligned:
            stable_start = anchor = None
        elif (
            anchor is None
            or max(abs(leader_pose[j] - anchor[j]) for j in leader_pose)
            > args.stable_delta
        ):
            anchor = leader_pose.copy()
            stable_start = time.monotonic()
        elif time.monotonic() - stable_start >= args.stable_seconds:
            return follower_pose, leader_pose
        time.sleep(args.settings.startup_poll_s)
    raise RuntimeError("Cancelled before tracking")


def track(args, io, leader, capture, offsets, stop):
    started = time.monotonic()
    lastlog = 0
    command = io.read_joints()
    with (args.runtime_dir / "observations.jsonl").open("a", buffering=1) as log:
        while not stop.is_set() and (
            args.seconds == 0 or time.monotonic() - started < args.seconds
        ):
            tick = time.monotonic()
            current = io.read_joints()
            measured_leader = leader.bus.sync_read("Present_Position")
            capture.poll(current)
            goal = {j: measured_leader[j] + offsets[j] for j in current}
            goal["gripper"] = min(100.0, max(0.0, measured_leader["gripper"]))
            step = (
                args.startup_step
                if args.startup_step is not None
                and tick - started < args.startup_ramp_seconds
                else args.step
            )
            path = interpolate(command, goal, step)
            if path:
                command = io.send_joints(path[0])
            if tick - lastlog >= args.settings.telemetry_interval_s:
                temperatures = io.robot.bus.sync_read("Present_Temperature")
                if max(temperatures.values()) >= args.temperature_limit:
                    raise RuntimeError(
                        "Temperature limit reached: " + json.dumps(temperatures)
                    )
                if not all(
                    value == 1
                    for value in io.robot.bus.sync_read("Torque_Enable").values()
                ):
                    raise RuntimeError("Follower torque lost")
                log.write(
                    json.dumps(
                        dict(
                            time=time.time(),
                            follower=current,
                            leader=measured_leader,
                            goal=goal,
                            command=command,
                            temperature=temperatures,
                        )
                    )
                    + "\n"
                )
                lastlog = tick
            time.sleep(max(0, 1 / args.fps - (time.monotonic() - tick)))


def run_session(args, io, leader, stop):
    follower = io.robot
    enabled = False
    capture = None
    offsets = json.loads(args.offsets_json.read_text()) if args.offsets_json else None
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    try:
        for device in (follower, leader):
            device.bus.connect()
        preflight(follower, leader, args.settings.expected_motor_model)
        fp = io.read_joints()
        lp = leader.bus.sync_read("Present_Position")
        print(
            "STARTUP_DELTA",
            json.dumps({j: lp[j] - fp[j] for j in fp if j != "gripper"}),
            flush=True,
        )
        if not args.dry_run:
            capture = LiveCapture(
                args.capture_request,
                args.snapshot_url,
                args.capture_max_delta,
                snapshot_timeout_s=args.settings.snapshot_timeout_s,
            )
            fp, lp = wait_for_alignment(args, io, leader, capture, offsets, stop)
        offsets = offsets if offsets else {j: fp[j] - lp[j] for j in fp}
        print(
            json.dumps(
                dict(
                    dry_run=args.dry_run,
                    follower_start=fp,
                    leader_start=lp,
                    offsets=offsets,
                )
            ),
            flush=True,
        )
        if not args.dry_run:
            raw = follower.bus.sync_read("Present_Position", normalize=False)
            follower.bus.sync_write("Goal_Position", raw, normalize=False)
            assert follower.bus.sync_read("Goal_Position", normalize=False) == raw, (
                "Goal readback mismatch"
            )
            io.set_torque(True)
            enabled = True
            assert all(v == 1 for v in follower.bus.sync_read("Torque_Enable").values())
            print(
                "ACTIVE: leader tracking; follower torque ON; leader torque OFF",
                flush=True,
            )
            track(args, io, leader, capture, offsets, stop)
    finally:
        try:
            if enabled and follower.bus.is_connected:
                follower.bus.sync_write(
                    "Goal_Position",
                    follower.bus.sync_read("Present_Position", normalize=False),
                    normalize=False,
                )
                print("STOPPED: holding current pose, torque retained", flush=True)
        finally:
            try:
                if capture is not None:
                    capture.close()
            finally:
                for device in (leader, follower):
                    if device.bus.is_connected:
                        device.bus.disconnect(disable_torque=False)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.ERROR)
    stop = threading.Event()
    previous = {
        sig: signal.signal(sig, lambda *_: stop.set())
        for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        io, leader = create_devices(args)
        run_session(args, io, leader, stop)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()

"""Publish FK from measured telemetry without serial access."""

import argparse
import json
import time
from pathlib import Path

from tools.session_io import atomic_json, parse_session_args, read_telemetry


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stale-after", type=float)
    parser.add_argument("--interval", type=float)
    args = parse_session_args(
        parser,
        argv,
        {
            "runtime_dir": "runtime_dir",
            "stale_after": "telemetry_stale_s",
            "interval": "poll_interval_s",
        },
    )
    if args.interval <= 0 or args.stale_after <= 0:
        parser.error("interval and stale-after must be positive")
    args.telemetry = args.telemetry or args.runtime_dir / "observations.jsonl"
    args.output = args.output or args.runtime_dir / "gripper-reference.json"
    return args


def reference_payload(row, kinematics, calibration, motors, stale_after, now):
    joints = [row["follower"][name] for name in motors]
    mm = kinematics.forward_kinematics(joints)[:3, 3] * 1000
    px = calibration.board_to_pixel(mm[:2].reshape(1, 2))[0]
    return dict(
        available=0 <= now - row["time"] < stale_after,
        frame="gripper_frame_link",
        source="Live measured teleop FK",
        projection="XY on calibrated block plane; not image detection",
        xyz_mm=mm.tolist(),
        pixel=px.tolist(),
        measured_at=row["time"],
        joint_degrees=joints,
    )


def publish_loop(args, kinematics, calibration, motors):
    last = None
    while True:
        try:
            row = read_telemetry(args.telemetry, args.settings.telemetry_tail_bytes)
            data = reference_payload(
                row, kinematics, calibration, motors, args.stale_after, time.time()
            )
            key = (row["time"], data["available"])
            if key != last:
                atomic_json(args.output, data)
                print(json.dumps(data), flush=True)
                last = key
        except (OSError, ValueError, KeyError, TypeError) as exc:
            atomic_json(args.output, dict(available=False, error=str(exc)))
            last = None
        time.sleep(args.interval)


def main(argv=None):
    args = parse_args(argv)
    from tools.record_calibration_point import load_kinematics, DEFAULT_URDF, ARM_MOTORS
    from perception import PlaneCalibration

    kinematics = load_kinematics(DEFAULT_URDF, "gripper_frame_link")
    calibration = PlaneCalibration.load(args.app_config.perception.calibration_path)
    try:
        publish_loop(args, kinematics, calibration, ARM_MOTORS)
    except KeyboardInterrupt:
        atomic_json(args.output, {"available": False})


if __name__ == "__main__":
    main()

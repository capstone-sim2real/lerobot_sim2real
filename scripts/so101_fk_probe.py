#!/usr/bin/env python3
"""Print live FK (x, y, z) at gripper_frame_link while you hand-position the
arm. Read-only — sends no commands. Use this BEFORE running
so101_calib_point.sh to check the grip height (z) looks consistent with
previous points (keep the wrist vertical/top-down each time; AGENTS.md §6).

    ./scripts/so101_fk_probe.sh
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = PROJECT_ROOT / "third_party/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
PORT = os.environ.get(
    "SO101_FOLLOWER_PORT",
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00",
)
ROBOT_ID = os.environ.get("SO101_FOLLOWER_ID", "my_follower")


def load_kinematics(urdf_path: Path) -> RobotKinematics:
    urdf_path = urdf_path.expanduser().resolve()
    old_cwd = Path.cwd()
    os.chdir(urdf_path.parent)  # placo resolves mesh paths relative to cwd
    try:
        return RobotKinematics(str(urdf_path), target_frame_name="gripper_frame_link", joint_names=MOTORS)
    finally:
        os.chdir(old_cwd)


def main() -> int:
    kinematics = load_kinematics(DEFAULT_URDF)
    robot = SOFollower(
        SOFollowerRobotConfig(id=ROBOT_ID, port=PORT, use_degrees=True, disable_torque_on_disconnect=False)
    )
    robot.bus.connect()  # bus only: no configure(), no torque re-enable (same as record_calibration_point.py)
    print("Live FK at gripper_frame_link. Move the arm by hand. Ctrl-C to stop.\n")
    try:
        prev_z = None
        while True:
            joints = robot.bus.sync_read("Present_Position")
            q = np.array([float(joints[m]) for m in MOTORS])
            xyz = kinematics.forward_kinematics(q)[:3, 3]
            x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
            note = ""
            if prev_z is not None:
                note = f"  (Δz vs last reading: {z_mm - prev_z:+.1f}mm)"
            prev_z = z_mm
            print(f"  x={x_mm:7.1f}mm  y={y_mm:7.1f}mm  z={z_mm:7.1f}mm{note}    ", end="\r", flush=True)
            time.sleep(0.15)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        robot.bus.disconnect(disable_torque=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

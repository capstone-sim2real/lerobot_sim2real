#!/usr/bin/env python3
"""Print live FK (x, y, z) at gripper_frame_link while you hand-position the
arm. Read-only — sends no commands. Use this BEFORE running
so101_calib_point.sh to check the grip height (z) looks consistent with
previous points (keep the wrist vertical/top-down each time; AGENTS.md §6).

    so101-fk
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = PROJECT_ROOT / "third_party/so101/so101.urdf"
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
PORT = os.environ.get(
    "SO101_FOLLOWER_PORT",
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00",
)
ROBOT_ID = os.environ.get("SO101_FOLLOWER_ID", "my_follower")
# how close to neutral wrist_roll a calibration grip must be
ROLL_NEUTRAL_TOL_DEG = 8.0


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
    print("Live FK at gripper_frame_link. Move the arm by hand. Ctrl-C to stop.")
    print(f"Keep wrist_roll within +-{ROLL_NEUTRAL_TOL_DEG:.0f} deg of 0 while recording calibration points.\n")
    try:
        while True:
            joints = robot.bus.sync_read("Present_Position")
            q = np.array([float(joints[m]) for m in MOTORS])
            xyz = kinematics.forward_kinematics(q)[:3, 3]
            x_mm, y_mm, z_mm = (xyz * 1000.0).tolist()
            # wrist_roll must be near 0 while recording calibration points:
            # gripper_frame_link sits ~8mm off the roll axis, so a roll that
            # varies between points injects a different offset into each
            # recorded position, and runtime IK grasps at neutral roll
            # anyway (AGENTS.md §6/§7).
            roll = float(q[4])
            mark = "OK " if abs(roll) <= ROLL_NEUTRAL_TOL_DEG else ">> TURN WRIST <<"
            print(
                f"  x={x_mm:7.1f}  y={y_mm:7.1f}  z={z_mm:7.1f} mm   "
                f"wrist_roll={roll:+7.1f} deg  {mark}    ",
                end="\r",
                flush=True,
            )
            time.sleep(0.15)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        robot.bus.disconnect(disable_torque=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

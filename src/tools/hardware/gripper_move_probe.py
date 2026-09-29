#!/usr/bin/env python3
"""Command the gripper to one target position and report before/after,
to isolate whether a commanded move actually reaches the motor.

    so101-gripper-move --to 60
"""
from __future__ import annotations

import argparse
import os
import time

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

PORT = os.environ.get(
    "SO101_FOLLOWER_PORT",
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00",
)
ROBOT_ID = os.environ.get("SO101_FOLLOWER_ID", "my_follower")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", type=float, required=True, help="target gripper position")
    ap.add_argument("--max-relative-target", type=float, default=150.0)
    args = ap.parse_args()

    robot = SOFollower(
        SOFollowerRobotConfig(
            id=ROBOT_ID, port=PORT, use_degrees=True,
            max_relative_target=args.max_relative_target,
            disable_torque_on_disconnect=False,
        )
    )
    robot.connect()  # full connect: torque comes ON (calibration file already matches, no auto-calibrate)
    try:
        before = robot.bus.sync_read("Present_Position")["gripper"]
        load_before = robot.bus.sync_read("Present_Load")["gripper"]
        print(f"before: position={before:7.2f}  load={load_before:6d}")
        print(f"commanding gripper -> {args.to:.1f} ...")
        sent = robot.send_action({"gripper.pos": args.to})
        print(f"  actually sent (post-clamp): {sent}")
        for i in range(10):
            time.sleep(0.2)
            pos = robot.bus.sync_read("Present_Position")["gripper"]
            load = robot.bus.sync_read("Present_Load")["gripper"]
            print(f"  t={0.2*(i+1):.1f}s  position={pos:7.2f}  load={load:6d}")
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

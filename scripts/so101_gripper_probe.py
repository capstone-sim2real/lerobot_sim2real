#!/usr/bin/env python3
"""Print live gripper Present_Position / Present_Load while you open/close it
by hand. Sends no commands to the robot — read-only, safe to run at any time.

Open the gripper all the way, note the number that stops changing (that is
"open"). Close it all the way, note that number too (that is "closed").
Ctrl-C to stop.
"""
from __future__ import annotations

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
    robot = SOFollower(
        SOFollowerRobotConfig(id=ROBOT_ID, port=PORT, use_degrees=True, disable_torque_on_disconnect=False)
    )
    robot.bus.connect()  # bus only: no configure(), no torque re-enable
    print("Reading gripper position/load. Open and close the gripper by hand now.")
    print("Ctrl-C to stop.\n")
    try:
        while True:
            pos = robot.bus.sync_read("Present_Position")
            load = robot.bus.sync_read("Present_Load")
            print(f"  gripper: position={pos['gripper']:7.2f}   load={load['gripper']:6d}", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        robot.bus.disconnect(disable_torque=False)  # leave torque state untouched
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

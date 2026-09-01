#!/usr/bin/env python3
"""Disable torque on every reachable follower motor so the arm can be moved
by hand. Tolerates a motor that is unreachable or faulted (e.g. an
overheat-protection trip) — it reports that motor's failure and still
disables the rest, instead of the normal full-handshake connect() failing
outright when one motor doesn't answer.

    ./scripts/so101_torque_off.sh
"""
from __future__ import annotations

import os

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
    robot.bus.connect(handshake=False)  # skip the all-motors-present check
    failed = []
    try:
        for name, motor in robot.bus.motors.items():
            try:
                robot.bus.disable_torque(name)
                print(f"  {name} (id={motor.id}): torque OFF")
            except Exception as e:
                print(f"  {name} (id={motor.id}): FAILED - {e}")
                failed.append(name)
    finally:
        robot.bus.disconnect(disable_torque=False)
    if failed:
        print(f"\n{len(failed)} motor(s) did not respond: {failed}")
        print("Do not command them further until resolved (e.g. let an overheated servo cool down).")
        return 1
    print("\nall motors torque OFF — safe to move the arm by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

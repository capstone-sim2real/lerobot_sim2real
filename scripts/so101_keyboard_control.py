#!/usr/bin/env python3
"""Keyboard control for the SO-101 follower arm over SSH.

This is intended for slow, manual remote testing without a leader arm.
It starts from the follower's current pose and applies small joint deltas.
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower


DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00"
DEFAULT_ID = "my_follower"

MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

KEYMAP = {
    "q": ("shoulder_pan", +1),
    "a": ("shoulder_pan", -1),
    "w": ("shoulder_lift", +1),
    "s": ("shoulder_lift", -1),
    "e": ("elbow_flex", +1),
    "d": ("elbow_flex", -1),
    "r": ("wrist_flex", +1),
    "f": ("wrist_flex", -1),
    "t": ("wrist_roll", +1),
    "g": ("wrist_roll", -1),
    "y": ("gripper", +1),
    "h": ("gripper", -1),
}


@dataclass
class Limits:
    start: dict[str, float]
    max_delta: float
    gripper_max_delta: float

    def clamp(self, motor: str, value: float) -> float:
        delta = self.gripper_max_delta if motor == "gripper" else self.max_delta
        low = self.start[motor] - delta
        high = self.start[motor] + delta
        value = min(max(value, low), high)
        if motor == "gripper":
            value = min(max(value, 0.0), 100.0)
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyboard-control the SO-101 follower arm.")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Follower serial port")
    parser.add_argument("--id", default=DEFAULT_ID, help="Follower calibration id")
    parser.add_argument("--step", type=float, default=2.0, help="Degrees per key press")
    parser.add_argument("--gripper-step", type=float, default=2.0, help="Gripper units per key press")
    parser.add_argument(
        "--max-delta",
        type=float,
        default=60.0,
        help="Max degrees each joint may move from the startup pose",
    )
    parser.add_argument(
        "--gripper-max-delta",
        type=float,
        default=50.0,
        help="Max gripper movement from the startup value",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=10.0,
        help="LeRobot per-command safety limit",
    )
    parser.add_argument(
        "--no-return-start",
        action="store_true",
        help="Do not return to the startup pose on exit",
    )
    return parser.parse_args()


def key_help() -> str:
    return """
키 조작:
  q/a  shoulder_pan   +/-
  w/s  shoulder_lift  +/-
  e/d  elbow_flex     +/-
  r/f  wrist_flex     +/-
  t/g  wrist_roll     +/-
  y/h  gripper        +/-

기타:
  space  현재 목표 위치 출력
  0      현재 실제 위치를 새 목표로 재설정
  ?      도움말
  x      종료
""".strip()


def read_key(timeout_s: float = 0.1) -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not readable:
        return None
    return sys.stdin.read(1)


def observation_to_goal(obs: dict[str, float]) -> dict[str, float]:
    return {motor: float(obs[f"{motor}.pos"]) for motor in MOTORS}


def action_from_goal(goal: dict[str, float]) -> dict[str, float]:
    return {f"{motor}.pos": goal[motor] for motor in MOTORS}


def print_goal(label: str, goal: dict[str, float]) -> None:
    values = "  ".join(f"{motor}={goal[motor]:7.2f}" for motor in MOTORS)
    print(f"\n{label}: {values}")


def main() -> int:
    args = parse_args()

    config = SOFollowerRobotConfig(
        id=args.id,
        port=args.port,
        max_relative_target=args.max_relative_target,
        use_degrees=True,
    )
    robot = SOFollower(config)
    start_goal: dict[str, float] | None = None

    old_term = termios.tcgetattr(sys.stdin)
    try:
        print("SO-101 follower 연결 중...")
        robot.connect()

        obs = robot.get_observation()
        goal = observation_to_goal(obs)
        start_goal = goal.copy()
        limits = Limits(goal.copy(), args.max_delta, args.gripper_max_delta)

        print("연결 완료. 시작 자세를 기준으로 제한 이동합니다.")
        print(key_help())
        print_goal("start", goal)

        tty.setcbreak(sys.stdin.fileno())
        while True:
            key = read_key()
            if key is None:
                continue

            if key == "x":
                print("\n종료합니다.")
                return 0

            if key == "?":
                print("\n" + key_help())
                continue

            if key == " ":
                print_goal("target", goal)
                continue

            if key == "0":
                obs = robot.get_observation()
                goal = observation_to_goal(obs)
                limits = Limits(goal.copy(), args.max_delta, args.gripper_max_delta)
                print_goal("reset", goal)
                continue

            if key not in KEYMAP:
                continue

            motor, direction = KEYMAP[key]
            step = args.gripper_step if motor == "gripper" else args.step
            next_value = limits.clamp(motor, goal[motor] + direction * step)
            if next_value == goal[motor]:
                print(f"\r{motor}: {goal[motor]:7.2f}  limit    ", end="", flush=True)
                continue

            goal[motor] = next_value
            robot.send_action(action_from_goal(goal))
            print(f"\r{motor}: {goal[motor]:7.2f}    ", end="", flush=True)
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nCtrl+C로 종료합니다.")
        return 130
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        if robot.is_connected:
            if start_goal is not None and not args.no_return_start:
                print("\n시작 자세로 복귀합니다.")
                robot.send_action(action_from_goal(start_goal))
                time.sleep(1.0)
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Keyboard control for the SO-101 follower arm over SSH.

This is intended for slow, manual remote testing without a leader arm.
It starts from the follower's current pose and applies small joint deltas.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower


DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00"
DEFAULT_ID = "my_follower"
DEFAULT_TARGET_FRAME = "gripper_frame_link"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF_CANDIDATES = [
    Path("third_party/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"),
    Path("~/third_party/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"),
]

MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
ARM_MOTORS = MOTORS[:-1]

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

CARTESIAN_KEYMAP = {
    "w": ("x", +1),
    "s": ("x", -1),
    "a": ("y", +1),
    "d": ("y", -1),
    "r": ("z", +1),
    "f": ("z", -1),
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


@dataclass
class CartesianLimits:
    start_pos: np.ndarray
    max_delta_m: float

    def clamp(self, pos: np.ndarray) -> np.ndarray:
        low = self.start_pos - self.max_delta_m
        high = self.start_pos + self.max_delta_m
        return np.clip(pos, low, high)


def parse_home_pose(pose_str: str) -> dict[str, float]:
    values = [float(v) for v in pose_str.split(",")]
    if len(values) != len(MOTORS):
        raise ValueError(
            f"--home-pose는 쉼표로 구분된 {len(MOTORS)}개 값이 필요합니다.\n"
            f"순서: {','.join(MOTORS)}\n"
            f"예: --home-pose 0,0,0,0,0,50"
        )
    return dict(zip(MOTORS, values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyboard-control the SO-101 follower arm.")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Follower serial port")
    parser.add_argument("--id", default=DEFAULT_ID, help="Follower calibration id")
    parser.add_argument(
        "--home-pose",
        type=str,
        default=None,
        metavar="DEG,DEG,DEG,DEG,DEG,DEG",
        help=(
            "시작 및 종료 시 복귀할 홈 자세. 쉼표 구분 6개 값(degrees).\n"
            f"순서: {','.join(MOTORS)}\n"
            "예: --home-pose 0,0,0,0,0,50"
        ),
    )
    parser.add_argument(
        "--cartesian",
        action="store_true",
        help="Control the end-effector in XYZ space using IK instead of direct joint deltas",
    )
    parser.add_argument(
        "--urdf-path",
        help="SO-101 URDF path for --cartesian mode. Defaults to common project/home paths",
    )
    parser.add_argument(
        "--target-frame",
        default=DEFAULT_TARGET_FRAME,
        help="URDF end-effector frame name for --cartesian mode",
    )
    parser.add_argument("--step", type=float, default=2.0, help="Degrees per key press")
    parser.add_argument(
        "--xyz-step",
        type=float,
        default=0.005,
        help="Meters per key press in --cartesian mode",
    )
    parser.add_argument("--gripper-step", type=float, default=2.0, help="Gripper units per key press")
    parser.add_argument(
        "--max-delta",
        type=float,
        default=60.0,
        help="Max degrees each joint may move from the startup pose",
    )
    parser.add_argument(
        "--max-xyz-delta",
        type=float,
        default=0.08,
        help="Max meters the end-effector may move from the startup pose in --cartesian mode",
    )
    parser.add_argument(
        "--max-ik-joint-step",
        type=float,
        default=8.0,
        help="Max degrees each arm joint may change per IK command in --cartesian mode",
    )
    parser.add_argument(
        "--orientation-weight",
        type=float,
        default=0.0,
        help="IK orientation weight. Keep 0.0 for position-only XYZ control",
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


def joint_key_help() -> str:
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


def cartesian_key_help() -> str:
    return """
좌표 키 조작:
  w/s  x +/-
  a/d  y +/-
  r/f  z +/-
  y/h  gripper +/-

기타:
  space  현재 목표 관절값과 목표 XYZ 출력
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


def print_cartesian_goal(label: str, goal: dict[str, float], pos: np.ndarray) -> None:
    print_goal(label, goal)
    print(f"{label}.xyz: x={pos[0]: .4f}m  y={pos[1]: .4f}m  z={pos[2]: .4f}m")


def resolve_urdf_path(path_arg: str | None) -> Path:
    candidates = [Path(path_arg)] if path_arg else DEFAULT_URDF_CANDIDATES
    for candidate in candidates:
        path = candidate.expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            return path

    searched = "\n".join(f"  - {candidate.expanduser()}" for candidate in candidates)
    raise FileNotFoundError(f"SO-101 URDF를 찾지 못했습니다. --urdf-path로 지정하세요.\n검색 경로:\n{searched}")


def load_kinematics(urdf_path: Path, target_frame: str):
    try:
        from lerobot.model.kinematics import RobotKinematics
    except ImportError as exc:
        raise RuntimeError(
            "좌표 제어에는 LeRobot kinematics 의존성이 필요합니다.\n"
            "설치 예:\n"
            "  cd ~/lerobot\n"
            "  source .venv/bin/activate\n"
            "  pip install -e '.[kinematics]'"
        ) from exc

    try:
        old_cwd = Path.cwd()
        os.chdir(urdf_path.parent)
        try:
            return RobotKinematics(
                str(urdf_path),
                target_frame_name=target_frame,
                joint_names=ARM_MOTORS,
            )
        finally:
            os.chdir(old_cwd)
    except ImportError as exc:
        raise RuntimeError(
            "좌표 제어에는 LeRobot kinematics 의존성이 필요합니다.\n"
            "설치 예:\n"
            "  cd ~/lerobot\n"
            "  source .venv/bin/activate\n"
            "  pip install -e '.[kinematics]'"
        ) from exc
    except ValueError as exc:
        if "Mesh assets/" in str(exc):
            raise RuntimeError(
                "URDF가 참조하는 mesh assets를 찾지 못했습니다.\n"
                f"URDF 위치: {urdf_path}\n"
                f"필요 위치: {urdf_path.parent / 'assets'}\n"
                "SO-ARM100 repo의 Simulation/SO101/assets 폴더를 URDF 옆에 둬야 합니다."
            ) from exc
        raise


def arm_goal_array(goal: dict[str, float]) -> np.ndarray:
    return np.array([goal[motor] for motor in ARM_MOTORS], dtype=float)


def compute_ik_goal(args: argparse.Namespace, kinematics, goal: dict[str, float], target_pose: np.ndarray) -> dict[str, float]:
    q_current = arm_goal_array(goal)
    try:
        q_target = kinematics.inverse_kinematics(
            q_current,
            target_pose,
            orientation_weight=args.orientation_weight,
        )
    except Exception as exc:
        raise RuntimeError(f"IK 계산 실패: {exc}") from exc

    next_goal = goal.copy()
    for i, motor in enumerate(ARM_MOTORS):
        raw = float(q_target[i])
        limited = min(max(raw, goal[motor] - args.max_ik_joint_step), goal[motor] + args.max_ik_joint_step)
        next_goal[motor] = limited
    return next_goal


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
    kinematics = None
    target_pose: np.ndarray | None = None
    target_pos: np.ndarray | None = None
    cartesian_limits: CartesianLimits | None = None

    if args.cartesian:
        urdf_path = resolve_urdf_path(args.urdf_path)
        print(f"좌표 제어 URDF: {urdf_path}")
        kinematics = load_kinematics(urdf_path, args.target_frame)

    old_term = termios.tcgetattr(sys.stdin)
    home_pose = parse_home_pose(args.home_pose) if args.home_pose else None

    try:
        print("SO-101 follower 연결 중...")
        robot.connect()

        if home_pose is not None:
            print("홈 자세로 이동합니다...")
            robot.send_action(action_from_goal(home_pose))
            time.sleep(1.5)

        obs = robot.get_observation()
        goal = observation_to_goal(obs)
        start_goal = home_pose.copy() if home_pose is not None else goal.copy()
        limits = Limits(start_goal.copy(), args.max_delta, args.gripper_max_delta)

        if args.cartesian:
            if kinematics is None:
                raise RuntimeError("좌표 제어 초기화 실패")
            target_pose = kinematics.forward_kinematics(arm_goal_array(goal))
            target_pos = target_pose[:3, 3].copy()
            cartesian_limits = CartesianLimits(target_pos.copy(), args.max_xyz_delta)
            print("연결 완료. 시작 그리퍼 좌표를 기준으로 제한 이동합니다.")
            print(cartesian_key_help())
            print_cartesian_goal("start", goal, target_pos)
        else:
            print("연결 완료. 시작 자세를 기준으로 제한 이동합니다.")
            print(joint_key_help())
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
                print("\n" + (cartesian_key_help() if args.cartesian else joint_key_help()))
                continue

            if key == " ":
                if args.cartesian and target_pos is not None:
                    print_cartesian_goal("target", goal, target_pos)
                else:
                    print_goal("target", goal)
                continue

            if key == "0":
                obs = robot.get_observation()
                goal = observation_to_goal(obs)
                limits = Limits(goal.copy(), args.max_delta, args.gripper_max_delta)
                if args.cartesian:
                    if kinematics is None:
                        raise RuntimeError("좌표 제어 초기화 실패")
                    target_pose = kinematics.forward_kinematics(arm_goal_array(goal))
                    target_pos = target_pose[:3, 3].copy()
                    cartesian_limits = CartesianLimits(target_pos.copy(), args.max_xyz_delta)
                    print_cartesian_goal("reset", goal, target_pos)
                else:
                    print_goal("reset", goal)
                continue

            if args.cartesian:
                if key not in CARTESIAN_KEYMAP:
                    continue

                axis, direction = CARTESIAN_KEYMAP[key]
                if axis == "gripper":
                    next_value = limits.clamp("gripper", goal["gripper"] + direction * args.gripper_step)
                    if next_value == goal["gripper"]:
                        print(f"\rgripper: {goal['gripper']:7.2f}  limit    ", end="", flush=True)
                        continue
                    goal["gripper"] = next_value
                    robot.send_action(action_from_goal(goal))
                    print(f"\rgripper: {goal['gripper']:7.2f}    ", end="", flush=True)
                    time.sleep(0.02)
                    continue

                if target_pose is None or target_pos is None or cartesian_limits is None or kinematics is None:
                    raise RuntimeError("좌표 제어 초기화 실패")

                axis_index = {"x": 0, "y": 1, "z": 2}[axis]
                requested_pos = target_pos.copy()
                requested_pos[axis_index] += direction * args.xyz_step
                next_pos = cartesian_limits.clamp(requested_pos)
                if np.array_equal(next_pos, target_pos):
                    print(f"\r{axis}: {target_pos[axis_index]: .4f}m  limit    ", end="", flush=True)
                    continue

                next_pose = target_pose.copy()
                next_pose[:3, 3] = next_pos
                next_goal = compute_ik_goal(args, kinematics, goal, next_pose)
                for motor in ARM_MOTORS:
                    next_goal[motor] = limits.clamp(motor, next_goal[motor])
                goal = next_goal
                target_pos = next_pos
                target_pose = next_pose
                robot.send_action(action_from_goal(goal))
                print(
                    f"\r{axis}: {target_pos[axis_index]: .4f}m  "
                    f"xyz=({target_pos[0]: .3f},{target_pos[1]: .3f},{target_pos[2]: .3f})    ",
                    end="",
                    flush=True,
                )
                time.sleep(0.02)
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
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

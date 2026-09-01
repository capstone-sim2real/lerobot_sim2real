#!/usr/bin/env python3
"""Move the SO-101 follower along a guarded Cartesian IK path."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = PROJECT_ROOT / "third_party/so101/so101.urdf"
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", required=True, type=float, help="Target gripper x in meters")
    parser.add_argument("--y", required=True, type=float, help="Target gripper y in meters")
    parser.add_argument("--z", required=True, type=float, help="Target gripper z in meters")
    parser.add_argument("--step-m", type=float, default=0.002, help="Maximum Cartesian advance per control cycle in meters")
    parser.add_argument("--max-joint-step", type=float, default=2.0, help="Maximum joint delta per control cycle in degrees")
    parser.add_argument("--step-delay", type=float, default=0.05, help="Control-cycle delay in seconds")
    parser.add_argument("--tolerance-m", type=float, default=0.001, help="Waypoint completion tolerance in meters")
    parser.add_argument("--max-cycles", type=int, default=1000, help="Maximum feedback-control cycles")
    parser.add_argument("--dry-run", action="store_true", help="Plan and validate without enabling torque or sending commands")
    parser.add_argument("--port", default=os.environ.get("SO101_FOLLOWER_PORT", "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00"))
    parser.add_argument("--robot-id", default=os.environ.get("SO101_FOLLOWER_ID", "my_follower"))
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--target-frame", default="gripper_frame_link")
    return parser.parse_args()


def load_kinematics(urdf_path: Path, target_frame: str) -> RobotKinematics:
    urdf_path = urdf_path.expanduser().resolve()
    old_cwd = Path.cwd()
    os.chdir(urdf_path.parent)
    try:
        return RobotKinematics(str(urdf_path), target_frame_name=target_frame, joint_names=MOTORS)
    finally:
        os.chdir(old_cwd)


def next_ik_command(
    kinematics: RobotKinematics,
    q_actual: np.ndarray,
    waypoint: np.ndarray,
    step_m: float,
    max_joint_step: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Plan only the next small move from the measured, not commanded, pose."""
    pose_actual = kinematics.forward_kinematics(q_actual)
    current_xyz = pose_actual[:3, 3]
    remaining = waypoint - current_xyz
    distance = float(np.linalg.norm(remaining))
    if distance <= step_m:
        next_xyz = waypoint
    else:
        next_xyz = current_xyz + remaining / distance * step_m

    pose_target = pose_actual.copy()
    pose_target[:3, 3] = next_xyz
    q_next = kinematics.inverse_kinematics(q_actual, pose_target, orientation_weight=1.0)
    largest_delta = float(np.max(np.abs(q_next - q_actual)))
    if largest_delta > max_joint_step:
        raise RuntimeError(
            f"Unsafe IK transition: {largest_delta:.2f}° exceeds "
            f"--max-joint-step={max_joint_step:.2f}°"
        )
    return q_next, next_xyz


def main() -> int:
    args = parse_args()
    if args.step_m <= 0 or args.max_joint_step <= 0:
        raise ValueError("--step-m and --max-joint-step must be positive")

    kinematics = load_kinematics(args.urdf_path, args.target_frame)
    robot = SOFollower(SOFollowerRobotConfig(id=args.robot_id, port=args.port, use_degrees=True, disable_torque_on_disconnect=False))
    try:
        robot.bus.connect()
        current = robot.bus.sync_read("Present_Position")
        q_start = np.array([float(current[motor]) for motor in MOTORS])
        pose_start = kinematics.forward_kinematics(q_start)
        start_xyz = pose_start[:3, 3].copy()
        target_xyz = np.array([args.x, args.y, args.z], dtype=float)

        # Rise first, then translate laterally at the requested safe height.
        waypoints = [np.array([start_xyz[0], start_xyz[1], target_xyz[2]]), target_xyz]

        def follow_waypoints(read_q, send_q) -> tuple[int, float]:
            cycles = 0
            max_delta = 0.0
            for waypoint in waypoints:
                while True:
                    if cycles >= args.max_cycles:
                        raise RuntimeError("Maximum feedback-control cycles reached")
                    q_actual = read_q()
                    current_xyz = kinematics.forward_kinematics(q_actual)[:3, 3]
                    if float(np.linalg.norm(waypoint - current_xyz)) <= args.tolerance_m:
                        break
                    q_next, _ = next_ik_command(
                        kinematics, q_actual, waypoint, args.step_m, args.max_joint_step
                    )
                    max_delta = max(max_delta, float(np.max(np.abs(q_next - q_actual))))
                    send_q(q_next)
                    cycles += 1
            return cycles, max_delta

        print(f"start_xyz_m={start_xyz}")
        print(f"target_xyz_m={target_xyz}")
        if args.dry_run:
            q_simulated = q_start.copy()

            def read_simulated() -> np.ndarray:
                return q_simulated

            def send_simulated(q_next: np.ndarray) -> None:
                nonlocal q_simulated
                q_simulated = q_next

            cycles, max_delta = follow_waypoints(read_simulated, send_simulated)
            print(f"planned_cycles={cycles}")
            print(f"max_joint_step_deg={max_delta:.3f}")
            print("dry-run complete; no torque or position command was sent")
            return 0

        robot.bus.enable_torque()
        def read_actual() -> np.ndarray:
            actual = robot.bus.sync_read("Present_Position")
            return np.array([float(actual[motor]) for motor in MOTORS])

        def send_actual(q_next: np.ndarray) -> None:
            robot.bus.sync_write("Goal_Position", dict(zip(MOTORS, q_next.tolist())))
            time.sleep(args.step_delay)

        cycles, max_delta = follow_waypoints(read_actual, send_actual)
        print(f"control_cycles={cycles}")
        print(f"max_joint_step_deg={max_delta:.3f}")
        print("motion complete")
        return 0
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        raise SystemExit(1) from exc

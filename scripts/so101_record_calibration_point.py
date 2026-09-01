#!/usr/bin/env python3
"""Record one SO-101 camera-to-robot calibration point.

The arm must already be positioned manually with the gripper reference point
over a chosen chessboard intersection. This script sends no motion command.
It saves the current top-camera image and appends the follower FK pose to CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = PROJECT_ROOT / "third_party/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
ARM_MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# Joint angles are recorded alongside the FK position because the position
# alone cannot be re-analysed later: gripper_frame_link sits ~8mm off the
# wrist_roll axis, so the same jaw placement yields different recorded xyz
# depending on wrist_roll, and without the joints that offset cannot be
# reconstructed or corrected for (AGENTS.md §6/§7).
CSV_FIELDS = [
    "name", "image", "u_px", "v_px", "x_m", "y_m", "z_m",
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Point label, e.g. P1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs/calibration",
        help="Directory for point images and points.csv",
    )
    parser.add_argument(
        "--snapshot-url",
        default="http://127.0.0.1:8090/snapshot/shoulder.jpg",
        help="Top-camera JPEG snapshot URL",
    )
    parser.add_argument(
        "--port",
        default=os.environ.get(
            "SO101_FOLLOWER_PORT",
            "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00",
        ),
        help="Follower serial port",
    )
    parser.add_argument("--robot-id", default=os.environ.get("SO101_FOLLOWER_ID", "my_follower"))
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--target-frame", default="gripper_frame_link")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing row and image with the same point name",
    )
    return parser.parse_args()


def load_kinematics(urdf_path: Path, target_frame: str) -> RobotKinematics:
    urdf_path = urdf_path.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    old_cwd = Path.cwd()
    os.chdir(urdf_path.parent)
    try:
        return RobotKinematics(str(urdf_path), target_frame_name=target_frame, joint_names=ARM_MOTORS)
    finally:
        os.chdir(old_cwd)


def read_follower_xyz(port: str, robot_id: str, kinematics: RobotKinematics) -> tuple[np.ndarray, np.ndarray]:
    robot = SOFollower(
        SOFollowerRobotConfig(
            id=robot_id,
            port=port,
            max_relative_target=10,
            use_degrees=True,
            disable_torque_on_disconnect=False,
        )
    )
    try:
        # Bypass SOFollower.connect(): it re-applies calibration and motor
        # configuration on every connection. Recording a point only needs a
        # calibrated position read, so avoid all motor writes here.
        robot.bus.connect()
        observation = robot.bus.sync_read("Present_Position")
        joints = np.array([float(observation[motor]) for motor in ARM_MOTORS])
        pose = kinematics.forward_kinematics(joints)
        return joints, pose[:3, 3].copy()
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)


def fetch_snapshot(url: str) -> bytes:
    with urlopen(url, timeout=5) as response:  # nosec B310 -- user-controlled local camera URL
        data = response.read()
    if not data.startswith(b"\xff\xd8"):
        raise RuntimeError(f"Snapshot from {url} is not a JPEG")
    return data


def update_csv(csv_path: Path, row: dict[str, str], overwrite: bool) -> None:
    rows: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

    exists = any(existing["name"] == row["name"] for existing in rows)
    if exists and not overwrite:
        raise RuntimeError(f"Point {row['name']} already exists; use --overwrite to replace it")
    rows = [existing for existing in rows if existing["name"] != row["name"]]
    rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / f"{args.name.lower()}_top.jpg"
    csv_path = args.output_dir / "points.csv"

    if image_path.exists() and not args.overwrite:
        raise RuntimeError(f"Image already exists: {image_path}; use --overwrite to replace it")

    print("Capturing top-camera frame...")
    image_path.write_bytes(fetch_snapshot(args.snapshot_url))
    print("Reading follower pose (no motion command)...")
    kinematics = load_kinematics(args.urdf_path, args.target_frame)
    joints, xyz = read_follower_xyz(args.port, args.robot_id, kinematics)

    update_csv(
        csv_path,
        {
            "name": args.name,
            "image": image_path.name,
            "u_px": "",
            "v_px": "",
            "x_m": f"{xyz[0]:.6f}",
            "y_m": f"{xyz[1]:.6f}",
            "z_m": f"{xyz[2]:.6f}",
            **{motor: f"{value:.3f}" for motor, value in zip(ARM_MOTORS, joints)},
            "notes": "FK at gripper_frame_link; pixel coordinate pending chessboard intersection selection",
        },
        args.overwrite,
    )
    print("joint_degrees=" + ", ".join(f"{motor}={value:.3f}" for motor, value in zip(ARM_MOTORS, joints)))
    print(f"saved={image_path}")
    print(f"gripper_xyz_m=x={xyz[0]:.6f}, y={xyz[1]:.6f}, z={xyz[2]:.6f}")
    print(f"csv={csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

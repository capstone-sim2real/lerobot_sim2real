"""Calibrate the SO-101 leader and/or follower through the LeRobot CLI."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


DEFAULT_LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6085435-if00"
DEFAULT_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("leader", "follower", "all"), help="device(s) to calibrate")
    parser.add_argument("--reset", action="store_true", help="delete existing calibration before calibrating")
    parser.add_argument("--leader-port", default=os.environ.get("SO101_LEADER_PORT", DEFAULT_LEADER_PORT))
    parser.add_argument("--follower-port", default=os.environ.get("SO101_FOLLOWER_PORT", DEFAULT_FOLLOWER_PORT))
    parser.add_argument("--leader-id", default=os.environ.get("SO101_LEADER_ID", "my_leader"))
    parser.add_argument("--follower-id", default=os.environ.get("SO101_FOLLOWER_ID", "my_follower"))
    return parser.parse_args()


def _calibration_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
    return root / "teleoperators" / "so_leader" / f"{args.leader_id}.json", root / "robots" / "so_follower" / f"{args.follower_id}.json"


def _run(*args: str) -> None:
    subprocess.run(["lerobot-calibrate", *args], check=True)


def main() -> int:
    args = parse_args()
    if args.reset:
        for path in _calibration_paths(args):
            path.unlink(missing_ok=True)
            print(f"removed {path}")
    if args.target in ("leader", "all"):
        _run("--teleop.type=so101_leader", f"--teleop.port={args.leader_port}", f"--teleop.id={args.leader_id}")
    if args.target in ("follower", "all"):
        _run("--robot.type=so101_follower", f"--robot.port={args.follower_port}", f"--robot.id={args.follower_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

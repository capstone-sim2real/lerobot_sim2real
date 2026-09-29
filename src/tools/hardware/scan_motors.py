"""Scan the configured SO-101 leader and follower serial buses."""

from __future__ import annotations

import argparse
import os

from lerobot.motors.feetech import FeetechMotorsBus


DEFAULT_LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6085435-if00"
DEFAULT_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", action="append", help="serial port to scan; repeat to scan more than one")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ports = args.port or [
        os.environ.get("SO101_LEADER_PORT", DEFAULT_LEADER_PORT),
        os.environ.get("SO101_FOLLOWER_PORT", DEFAULT_FOLLOWER_PORT),
    ]
    for port in dict.fromkeys(ports):
        print(f"\nPORT {port}")
        print(FeetechMotorsBus.scan_port(port, protocol_version=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

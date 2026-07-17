"""Record named joint poses into poses.yaml by hand-positioning the arm.

    # torque off -> move the arm by hand -> Enter -> saved
    python -m pick_stack.tools.record_pose --name home

    # keep torque on and snapshot the current pose (e.g. after teleop)
    python -m pick_stack.tools.record_pose --name retreat --keep-torque

    # list what has been recorded so far
    python -m pick_stack.tools.record_pose --list

Record home/retreat BEFORE collecting episodes: the same numbers must anchor
the teleop convention and the runtime FSM (EPISODE.md §1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pick_stack.config import load_config
from pick_stack.control import So101RobotIO
from pick_stack.control.poses import PoseRegistry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", help="pose name to record (e.g. home, retreat, slot_0, tower_descent_3)")
    parser.add_argument("--list", action="store_true", help="print recorded poses and exit")
    parser.add_argument("--keep-torque", action="store_true", help="do not disable torque (snapshot current pose)")
    parser.add_argument("--config", default="src/pick_stack/configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="key.path=value")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    poses_path = Path(cfg.motion.poses_path)
    registry = PoseRegistry.load(poses_path) if poses_path.exists() else PoseRegistry(path=poses_path)

    if args.list:
        if not registry.names():
            print(f"No poses recorded yet in {poses_path}")
            return 0
        for name in registry.names():
            pose = registry.get(name)
            print(f"{name:20s} " + "  ".join(f"{j}={v:7.2f}" for j, v in pose.items()))
        return 0

    if not args.name:
        parser.error("--name is required unless --list is given")

    cfg.robot.cameras = {}  # motor bus only
    robot = So101RobotIO(cfg.robot)
    try:
        robot.connect()
    except Exception as e:
        print(f"Robot connect failed: {e}", file=sys.stderr)
        return 1
    try:
        if args.name in registry:
            old = registry.get(args.name)
            print(f"'{args.name}'은 이미 기록됨: " + "  ".join(f"{j}={v:.1f}" for j, v in old.items()))
            if input("덮어쓸까요? [y/N] > ").strip().lower() != "y":
                return 0
        if not args.keep_torque:
            robot.set_torque(False)
            print("토크 해제됨 — 팔을 손으로 원하는 자세로 옮기세요.")
        if input(f"Enter=현재 자세를 '{args.name}'으로 저장, q=취소 > ").strip().lower() == "q":
            return 0
        pose = robot.read_joints()
        registry.set(args.name, pose)
        registry.save()
        print(f"저장됨: {args.name} -> {poses_path}")
        print("  " + "  ".join(f"{j}={v:7.2f}" for j, v in pose.items()))
        if not args.keep_torque:
            print("주의: 토크는 해제 상태로 종료합니다. 팔을 안전한 자세로 받쳐 두세요.")
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Threshold tuning CLI for grasp verification and contact detection.

Grasp trials (label each trial, then read the two distributions apart):

    python -m tools.tune_gripper_load --mode grasp --csv /tmp/grasp.csv

    각 트라이얼: Enter → 그리퍼 열림 → 블록을 물리거나(held) 빈손으로 두고
    Enter → 닫힘 → 측정값 + 판정 출력 → 실제로 잡혔는지 y/n 입력(정답 라벨)

Passive watch (hand-move the arm, press the gripper on a block, watch loads):

    python -m tools.tune_gripper_load --mode watch --hz 5

Cameras are not opened — this tool only touches the motor bus, so it can run
while camera pipelines are being debugged elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from config import load_config
from control import So101RobotIO, check_grasp
from control.robot_io import JOINT_NAMES


def _open_csv(path: str | None, header: list[str]):
    if path is None:
        return None, None
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(header)
    return f, writer


def run_grasp_trials(robot: So101RobotIO, cfg, csv_path: str | None) -> None:
    f, writer = _open_csv(
        csv_path,
        ["trial", "actually_held", "verdict", "gripper_pos", "gripper_load_abs", "pos_says_held", "load_says_held"],
    )
    trial = 0
    print(f"mode={cfg.sensing.grasp_check_mode}  empty_closed_max={cfg.sensing.gripper_empty_closed_max}  load_min={cfg.sensing.gripper_load_min}")
    print("Enter=트라이얼 시작, q=종료")
    try:
        while True:
            if input(f"\n[trial {trial}] Enter=그리퍼 열기 > ").strip().lower() == "q":
                break
            robot.send_joints({"gripper": cfg.sensing.gripper_open_pos})
            if input("블록을 물릴 위치에 두고 Enter=닫기 (빈손 트라이얼이면 그냥 Enter) > ").strip().lower() == "q":
                break
            robot.send_joints({"gripper": cfg.sensing.gripper_close_pos})
            result = check_grasp(robot, cfg.sensing)
            print(
                f"  verdict={'HELD' if result.grasped else 'EMPTY'}  pos={result.gripper_pos:.1f} "
                f"load={result.gripper_load_abs:.0f}  (pos_says={result.pos_says_held} load_says={result.load_says_held})"
            )
            label = input("  실제로 잡혔나요? [y/n] > ").strip().lower()
            if writer:
                writer.writerow(
                    [trial, label == "y", result.grasped, f"{result.gripper_pos:.2f}",
                     f"{result.gripper_load_abs:.1f}", result.pos_says_held, result.load_says_held]
                )
                f.flush()
            trial += 1
    finally:
        robot.send_joints({"gripper": cfg.sensing.gripper_open_pos})
        if f:
            f.close()
            print(f"CSV saved: {csv_path}")


def run_watch(robot: So101RobotIO, cfg, hz: float, csv_path: str | None) -> None:
    f, writer = _open_csv(csv_path, ["t", *(f"{j}.pos" for j in JOINT_NAMES), *(f"{j}.load" for j in JOINT_NAMES)])
    period = 1.0 / hz
    t0 = time.monotonic()
    print("Ctrl+C=종료")
    try:
        while True:
            joints = robot.read_joints()
            loads = robot.read_loads()
            line = "  ".join(f"{j}:{joints[j]:6.1f}/{loads[j]:5.0f}" for j in JOINT_NAMES)
            print(f"\r{line}", end="", flush=True)
            if writer:
                t = time.monotonic() - t0
                writer.writerow([f"{t:.3f}", *(f"{joints[j]:.2f}" for j in JOINT_NAMES), *(loads[j] for j in JOINT_NAMES)])
            time.sleep(period)
    except KeyboardInterrupt:
        print()
    finally:
        if f:
            f.close()
            print(f"CSV saved: {csv_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["grasp", "watch"], default="grasp")
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="key.path=value")
    parser.add_argument("--hz", type=float, default=5.0, help="watch mode sample rate")
    parser.add_argument("--csv", default=None, help="append measurements to this CSV")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    cfg.robot.cameras = {}  # motor bus only
    robot = So101RobotIO(cfg.robot)
    try:
        robot.connect()
    except Exception as e:
        print(f"Robot connect failed: {e}", file=sys.stderr)
        return 1
    try:
        if args.mode == "grasp":
            run_grasp_trials(robot, cfg, args.csv)
        else:
            run_watch(robot, cfg, args.hz, args.csv)
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

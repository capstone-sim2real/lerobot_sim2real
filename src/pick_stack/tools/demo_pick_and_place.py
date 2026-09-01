"""End-to-end smoke test: detect one block by colour, pick it, move it to a
named calibration point. This is the manual rehearsal for what
fsm/ik_handlers.py will do automatically — everything here is glue over
already-existing pieces (perception detector/homography, TopDownIK,
TrajectoryPlayer), nothing new is implemented.

    # plan only, no hardware connection, no motion:
    python -m pick_stack.tools.demo_pick_and_place --color green --to P5 --dry-run

    # the real thing:
    python -m pick_stack.tools.demo_pick_and_place --color green --to P5

Preconditions: so101_camera_web.sh running (top camera), venue calibration
at --calib, arm powered and reachable, workspace clear of obstructions.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from pick_stack.config import AppConfig, load_config
from pick_stack.control.ik import TopDownIK
from pick_stack.perception.homography import PlaneCalibration
from pick_stack.tools._capture import grab
from pick_stack.tools.auto_pick_pixels import find_block_centroid

HOVER_CLEARANCE_MM = 60.0


def load_named_point(points_csv: Path, name: str) -> tuple[float, float, float]:
    rows = list(csv.DictReader(points_csv.open(newline="", encoding="utf-8")))
    for r in rows:
        if r["name"] == name:
            return float(r["x_m"]) * 1000.0, float(r["y_m"]) * 1000.0, float(r["z_m"]) * 1000.0
    raise KeyError(f"'{name}' not found in {points_csv} (have: {[r['name'] for r in rows]})")


def plan(cfg: AppConfig, color: str, dest_name: str, snapshot_url: str, points_csv: Path):
    calib = PlaneCalibration.load(cfg.perception.calibration_path)
    frame = grab(snapshot_url)
    result = find_block_centroid(frame, color)
    if result is None:
        raise RuntimeError(f"No {color} block found in the current camera view ({snapshot_url})")
    cx, cy, _contour, area = result
    xy_mm = calib.pixel_to_board(np.array([[cx, cy]]))[0]
    pick_x, pick_y = float(xy_mm[0]), float(xy_mm[1])
    grasp_z = float(calib.meta.get("grasp_z_mm_mean", 5.0))

    place_x, place_y, place_z = load_named_point(points_csv, dest_name)

    ik = TopDownIK(cfg.ik, project_root=".")
    hover_z = grasp_z + HOVER_CLEARANCE_MM
    waypoints = {
        "pick_hover": ik.solve(pick_x, pick_y, hover_z),
        "pick_grasp": ik.solve(pick_x, pick_y, grasp_z),
        "place_hover": ik.solve(place_x, place_y, place_z + HOVER_CLEARANCE_MM),
        "place_drop": ik.solve(place_x, place_y, place_z),
    }
    return {
        "pixel": (cx, cy),
        "area_px": area,
        "pick_xy_mm": (pick_x, pick_y),
        "grasp_z_mm": grasp_z,
        "place_xy_mm": (place_x, place_y),
        "place_z_mm": place_z,
        "waypoints": waypoints,
    }


def print_plan(p: dict) -> None:
    print(f"detected {p['area_px']:.0f}px blob at pixel {p['pixel']}")
    print(f"pick target : x={p['pick_xy_mm'][0]:7.1f}mm y={p['pick_xy_mm'][1]:7.1f}mm grasp_z={p['grasp_z_mm']:.1f}mm")
    print(f"place target: x={p['place_xy_mm'][0]:7.1f}mm y={p['place_xy_mm'][1]:7.1f}mm z={p['place_z_mm']:.1f}mm")
    print()
    print(f"{'waypoint':12s} {'pos_err_mm':>10s} {'tilt_err_deg':>12s}  joints")
    for name, r in p["waypoints"].items():
        flag = "  <-- CHECK" if r.position_error_mm > 15.0 or r.tilt_error_deg > 6.0 else ""
        joints = "  ".join(f"{j}={v:.1f}" for j, v in r.joints.items())
        print(f"{name:12s} {r.position_error_mm:10.2f} {r.tilt_error_deg:12.2f}  {joints}{flag}")


def execute(cfg: AppConfig, p: dict) -> None:
    from pick_stack.control.poses import PoseRegistry
    from pick_stack.control.robot_io import So101RobotIO
    from pick_stack.control.trajectory import TrajectoryPlayer

    cfg.robot.cameras = {}  # motor bus only; the top camera is already held by camera_web.py
    robot = So101RobotIO(cfg.robot)
    robot.connect()
    player = TrajectoryPlayer(robot, cfg.motion)
    try:
        wp = p["waypoints"]
        # start from a known pose: whatever the arm was left at after manual
        # calibration handling could be arbitrarily far from the pick target,
        # and a huge first jump can outrun move_timeout_s (AGENTS.md §11 —
        # a trajectory always starts from measured current pose, but SELECT
        # goes through home first for exactly this reason).
        poses = PoseRegistry.load(cfg.motion.poses_path)
        print("going home first...")
        player.move_to(poses.get(cfg.motion.home_pose), max_step=1.0)
        print("opening gripper, moving to pick_hover...")
        player.move_to({"gripper": cfg.sensing.gripper_open_pos})
        player.move_to(wp["pick_hover"].joints, max_step=1.0)
        print("descending to grasp...")
        player.move_to(wp["pick_grasp"].joints, max_step=cfg.motion.descent_step_per_tick)
        print("closing gripper...")
        player.move_to({"gripper": cfg.sensing.gripper_close_pos})
        time.sleep(cfg.sensing.grasp_settle_s)
        print("lifting...")
        player.move_to(wp["pick_hover"].joints)
        print("moving to place_hover...")
        player.move_to(wp["place_hover"].joints)
        print("descending to place...")
        player.move_to(wp["place_drop"].joints, max_step=cfg.motion.descent_step_per_tick)
        print("opening gripper (release)...")
        player.move_to({"gripper": cfg.sensing.gripper_open_pos})
        time.sleep(cfg.motion.place_settle_s)
        print("lifting clear...")
        player.move_to(wp["place_hover"].joints)
        print("done.")
    finally:
        try:
            player.move_to({"gripper": cfg.sensing.gripper_open_pos})
        except Exception as e:
            print(f"warning: safe-open on shutdown failed: {e}", file=sys.stderr)
        robot.disconnect()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--color", required=True)
    ap.add_argument("--to", required=True, help="destination point name in --points (e.g. P5)")
    ap.add_argument("--config", default="src/pick_stack/configs/default.yaml")
    ap.add_argument("--points", type=Path, default=Path("docs/calibration/points.csv"))
    ap.add_argument("--snapshot-url", default="http://127.0.0.1:8090/snapshot/shoulder.jpg")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    p = plan(cfg, args.color, args.to, args.snapshot_url, args.points)
    print_plan(p)

    if args.dry_run:
        print("\n--dry-run: no hardware connection, no motion.")
        return 0

    for r in p["waypoints"].values():
        if r.position_error_mm > 15.0 or r.tilt_error_deg > 6.0:
            print("\nABORT: a waypoint exceeds the IK error gate — check the printout above.", file=sys.stderr)
            return 1

    if input("\nAbout to move the real arm. Type 'go' to proceed: ").strip().lower() != "go":
        print("cancelled.")
        return 0
    execute(cfg, p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

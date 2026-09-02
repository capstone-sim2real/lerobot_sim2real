"""End-to-end smoke test: detect one block by colour, pick it, move it to a
named calibration point. This is the manual rehearsal for what
fsm/ik_handlers.py will do automatically — everything here is glue over
already-existing pieces (perception detector/homography, TopDownIK,
TrajectoryPlayer, control/grasp.py), nothing new is implemented.

    # plan only, no hardware connection, no motion:
    python -m tools.demo_pick_and_place --color green --to P5 --dry-run

    # the real thing:
    python -m tools.demo_pick_and_place --color green --to P5

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

from config import AppConfig, load_config
from camera.client import DEFAULT_SHOULDER_SNAPSHOT_URL, fetch_snapshot
from control.grasp import biased_grasp_xy, highest_reachable_hover, plan_grasp_attempts
from control.ik import IkResult, TopDownIK
from perception import detect_blocks
from perception.homography import PlaneCalibration


def pull_in(x_mm: float, y_mm: float, radius_mm: float) -> tuple[float, float]:
    """The same azimuth, but no further out than ``radius_mm``.

    The top-down envelope collapses with reach — wrist_flex is on its +-95
    deg limit everywhere, so lift is bought purely by folding the arm in.
    Measured: ~90mm of lift at 195mm reach against ~50mm at 285mm.
    """
    r = float(np.hypot(x_mm, y_mm))
    if radius_mm <= 0.0 or r <= radius_mm:
        return (x_mm, y_mm)
    scale = radius_mm / r
    return (x_mm * scale, y_mm * scale)


def load_named_point(points_csv: Path, name: str) -> tuple[float, float, float]:
    rows = list(csv.DictReader(points_csv.open(newline="", encoding="utf-8")))
    for r in rows:
        if r["name"] == name:
            return float(r["x_m"]) * 1000.0, float(r["y_m"]) * 1000.0, float(r["z_m"]) * 1000.0
    raise KeyError(f"'{name}' not found in {points_csv} (have: {[r['name'] for r in rows]})")


def plan(cfg: AppConfig, color: str, dest_name: str, snapshot_url: str, points_csv: Path):
    calib = PlaneCalibration.load(cfg.perception.calibration_path)
    frame = fetch_snapshot(snapshot_url)
    detections = [d for d in detect_blocks(frame, calib, cfg.perception, is_rgb=False) if d.color == color]
    if not detections:
        raise RuntimeError(f"No {color} block found in the current camera view ({snapshot_url})")
    if len(detections) > 1:
        raise RuntimeError(f"Found {len(detections)} {color} blocks; use the FSM selector for multi-block runs")
    detection = detections[0]
    det_x, det_y = detection.center_mm
    grasp_z = float(calib.meta.get("grasp_z_mm_mean", 5.0))

    place_x, place_y, place_z = load_named_point(points_csv, dest_name)

    ik = TopDownIK(cfg.ik, project_root=".")
    # search the hover height at the biased aim point, not the raw detection:
    # that is where the arm will actually hold station.
    aim_x, aim_y = biased_grasp_xy(cfg.motion, det_x, det_y)
    place_hover_z = highest_reachable_hover(ik, place_x, place_y, place_z, cfg)
    return {
        "color": detection.color,
        "area_mm2": detection.area_mm2,
        "grasp_plan": plan_grasp_attempts(ik, cfg, det_x, det_y, grasp_z, log=print),
        "place_xy_mm": (place_x, place_y),
        "place_z_mm": place_z,
        "place_hover_z_mm": place_hover_z,
        "place_waypoints": {
            "place_hover": ik.solve(place_x, place_y, place_hover_z),
            "place_drop": ik.solve(place_x, place_y, place_z),
        },
        # Carried across folded in, where the envelope is tallest, instead of
        # at the reach the block was picked from.
        "carry_waypoints": _carry_waypoints(
            ik, cfg, (aim_x, aim_y), grasp_z, (place_x, place_y), place_z
        ),
    }


def _carry_waypoints(ik, cfg, pick_xy, grasp_z, place_xy, place_z) -> dict:
    """Apex poses over the pick and place columns, at the retracted radius.

    Only kept when the retraction actually buys height and the pose clears
    the IK gate; otherwise the carry falls back to the direct move.
    """
    out: dict[str, IkResult] = {}
    r = cfg.motion.transit_apex_radius_mm
    for name, (x, y), base_z in (("apex_pick", pick_xy, grasp_z), ("apex_place", place_xy, place_z)):
        ax, ay = pull_in(x, y, r)
        if (ax, ay) == (x, y):
            continue  # already folded in this far; nothing to gain
        az = highest_reachable_hover(ik, ax, ay, base_z, cfg)
        solved = ik.solve(ax, ay, az)
        if not _over_gate(cfg, solved):
            out[name] = solved
    return out


def _over_gate(cfg: AppConfig, r: IkResult) -> bool:
    return (
        r.position_error_mm > cfg.ik.max_position_error_mm
        or r.tilt_error_deg > cfg.ik.max_tilt_error_deg
    )


def print_plan(cfg: AppConfig, p: dict) -> None:
    gp = p["grasp_plan"]
    det_x, det_y = gp.detected_xy_mm
    aim_x, aim_y = gp.biased_xy_mm
    side = "left" if det_y > cfg.motion.left_half_y_mm else "right"
    print(f"detected {p['color']} block, area={p['area_mm2']:.0f}mm²")
    print(f"detected   : x={det_x:7.1f}mm y={det_y:7.1f}mm  ({side} half)")
    print(
        f"aim point  : x={aim_x:7.1f}mm y={aim_y:7.1f}mm  "
        f"(gripper-frame bias, shifted {np.hypot(aim_x - det_x, aim_y - det_y):.1f}mm)"
    )
    print(f"grasp_z    : {gp.grasp_z_mm:.1f}mm   hover_z: {gp.hover_z_mm:.0f}mm (highest top-down reachable)")
    print(f"place      : x={p['place_xy_mm'][0]:7.1f}mm y={p['place_xy_mm'][1]:7.1f}mm z={p['place_z_mm']:.1f}mm "
          f"(hover {p['place_hover_z_mm']:.0f}mm)")
    if p["carry_waypoints"]:
        print(f"carry via  : {', '.join(p['carry_waypoints'])} "
              f"(folded to r={cfg.motion.transit_apex_radius_mm:.0f}mm, where the envelope is tallest)")
    print()
    print(f"{'grasp point':14s} {'radial':>7s} {'tangnt':>7s} {'x_mm':>8s} {'y_mm':>8s} {'pos_err':>8s} {'tilt':>6s}  ")
    for a in gp.attempts:
        worst = max((a.hover, a.grasp), key=lambda r: r.position_error_mm)
        flag = "" if a.reachable else "  <-- UNREACHABLE, will be skipped"
        print(
            f"{a.label:14s} {a.offset_mm[0]:7.1f} {a.offset_mm[1]:7.1f} "
            f"{a.xy_mm[0]:8.1f} {a.xy_mm[1]:8.1f} "
            f"{worst.position_error_mm:8.2f} {worst.tilt_error_deg:6.2f}{flag}"
        )
    print()
    print(f"{'waypoint':12s} {'pos_err_mm':>10s} {'tilt_err_deg':>12s}  joints")
    for name, r in {**p["carry_waypoints"], **p["place_waypoints"]}.items():
        flag = "  <-- CHECK" if _over_gate(cfg, r) else ""
        joints = "  ".join(f"{j}={v:.1f}" for j, v in r.joints.items())
        print(f"{name:12s} {r.position_error_mm:10.2f} {r.tilt_error_deg:12.2f}  {joints}{flag}")


def execute(cfg: AppConfig, p: dict) -> bool:
    from control.poses import PoseRegistry
    from control.grasp import run_grasp_attempts
    from control.robot_io import So101RobotIO
    from control.trajectory import TrajectoryPlayer

    cfg.robot.cameras = {}  # motor bus only; the top camera is already held by camera_web.py
    robot = So101RobotIO(cfg.robot)
    robot.connect()
    player = TrajectoryPlayer(robot, cfg.motion)
    try:
        wp = p["place_waypoints"]
        # start from a known pose: whatever the arm was left at after manual
        # calibration handling could be arbitrarily far from the pick target,
        # and a huge first jump can outrun move_timeout_s (AGENTS.md §11 —
        # a trajectory always starts from measured current pose, but SELECT
        # goes through home first for exactly this reason).
        poses = PoseRegistry.load(cfg.motion.poses_path)
        print("going home first...")
        player.move_to(poses.get(cfg.motion.home_pose), max_step=1.0)
        print("picking...")
        held = run_grasp_attempts(player, robot, cfg, p["grasp_plan"])
        if held is None:
            print("  no block in the jaws at any grasp point — aborting before transport (AGENTS.md §3).")
            return False
        print(f"held at '{held.label}'; lifting...")
        player.move_to(held.hover.joints, tol=cfg.motion.transit_arrival_tol)
        # Fold in before swinging across: at the reach a far block is picked
        # from, the top-down envelope only allows ~50mm of lift, which drags
        # the block. Retracting first buys ~40mm for the swing itself.
        carry = p["carry_waypoints"]
        for name in ("apex_pick", "apex_place"):
            if name in carry:
                print(f"carrying via {name}...")
                player.move_to(carry[name].joints, tol=cfg.motion.transit_arrival_tol)
        print("moving to place_hover...")
        player.move_to(wp["place_hover"].joints, tol=cfg.motion.transit_arrival_tol)
        print("descending to place...")
        player.move_to(wp["place_drop"].joints, max_step=cfg.motion.descent_step_per_tick)
        print("opening gripper (release)...")
        player.set_gripper(cfg.sensing.gripper_open_pos)
        time.sleep(cfg.motion.place_settle_s)
        print("lifting clear...")
        player.move_to(wp["place_hover"].joints, tol=cfg.motion.transit_arrival_tol)
        print("done.")
        return True
    finally:
        # Always end at home, success or failure: the next run detects the
        # block *before* it moves the arm, so an arm left hovering over the
        # workspace would occlude the very block it is about to pick.
        try:
            player.set_gripper(cfg.sensing.gripper_open_pos)
            player.move_to(PoseRegistry.load(cfg.motion.poses_path).get(cfg.motion.home_pose), max_step=1.0, tol=cfg.motion.transit_arrival_tol)
        except Exception as e:
            print(f"warning: safe return home failed: {e}", file=sys.stderr)
        robot.disconnect()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--color", required=True)
    ap.add_argument("--to", required=True, help="destination point name in --points (e.g. P5)")
    ap.add_argument("--config", default="src/configs/default.yaml")
    ap.add_argument("--points", type=Path, default=Path("docs/calibration/points.csv"))
    ap.add_argument("--snapshot-url", default=DEFAULT_SHOULDER_SNAPSHOT_URL)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    p = plan(cfg, args.color, args.to, args.snapshot_url, args.points)
    print_plan(cfg, p)

    if args.dry_run:
        print("\n--dry-run: no hardware connection, no motion.")
        return 0

    # Only the first grasp point and the place waypoints have to clear the
    # gate; an unreachable retry is dropped from the queue by
    # run_grasp_attempts rather than aborting the pick.
    gp = p["grasp_plan"]
    if not gp.attempts[0].reachable or any(_over_gate(cfg, r) for r in p["place_waypoints"].values()):
        print("\nABORT: a required waypoint exceeds the IK error gate — check the printout above.", file=sys.stderr)
        return 1
    usable = sum(1 for a in gp.attempts if a.reachable)
    print(f"\n{usable} of {len(gp.attempts)} grasp points usable.")

    if input("\nAbout to move the real arm. Enter to proceed, 'n' to cancel: ").strip().lower() in ("n", "no"):
        print("cancelled.")
        return 0
    grasped = execute(cfg, p)
    return 0 if grasped else 1


if __name__ == "__main__":
    raise SystemExit(main())

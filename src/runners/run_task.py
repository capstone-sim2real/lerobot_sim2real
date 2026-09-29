"""so101-run: Task 1 (gather into the zone) and Task 2 (stack) on the CV+IK FSM.

    so101-run --task 1 --dry-run        # zone slots, IK and detections; no robot
    so101-run --task 1
    so101-run --task 2 --dry-run        # tower ladder: how many levels are reachable
    so101-run --task 2 --set task2.max_levels=1
    so101-run --task 3                  # alias for so101-collect
    so101-run --task 1 --flow pick_lift_lower --color green   # one-block grasp smoke test

Requires the venue calibration (so101-calibrate, so101-zone-calibrate) and a
recorded home pose (tools.hardware.record_pose).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

from config import AppConfig, load_config
from camera.autostart import base_url_of, ensure_camera_server
from camera.client import fetch_snapshot
from control import MotionController, PoseRegistry, So101RobotIO
from control.ik import TopDownIK
from control.task1_transport import Task1TransportPlanner
from control.task2_stack import Task2StackPlanner
from fsm.flows import build_pick_lift_lower_states, build_task1_states, build_task2_stack_states
from fsm.ik_handler import CvIkSelectState
from fsm.machine import StateMachine, TransitionLogger
from fsm.states import RunContext
from perception import PlaneCalibration, detect_blocks, select_target
from perception.zone import point_in_zone
from session.factories import make_pick_state, make_task1_perceive
from session.report import print_detections, print_slot_table

logger = logging.getLogger("run")


def make_perceive(calib: PlaneCalibration, cfg: AppConfig, *, target_color: str | None = None):
    def perceive(skipped: set[str]):
        frame = fetch_snapshot(cfg.perception.snapshot_url)
        detections = detect_blocks(frame, calib, cfg.perception, is_rgb=False)
        if target_color is not None:
            detections = [d for d in detections if d.color == target_color]
        return select_target(detections, calib, cfg.select, skipped=skipped)

    return perceive


def run(
    task: int,
    cfg: AppConfig,
    run_id: str,
    *,
    flow: str = "task",
    target_color: str | None = None,
) -> RunContext:
    calib_path = Path(cfg.perception.calibration_path)
    if not calib_path.exists():
        raise FileNotFoundError(
            f"Venue calibration not found: {calib_path}. Run so101-calibrate first."
        )
    calib = PlaneCalibration.load(calib_path)
    poses = PoseRegistry.load(cfg.motion.poses_path)

    camera_proc = None
    if cfg.camera.auto_start:
        camera_proc = ensure_camera_server(
            base_url_of(cfg.perception.snapshot_url), extra_args=cfg.camera.extra_args
        )

    robot = So101RobotIO(cfg.robot)
    robot.connect()
    motion = None
    try:
        motion = MotionController(robot, poses, cfg.motion, cfg.sensing)
        motion.validate_poses(required=[cfg.motion.home_pose])
        smoke_test = flow == "pick_lift_lower"
        task1_gather = task == 1 and not smoke_test
        if smoke_test and not target_color:
            raise ValueError("pick_lift_lower flow requires --color <detected-colour>")
        if not smoke_test and not calib.zone_polygon_mm:
            raise ValueError(f"Task {task} requires zone_polygon_mm; run so101-zone-calibrate --write")

        shared_ik = TopDownIK(cfg.ik, project_root=".")
        pick_state = make_pick_state(
            robot=robot,
            motion=motion,
            cfg=cfg,
            calib=calib,
            retreat_after_grasp=not smoke_test,
            radial_tilt_extra_key=None if smoke_test else "task1_pick_radial_tilt_deg",
            ik=shared_ik,
        )

        if smoke_test:
            perceive = make_perceive(calib, cfg, target_color=target_color)
            states = build_pick_lift_lower_states(
                robot=robot, motion=motion, perceive=perceive, pick_state=pick_state, cfg=cfg,
                select_state=CvIkSelectState(motion, perceive),
            )
        elif task == 1:
            states = build_task1_states(
                robot=robot, motion=motion, perceive=make_task1_perceive(calib, cfg),
                pick_state=pick_state, cfg=cfg, calib=calib,
                planner=Task1TransportPlanner(calib, cfg, shared_ik),
            )
        else:
            stack_planner = Task2StackPlanner(calib, cfg, shared_ik)
            logger.info(
                "Task-2 tower at x=%.1f y=%.1f (reach %.0fmm): %d of %d levels reachable",
                stack_planner.stack_xy_mm[0], stack_planner.stack_xy_mm[1],
                math.dist(stack_planner.stack_xy_mm, calib.base_xy_mm or (0.0, 0.0)),
                stack_planner.reachable_levels, cfg.task2.max_levels,
            )
            states = build_task2_stack_states(
                robot=robot, motion=motion, perceive=make_task1_perceive(calib, cfg),
                pick_state=pick_state, cfg=cfg, calib=calib, planner=stack_planner,
            )

        log_dir = Path(cfg.logging.log_dir)
        transitions_csv = log_dir / f"{run_id}_transitions.csv" if cfg.logging.save_transitions else None
        ctx = RunContext(fsm=cfg.fsm)
        machine = StateMachine(
            states,
            ctx,
            transition_logger=TransitionLogger(transitions_csv),
            enforce_time_budget=not task1_gather,
        )

        budget = (
            f"until outside region is empty for {cfg.task1.empty_timeout_s:g}s"
            if task1_gather
            else f"budget {cfg.fsm.time_budget_s:.0f}s"
        )
        logger.info("Task %d / %s starting (run %s, %s)", task, flow, run_id, budget)
        machine.run()
        return ctx
    finally:
        if motion is not None:
            try:
                motion.open_gripper()
                motion.go_home()
            except Exception as e:
                logger.warning("Safe-shutdown motion failed: %s", e)
        robot.disconnect()
        if camera_proc is not None:
            camera_proc.stop()


def write_summary(ctx: RunContext, task: int, run_id: str, cfg: AppConfig) -> Path:
    summary = {
        "run_id": run_id,
        "task": task,
        "task1_complete": ctx.extras.get("task1_complete"),
        "task1_place_actions": ctx.extras.get("task1_place_actions"),
        "task1_slot_by_color": ctx.extras.get("task1_slot_by_color"),
        "task1_attempts_total": ctx.extras.get("task1_attempts_total"),
        "task2_place_actions": ctx.extras.get("task2_place_actions"),
        "task2_tower_height": ctx.extras.get("task2_tower_height"),
        "task2_max_height_reached": ctx.extras.get("task2_max_height_reached"),
        "task2_stop_reason": ctx.extras.get("task2_stop_reason"),
        # Levels flown despite the ladder calling them unreachable.
        "task2_forced_levels": ctx.extras.get("task2_forced_levels", []),
        "elapsed_s": round(ctx.elapsed_s(), 1),
        "attempts": ctx.attempts,
        "skipped": sorted(ctx.skipped),
        "stack_contacts": ctx.extras.get("stack_contacts"),
    }
    if task != 1:
        summary["placed_count"] = ctx.placed_count
    path = Path(cfg.logging.log_dir) / f"{run_id}_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    return path


def dry_run_task1(cfg: AppConfig) -> int:
    """Preflight the complete Task-1 geometry without touching the motor bus."""
    calib = PlaneCalibration.load(cfg.perception.calibration_path)
    if not calib.zone_polygon_mm:
        raise ValueError("Task 1 requires zone_polygon_mm; run so101-zone-calibrate --write")
    ik = TopDownIK(cfg.ik, project_root=".")
    planner = Task1TransportPlanner(calib, cfg, ik)

    camera_proc = None
    if cfg.camera.auto_start:
        camera_proc = ensure_camera_server(
            base_url_of(cfg.perception.snapshot_url), extra_args=cfg.camera.extra_args
        )
    try:
        frame = fetch_snapshot(cfg.perception.snapshot_url)
    finally:
        if camera_proc is not None:
            camera_proc.stop()
    detections = detect_blocks(frame, calib, cfg.perception, is_rgb=False)

    print("Task 1 dry-run (no robot connection, no motion)")
    print_detections(detections)
    print_slot_table(planner.slots)
    return 0


def dry_run_task2(cfg: AppConfig) -> int:
    """Preflight the whole Task-2 tower ladder without touching the motor bus.

    This is the instrument that answers "how many levels fit". Top-down lift
    collapses with reach, so the level count is an *output* of this command,
    not an input to the design -- read it before running the arm.
    """
    calib = PlaneCalibration.load(cfg.perception.calibration_path)
    if not calib.zone_polygon_mm:
        raise ValueError("Task 2 requires zone_polygon_mm; run so101-zone-calibrate --write")
    ik = TopDownIK(cfg.ik, project_root=".")
    planner = Task2StackPlanner(calib, cfg, ik)

    print("Task 2 dry-run (no robot connection, no motion)")
    print(planner.describe())
    inside = point_in_zone(planner.raw_xy_mm, calib)
    print(
        "\nlanding point inside zone_polygon_mm: "
        + ("yes" if inside else "NO -- placed blocks stay visible to SELECT")
    )

    camera_proc = None
    if cfg.camera.auto_start:
        camera_proc = ensure_camera_server(
            base_url_of(cfg.perception.snapshot_url), extra_args=cfg.camera.extra_args
        )
    try:
        frame = fetch_snapshot(cfg.perception.snapshot_url)
    finally:
        if camera_proc is not None:
            camera_proc.stop()
    detections = detect_blocks(frame, calib, cfg.perception, is_rgb=False)
    print_detections(detections, title="\nactive outside-zone detections:")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--flow", choices=["task", "pick_lift_lower"], default="task")
    parser.add_argument("--color", help="Only select this colour (required by pick_lift_lower)")
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="key.path=value")
    parser.add_argument("--dry-run", action="store_true", help="Task 1: inspect zone slots/IK/detections; Task 2: inspect the tower ladder. No robot connection")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.task == 3:
        if args.flow != "task" or args.color is not None:
            parser.error("Task 3 uses its fixed collection flow; omit --flow and --color")
        from runners.run_task3 import main as collect_main

        collect_argv = ["--config", args.config]
        for override in args.overrides:
            collect_argv.extend(["--set", override])
        if args.dry_run:
            collect_argv.append("--dry-run")
        return collect_main(collect_argv)

    cfg = load_config(args.config, overrides=args.overrides)
    run_id = time.strftime(f"task{args.task}_%Y%m%d_%H%M%S")

    if args.dry_run:
        if not (args.task in (1, 2) and args.flow == "task"):
            parser.error("--dry-run is supported for the Task 1 / Task 2 flows")
        try:
            return dry_run_task1(cfg) if args.task == 1 else dry_run_task2(cfg)
        except Exception as e:
            logger.error("Dry-run aborted: %s", e)
            return 1

    try:
        ctx = run(args.task, cfg, run_id, flow=args.flow, target_color=args.color)
    except Exception as e:
        logger.error("Run aborted: %s", e)
        return 1

    summary_path = write_summary(ctx, args.task, run_id, cfg)
    if args.task == 1 and args.flow == "task":
        print(f"\n=== Task 1 finished: outside region empty for {ctx.extras.get('task1_empty_for_s', 0.0):.1f}s "
              f"after {ctx.extras.get('task1_place_actions', 0)} place action(s) in {ctx.elapsed_s():.0f}s ===")
    elif args.task == 2 and args.flow == "task":
        print(f"\n=== Task 2 finished: tower {ctx.placed_count} high in {ctx.elapsed_s():.0f}s "
              f"({ctx.extras.get('task2_stop_reason') or 'outside region empty'}) ===")
    else:
        print(f"\n=== Task {args.task} finished: {ctx.placed_count}/{cfg.fsm.num_blocks} placed "
              f"in {ctx.elapsed_s():.0f}s (skipped: {sorted(ctx.skipped) or 'none'}) ===")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Task entrypoint: wire everything and run the FSM.

    # Task 1 (transport into the zone)
    # CV+IK (default; no policy server required)
    python -m runners.run_task --task 1 --pick-mode cv_ik

    # Legacy ACT path
    python -m runners.run_task --task 1 --pick-mode act \
        --set policy.server_address=100.99.252.112:8080 \
        --set policy.pretrained_name_or_path=/home/user/.../pretrained_model

    # Task 2 (stack)
    python -m runners.run_task --task 2 ...

    # One-block CV+IK grasp smoke test; no destination poses required
    python -m runners.run_task --task 1 --flow pick_lift_lower --color green

Preconditions (fail fast otherwise):
  - venue calibration JSON exists (tools/calibrate_homography.py)
  - required poses recorded (tools/record_pose.py)
  - ACT mode only: policy_server reachable and model path valid on the server machine
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from config import AppConfig, load_config
from camera.client import fetch_snapshot, fetch_snapshot_with_metadata
from control import MotionController, PoseRegistry, So101RobotIO
from control.ik import TopDownIK
from control.task1_transport import Task1TransportPlanner
from fsm.act_handler import ActPickState
from fsm.flows import build_pick_lift_lower_states, build_task1_states, build_task2_states
from fsm.ik_handler import CvIkPickState, CvIkSelectState
from fsm.machine import StateMachine, TransitionLogger
from fsm.states import RunContext
from fsm.task1 import Task1Perception
from perception import PlaneCalibration, detect_blocks, select_target
from policy import ActPolicyClient, GrpcPolicyTransport

logger = logging.getLogger("run")


def make_perceive(calib: PlaneCalibration, cfg: AppConfig, *, target_color: str | None = None):
    def perceive(skipped: set[str]):
        frame = fetch_snapshot(cfg.perception.snapshot_url)
        detections = detect_blocks(frame, calib, cfg.perception, is_rgb=False)
        if target_color is not None:
            detections = [d for d in detections if d.color == target_color]
        return select_target(detections, calib, cfg.select, skipped=skipped)

    return perceive


def make_task1_perceive(calib: PlaneCalibration, cfg: AppConfig):
    def perceive() -> Task1Perception:
        snapshot = fetch_snapshot_with_metadata(cfg.perception.snapshot_url)
        detections = detect_blocks(snapshot.frame, calib, cfg.perception, is_rgb=False)
        return Task1Perception(detections, snapshot.frame_seq, snapshot.captured_at)

    return perceive


def make_pick_state(
    pick_mode: str,
    *,
    robot: So101RobotIO,
    motion: MotionController,
    cfg: AppConfig,
    calib: PlaneCalibration,
    retreat_pose,
    retreat_after_grasp: bool = True,
    radial_tilt_extra_key: str | None = None,
    client: ActPolicyClient | None = None,
    ik: TopDownIK | None = None,
):
    """Build the PICK implementation without coupling common FSM states to ACT."""
    if pick_mode == "cv_ik":
        try:
            grasp_z_mm = float(calib.meta["grasp_z_mm_mean"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Calibration metadata is missing grasp_z_mm_mean; rerun base-frame calibration "
                "with the gripper on the block top plane."
            ) from exc
        return CvIkPickState(
            robot=robot,
            motion=motion,
            cfg=cfg,
            grasp_z_mm=grasp_z_mm,
            retreat_pose=retreat_pose,
            retreat_after_grasp=retreat_after_grasp,
            radial_tilt_extra_key=radial_tilt_extra_key,
            ik=ik,
        )
    if pick_mode == "act":
        if client is None:
            raise ValueError("ACT pick mode requires a connected policy client")
        return ActPickState(client, motion, retreat_pose)
    raise ValueError(f"Unknown pick mode: {pick_mode!r}")


def run(
    task: int,
    cfg: AppConfig,
    run_id: str,
    *,
    pick_mode: str = "cv_ik",
    flow: str = "task",
    target_color: str | None = None,
) -> RunContext:
    calib_path = Path(cfg.perception.calibration_path)
    if not calib_path.exists():
        raise FileNotFoundError(
            f"Venue calibration not found: {calib_path}. Run tools/calibrate_homography.py first."
        )
    calib = PlaneCalibration.load(calib_path)
    poses = PoseRegistry.load(cfg.motion.poses_path)

    robot = So101RobotIO(cfg.robot)
    robot.connect()
    client = None
    motion = None
    try:
        motion = MotionController(robot, poses, cfg.motion, cfg.sensing)
        task1_gather = task == 1 and flow == "task"
        if task1_gather:
            if pick_mode != "cv_ik":
                raise ValueError("Task 1 zone gathering currently requires --pick-mode cv_ik")
            if not calib.zone_polygon_mm:
                raise ValueError("Task 1 requires zone_polygon_mm; run so101-zone-calibrate --write")
            motion.validate_poses(required=[cfg.motion.home_pose])
            retreat_pose = None
        elif flow == "pick_lift_lower":
            if pick_mode != "cv_ik":
                raise ValueError("pick_lift_lower flow requires --pick-mode cv_ik")
            if not target_color:
                raise ValueError("pick_lift_lower flow requires --color <detected-colour>")
            motion.validate_poses(required=[cfg.motion.home_pose])
            retreat_pose = None
        else:
            motion.validate_poses(task=task)
            retreat_pose = poses.get(cfg.motion.retreat_pose)

        if pick_mode == "act":
            client = ActPolicyClient(robot, GrpcPolicyTransport(robot.robot, cfg.policy), cfg.policy)
            client.connect()  # server loads the model here, once per session

        shared_ik = TopDownIK(cfg.ik, project_root=".") if task1_gather else None
        pick_state = make_pick_state(
            pick_mode,
            robot=robot,
            motion=motion,
            cfg=cfg,
            calib=calib,
            retreat_pose=retreat_pose,
            retreat_after_grasp=flow != "pick_lift_lower",
            radial_tilt_extra_key="task1_pick_radial_tilt_deg" if task1_gather else None,
            client=client,
            ik=shared_ik,
        )

        perceive = make_perceive(calib, cfg, target_color=target_color)
        # The CV+IK pick opens the jaws itself, so its SELECT homes without
        # commanding the gripper; the ACT path keeps the recorded home pose
        # intact so the policy starts in distribution.
        select_state = CvIkSelectState(motion, perceive) if pick_mode == "cv_ik" else None
        if flow == "pick_lift_lower":
            states = build_pick_lift_lower_states(
                robot=robot, motion=motion, perceive=perceive, pick_state=pick_state, cfg=cfg,
                select_state=select_state,
            )
        elif task == 1:
            assert shared_ik is not None
            planner = Task1TransportPlanner(calib, cfg, shared_ik)
            states = build_task1_states(
                robot=robot, motion=motion, perceive=make_task1_perceive(calib, cfg),
                pick_state=pick_state, cfg=cfg, calib=calib, planner=planner,
            )
        else:
            states = build_task2_states(
                robot=robot, motion=motion, perceive=perceive, pick_state=pick_state, sensing_cfg=cfg.sensing,
                select_state=select_state,
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
        logger.info("Task %d / %s starting with %s PICK (run %s, %s)", task, flow, pick_mode, run_id, budget)
        machine.run()
        return ctx
    finally:
        # best-effort safe shutdown, also on exceptions mid-run
        if motion is not None:
            try:
                motion.open_gripper()
                motion.go_home()
            except Exception as e:
                logger.warning("Safe-shutdown motion failed: %s", e)
        if client is not None:
            client.close()
        robot.disconnect()


def write_summary(ctx: RunContext, task: int, run_id: str, cfg: AppConfig) -> Path:
    summary = {
        "run_id": run_id,
        "task": task,
        "task1_complete": ctx.extras.get("task1_complete"),
        "task1_place_actions": ctx.extras.get("task1_place_actions"),
        "task1_slot_by_color": ctx.extras.get("task1_slot_by_color"),
        "task1_attempts_total": ctx.extras.get("task1_attempts_total"),
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
    frame = fetch_snapshot(cfg.perception.snapshot_url)
    detections = detect_blocks(frame, calib, cfg.perception, is_rgb=False)

    print("Task 1 dry-run (no robot connection, no motion)")
    print("active outside-zone detections:")
    if detections:
        for detection in detections:
            x, y = detection.center_mm
            print(f"  {detection.color:6s} x={x:7.1f} y={y:7.1f} area={detection.area_mm2:.0f}mm2")
    else:
        print("  none")
    print("placement slots (far row first):")
    for slot in planner.slots:
        print(
            f"  {slot.index}: x={slot.xy_mm[0]:7.1f} y={slot.xy_mm[1]:7.1f} "
            f"drop_z={slot.drop_z_mm:.1f} hover_z={slot.hover_z_mm:.1f} "
            f"tilt={slot.radial_tilt_deg:.1f}deg "
            f"errors=({slot.drop.position_error_mm:.2f}, {slot.hover.position_error_mm:.2f})mm"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int, choices=[1, 2], required=True)
    parser.add_argument("--pick-mode", choices=["cv_ik", "act"], default="cv_ik")
    parser.add_argument("--flow", choices=["task", "pick_lift_lower"], default="task")
    parser.add_argument("--color", help="Only select this colour (required by pick_lift_lower)")
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="key.path=value")
    parser.add_argument("--dry-run", action="store_true", help="Task 1: inspect zone slots/IK/detections without connecting the robot")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(args.config, overrides=args.overrides)
    run_id = time.strftime(f"task{args.task}_%Y%m%d_%H%M%S")

    if args.dry_run:
        if not (args.task == 1 and args.flow == "task" and args.pick_mode == "cv_ik"):
            parser.error("--dry-run is supported for the default Task 1 CV+IK flow")
        try:
            return dry_run_task1(cfg)
        except Exception as e:
            logger.error("Dry-run aborted: %s", e)
            return 1

    try:
        ctx = run(args.task, cfg, run_id, pick_mode=args.pick_mode, flow=args.flow, target_color=args.color)
    except Exception as e:
        logger.error("Run aborted: %s", e)
        return 1

    summary_path = write_summary(ctx, args.task, run_id, cfg)
    if args.task == 1 and args.flow == "task":
        print(f"\n=== Task 1 finished: outside region empty for {ctx.extras.get('task1_empty_for_s', 0.0):.1f}s "
              f"after {ctx.extras.get('task1_place_actions', 0)} place action(s) in {ctx.elapsed_s():.0f}s ===")
    else:
        print(f"\n=== Task {args.task} finished: {ctx.placed_count}/{cfg.fsm.num_blocks} placed "
              f"in {ctx.elapsed_s():.0f}s (skipped: {sorted(ctx.skipped) or 'none'}) ===")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
import sys
import time
from pathlib import Path

from config import AppConfig, load_config
from camera.client import fetch_snapshot
from control import MotionController, PoseRegistry, So101RobotIO
from fsm.act_handler import ActPickState
from fsm.flows import build_pick_lift_lower_states, build_task1_states, build_task2_states
from fsm.ik_handler import CvIkPickState
from fsm.machine import StateMachine, TransitionLogger
from fsm.states import RunContext
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


def make_pick_state(
    pick_mode: str,
    *,
    robot: So101RobotIO,
    motion: MotionController,
    cfg: AppConfig,
    calib: PlaneCalibration,
    retreat_pose,
    retreat_after_grasp: bool = True,
    client: ActPolicyClient | None = None,
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
        if flow == "pick_lift_lower":
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

        pick_state = make_pick_state(
            pick_mode,
            robot=robot,
            motion=motion,
            cfg=cfg,
            calib=calib,
            retreat_pose=retreat_pose,
            retreat_after_grasp=flow != "pick_lift_lower",
            client=client,
        )

        perceive = make_perceive(calib, cfg, target_color=target_color)
        if flow == "pick_lift_lower":
            states = build_pick_lift_lower_states(
                robot=robot, motion=motion, perceive=perceive, pick_state=pick_state, cfg=cfg
            )
        elif task == 1:
            states = build_task1_states(
                robot=robot, motion=motion, perceive=perceive, pick_state=pick_state, sensing_cfg=cfg.sensing
            )
        else:
            states = build_task2_states(
                robot=robot, motion=motion, perceive=perceive, pick_state=pick_state, sensing_cfg=cfg.sensing
            )

        log_dir = Path(cfg.logging.log_dir)
        transitions_csv = log_dir / f"{run_id}_transitions.csv" if cfg.logging.save_transitions else None
        ctx = RunContext(fsm=cfg.fsm)
        machine = StateMachine(states, ctx, transition_logger=TransitionLogger(transitions_csv))

        logger.info("Task %d / %s starting with %s PICK (run %s, budget %.0fs)", task, flow, pick_mode, run_id, cfg.fsm.time_budget_s)
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
        "placed_count": ctx.placed_count,
        "elapsed_s": round(ctx.elapsed_s(), 1),
        "attempts": ctx.attempts,
        "skipped": sorted(ctx.skipped),
        "stack_contacts": ctx.extras.get("stack_contacts"),
    }
    path = Path(cfg.logging.log_dir) / f"{run_id}_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int, choices=[1, 2], required=True)
    parser.add_argument("--pick-mode", choices=["cv_ik", "act"], default="cv_ik")
    parser.add_argument("--flow", choices=["task", "pick_lift_lower"], default="task")
    parser.add_argument("--color", help="Only select this colour (required by pick_lift_lower)")
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="key.path=value")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(args.config, overrides=args.overrides)
    run_id = time.strftime(f"task{args.task}_%Y%m%d_%H%M%S")

    try:
        ctx = run(args.task, cfg, run_id, pick_mode=args.pick_mode, flow=args.flow, target_color=args.color)
    except Exception as e:
        logger.error("Run aborted: %s", e)
        return 1

    summary_path = write_summary(ctx, args.task, run_id, cfg)
    print(f"\n=== Task {args.task} finished: {ctx.placed_count}/{cfg.fsm.num_blocks} placed "
          f"in {ctx.elapsed_s():.0f}s (skipped: {sorted(ctx.skipped) or 'none'}) ===")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

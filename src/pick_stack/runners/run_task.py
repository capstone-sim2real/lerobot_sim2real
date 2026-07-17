"""Task entrypoint: wire everything and run the FSM.

    # Task 1 (transport into the zone)
    python -m pick_stack.runners.run_task --task 1 \
        --set policy.server_address=100.99.252.112:8080 \
        --set policy.pretrained_name_or_path=/home/user/.../pretrained_model

    # Task 2 (stack)
    python -m pick_stack.runners.run_task --task 2 ...

Preconditions (fail fast otherwise):
  - venue calibration JSON exists (tools/calibrate_homography.py)
  - required poses recorded (tools/record_pose.py)
  - policy_server reachable and model path valid on the server machine
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from pick_stack.config import AppConfig, load_config
from pick_stack.control import MotionController, PoseRegistry, So101RobotIO
from pick_stack.fsm.handlers import SlotPlaceStrategy, StackPlaceStrategy, build_states
from pick_stack.fsm.machine import StateMachine, TransitionLogger
from pick_stack.fsm.states import RunContext
from pick_stack.perception import PlaneCalibration, detect_blocks, select_target
from pick_stack.policy import ActPolicyClient, GrpcPolicyTransport

logger = logging.getLogger("pick_stack.run")


def make_perceive(robot: So101RobotIO, calib: PlaneCalibration, cfg: AppConfig):
    def perceive(skipped: set[str]):
        observation = robot.read_observation()
        frame = observation.get(cfg.perception.top_camera_key)
        if frame is None:
            raise RuntimeError(
                f"Top camera key '{cfg.perception.top_camera_key}' missing from observation "
                f"(have: {[k for k in observation if not k.endswith('.pos')]})"
            )
        detections = detect_blocks(frame, calib, cfg.perception, is_rgb=True)
        return select_target(detections, calib, cfg.select, skipped=skipped)

    return perceive


def run(task: int, cfg: AppConfig, run_id: str) -> RunContext:
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
        motion.validate_poses(task=task)
        retreat_pose = poses.get(cfg.motion.retreat_pose)

        client = ActPolicyClient(robot, GrpcPolicyTransport(robot.robot, cfg.policy), cfg.policy)
        client.connect()  # server loads the model here, once per session

        place_strategy = SlotPlaceStrategy(motion) if task == 1 else StackPlaceStrategy(motion)
        states = build_states(
            robot=robot,
            motion=motion,
            perceive=make_perceive(robot, calib, cfg),
            client=client,
            retreat_pose=retreat_pose,
            sensing_cfg=cfg.sensing,
            place_strategy=place_strategy,
        )

        log_dir = Path(cfg.logging.log_dir)
        transitions_csv = log_dir / f"{run_id}_transitions.csv" if cfg.logging.save_transitions else None
        ctx = RunContext(fsm=cfg.fsm)
        machine = StateMachine(states, ctx, transition_logger=TransitionLogger(transitions_csv))

        logger.info("Task %d starting (run %s, budget %.0fs)", task, run_id, cfg.fsm.time_budget_s)
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
    parser.add_argument("--config", default="src/pick_stack/configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="key.path=value")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(args.config, overrides=args.overrides)
    run_id = time.strftime(f"task{args.task}_%Y%m%d_%H%M%S")

    try:
        ctx = run(args.task, cfg, run_id)
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

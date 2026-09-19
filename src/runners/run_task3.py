"""Task 3 entrypoint: collect an ACT dataset by letting the arm demonstrate.

    # preflight -- no motor bus, no dataset written
    so101-collect --dry-run

    # collect (so101-camera is started automatically if not already running)
    so101-collect
    so101-collect --set task3.repo_id=local/so101_task3_green

    # append to an existing dataset (needs the exact stamped name + root)
    so101-collect --resume \
        --set task3.repo_id=local/so101_task3_20260913_101500 \
        --set task3.root=/home/mseoky/.cache/huggingface/lerobot/local/so101_task3_20260913_101500

The operator places five blocks; the arm picks one, carries it to a zone
slot, returns home, and that cycle is one episode. A grasp that fails is
tried once and the episode is discarded -- a failed demonstration teaches
the failure. When no block remains outside the zone the run stops and asks
for a new arrangement; Ctrl-C ends it, keeping every episode already saved.

Preconditions (fail fast otherwise):
  - so101-camera serving the configured MJPEG streams (started automatically
    if not already running; set camera.auto_start=false to require it be
    started by hand instead)
  - venue calibration JSON with zone_polygon_mm (so101-zone-calibrate --write)
  - home pose recorded (src/configs/poses.yaml)
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import signal
import threading
import time
from pathlib import Path

from camera.autostart import ManagedCamera, base_url_of, ensure_camera_server
from camera.client import fetch_snapshot
from camera.frame_source import build_frame_sources
from config import AppConfig, load_config
from control import MotionController, PoseRegistry, So101RobotIO
from control.ik import TopDownIK
from control.task1_transport import Task1TransportPlanner
from data.episode_recorder import (
    EpisodeRecorder,
    LeRobotEpisodeSink,
    RecordingRobotIO,
    StopRecording,
    create_dataset,
    format_interval_report,
    remove_empty_dataset,
    resolve_dataset_root,
)
from fsm.flows import build_task3_states
from fsm.machine import StateMachine, TransitionLogger
from fsm.states import RunContext
from perception import PlaneCalibration, detect_blocks
from session.factories import make_pick_state, make_task1_perceive
from session.report import print_detections, print_slot_table

logger = logging.getLogger("collect")


def resolve_repo_id(cfg: AppConfig, *, resume: bool) -> str:
    """Stamped repo id, so two runs never collide on one directory."""
    if resume or not cfg.task3.stamp_repo_id:
        return cfg.task3.repo_id
    return f"{cfg.task3.repo_id}_{time.strftime('%Y%m%d_%H%M%S')}"


def start_frame_sources(cfg: AppConfig) -> tuple[dict, ManagedCamera | None]:
    """Open every configured MJPEG stream and prove each one delivers."""
    camera_proc = None
    if cfg.camera.auto_start and cfg.task3.cameras:
        base_url = base_url_of(next(iter(cfg.task3.cameras.values())))
        camera_proc = ensure_camera_server(base_url, extra_args=cfg.camera.extra_args)

    sources = build_frame_sources(
        cfg.task3.cameras,
        width=cfg.task3.image_width,
        height=cfg.task3.image_height,
    )
    for source in sources.values():
        source.start()
    for name, source in sources.items():
        if not source.wait_for_first_frame(timeout_s=15.0):
            for other in sources.values():
                other.stop()
            if camera_proc is not None:
                camera_proc.stop()
            raise RuntimeError(
                f"No frame from camera {name!r} at {source.url}. Is so101-camera running "
                f"and serving that stream?"
            )
        logger.info("camera %s: %s", name, source.status())
    return sources, camera_proc


def install_stop_handler(stop_event: threading.Event) -> bool:
    """Ctrl-C sets a flag; the next tick or operator prompt stops the run.

    Unwinding from a signal handler could land anywhere -- mid bus write,
    mid image-writer flush. Raising at a tick boundary instead gives the
    runner a single, known place to discard the in-flight episode from. The
    handler also disarms itself, so a second Ctrl-C during shutdown cannot
    interrupt the parquet/video finalize and leave a corrupt dataset.

    Returns False without installing anything off the main thread, where
    ``signal.signal`` raises; the caller (the LLM agent) then owns the stop
    path through ``stop_event`` itself.
    """
    if threading.current_thread() is not threading.main_thread():
        return False

    def handler(signum, frame):  # noqa: ARG001 - signal handler signature
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop_event.set()
        print(
            "\n중지 요청됨 — 진행 중인 에피소드를 버리고 home으로 복귀한 뒤 저장을 마칩니다...",
            flush=True,
        )

    signal.signal(signal.SIGINT, handler)
    return True


def interruptible_prompt(
    message: str,
    stop_event: threading.Event,
    *,
    input_fn=input,
    poll_s: float = 0.1,
) -> str:
    """Read an operator prompt without trapping a pending Ctrl-C.

    Python may restart the blocking ``input()`` system call after our SIGINT
    handler returns. Run that read on a daemon thread so the control thread
    can observe ``stop_event`` and unwind through the normal dataset/video
    finalizers even while no robot command ticks are occurring.
    """
    if stop_event.is_set():
        raise StopRecording("stop requested while waiting for rearrangement")

    result: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def read_input() -> None:
        try:
            result.put((True, input_fn(message)))
        except BaseException as exc:  # propagate EOF and input failures on the main thread
            result.put((False, exc))

    threading.Thread(target=read_input, name="task3-operator-prompt", daemon=True).start()
    while True:
        if stop_event.wait(poll_s):
            raise StopRecording("stop requested while waiting for rearrangement")
        try:
            ok, value = result.get_nowait()
        except queue.Empty:
            continue
        if ok:
            return str(value)
        assert isinstance(value, BaseException)
        raise value


def run(cfg: AppConfig, run_id: str, *, resume: bool = False) -> dict:
    from lerobot.datasets import VideoEncodingManager

    calib_path = Path(cfg.perception.calibration_path)
    if not calib_path.exists():
        raise FileNotFoundError(
            f"Venue calibration not found: {calib_path}. Run tools/calibrate_homography.py first."
        )
    calib = PlaneCalibration.load(calib_path)
    if not calib.zone_polygon_mm:
        raise ValueError("Task 3 requires zone_polygon_mm; run so101-zone-calibrate --write")
    poses = PoseRegistry.load(cfg.motion.poses_path)

    repo_id = resolve_repo_id(cfg, resume=resume)
    root = resolve_dataset_root(cfg.task3, repo_id)

    sources, camera_proc = start_frame_sources(cfg)
    dataset = create_dataset(cfg.task3, repo_id, root, resume=resume)
    dataset_root = Path(dataset.root)
    recorder = EpisodeRecorder(LeRobotEpisodeSink(dataset), sources, cfg.task3)

    stop_event = threading.Event()
    install_stop_handler(stop_event)

    # TrajectoryPlayer's flat 1/fps sleep cannot hold a rate; RecordingRobotIO
    # paces to task3.record_fps instead. See its docstring.
    cfg.motion.fps = cfg.task3.motion_fps_override

    inner_robot = So101RobotIO(cfg.robot)
    inner_robot.connect()
    robot = RecordingRobotIO(
        inner_robot, recorder, record_fps=cfg.task3.record_fps, stop_event=stop_event
    )

    motion = None
    ctx = RunContext(fsm=cfg.fsm)
    stopped = False
    try:
        with VideoEncodingManager(dataset):
            try:
                motion = MotionController(robot, poses, cfg.motion, cfg.sensing)
                motion.validate_poses(required=[cfg.motion.home_pose])

                # make_pick_state reads grasp_z_mm_mean from calib itself and
                # raises a pointed error if the calibration predates it.
                shared_ik = TopDownIK(cfg.ik, project_root=".")
                pick_state = make_pick_state(
                    "cv_ik",
                    robot=robot,
                    motion=motion,
                    cfg=cfg,
                    calib=calib,
                    retreat_pose=None,
                    radial_tilt_extra_key="task1_pick_radial_tilt_deg",
                    max_grasp_attempts=cfg.task3.max_grasp_attempts,
                    ik=shared_ik,
                )
                planner = Task1TransportPlanner(calib, cfg, shared_ik)
                states = build_task3_states(
                    robot=robot,
                    motion=motion,
                    perceive=make_task1_perceive(calib, cfg),
                    pick_state=pick_state,
                    cfg=cfg,
                    calib=calib,
                    planner=planner,
                    recorder=recorder,
                    prompt=lambda message: interruptible_prompt(
                        message, stop_event
                    ),
                    stop_requested=stop_event.is_set,
                )

                log_dir = Path(cfg.logging.log_dir)
                transitions_csv = (
                    log_dir / f"{run_id}_transitions.csv" if cfg.logging.save_transitions else None
                )
                machine = StateMachine(
                    states,
                    ctx,
                    transition_logger=TransitionLogger(transitions_csv),
                    # Task 3 has no deadline: it runs until the operator stops
                    # it, across as many arrangements as they care to lay out.
                    enforce_time_budget=False,
                )
                logger.info(
                    "Task 3 collecting into %s at %g fps (grasp attempts: %d)",
                    dataset_root, cfg.task3.record_fps, cfg.task3.max_grasp_attempts,
                )
                print("\n블록 5개를 배치한 뒤 수집이 시작됩니다. 종료는 Ctrl-C.\n", flush=True)
                machine.run()
            except (StopRecording, KeyboardInterrupt):
                stopped = True
                # Rule 4: only a completed pick-and-carry is an episode.
                recorder.abort_episode("interrupted")
            finally:
                # Safe shutdown must not be recorded, and must not be stopped
                # by the flag that just fired.
                recorder.abort_episode("shutdown")
                stop_event.clear()
                if motion is not None:
                    try:
                        motion.open_gripper()
                        motion.go_home()
                    except Exception as e:  # noqa: BLE001 - best effort
                        logger.warning("Safe-shutdown motion failed: %s", e)
    finally:
        inner_robot.disconnect()
        for source in sources.values():
            source.stop()
        if camera_proc is not None:
            camera_proc.stop()

    if recorder.saved_total == 0 and not resume:
        remove_empty_dataset(dataset_root)

    return {
        "run_id": run_id,
        "stopped_by_operator": stopped,
        "repo_id": repo_id,
        "dataset_root": str(dataset_root),
        "episodes_saved": recorder.saved_total,
        "episodes_saved_by_color": recorder.saved_by_color,
        "episodes_discarded_by_color": recorder.discarded_by_color,
        "discard_reasons": recorder.discard_reasons,
        "rounds": ctx.extras.get("task3_rounds", 0),
        "record_fps": cfg.task3.record_fps,
        "tick_interval": recorder.interval_stats(),
        "tick_report": format_interval_report(recorder.interval_stats(), cfg.task3.record_fps),
        "elapsed_s": round(ctx.elapsed_s(), 1),
        "attempts_total": ctx.extras.get("task1_attempts_total"),
    }


def write_summary(summary: dict, run_id: str, cfg: AppConfig) -> Path:
    path = Path(cfg.logging.log_dir) / f"{run_id}_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return path


def dry_run(cfg: AppConfig, *, resume: bool) -> int:
    """Preflight everything Task 3 needs, touching neither bus nor dataset."""
    calib = PlaneCalibration.load(cfg.perception.calibration_path)
    if not calib.zone_polygon_mm:
        raise ValueError("Task 3 requires zone_polygon_mm; run so101-zone-calibrate --write")
    ik = TopDownIK(cfg.ik, project_root=".")
    planner = Task1TransportPlanner(calib, cfg, ik)

    print("Task 3 dry-run (no robot connection, no motion, no dataset written)")
    print(f"\nrecord_fps={cfg.task3.record_fps:g}  motion.fps override={cfg.task3.motion_fps_override:g}")
    print(f"grasp attempts per episode: {cfg.task3.max_grasp_attempts}")
    print(f"episode frame bounds: {cfg.task3.min_episode_frames}..{cfg.task3.max_episode_frames}")

    repo_id = resolve_repo_id(cfg, resume=resume)
    root = resolve_dataset_root(cfg.task3, repo_id)
    print(f"\ndataset repo_id: {repo_id}")
    print(f"dataset root: {root if root is not None else '$HF_LEROBOT_HOME/' + repo_id}")

    print("\ncameras:")
    sources, camera_proc = start_frame_sources(cfg)
    try:
        for name, source in sources.items():
            before = source.latest()
            time.sleep(1.0)
            after = source.latest()
            fps = (after.seq - before.seq) if (before and after) else 0
            print(f"  {name:6s} shape={after.image.shape if after else None} ~{fps} fps  {source.url}")
    finally:
        for source in sources.values():
            source.stop()

    try:
        from data.episode_recorder import dataset_features

        features = dataset_features(cfg.task3.cameras, cfg.task3.image_width, cfg.task3.image_height)
        print("\ndataset features:")
        for key, spec in features.items():
            print(f"  {key:32s} {spec['dtype']:8s} {tuple(spec['shape'])}")
    except ImportError as exc:
        print(f"\ndataset features: unavailable without lerobot ({exc})")

    print()
    print_slot_table(planner.slots, show_tilt=False)

    try:
        frame = fetch_snapshot(cfg.perception.snapshot_url)
    finally:
        if camera_proc is not None:
            camera_proc.stop()
    detections = detect_blocks(frame, calib, cfg.perception, is_rgb=False)
    print_detections(
        detections,
        title="\nactive outside-zone detections:",
        extra=lambda d: f" task={cfg.task3.task_templates.get(d.color, '<MISSING>')!r}",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="key.path=value")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to the dataset named by task3.repo_id (needs task3.root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check cameras, calibration, slot IK and dataset features without moving the arm",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(args.config, overrides=args.overrides)
    run_id = time.strftime("task3_%Y%m%d_%H%M%S")

    if args.dry_run:
        try:
            return dry_run(cfg, resume=args.resume)
        except Exception as e:  # noqa: BLE001 - report and exit non-zero
            logger.error("Dry-run aborted: %s", e)
            return 1

    try:
        summary = run(cfg, run_id, resume=args.resume)
    except Exception as e:  # noqa: BLE001 - report and exit non-zero
        logger.error("Collection aborted: %s", e)
        return 1

    summary_path = write_summary(summary, run_id, cfg)
    by_color = ", ".join(f"{c} {n}" for c, n in sorted(summary["episodes_saved_by_color"].items()))
    print(
        f"\n=== Task 3 finished: {summary['episodes_saved']} episode(s) saved "
        f"over {summary['rounds']} round(s) in {summary['elapsed_s']:.0f}s ==="
    )
    print(f"by colour : {by_color or 'none'}")
    print(f"discarded : {summary['discard_reasons'] or 'none'}")
    print(f"rate      : {summary['tick_report']}")
    print(f"dataset   : {summary['dataset_root']}")
    print(f"summary   : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

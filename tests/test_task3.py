"""Task 3 contracts: episode boundaries, discard rules, pacing, and rounds.

Everything here runs without lerobot, torch, or hardware (AGENTS.md §13):
the dataset is an ``EpisodeSink`` double and the arm is ``MockRobotIO``.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

import control.grasp as grasp_mod
from runners import run_task, run_task3
from config import AppConfig, validate_task3
from control.grasp import GraspAttempt, GraspOutcome, GraspPlan, run_grasp_attempts
from control.ik import IkResult
from control.robot_io import JOINT_NAMES, MockRobotIO
from data.episode_recorder import EpisodeRecorder, RecordingRobotIO, StopRecording
from fsm.states import RunContext, StateName
from fsm.task1 import Task1Perception
from fsm.task3 import EPISODE_OK_KEY, Task3PlaceState, Task3SelectState
from perception import BlockDetection, PlaneCalibration


# ── doubles ──────────────────────────────────────────────────────────


class _Sink:
    """Records what the dataset was asked to do, in order."""

    def __init__(self):
        self.calls: list[str] = []
        self.frames: list[dict] = []
        self.saved: list[list[dict]] = []
        self._buffer: list[dict] = []

    @property
    def features(self) -> dict:
        return {}

    def add_frame(self, frame: dict) -> None:
        self.calls.append("add_frame")
        self._buffer.append(frame)
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.calls.append("save_episode")
        self.saved.append(list(self._buffer))
        self._buffer = []

    def clear_episode_buffer(self) -> None:
        self.calls.append("clear_episode_buffer")
        self._buffer = []

    @property
    def save_count(self) -> int:
        return self.calls.count("save_episode")


class _Frame:
    def __init__(self, received_at: float):
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.seq = 1
        self.received_at = received_at


class _Source:
    """Frame source whose freshness the test controls."""

    def __init__(self, clock: dict, age_s: float = 0.0):
        self._clock = clock
        self.age_s_value = age_s

    def latest(self):
        return _Frame(self._clock["mono"] - self.age_s_value)


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.task3.min_episode_frames = 2
    cfg.task3.max_episode_frames = 100
    cfg.task1.scan_interval_s = 0.0
    return cfg


def _recorder(cfg: AppConfig, sink: _Sink, sources: dict) -> EpisodeRecorder:
    return EpisodeRecorder(sink, sources, cfg.task3)


def _joints(value: float = 1.0) -> dict[str, float]:
    return {joint: value for joint in JOINT_NAMES}


class _Motion:
    def __init__(self):
        self.home_calls = 0

    def go_home(self, *, include_gripper=True):
        assert include_gripper is False
        self.home_calls += 1


def _calibration() -> PlaneCalibration:
    return PlaneCalibration(
        H=np.eye(3),
        image_size=(500, 400),
        square_mm=1.0,
        base_xy_mm=(0.0, 0.0),
        zone_polygon_mm=[(300.0, 100.0), (300.0, -100.0), (200.0, -100.0), (200.0, 100.0)],
        meta={"grasp_z_mm_mean": 10.0},
    )


def _block(color: str, x: float, y: float) -> BlockDetection:
    return BlockDetection(color, (x, y), 1600.0, 1.0, 1.0, 1.0, [])


class _Samples:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


# ── the success rule ─────────────────────────────────────────────────


def test_a_failed_grasp_discards_the_episode_and_saves_nothing(monkeypatch):
    cfg = _cfg()
    clock = {"mono": 100.0}
    monkeypatch.setattr("data.episode_recorder.time.monotonic", lambda: clock["mono"])
    sink = _Sink()
    recorder = _recorder(cfg, sink, {"top": _Source(clock)})

    recorder.begin_episode("green")
    for _ in range(10):
        recorder.record_tick(_joints(), _joints(2.0))
    assert recorder.finish_episode(success=False) is False

    assert sink.save_count == 0
    assert sink.calls.count("clear_episode_buffer") == 2  # begin + discard
    assert recorder.discard_reasons == {"pick_failed": 1}
    assert recorder.discarded_by_color == {"green": 1}


# ── frame content ────────────────────────────────────────────────────


def test_recorded_action_is_the_post_safety_value_the_robot_actually_sent(monkeypatch):
    cfg = _cfg()
    clock = {"mono": 100.0}
    monkeypatch.setattr("data.episode_recorder.time.monotonic", lambda: clock["mono"])
    monkeypatch.setattr("data.episode_recorder.time.perf_counter", lambda: clock["mono"])
    sink = _Sink()
    recorder = _recorder(cfg, sink, {"top": _Source(clock)})

    class ClampingRobot(MockRobotIO):
        def send_joints(self, positions):
            sent = {
                joint: max(self.joints[joint] - 10.0, min(self.joints[joint] + 10.0, value))
                for joint, value in positions.items()
            }
            self.joints.update(sent)
            return sent

    robot = RecordingRobotIO(ClampingRobot(_joints(0.0)), recorder, record_fps=1000.0)
    recorder.begin_episode("green")
    returned = robot.send_joints({"gripper": 50.0})

    gripper_index = list(JOINT_NAMES).index("gripper")
    assert returned["gripper"] == pytest.approx(10.0)
    assert sink.frames[0]["observation.state"][gripper_index] == pytest.approx(0.0)
    assert sink.frames[0]["action"][gripper_index] == pytest.approx(10.0)


# ── camera staleness ─────────────────────────────────────────────────


def test_a_stale_frame_is_skipped_and_a_dead_camera_discards_the_episode(monkeypatch):
    cfg = _cfg()
    cfg.task3.max_frame_age_s = 0.5
    cfg.task3.max_stale_ticks = 3
    clock = {"mono": 100.0}
    monkeypatch.setattr("data.episode_recorder.time.monotonic", lambda: clock["mono"])
    sink = _Sink()
    source = _Source(clock)
    recorder = _recorder(cfg, sink, {"top": source})

    recorder.begin_episode("green")
    recorder.record_tick(_joints(), _joints())
    assert len(sink.frames) == 1

    source.age_s_value = 2.0  # camera fell behind
    recorder.record_tick(_joints(), _joints())
    recorder.record_tick(_joints(), _joints())
    assert len(sink.frames) == 1  # skipped, not recorded against a stale image

    recorder.record_tick(_joints(), _joints())  # third strike
    source.age_s_value = 0.0
    recorder.record_tick(_joints(), _joints())
    assert len(sink.frames) == 1  # aborted episodes stop accepting frames

    assert recorder.finish_episode(success=True) is False
    assert recorder.discard_reasons == {"stale_camera": 1}


# ── pacing and stop ──────────────────────────────────────────────────


def test_rearrangement_prompt_exits_when_stop_arrives_during_blocking_input():
    stop = threading.Event()
    input_started = threading.Event()
    release_input = threading.Event()

    def blocking_input(_message):
        input_started.set()
        release_input.wait()
        return ""

    def request_stop():
        assert input_started.wait(timeout=1.0)
        stop.set()

    threading.Thread(target=request_stop, daemon=True).start()
    try:
        with pytest.raises(StopRecording):
            run_task3.interruptible_prompt(
                "rearrange", stop, input_fn=blocking_input, poll_s=0.001
            )
    finally:
        release_input.set()


def test_an_interrupted_episode_is_discarded_not_saved(monkeypatch):
    cfg = _cfg()
    clock = {"mono": 100.0}
    monkeypatch.setattr("data.episode_recorder.time.monotonic", lambda: clock["mono"])
    sink = _Sink()
    recorder = _recorder(cfg, sink, {"top": _Source(clock)})

    recorder.begin_episode("green")
    for _ in range(10):
        recorder.record_tick(_joints(), _joints())
    assert recorder.abort_episode("interrupted") is False

    assert sink.save_count == 0
    assert recorder.discard_reasons == {"interrupted": 1}
    # A second abort during safe shutdown must not double-count.
    assert recorder.abort_episode("shutdown") is False
    assert recorder.discard_reasons == {"interrupted": 1}


# ── one grasp attempt ────────────────────────────────────────────────


def _attempt(label: str, offset=(0.0, 0.0)) -> GraspAttempt:
    result = IkResult({"wrist_flex": 0.0}, position_error_mm=0.0, tilt_error_deg=0.0)
    return GraspAttempt(label, offset, (200.0, 0.0), result, result, True)


def _cardinal_plan() -> GraspPlan:
    return GraspPlan(
        (200, 0), (212, 0), 9.0, 80.0,
        [
            _attempt("centre"),
            _attempt("left", (0.0, 10.0)),
            _attempt("back", (-10.0, 0.0)),
            _attempt("right", (0.0, -10.0)),
            _attempt("front", (10.0, 0.0)),
        ],
    )


def _run_queue(monkeypatch, max_attempts):
    tried = []

    def fake_attempt(_player, _robot, _cfg, attempt, **_kwargs):
        tried.append(attempt.label)
        return GraspOutcome.EMPTY, None

    monkeypatch.setattr(grasp_mod, "attempt_grasp", fake_attempt)
    held = run_grasp_attempts(
        None, None, AppConfig(), _cardinal_plan(),
        max_attempts=max_attempts, log=lambda _m: None,
    )
    return held, tried


# ── episode boundaries in the FSM ────────────────────────────────────


def _select(
    cfg,
    recorder,
    samples,
    prompt=lambda _msg: None,
    motion=None,
    stop_requested=lambda: False,
):
    return Task3SelectState(
        motion or _Motion(),
        samples,
        _calibration(),
        cfg,
        recorder,
        prompt=prompt,
        stop_requested=stop_requested,
    )


def test_the_episode_closes_after_the_arm_has_returned_home(monkeypatch):
    """Home return is the episode's tail, so finish must follow go_home."""
    cfg = _cfg()
    clock = {"mono": 100.0}
    monkeypatch.setattr("data.episode_recorder.time.monotonic", lambda: clock["mono"])
    sink = _Sink()
    recorder = _recorder(cfg, sink, {"top": _Source(clock)})
    order: list[str] = []

    class _OrderedMotion(_Motion):
        def go_home(self, *, include_gripper=True):
            super().go_home(include_gripper=include_gripper)
            order.append("go_home")
            recorder.record_tick(_joints(), _joints())

    class _OrderedSink(_Sink):
        def save_episode(self):
            order.append("save_episode")
            super().save_episode()

    sink = _OrderedSink()
    recorder = _recorder(cfg, sink, {"top": _Source(clock)})
    state = _select(cfg, recorder, lambda: None, motion=_OrderedMotion())

    ctx = RunContext(cfg.fsm)
    recorder.begin_episode("green")
    recorder.record_tick(_joints(), _joints())
    recorder.record_tick(_joints(), _joints())
    ctx.extras[EPISODE_OK_KEY] = True
    state.enter(ctx)

    assert order == ["go_home", "save_episode"]
    assert len(sink.saved[0]) == 3  # the home-return tick is in the episode
    assert EPISODE_OK_KEY not in ctx.extras  # the flag never carries over


# ── the round boundary ───────────────────────────────────────────────


# ── configuration ────────────────────────────────────────────────────


# ── end to end ───────────────────────────────────────────────────────


class _RoundDone(Exception):
    """Ends the end-to-end run at the first arrangement boundary."""


def test_a_whole_collection_round_cycles_through_the_real_state_machine(monkeypatch):
    """SELECT -> PICK -> VERIFY -> TRANSPORT -> PLACE -> SELECT, recorded.

    The individual pieces are covered above; this is the wiring. Three
    blocks are offered, the middle one fails its single grasp attempt, and
    the arrangement then runs out. What has to come out is one saved episode
    per delivered block -- with the right colour sentence -- one discarded
    episode, and a prompt where Task 1 would have returned DONE.
    """
    from control.motion import MotionController
    from control.task1_transport import Task1SlotPlan, Task1TransportPlan
    from fsm.flows import build_task3_states
    from fsm.machine import StateMachine
    from fsm.states import State

    cfg = _cfg()
    cfg.motion.fps = 0
    cfg.motion.place_settle_s = 0.0
    cfg.sensing.gripper_action_wait_s = 0.0
    cfg.task3.min_episode_frames = 1

    clock = {"wall": 1000.0, "mono": 10.0}
    monkeypatch.setattr("fsm.task1.time.time", lambda: clock["wall"])
    monkeypatch.setattr("fsm.task1.time.monotonic", lambda: clock["mono"])
    monkeypatch.setattr("data.episode_recorder.time.monotonic", lambda: clock["mono"])
    monkeypatch.setattr("data.episode_recorder.time.perf_counter", lambda: clock["mono"])
    monkeypatch.setattr("data.episode_recorder.time.sleep", lambda _s: None)

    sink = _Sink()
    inner = MockRobotIO(_joints(0.0))
    recorder = _recorder(cfg, sink, {"top": _Source(clock)})
    robot = RecordingRobotIO(inner, recorder, record_fps=30.0)

    # green is taken, blue fails once then succeeds, wood is taken.
    sweeps = [["green", "blue", "wood"], ["blue", "wood"], ["blue", "wood"], ["wood"]]
    frames = [
        Task1Perception(
            [_block(color, 120.0 + 10 * n, 0.0) for n, color in enumerate(colors)], i + 1, 1000.0
        )
        for i, colors in enumerate(sweeps)
    ]
    frames += [Task1Perception([], 98, 1000.0), Task1Perception([], 99, 1000.0)]

    def perceive():
        sample = frames.pop(0)
        if not sample.detections:
            clock["mono"] += cfg.task1.empty_timeout_s
        return sample

    result = IkResult({j: 0.0 for j in JOINT_NAMES if j != "gripper"}, 0.0, 0.0)

    class _StubPick(State):
        """Holds everything except blue's first attempt -- and never retries."""

        name = StateName.PICK

        def __init__(self):
            self.attempted: list[str] = []

        def step(self, ctx):
            self.attempted.append(ctx.target_id)
            robot.send_joints({"gripper": cfg.sensing.gripper_close_pos})
            if ctx.target_id == "blue" and self.attempted.count("blue") == 1:
                return StateName.SELECT
            ctx.extras["ik_pick_attempt"] = GraspAttempt(
                "centre", (0.0, 0.0), (120.0, 0.0), result, result, True
            )
            return StateName.VERIFY

    class _StubVerify(State):
        """The real VerifyState reads loads MockRobotIO does not simulate."""

        name = StateName.VERIFY

        def step(self, ctx):
            return StateName.TRANSPORT

    class _Planner:
        def plan(self, _held, slot_index):
            slot = Task1SlotPlan(
                index=slot_index, xy_mm=(250.0, 0.0), drop_z_mm=15.0,
                hover_z_mm=55.0, radial_tilt_deg=0.0, hover=result, drop=result,
            )
            return Task1TransportPlan(slot=slot, carry=())

    class _Poses:
        def get(self, _name):
            return _joints(0.0)

        def require(self, _names):
            pass

    prompts: list[str] = []

    def stop_after_one_round(message):
        prompts.append(message)
        raise _RoundDone

    motion = MotionController(robot, _Poses(), cfg.motion, cfg.sensing)
    pick = _StubPick()
    states = build_task3_states(
        robot=robot, motion=motion, perceive=perceive, pick_state=pick,
        cfg=cfg, calib=_calibration(), planner=_Planner(), recorder=recorder,
        prompt=stop_after_one_round,
    )
    states[StateName.VERIFY] = _StubVerify()

    ctx = RunContext(cfg.fsm)
    with pytest.raises(_RoundDone):
        StateMachine(states, ctx, enforce_time_budget=False).run()

    assert recorder.saved_by_color == {"green": 1, "blue": 1, "wood": 1}
    assert recorder.discard_reasons == {"pick_failed": 1}
    assert sink.save_count == 3
    assert len(prompts) == 1
    assert ctx.extras["task3_rounds"] == 1
    # One grasp attempt per selection: blue was re-selected, never retried
    # inside a single PICK.
    assert pick.attempted == ["green", "blue", "blue", "wood"]
    # Each saved episode carries exactly one colour's sentence.
    sentences = [{frame["task"] for frame in episode} for episode in sink.saved]
    assert [len(s) for s in sentences] == [1, 1, 1]
    assert [s.pop() for s in sentences] == [
        cfg.task3.task_templates[c] for c in ("green", "blue", "wood")
    ]
    # Slots were handed out in delivery order, and the failed pick burned none.
    assert ctx.extras["task1_slot_by_color"] == {"green": 0, "blue": 1, "wood": 2}

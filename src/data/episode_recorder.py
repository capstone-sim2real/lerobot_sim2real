"""Record the arm's own motion as LeRobot episodes, without touching the FSM.

Every arm command in this project passes through ``BaseRobotIO.send_joints``
— ``TrajectoryPlayer.move_to``/``settle``/``descend``/``set_gripper`` have no
other way to reach the bus. So a decorator around the robot is the single
point where a whole run can be recorded, and the FSM, motion, and grasp code
stay byte-for-byte the code that Task 1 and Task 2 already validate.

Three things live here:

``RecordingRobotIO``
    The decorator. It is also the loop's **metronome** — see the note on
    ``_pace`` for why the recorder, not ``TrajectoryPlayer``, has to own the
    tick rate.

``EpisodeRecorder``
    Episode lifecycle and the success rule: an episode is saved only when the
    block was actually grasped *and* delivered. Everything else is dropped,
    because a failed demonstration teaches the failure.

``EpisodeSink``
    The narrow protocol ``EpisodeRecorder`` needs from a dataset, so tests can
    substitute a recording double and run with no lerobot installed.
"""

from __future__ import annotations

import logging
import math
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from config import Task3Config
from control.robot_io import JOINT_NAMES, BaseRobotIO

logger = logging.getLogger(__name__)


class StopRecording(Exception):
    """Raised at a tick boundary when the operator asked to stop.

    Deliberately not ``KeyboardInterrupt``: the signal handler only sets a
    flag, and the flag is checked in ``send_joints``. That makes the unwind
    point deterministic — between two commands, never halfway through a bus
    write — while still reaching the runner's safe-shutdown ``finally``.
    """


class EpisodeSink(Protocol):
    """What :class:`EpisodeRecorder` needs from a dataset."""

    @property
    def features(self) -> dict[str, dict]: ...

    def add_frame(self, frame: dict) -> None: ...

    def save_episode(self) -> None: ...

    def clear_episode_buffer(self) -> None: ...


class LeRobotEpisodeSink:
    """:class:`EpisodeSink` backed by a real ``LeRobotDataset``.

    lerobot is imported inside ``__init__`` on purpose: importing this module
    must not require lerobot, torch, or ffmpeg (AGENTS.md §13).
    """

    def __init__(self, dataset: Any):
        self._dataset = dataset

    @property
    def dataset(self) -> Any:
        return self._dataset

    @property
    def features(self) -> dict[str, dict]:
        return self._dataset.features

    def add_frame(self, frame: dict) -> None:
        self._dataset.add_frame(frame)

    def save_episode(self) -> None:
        self._dataset.save_episode()

    def clear_episode_buffer(self) -> None:
        self._dataset.clear_episode_buffer()


def dataset_features(cameras: dict[str, str], width: int, height: int) -> dict[str, dict]:
    """Build the LeRobot feature dict for the SO-101 plus recorded cameras.

    ``robot.observation_features`` cannot be used: ``robot.cameras`` is empty
    because ``camera.server`` owns the devices (AGENTS.md §8), so the camera
    features are declared here from the configured streams instead.
    """
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import hw_to_dataset_features

    motors: dict[str, Any] = {f"{joint}.pos": float for joint in JOINT_NAMES}
    cams: dict[str, Any] = {name: (height, width, 3) for name in cameras}
    return {
        **hw_to_dataset_features(motors, ACTION, use_video=True),
        **hw_to_dataset_features({**motors, **cams}, OBS_STR, use_video=True),
    }


def create_dataset(cfg: Task3Config, repo_id: str, root: Path | str | None, *, resume: bool):
    """Create (or resume) the LeRobotDataset Task 3 records into."""
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset

    if resume:
        if root is None:
            raise ValueError(
                "--resume needs task3.root: LeRobotDataset.resume cannot locate a "
                "dataset from repo_id alone"
            )
        return LeRobotDataset.resume(
            repo_id,
            root=root,
            image_writer_processes=0,
            image_writer_threads=4 * max(1, len(cfg.cameras)),
            rgb_encoder=RGBEncoderConfig(vcodec=cfg.video_codec),
        )
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(round(cfg.record_fps)),
        features=dataset_features(cfg.cameras, cfg.image_width, cfg.image_height),
        root=root,
        robot_type=cfg.robot_type,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4 * max(1, len(cfg.cameras)),
        # Real-time encoding steals CPU from the control loop on the Orin
        # (docs/guide/SO101_데이터수집_관리.md §3).
        streaming_encoding=False,
        rgb_encoder=RGBEncoderConfig(vcodec=cfg.video_codec),
    )


class EpisodeRecorder:
    """Owns one episode at a time and decides whether it is worth keeping.

    ``frame_sources`` maps a dataset camera name to anything with
    ``latest()`` (see :mod:`camera.frame_source`).
    """

    def __init__(
        self,
        sink: EpisodeSink,
        frame_sources: dict[str, Any],
        cfg: Task3Config,
    ):
        self._sink = sink
        self._sources = frame_sources
        self._cfg = cfg
        self._open = False
        self._color: str | None = None
        self._task_text = ""
        self._frames = 0
        self._stale_run = 0
        self._abort_reason: str | None = None
        # Per-run bookkeeping, surfaced in the summary JSON.
        self.saved_by_color: dict[str, int] = {}
        self.discarded_by_color: dict[str, int] = {}
        self.discard_reasons: dict[str, int] = {}
        self.tick_intervals: list[float] = []
        self._last_tick_at: float | None = None

    # ── state ────────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def abort_reason(self) -> str | None:
        """Latched recording quality failure, for orchestration to stop a take."""
        return self._abort_reason

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def color(self) -> str | None:
        return self._color

    @property
    def saved_total(self) -> int:
        return sum(self.saved_by_color.values())

    def task_text_for(self, color: str) -> str:
        try:
            return self._cfg.task_templates[color]
        except KeyError as exc:  # pragma: no cover - validate_task3 rejects this
            raise KeyError(
                f"task3.task_templates has no sentence for colour {color!r}"
            ) from exc

    # ── lifecycle ────────────────────────────────────────────────────

    def begin_episode(self, color: str) -> None:
        """Start recording; the arm is expected to be at home.

        An already-open episode here means a state left without resolving it,
        which is a bug rather than a data condition — but dropping it is
        still strictly better than splicing two attempts into one episode.
        """
        if self._open:
            logger.warning("Episode for %s was never finished; discarding it", self._color)
            self._discard("unfinished")
        self._sink.clear_episode_buffer()
        self._open = True
        self._color = color
        self._task_text = self.task_text_for(color)
        self._frames = 0
        self._stale_run = 0
        self._abort_reason = None
        self._last_tick_at = None
        logger.info("Recording episode: %s", self._task_text)

    def record_tick(self, state: dict[str, float], action: dict[str, float]) -> None:
        """Add one (observation, action) pair. No-op when no episode is open."""
        pending = self.capture_tick(state)
        self.commit_tick(pending, action)

    def capture_tick(self, state: dict[str, float]) -> dict[str, Any] | None:
        """Capture the pre-action observation, but do not append it yet.

        Real hardware may clamp the requested target. ``RecordingRobotIO``
        therefore captures state/images first, sends the command, then calls
        :meth:`commit_tick` with the action the robot says it actually sent.
        """
        if not self._open:
            return None
        if self._abort_reason is not None:
            return None
        images = self._collect_images()
        if images is None:
            self._stale_run += 1
            if self._stale_run >= self._cfg.max_stale_ticks:
                # A gap is worse than a missing episode: the policy would
                # learn a jump between two frames that were seconds apart.
                self._abort_reason = "stale_camera"
                logger.warning(
                    "Discarding %s episode: no fresh camera frame for %d ticks",
                    self._color, self._stale_run,
                )
            return None
        self._stale_run = 0

        if self._frames >= self._cfg.max_episode_frames:
            self._abort_reason = "too_long"
            logger.warning(
                "Discarding %s episode: exceeded task3.max_episode_frames (%d)",
                self._color, self._cfg.max_episode_frames,
            )
            return None

        frame: dict[str, Any] = {
            "observation.state": self._vector(state),
            "task": self._task_text,
        }
        for name, image in images.items():
            frame[f"observation.images.{name}"] = image
        return frame

    def commit_tick(
        self, frame: dict[str, Any] | None, action: dict[str, float]
    ) -> None:
        """Append a captured observation with the post-safety action."""
        if frame is None:
            return
        frame["action"] = self._vector(action)
        self._sink.add_frame(frame)
        self._frames += 1

        now = time.monotonic()
        if self._last_tick_at is not None:
            self.tick_intervals.append(now - self._last_tick_at)
        self._last_tick_at = now

    def finish_episode(self, *, success: bool) -> bool:
        """Close the open episode. Returns True if it was saved."""
        if not self._open:
            return False
        if self._abort_reason is not None:
            return self._discard(self._abort_reason)
        if not success:
            return self._discard("pick_failed")
        if self._frames < self._cfg.min_episode_frames:
            logger.warning(
                "Discarding %s episode: %d frames is under task3.min_episode_frames (%d)",
                self._color, self._frames, self._cfg.min_episode_frames,
            )
            return self._discard("too_short")

        color = self._color or "unknown"
        self._sink.save_episode()
        self.saved_by_color[color] = self.saved_by_color.get(color, 0) + 1
        logger.info(
            "Saved %s episode (%d frames); %d saved this run",
            color, self._frames, self.saved_total,
        )
        self._reset()
        return True

    def abort_episode(self, reason: str = "interrupted") -> bool:
        """Drop the open episode, if any. Returns True if one was dropped."""
        if not self._open:
            return False
        return self._discard(reason)

    # ── internals ────────────────────────────────────────────────────

    def _discard(self, reason: str) -> bool:
        color = self._color or "unknown"
        self._sink.clear_episode_buffer()
        self.discarded_by_color[color] = self.discarded_by_color.get(color, 0) + 1
        self.discard_reasons[reason] = self.discard_reasons.get(reason, 0) + 1
        logger.info("Discarded %s episode after %d frames (%s)", color, self._frames, reason)
        self._reset()
        return False

    def _reset(self) -> None:
        self._open = False
        self._color = None
        self._task_text = ""
        self._frames = 0
        self._stale_run = 0
        self._abort_reason = None
        self._last_tick_at = None

    def _collect_images(self) -> dict[str, np.ndarray] | None:
        """Latest frame per camera, or None if any of them is stale."""
        now = time.monotonic()
        images: dict[str, np.ndarray] = {}
        for name, source in self._sources.items():
            frame = source.latest()
            if frame is None or now - frame.received_at > self._cfg.max_frame_age_s:
                return None
            images[name] = frame.image
        return images

    @staticmethod
    def _vector(values: dict[str, float]) -> np.ndarray:
        # build_dataset_frame would do this from "<joint>.pos" keys; building
        # it directly keeps the hot path free of the feature-dict walk and
        # pins the joint order to JOINT_NAMES, which is the order the
        # features were declared in.
        return np.array([values[joint] for joint in JOINT_NAMES], dtype=np.float32)

    def interval_stats(self) -> dict[str, float]:
        """Measured tick spacing — the check that ``record_fps`` is honest."""
        if not self.tick_intervals:
            return {}
        ordered = sorted(self.tick_intervals)
        def pct(p: float) -> float:
            return ordered[min(len(ordered) - 1, int(p * len(ordered)))]
        return {
            "mean_s": round(sum(ordered) / len(ordered), 4),
            "p50_s": round(pct(0.50), 4),
            "p95_s": round(pct(0.95), 4),
            "p99_s": round(pct(0.99), 4),
            "max_s": round(ordered[-1], 4),
            "samples": len(ordered),
        }


class RecordingRobotIO(BaseRobotIO):
    """A robot that records every command it is given, and paces the loop.

    **Why the pacing lives here.** ``TrajectoryPlayer._tick_sleep`` sleeps a
    flat ``1/motion.fps`` *after* the work of the tick, so the real period is
    ``1/fps + bus time`` — it cannot hold a rate. LeRobotDataset synthesises
    ``timestamp`` as ``frame_index / fps``, so a dataset labelled 30 fps whose
    frames are really 25 Hz apart trains a policy that replays 20% fast. Task 3
    therefore raises ``motion.fps`` until that sleep is negligible
    (``task3.motion_fps_override``) and this wrapper sleeps to an absolute
    deadline instead.

    The deadline is dropped whenever a tick arrives late by more than two
    periods — SELECT's camera polling and the rearrangement prompt are not
    part of any episode, and carrying a stale deadline across them would only
    produce a burst of catch-up commands.
    """

    def __init__(
        self,
        inner: BaseRobotIO,
        recorder: EpisodeRecorder,
        *,
        record_fps: float,
        stop_event: threading.Event | None = None,
    ):
        if record_fps <= 0:
            raise ValueError(f"record_fps must be positive, got {record_fps}")
        self._inner = inner
        self._recorder = recorder
        self._period_s = 1.0 / record_fps
        self._stop_event = stop_event or threading.Event()
        self._next_deadline: float | None = None
        # Latched full command. ``interpolate`` only emits the joints named in
        # the goal (``set_gripper`` sends the gripper alone, an arm move omits
        # it), but an ACT action vector has to be all six every tick — and the
        # physically true value for an uncommanded joint is the last thing it
        # was told, not its measured position.
        self._last_command: dict[str, float] | None = None

    # ── BaseRobotIO passthrough ──────────────────────────────────────

    def connect(self) -> None:
        self._inner.connect()

    def disconnect(self) -> None:
        self._inner.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._inner.is_connected

    def read_joints(self) -> dict[str, float]:
        return self._inner.read_joints()

    def read_observation(self) -> dict[str, Any]:
        return self._inner.read_observation()

    def read_loads(self) -> dict[str, int]:
        return self._inner.read_loads()

    def set_torque(self, enabled: bool) -> None:
        self._inner.set_torque(enabled)

    # ── the recorded tick ────────────────────────────────────────────

    def request_stop(self) -> None:
        self._stop_event.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def send_joints(self, positions: dict[str, float]) -> dict[str, float]:
        if self._stop_event.is_set():
            raise StopRecording("stop requested")

        # Read before commanding: the observation an action is conditioned on
        # must be the state the arm was in when the action was chosen.
        measured = self._inner.read_joints()
        if self._last_command is None:
            self._last_command = dict(measured)
        pending = self._recorder.capture_tick(measured)
        sent = self._inner.send_joints(positions)
        self._last_command.update(sent)
        self._recorder.commit_tick(pending, dict(self._last_command))
        self._pace()
        return sent

    def _pace(self) -> None:
        now = time.perf_counter()
        deadline = self._next_deadline
        if deadline is None or now - deadline > 2 * self._period_s:
            self._next_deadline = now + self._period_s
            return
        remaining = deadline - now
        if remaining > 0:
            time.sleep(remaining)
            self._next_deadline = deadline + self._period_s
        else:
            # Late but within a couple of periods: catch up without sleeping
            # so a single slow tick does not permanently shift the phase.
            self._next_deadline = deadline + self._period_s


def resolve_dataset_root(cfg: Task3Config, repo_id: str) -> Path | None:
    """Where this run's dataset goes; None means lerobot's default location."""
    if not cfg.root:
        return None
    root = Path(cfg.root)
    # A configured root is a parent directory when it does not already name
    # this run, so two stamped runs cannot collide inside it.
    return root if root.name == repo_id.split("/")[-1] else root / repo_id.split("/")[-1]


def remove_empty_dataset(root: Path | None) -> None:
    """Delete a dataset directory that never received an episode.

    A run that is interrupted before the first save leaves a skeleton that
    ``LeRobotDataset.create`` would then refuse to reuse, so the next attempt
    would fail on a directory holding nothing.
    """
    if root is None or not root.exists():
        return
    try:
        shutil.rmtree(root)
        logger.info("Removed empty dataset directory %s", root)
    except OSError as exc:
        logger.warning("Could not remove empty dataset directory %s: %s", root, exc)


def format_interval_report(stats: dict[str, float], record_fps: float) -> str:
    """Human-readable verdict on whether the recorded rate was real."""
    if not stats:
        return "no frames recorded"
    target = 1.0 / record_fps
    mean = stats["mean_s"]
    drift = (mean - target) / target * 100.0
    verdict = "ok" if abs(drift) < 10.0 and not math.isnan(drift) else "OFF TARGET"
    return (
        f"tick mean {mean * 1000:.1f}ms (target {target * 1000:.1f}ms, {drift:+.1f}%) "
        f"p95 {stats['p95_s'] * 1000:.1f}ms max {stats['max_s'] * 1000:.1f}ms "
        f"over {stats['samples']} frames -- {verdict}"
    )

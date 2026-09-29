"""Dataset lifecycle for composed primitives; no Task-3 runner or FSM invocation.

Only RecordingRobotIO records commands. The worker supplies idle hold ticks on
its own thread while the model is thinking. Timing gaps invalidate a take;
missing wall time is never silently relabelled as a continuous 30-Hz motion.
"""
from __future__ import annotations

import copy
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from control.robot_io import BaseRobotIO
from data.episode_recorder import EpisodeRecorder, LeRobotEpisodeSink, RecordingRobotIO


class RecordingQualityError(ValueError):
    """The take is unusable; this is not a motor or bus failure."""


@dataclass
class CollectionResources:
    recorder: EpisodeRecorder
    root: Path
    finish: object


def open_resources(cfg):
    # Import before starting streams, so missing dataset extras have no side effects.
    from lerobot.datasets import VideoEncodingManager
    from camera.frame_source import build_frame_sources
    from data.episode_recorder import create_dataset
    from config import validate_task3

    validate_task3(cfg)
    task = copy.deepcopy(cfg.task3)
    repo_id = task.repo_id + time.strftime("_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    root = Path(cfg.agent.collection.root).expanduser().resolve() / repo_id.split("/")[-1]
    root.parent.mkdir(parents=True, exist_ok=True)
    sources = build_frame_sources(task.cameras, width=task.image_width, height=task.image_height)
    started = []
    try:
        for source in sources.values():
            source.start()
            started.append(source)
        for source in sources.values():
            if not source.wait_for_first_frame(timeout_s=cfg.agent.camera_fresh_timeout_s):
                raise RuntimeError("Dataset camera unavailable: " + source.url)
        dataset = create_dataset(task, repo_id, root, resume=False)
        manager = VideoEncodingManager(dataset)
        manager.__enter__()
    except BaseException:
        for source in started:
            source.stop()
        raise

    def finish():
        try:
            manager.__exit__(None, None, None)
        finally:
            for source in started:
                source.stop()  # only our HTTP reader; never camera.server
    return CollectionResources(EpisodeRecorder(LeRobotEpisodeSink(dataset), sources, task), root, finish)


class CollectionRobotIO(BaseRobotIO):
    """Switch the existing recorder on without changing any motion implementation."""
    def __init__(self, inner, owner):
        self.inner = inner
        self.owner = owner
        self.last_command = None

    def connect(self): self.inner.connect()
    def disconnect(self): self.inner.disconnect()
    @property
    def is_connected(self): return self.inner.is_connected
    def read_joints(self): return self.inner.read_joints()
    def read_observation(self): return self.inner.read_observation()
    def read_loads(self): return self.inner.read_loads()
    def set_torque(self, enabled): self.inner.set_torque(enabled)

    def send_joints(self, positions):
        if self.last_command is None:
            self.last_command = dict(self.inner.read_joints())
        owner = self.owner
        if owner.recording:
            try:
                owner.check_tick()
            except RecordingQualityError:
                # Losing a take must not interrupt a verified transfer while
                # holding a block. Finish this tool without recording; the
                # sequence reports the discarded take before its next step.
                sent = self.inner.send_joints(positions)
                self.last_command.update(sent)
                return sent
            sent = owner.writer.send_joints(positions)
            if owner.recorder.abort_reason is not None:
                reason = owner.recorder.abort_reason
                owner.discard(reason)
                raise ValueError(f"Recording invalidated: {reason}")
        else:
            sent = self.inner.send_joints(positions)
        self.last_command.update(sent)
        return sent


class Collection:
    def __init__(self, session, *, resource_factory=open_resources, clock=time.monotonic):
        self.s = session
        self.cfg = session.cfg
        self.factory = resource_factory
        self.clock = clock
        self.resources = None
        self.writer = None
        self.io = CollectionRobotIO(session.robot, self)
        self.recording = False
        self.last_error = None
        self.last_finished = None
        self.finalizing = False
        self._original_fps = None
        self._reset_evidence()

    @property
    def recorder(self):
        return self.resources.recorder if self.resources else None

    def _reset_evidence(self):
        self.color = None
        self.sequence_mode = False
        self.stationary_pauses = []
        self.recording_gaps = []
        self.missing_motion_s = 0.0
        self.grasped = False
        self.delivered = False
        self.destination_in_zone = False
        self.contacted = False
        self.home = False
        self.verified_at = None
        self.last_tick = None
        self.intervals = []
        self.events = []

    def status(self):
        if self.resources is None and self.last_finished is not None:
            return dict(self.last_finished)
        recorder = self.recorder
        return {
            "dataset_root": str(self.resources.root) if self.resources else None,
            "episode_open": bool(recorder and recorder.is_open),
            "recording": self.recording,
            "frames": recorder.frames if recorder else 0,
            "episodes_saved": recorder.saved_total if recorder else 0,
            "saved_by_color": dict(recorder.saved_by_color) if recorder else {},
            "discard_reasons": dict(recorder.discard_reasons) if recorder else {},
            "record_fps": self.cfg.task3.record_fps,
            "color": self.color, "sequence_mode": self.sequence_mode,
            "stationary_pauses": list(self.stationary_pauses),
            "recording_gaps": list(self.recording_gaps),
            "missing_motion_s": round(self.missing_motion_s, 4),
            "grasp_verified": self.grasped,
            "delivered": self.delivered, "returned_home": self.home,
            "placement_observed": self.verified_at is not None,
            "last_error": self.last_error,
            "training_started": False, "finalizing": self.finalizing,
        }

    def begin(self, color, *, task_text=None, sequence_mode=False):
        if self.finalizing:
            raise ValueError("Dataset finalization is unresolved; retry finish_dataset before new recording")
        if self.recorder and self.recorder.is_open:
            raise ValueError("Resolve the current episode before starting another")
        if self.s.held is not None or not self.s.arm_at_home():
            raise ValueError("Begin an episode at home with an empty gripper")
        if self.resources is None:
            self.resources = self.factory(self.cfg)
            self.last_finished = None
        self._reset_evidence()
        self.last_error = None
        self.color = color
        self.sequence_mode = sequence_mode
        self.recorder.begin_episode(color, task_text=task_text)
        self.writer = RecordingRobotIO(self.io.inner, self.recorder,
                                        record_fps=self.cfg.task3.record_fps,
                                        stop_event=self.s.cancel.event)
        self._original_fps = self.cfg.motion.fps
        self.cfg.motion.fps = self.cfg.task3.motion_fps_override
        self.recording = True
        # Include a measured home sample, not just the next movement's start.
        self.io.send_joints(self.s.robot.read_joints())
        self._summary()

    def check_tick(self):
        now = self.clock()
        if self.last_tick is not None:
            gap = now - self.last_tick
            cfg = self.cfg.agent.collection
            period = 1.0 / self.cfg.task3.record_fps
            missing_ticks = max(0, int(gap / period + 0.5) - 1)
            short_limit = max(cfg.max_tick_gap_s, (cfg.tolerated_missing_ticks + 1) * period)
            drift = None
            if gap > short_limit + 1e-9:
                if self.writer is not None and self.writer.last_measured is not None:
                    current = self.io.inner.read_joints()
                    drift = max(abs(current[j] - value)
                                for j, value in self.writer.last_measured.items())
                if self.sequence_mode and drift is not None and drift <= cfg.max_stationary_drift:
                    self.stationary_pauses.append({"duration_s": round(gap, 3),
                                                   "max_joint_drift": round(drift, 3)})
                    self.recorder.reset_tick_interval()
                    self.last_tick = now
                    return
            if missing_ticks:
                missing_s = max(0.0, gap - period)
                self.missing_motion_s += missing_s
                # One/two-tick jitter is explicitly tolerated. Summing only
                # positive jitter makes a well-paced long take fail even when
                # the following short ticks catch up. Keep all gaps in the
                # diagnostics, but charge only longer gaps to this budget.
                budget_missing_s = sum(item["missing_s"] for item in self.recording_gaps
                                       if item["duration_s"] > short_limit + 1e-9)
                if gap > short_limit + 1e-9:
                    budget_missing_s += missing_s
                over_budget = (gap > max(short_limit, cfg.max_moving_gap_s) + 1e-9
                               or budget_missing_s > cfg.max_total_missing_motion_s + 1e-9)
                self.recording_gaps.append({
                    "duration_s": round(gap, 4), "estimated_missing_ticks": missing_ticks,
                    "missing_s": round(missing_s, 4),
                    "max_joint_drift": round(drift, 3) if drift is not None else None,
                    "accepted": not over_budget,
                })
                if over_budget:
                    total = self.missing_motion_s
                    self.discard("moving_timing_gap" if self.sequence_mode else "timing_gap")
                    raise RecordingQualityError(
                        f"Recording gap budget exceeded: gap={gap:.3f}s, "
                        f"missing_total={total:.3f}s, budget_missing={budget_missing_s:.3f}s; episode discarded")
            # Accepted motion gaps remain in timing statistics. Do not fabricate
            # frames or label missing movement as a stationary pause.
            self.intervals.append(gap)
        self.last_tick = now

    def idle_tick(self):
        if not self.recording:
            return
        if self.s.cancel.is_set():
            self.discard("cancelled")
            return
        try:
            # Reissue the actual latched action, including the gripper target.
            self.io.send_joints(dict(self.io.last_command or self.s.robot.read_joints()))
        except Exception as exc:
            self.discard("recording_error")
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.s.cancel.set()
            self._summary()

    def settle_for_sequence_pause(self):
        """Record the servo tail before a composite tool enters a long IK plan.

        IK runs on the robot thread and cannot capture frames concurrently. A
        measured still interval makes the following gap safely compressible;
        moving through that gap must still discard the episode.
        """
        if not self.recording or not self.sequence_mode:
            return
        cfg = self.cfg.agent.collection
        reference = dict(self.writer.last_measured or self.s.robot.read_joints())
        start = self.clock()
        stable_since = start
        command = dict(self.io.last_command or self.s.robot.read_joints())
        while self.clock() - start < cfg.sequence_settle_timeout_s:
            self.io.send_joints(command)
            measured = self.writer.last_measured
            if max(abs(measured[joint] - value) for joint, value in reference.items()) > cfg.sequence_settle_max_drift:
                reference = dict(measured)
                stable_since = self.clock()
            elif self.clock() - stable_since >= cfg.sequence_settle_window_s:
                return
        self.discard("sequence_not_settled")
        raise TimeoutError("Recorded sequence did not settle before IK planning")

    def _pause(self):
        self.recording = False
        if self._original_fps is not None:
            self.cfg.motion.fps = self._original_fps
            self._original_fps = None

    def note(self, action, result):
        recorder = self.recorder
        if not recorder or not recorder.is_open:
            return
        self.events.append({"t": self.clock(), "action": action, "ok": result.ok, "reason": result.reason})
        if self.sequence_mode:
            if not result.ok:
                self.discard(result.reason)
            return
        if self.home and action in {"close_gripper", "move_relative", "move_to_target", "align_gripper", "descend_until_contact", "open_gripper"}:
            self.discard("motion_after_episode_end")
            return
        if not result.ok and action not in {"save_episode", "begin_episode", "finish_dataset", "collection_status"}:
            self.discard(result.reason)
            return
        if action == "close_gripper":
            self.grasped = result.ok and self.s.held is not None and self.s.held.color == self.color
            if not self.grasped:
                self.discard("wrong_or_unknown_object")
        elif action == "move_to_target" and self.s.held is not None:
            # The skill supplies this fact from its resolved target, not tool arguments.
            self.destination_in_zone = bool(result.data.get("collection_zone_destination"))
            self.contacted = False
        elif action in {"descend_until_contact", "drop_at_zone_target"}:
            self.contacted = bool(result.ok and (
                result.data.get("contact", {}).get("contact")
                or result.data.get("release_mode") == "height_drop"
            ))
        elif action == "move_relative":
            self.contacted = False
        elif action == "open_gripper" and self.grasped:
            self.delivered = self.contacted and self.s.held is None
            if not self.delivered:
                self.discard("unverified_delivery")
        elif action in {"place_at_pixel", "place_at_cell", "place_at_slot", "place_on_table", "place_here"} and self.grasped:
            self.delivered = result.ok and self.s.held is None
        elif action == "return_to_home" and self.delivered:
            if self.s.held is None and self.s.arm_at_home():
                self.io.send_joints(dict(self.io.last_command or self.s.robot.read_joints()))
                self.home = True
                self._pause()  # home-to-home episode ends before camera polling
        elif action == "observe_scene" and self.home and self.delivered:
            scene = self.s.last_scene
            self.verified_at = self.clock() if scene and self.color in scene.inside else None
        elif self.home and action in {"close_gripper", "move_relative", "move_to_target", "align_gripper", "descend_until_contact", "open_gripper"}:
            self.discard("motion_after_episode_end")
        self._summary()

    def save(self):
        if not self.recorder or not self.recorder.is_open:
            raise ValueError("No open episode")
        if self.s.cancel.is_set():
            self.discard("cancelled")
            raise ValueError("Stopped episode discarded")
        if not (self.grasped and self.delivered and self.home and self.s.held is None and self.s.arm_at_home()):
            raise ValueError("Save requires verified grasp, release completion and home return")
        if self.intervals:
            relative_error = abs(sum(self.intervals)/len(self.intervals)*self.cfg.task3.record_fps - 1)
            if relative_error > self.cfg.agent.collection.max_mean_period_error:
                self.discard("timing_drift")
                return False
        saved = self.recorder.finish_episode(success=True)
        self._pause()
        self._summary()
        return saved

    def save_sequence(self):
        """Save a completed preplanned tool run, without zone-specific assumptions."""
        if not self.sequence_mode or not self.recorder or not self.recorder.is_open:
            raise ValueError("No open tool-sequence episode")
        if self.s.cancel.is_set() or self.s.held is not None or not self.s.arm_at_home():
            self.discard("incomplete_sequence")
            raise ValueError("Recorded sequence must finish empty-handed at home")
        # Include measured settled home as the final observation/action pair.
        self.io.send_joints(dict(self.io.last_command or self.s.robot.read_joints()))
        if self.intervals:
            mean = sum(self.intervals) / len(self.intervals)
            if abs(mean * self.cfg.task3.record_fps - 1) > self.cfg.agent.collection.max_mean_period_error:
                self.discard("timing_drift")
                return False
        saved = self.recorder.finish_episode(success=True)
        self._pause()
        self._summary()
        return saved

    def discard(self, reason):
        self._pause()
        if self.recorder and self.recorder.is_open:
            self.recorder.abort_episode(reason)
            self.last_error = reason
            self._summary()

    def _summary(self):
        if self.resources:
            path = self.resources.root / "agent_collection_summary.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {**self.status(), "tick_interval": self.recorder.interval_stats(), "last_episode_events": self.events}
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            temp.replace(path)

    def finish(self):
        if self.recorder and self.recorder.is_open:
            raise ValueError("Save or discard the current episode before finishing the dataset")
        if self.resources:
            self.finalizing = True
            try:
                self.resources.finish()
            except Exception as exc:
                self.last_error = f"finalization_failed: {exc}"
                self._summary()
                raise
            self.finalizing = False
            self._summary()
            status = {**self.status(), "finalized": True}
            self.last_finished = status
            self.resources = None
            self.writer = None
            return status
        return self.status()

    def close(self):
        self.discard("shutdown")
        self.finish()

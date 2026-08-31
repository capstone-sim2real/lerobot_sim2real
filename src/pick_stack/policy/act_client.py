"""PICK policy client: runs one bounded ACT episode per call.

Adapted from lerobot.async_inference.robot_client.RobotClient (Apache-2.0)
with three deliberate differences:

  1. the robot is injected — the FSM owns the single serial-bus/camera
     connection and lends it to this client during PICK only;
  2. execution is per-episode (``run_pick``), terminating when the arm holds
     the fixed retreat pose (episodes are trained to end there, EPISODE.md)
     or on timeout — not an infinite control loop;
  3. the queue/aggregation core is dependency-free (plain float lists), so
     it unit-tests without torch/grpc; the wire specifics live in
     ``transport.GrpcPolicyTransport``.

The validated async-chain behaviour is preserved: actions are keyed by
timestep, chunks overlap-aggregate (``weighted_average``/``latest``), a new
observation is sent when the queue drains below ``chunk_size_threshold``,
and an observation is flagged ``must_go`` when the queue is empty.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from pick_stack.config import PolicyConfig
from pick_stack.control.robot_io import JOINT_NAMES, BaseRobotIO
from pick_stack.policy.transport import ActionStep, PolicyTransport

logger = logging.getLogger(__name__)


@dataclass
class PickResult:
    # "retreat_reached" = successful handoff; "timeout" = policy never settled
    outcome: str
    duration_s: float
    ticks: int
    actions_executed: int
    final_joints: dict[str, float] = field(default_factory=dict)

    @property
    def reached_retreat(self) -> bool:
        return self.outcome == "retreat_reached"


class RetreatDetector:
    """Fires after K consecutive ticks with every checked joint within
    tolerance of the retreat pose (gripper excluded — it holds the block)."""

    def __init__(self, retreat_pose: dict[str, float], tol: float, hold_ticks: int, joints: list[str]):
        missing = [j for j in joints if j not in retreat_pose]
        if missing:
            raise ValueError(f"Retreat pose is missing checked joint(s): {missing}")
        self._pose = dict(retreat_pose)
        self._tol = tol
        self._hold_ticks = max(1, hold_ticks)
        self._joints = list(joints)
        self._streak = 0

    def reset(self) -> None:
        self._streak = 0

    def update(self, joints: dict[str, float]) -> bool:
        within = all(abs(joints[j] - self._pose[j]) <= self._tol for j in self._joints)
        self._streak = self._streak + 1 if within else 0
        return self._streak >= self._hold_ticks


def _make_aggregate_fn(name: str, weight: float):
    if name == "latest":
        return lambda old, new: new
    if name == "weighted_average":
        return lambda old, new: [(1.0 - weight) * o + weight * n for o, n in zip(old, new)]
    raise ValueError(f"Unknown aggregate_fn_name {name!r} (expected 'latest' or 'weighted_average')")


class ActPolicyClient:
    def __init__(self, robot: BaseRobotIO, transport: PolicyTransport, cfg: PolicyConfig):
        self._robot = robot
        self._transport = transport
        self._cfg = cfg
        self._aggregate = _make_aggregate_fn(cfg.aggregate_fn_name, cfg.aggregate_weight)

        self._queue: dict[int, list[float]] = {}
        self._queue_lock = threading.Lock()
        self._latest_timestep = -1
        self._chunk_size = max(1, cfg.actions_per_chunk)
        self._must_go = threading.Event()
        self._stop_receiver = threading.Event()

    # -- session ------------------------------------------------------------

    def connect(self) -> None:
        """Handshake + model load on the server. Call once per session, not
        per pick — reloading the policy per block would eat the time budget."""
        self._transport.connect()

    def ping(self) -> bool:
        return self._transport.ping()

    def close(self) -> None:
        self._transport.close()

    # -- queue core (mirrors lerobot's aggregation semantics) ----------------

    def _merge_actions(self, incoming: list[ActionStep]) -> None:
        with self._queue_lock:
            for timestep, action in incoming:
                if timestep <= self._latest_timestep:
                    continue
                if timestep in self._queue:
                    self._queue[timestep] = self._aggregate(self._queue[timestep], action)
                else:
                    self._queue[timestep] = list(action)

    def _pop_next_action(self) -> tuple[int, list[float]] | None:
        with self._queue_lock:
            if not self._queue:
                return None
            timestep = min(self._queue)
            action = self._queue.pop(timestep)
            self._latest_timestep = timestep
            return timestep, action

    def _ready_to_send_observation(self) -> bool:
        with self._queue_lock:
            return len(self._queue) / self._chunk_size <= self._cfg.chunk_size_threshold

    def _action_to_joints(self, action: list[float]) -> dict[str, float]:
        # action vector order == robot action_features order == JOINT_NAMES
        if len(action) != len(JOINT_NAMES):
            raise ValueError(f"Action length {len(action)} != {len(JOINT_NAMES)} joints")
        return {name: float(action[i]) for i, name in enumerate(JOINT_NAMES)}

    def _receive_loop(self) -> None:
        while not self._stop_receiver.is_set():
            incoming = self._transport.poll_actions()
            if incoming:
                self._merge_actions(incoming)
                self._must_go.set()
            else:
                time.sleep(0.005)

    # -- one PICK episode -----------------------------------------------------

    def run_pick(self, retreat_pose: dict[str, float]) -> PickResult:
        detector = RetreatDetector(
            retreat_pose, self._cfg.retreat_tol, self._cfg.retreat_hold_ticks, self._cfg.retreat_check_joints
        )
        with self._queue_lock:
            self._queue.clear()
        self._must_go.set()
        self._stop_receiver.clear()
        receiver = threading.Thread(target=self._receive_loop, daemon=True, name="pick-action-receiver")
        receiver.start()

        tick_s = 1.0 / self._cfg.fps if self._cfg.fps > 0 else 0.0
        start = time.monotonic()
        deadline = start + self._cfg.pick_timeout_s
        ticks = 0
        executed = 0
        outcome = "timeout"
        joints: dict[str, float] = {}
        try:
            while time.monotonic() < deadline:
                tick_start = time.monotonic()
                ticks += 1

                popped = self._pop_next_action()
                if popped is not None:
                    _, action = popped
                    self._robot.send_joints(self._action_to_joints(action))
                    executed += 1

                if self._ready_to_send_observation():
                    raw_obs = self._robot.read_observation()
                    raw_obs["task"] = self._cfg.task
                    with self._queue_lock:
                        must_go = self._must_go.is_set() and not self._queue
                    if self._transport.send_observation(raw_obs, max(self._latest_timestep, 0), must_go):
                        if must_go:
                            self._must_go.clear()

                joints = self._robot.read_joints()
                if detector.update(joints):
                    outcome = "retreat_reached"
                    break

                if tick_s > 0:
                    time.sleep(max(0.0, tick_s - (time.monotonic() - tick_start)))
        finally:
            self._stop_receiver.set()
            receiver.join(timeout=1.0)

        result = PickResult(
            outcome=outcome,
            duration_s=time.monotonic() - start,
            ticks=ticks,
            actions_executed=executed,
            final_joints=joints,
        )
        logger.info(
            "PICK finished: %s (%.1fs, %d ticks, %d actions)",
            result.outcome, result.duration_s, result.ticks, result.actions_executed,
        )
        return result

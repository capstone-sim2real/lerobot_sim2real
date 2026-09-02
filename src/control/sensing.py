"""Load/position sensing: grasp verification + contact detection.

One utility, two callers (AGENTS.md §10):
  - VERIFY state calls ``check_grasp`` at the retreat pose,
  - the Task-2 stack descent polls ``ContactMonitor`` between steps.

Everything here is *read-only* on the bus — commanding the gripper or the
descent is the caller's job. That keeps the sensing thresholds tunable with
tools/tune_gripper_load.py without moving the arm.

Never advance to PLACE without ``GraspCheck.grasped`` — hard rule.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from config import SensingConfig
from control.robot_io import BaseRobotIO

_GRASP_MODES = ("position_only", "load_only", "position_and_load", "position_or_load")


@dataclass
class GraspCheck:
    grasped: bool
    gripper_pos: float
    gripper_load_abs: float
    pos_says_held: bool
    load_says_held: bool
    mode: str


@dataclass
class ContactReading:
    contact: bool
    # per-joint |load - baseline| for the monitored joints
    deltas: dict[str, float] = field(default_factory=dict)
    loads: dict[str, float] = field(default_factory=dict)


def _sample_mean(robot: BaseRobotIO, samples: int, interval_s: float) -> tuple[float, float]:
    """Mean gripper Present_Position and |Present_Load| over a few reads."""
    pos_acc, load_acc = 0.0, 0.0
    for i in range(samples):
        if i > 0 and interval_s > 0:
            time.sleep(interval_s)
        pos_acc += robot.read_joints()["gripper"]
        load_acc += abs(robot.read_loads()["gripper"])
    return pos_acc / samples, load_acc / samples


def check_grasp(robot: BaseRobotIO, cfg: SensingConfig, *, settle: bool = True) -> GraspCheck:
    """Decide whether the gripper is holding a block.

    Assumes the close command was already sent. Position signal: an empty
    gripper reaches ``gripper_empty_closed_max`` or below; a 20 mm block
    stops it earlier. Load signal: a held block keeps |Present_Load| high.
    """
    if cfg.grasp_check_mode not in _GRASP_MODES:
        raise ValueError(f"Unknown grasp_check_mode {cfg.grasp_check_mode!r}, expected one of {_GRASP_MODES}")
    if settle and cfg.grasp_settle_s > 0:
        time.sleep(cfg.grasp_settle_s)
    pos, load_abs = _sample_mean(robot, max(1, cfg.grasp_samples), cfg.sample_interval_s)
    pos_says_held = pos > cfg.gripper_empty_closed_max
    load_says_held = load_abs >= cfg.gripper_load_min
    grasped = {
        "position_only": pos_says_held,
        "load_only": load_says_held,
        "position_and_load": pos_says_held and load_says_held,
        "position_or_load": pos_says_held or load_says_held,
    }[cfg.grasp_check_mode]
    return GraspCheck(
        grasped=grasped,
        gripper_pos=pos,
        gripper_load_abs=load_abs,
        pos_says_held=pos_says_held,
        load_says_held=load_says_held,
        mode=cfg.grasp_check_mode,
    )


class ContactMonitor:
    """Detects the load spike when a held block touches the surface below.

    Usage: hold still above the target, call ``start()`` to capture the
    baseline, then poll ``check()`` after each small descent step.
    """

    def __init__(self, robot: BaseRobotIO, cfg: SensingConfig):
        self._robot = robot
        self._cfg = cfg
        self._baseline: dict[str, float] | None = None

    def start(self) -> dict[str, float]:
        acc = {joint: 0.0 for joint in self._cfg.contact_joints}
        samples = max(1, self._cfg.contact_baseline_samples)
        for i in range(samples):
            if i > 0 and self._cfg.sample_interval_s > 0:
                time.sleep(self._cfg.sample_interval_s)
            loads = self._robot.read_loads()
            for joint in acc:
                acc[joint] += loads[joint]
        self._baseline = {joint: total / samples for joint, total in acc.items()}
        return dict(self._baseline)

    def check(self) -> ContactReading:
        if self._baseline is None:
            raise RuntimeError("ContactMonitor.check() before start(); capture a baseline first")
        loads = self._robot.read_loads()
        deltas = {joint: abs(loads[joint] - self._baseline[joint]) for joint in self._cfg.contact_joints}
        contact = any(delta >= self._cfg.contact_load_delta for delta in deltas.values())
        return ContactReading(
            contact=contact,
            deltas=deltas,
            loads={joint: float(loads[joint]) for joint in self._cfg.contact_joints},
        )

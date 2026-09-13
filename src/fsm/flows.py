"""Composable FSM flows built from the shared state handlers.

The individual states know how to select, pick, verify, or move.  This
module is the only place that decides *which state follows which*, so a
hardware smoke test does not need to fork the production Task 1 FSM.
"""

from __future__ import annotations

from typing import Callable

from config import AppConfig, SensingConfig
from control.motion import MotionController
from control.robot_io import BaseRobotIO
from control.trajectory import TrajectoryPlayer
from control.task1_transport import Task1TransportPlanner
from control.task2_stack import Task2StackPlanner
from fsm.handlers import (
    ContextMotionState,
    PerceiveFn,
    PlaceState,
    ReleaseState,
    SelectState,
    StackPlaceStrategy,
    TransportState,
    VerifyState,
)
from fsm.states import RunContext, State, StateName
from fsm.task1 import (
    Task1PerceiveFn,
    Task1PlaceState,
    Task1SelectState,
    Task1TransportState,
)
from fsm.task2 import Task2PlaceState, Task2SelectState, Task2TransportState
from fsm.task3 import Task3PlaceState, Task3SelectState
from data.episode_recorder import EpisodeRecorder
from control.grasp import GraspAttempt
from perception.homography import PlaneCalibration


def _common_states(
    *,
    robot: BaseRobotIO,
    motion: MotionController,
    perceive: PerceiveFn,
    pick_state: State,
    sensing_cfg: SensingConfig,
    after_verified: StateName,
    select_state: State | None = None,
) -> dict[StateName, State]:
    if pick_state.name is not StateName.PICK:
        raise ValueError("pick_state must implement the PICK state")
    return {
        StateName.SELECT: select_state or SelectState(motion, perceive),
        StateName.PICK: pick_state,
        StateName.VERIFY: VerifyState(
            robot, sensing_cfg, motion, on_grasped=after_verified
        ),
    }


def build_task1_states(
    *,
    robot: BaseRobotIO,
    motion: MotionController,
    perceive: Task1PerceiveFn,
    pick_state: State,
    cfg: AppConfig,
    calib: PlaneCalibration,
    planner: Task1TransportPlanner,
) -> dict[StateName, State]:
    """Gather until fresh perception proves no outside-zone blocks remain."""
    if pick_state.name is not StateName.PICK:
        raise ValueError("pick_state must implement the PICK state")
    player = TrajectoryPlayer(robot, cfg.motion)
    return {
        StateName.SELECT: Task1SelectState(motion, perceive, calib, cfg),
        StateName.PICK: pick_state,
        StateName.VERIFY: VerifyState(robot, cfg.sensing, motion, on_grasped=StateName.TRANSPORT),
        StateName.TRANSPORT: Task1TransportState(planner, player, cfg),
        StateName.PLACE: Task1PlaceState(motion, player, cfg),
    }


def build_task3_states(
    *,
    robot: BaseRobotIO,
    motion: MotionController,
    perceive: Task1PerceiveFn,
    pick_state: State,
    cfg: AppConfig,
    calib: PlaneCalibration,
    planner: Task1TransportPlanner,
    recorder: EpisodeRecorder,
    prompt=input,
    stop_requested: Callable[[], bool] = lambda: False,
) -> dict[StateName, State]:
    """Task 1's gather loop, recorded as ACT episodes.

    PICK, VERIFY and TRANSPORT are Task 1's own handlers, unmodified: the
    trajectories being recorded have to be the ones the mission actually
    runs. Recording happens a layer below, in the ``RecordingRobotIO`` the
    caller wraps ``robot`` with, so nothing in this flow knows about it
    except the two states that own the episode boundary.
    """
    if pick_state.name is not StateName.PICK:
        raise ValueError("pick_state must implement the PICK state")
    player = TrajectoryPlayer(robot, cfg.motion)
    return {
        StateName.SELECT: Task3SelectState(
            motion,
            perceive,
            calib,
            cfg,
            recorder,
            prompt=prompt,
            stop_requested=stop_requested,
        ),
        StateName.PICK: pick_state,
        StateName.VERIFY: VerifyState(robot, cfg.sensing, motion, on_grasped=StateName.TRANSPORT),
        StateName.TRANSPORT: Task1TransportState(planner, player, cfg),
        StateName.PLACE: Task3PlaceState(motion, player, cfg),
    }


def build_task2_stack_states(
    *,
    robot: BaseRobotIO,
    motion: MotionController,
    perceive: Task1PerceiveFn,
    pick_state: State,
    cfg: AppConfig,
    calib: PlaneCalibration,
    planner: Task2StackPlanner,
) -> dict[StateName, State]:
    """Stack every block at one point; SELECT/PICK/VERIFY/TRANSPORT are Task 1's.

    The CV+IK Task 2. ``build_task2_states`` below is the earlier
    recorded-pose generation, kept for the pose-registry path.
    """
    if pick_state.name is not StateName.PICK:
        raise ValueError("pick_state must implement the PICK state")
    player = TrajectoryPlayer(robot, cfg.motion)
    return {
        StateName.SELECT: Task2SelectState(motion, perceive, calib, cfg),
        StateName.PICK: pick_state,
        StateName.VERIFY: VerifyState(robot, cfg.sensing, motion, on_grasped=StateName.TRANSPORT),
        StateName.TRANSPORT: Task2TransportState(planner, player, cfg),
        StateName.PLACE: Task2PlaceState(robot, motion, player, cfg),
    }


def build_task2_states(
    *,
    robot: BaseRobotIO,
    motion: MotionController,
    perceive: PerceiveFn,
    pick_state: State,
    sensing_cfg: SensingConfig,
    select_state: State | None = None,
) -> dict[StateName, State]:
    """Production stacking flow: SELECT → PICK → VERIFY → TRANSPORT → PLACE."""
    states = _common_states(
        robot=robot,
        motion=motion,
        perceive=perceive,
        pick_state=pick_state,
        sensing_cfg=sensing_cfg,
        after_verified=StateName.TRANSPORT,
        select_state=select_state,
    )
    states.update(
        {
            StateName.TRANSPORT: TransportState(motion),
            StateName.PLACE: PlaceState(StackPlaceStrategy(motion)),
        }
    )
    return states


def _held_attempt(ctx: RunContext) -> GraspAttempt:
    held = ctx.extras.get("ik_pick_attempt")
    if not isinstance(held, GraspAttempt):
        raise RuntimeError("Pick-test flow has no successful CV+IK grasp plan")
    return held


def build_pick_lift_lower_states(
    *,
    robot: BaseRobotIO,
    motion: MotionController,
    perceive: PerceiveFn,
    pick_state: State,
    cfg: AppConfig,
    select_state: State | None = None,
) -> dict[StateName, State]:
    """One-block smoke test with no destination poses.

    SELECT → PICK → VERIFY → LIFT → LOWER → RELEASE → RETURN → DONE.
    The lift target is the highest top-down-reachable hover calculated for
    this grasp (up to ``motion.hover_clearance_mm``), not a blind fixed height.
    """
    states = _common_states(
        robot=robot,
        motion=motion,
        perceive=perceive,
        pick_state=pick_state,
        sensing_cfg=cfg.sensing,
        after_verified=StateName.LIFT,
        select_state=select_state,
    )
    player = TrajectoryPlayer(robot, cfg.motion)
    transit = {"max_step": 1.0, "tol": cfg.motion.transit_arrival_tol}
    states.update(
        {
            StateName.LIFT: ContextMotionState(
                StateName.LIFT,
                player,
                lambda ctx: _held_attempt(ctx).hover.joints,
                next_state=StateName.LOWER,
                **transit,
            ),
            StateName.LOWER: ContextMotionState(
                StateName.LOWER,
                player,
                lambda ctx: _held_attempt(ctx).grasp.joints,
                next_state=StateName.RELEASE,
                max_step=cfg.motion.descent_step_per_tick,
                tol=cfg.motion.transit_arrival_tol,
            ),
            StateName.RELEASE: ReleaseState(motion, next_state=StateName.RETURN),
            StateName.RETURN: ContextMotionState(
                StateName.RETURN,
                player,
                lambda ctx: _held_attempt(ctx).hover.joints,
                next_state=StateName.DONE,
                **transit,
            ),
        }
    )
    return states

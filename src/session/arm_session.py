"""One connected arm and everything it needs, built once, used from one thread.

``ArmSession`` owns what ``runners.run_task.run`` builds inline: calibration,
poses, the robot connection, the one ``TopDownIK`` (its seed table is
expensive and ``_load_kinematics`` does ``os.chdir``), the planners and the
CV+IK PICK state. The LLM agent keeps one session open for the life of the
server and calls ``session.skills.Skills`` on it.

Rules this class keeps:

- ``cfg`` is a private deep copy. Overrides go through ``config.apply_override``
  on that copy, so one skill can never leak a config change into the next
  (``runners/run_task3.py`` mutates ``cfg.motion.fps`` in place; the agent
  must not inherit that).
- Every arm command goes through ``CancellableRobotIO``: STOP raises
  ``Cancelled`` at the next bus write, with no hook in control/ or fsm/.
- Construct, use and close it on the same thread.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

from camera.client import CameraSnapshot, fetch_snapshot_with_metadata
from config import AppConfig, apply_override
from control.grasp import GraspAttempt, highest_reachable_hover
from control.ik import TopDownIK
from control.motion import MotionController
from control.poses import PoseRegistry
from control.robot_io import BaseRobotIO
from control.task1_transport import (
    Task1SlotPlan,
    Task1TransportPlan,
    carry_waypoints,
    fly_carry,
    over_ik_gate,
    push_out_from_base,
    release_at,
    solve_place_point,
)
from control.trajectory import TrajectoryPlayer
from perception.detector import point_in_workspace
from perception.homography import PlaneCalibration
from perception.scene import Scene, detect_scene
from perception.zone import point_in_zone, zone_slot_centres
from session.cancel import CancellableRobotIO, Cancelled, CancelToken
from session.factories import calibration_grasp_z_mm
from session.lock import RobotBusLock

logger = logging.getLogger(__name__)

XY = tuple[float, float]
PICK_TILT_KEY = "task1_pick_radial_tilt_deg"


class CameraError(RuntimeError):
    """The camera service could not provide a usable, fresh frame."""

    def __init__(self, message: str, *, stale: bool = False):
        super().__init__(message)
        self.stale = stale


@dataclass
class HeldBlock:
    # None for a manual claw-machine grab (pick_here): something is held but
    # its colour was never looked up, since a photo taken from directly above
    # it would just show the arm's own gripper.
    color: str | None
    attempt: GraspAttempt
    # where the block was detected when picked (before any correction)
    picked_xy_mm: XY
    from_zone: bool
    # where the gripper currently hovers; move_arm updates it
    over_xy_mm: XY


@dataclass
class LastPick:
    color: str
    forward_mm: float
    left_mm: float


class ArmSession:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        calib: PlaneCalibration,
        poses: PoseRegistry,
        robot: BaseRobotIO,
        cancel: CancelToken | None = None,
        ik: TopDownIK | None = None,
        snapshot_fn: Callable[[], CameraSnapshot] | None = None,
        scene_fn: Callable[[], Scene] | None = None,
        lock: RobotBusLock | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.cfg = cfg
        self.calib = calib
        self.poses = poses
        self.cancel = cancel or CancelToken()
        self._inner_robot = robot.inner if isinstance(robot, CancellableRobotIO) else robot
        self.robot: BaseRobotIO = (
            robot if isinstance(robot, CancellableRobotIO) else CancellableRobotIO(robot, self.cancel)
        )
        self.player = TrajectoryPlayer(self.robot, cfg.motion)
        self.motion = MotionController(self.robot, poses, cfg.motion, cfg.sensing)
        self.grasp_z_mm = calibration_grasp_z_mm(calib)
        self.drop_z_mm = self.grasp_z_mm + cfg.task1.release_clearance_mm
        base = calib.base_xy_mm or (0.0, 0.0)
        self.base_xy: XY = (float(base[0]), float(base[1]))
        self._ik = ik
        self._transport = None
        self._stack = None
        self._pick_state = None
        self._snapshot_fn = snapshot_fn or (
            lambda: fetch_snapshot_with_metadata(self.cfg.perception.snapshot_url)
        )
        self._scene_fn = scene_fn
        self._lock = lock
        self._clock = clock
        self._closed = False
        self.held: HeldBlock | None = None
        self.last_pick: LastPick | None = None
        self.last_block_color: str | None = None
        self.last_scene: Scene | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    @classmethod
    def open(
        cls,
        cfg: AppConfig,
        *,
        robot: BaseRobotIO | None = None,
        overrides: Sequence[str] = (),
        acquire_bus_lock: bool = True,
        prebuild_ik: bool = False,
        **kwargs,
    ) -> "ArmSession":
        cfg = copy.deepcopy(cfg)
        for override in overrides:
            apply_override(cfg, override)
        calib_path = Path(cfg.perception.calibration_path)
        if not calib_path.exists():
            raise FileNotFoundError(
                f"Venue calibration not found: {calib_path}. Run tools/calibrate_homography.py first."
            )
        calib = PlaneCalibration.load(calib_path)
        if not calib.zone_polygon_mm:
            raise ValueError("The agent requires zone_polygon_mm; run so101-zone-calibrate --write")
        poses = PoseRegistry.load(cfg.motion.poses_path)
        poses.require([cfg.motion.home_pose])

        lock = RobotBusLock(cfg.agent.lock_path) if acquire_bus_lock else None
        if lock is not None:
            lock.acquire()
        inner = robot
        try:
            if inner is None:
                from control.robot_io import So101RobotIO

                inner = So101RobotIO(cfg.robot)
            inner.connect()
            session = cls(cfg, calib=calib, poses=poses, robot=inner, lock=lock, **kwargs)
            if prebuild_ik:
                session.transport  # noqa: B018 - builds IK seeds and solves every slot
            return session
        except BaseException:
            if inner is not None:
                try:
                    inner.disconnect()
                except Exception:  # noqa: BLE001 - best effort
                    pass
            if lock is not None:
                lock.release()
            raise

    def close(self) -> None:
        """The single safe shutdown. Never stopped by the STOP flag it clears.

        Unlike the rule-based runners this does not open the gripper: a
        block still held at shutdown is carried home rather than dropped.
        """
        if self._closed:
            return
        self._closed = True
        self.cancel.clear()
        try:
            self.return_home_safely()
        except Exception as exc:  # noqa: BLE001 - best effort
            logger.warning("Safe-shutdown motion failed: %s", exc)
        finally:
            try:
                self._inner_robot.disconnect()
            finally:
                if self._lock is not None:
                    self._lock.release()

    def __enter__(self) -> "ArmSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── lazily built collaborators ───────────────────────────────────

    @property
    def ik(self) -> TopDownIK:
        if self._ik is None:
            self._ik = TopDownIK(self.cfg.ik, project_root=".")
        return self._ik

    @property
    def transport(self):
        if self._transport is None:
            from control.task1_transport import Task1TransportPlanner

            self._transport = Task1TransportPlanner(self.calib, self.cfg, self.ik)
        return self._transport

    @property
    def stack(self):
        if self._stack is None:
            from control.task2_stack import Task2StackPlanner

            self._stack = Task2StackPlanner(self.calib, self.cfg, self.ik)
        return self._stack

    @property
    def pick_state(self):
        if self._pick_state is None:
            from fsm.ik_handler import CvIkPickState

            self._pick_state = CvIkPickState(
                robot=self.robot,
                motion=self.motion,
                cfg=self.cfg,
                grasp_z_mm=self.grasp_z_mm,
                retreat_pose=None,
                retreat_after_grasp=True,
                radial_tilt_extra_key=PICK_TILT_KEY,
                ik=self.ik,
                player=self.player,
            )
        return self._pick_state

    # ── geometry ─────────────────────────────────────────────────────

    @property
    def slot_centres(self) -> list[XY]:
        """Zone slot centres exactly as Task1TransportPlanner places them."""
        raw = zone_slot_centres(self.calib, self.cfg.task1.slot_uv)
        return [
            tuple(push_out_from_base(xy, self.base_xy, offset))
            for xy, offset in zip(raw, self.cfg.task1.slot_radial_offset_mm, strict=True)
        ]

    def in_zone(self, xy: XY, margin_mm: float = 0.0) -> bool:
        return point_in_zone(xy, self.calib, margin_mm)

    def in_workspace(self, xy: XY) -> bool:
        return point_in_workspace(xy, self.cfg.perception, self.base_xy)

    def solve_place(self, xy: XY, *, label: str = "placement") -> Task1SlotPlan:
        """Raises ValueError when the drop or hover pose misses the IK gate."""
        return solve_place_point(
            self.ik, self.cfg, xy, self.drop_z_mm, base_xy_mm=self.base_xy, label=label
        )

    def arm_position_mm(self) -> tuple[float, float, float]:
        return self.ik.forward_position_mm(self.robot.read_joints())

    def arm_at_home(self) -> bool:
        home = self.poses.get(self.cfg.motion.home_pose)
        joints = self.robot.read_joints()
        return all(
            abs(joints[name] - value) <= self.cfg.motion.transit_arrival_tol
            for name, value in home.items()
            if name != "gripper" and name in joints
        )

    # ── perception ───────────────────────────────────────────────────

    def observe(self, *, after: float | None = None) -> Scene:
        """A fresh scene. ``after``: wall time the frame must be captured after.

        Callers pass the moment the arm stopped moving, so a frame showing the
        arm still in the camera's view is never used.
        """
        self.cancel.raise_if_set()
        if self._scene_fn is not None:
            scene = self._scene_fn()
            self.last_scene = scene
            return scene
        deadline = time.monotonic() + self.cfg.agent.camera_fresh_timeout_s
        while True:
            self.cancel.raise_if_set()
            try:
                snapshot = self._snapshot_fn()
            except (OSError, RuntimeError, ValueError) as exc:
                raise CameraError(f"camera snapshot failed: {exc}") from exc
            age = self._clock() - snapshot.captured_at
            fresh_enough = age <= self.cfg.task1.max_frame_age_s
            late_enough = after is None or snapshot.captured_at >= after
            if fresh_enough and late_enough:
                break
            if time.monotonic() > deadline:
                raise CameraError(
                    f"no fresh camera frame (age {age:.1f}s); is so101-camera running?",
                    stale=True,
                )
            time.sleep(self.cfg.task1.scan_interval_s)
        scene = detect_scene(
            snapshot.frame,
            self.calib,
            self.cfg.perception,
            self.slot_centres,
            zone_max_per_color=self.cfg.agent.zone_scan_max_per_color,
            snap_radius_mm=self.cfg.agent.slot_snap_radius_mm,
            is_rgb=False,
            frame_seq=snapshot.frame_seq,
            captured_at=snapshot.captured_at,
        )
        self.last_scene = scene
        return scene

    def task1_perceive(self):
        """The ``perceive`` the Task 1/2 FSM polls: real camera, or the injected scene."""
        if self._scene_fn is None:
            from session.factories import make_task1_perceive

            return make_task1_perceive(self.calib, self.cfg)
        from fsm.task1 import Task1Perception

        counter = {"seq": 0}

        def perceive() -> Task1Perception:
            counter["seq"] += 1
            scene = self._scene_fn()
            return Task1Perception(
                [block.detection for block in scene.outside.values()], counter["seq"], self._clock()
            )

        return perceive

    def home_and_observe(self) -> Scene:
        """Clear the camera's view first, as Task1SelectState.enter does."""
        self.go_home()
        return self.observe(after=self._clock())

    # ── motion ───────────────────────────────────────────────────────

    def go_home(self) -> None:
        # CV+IK callers never home the gripper (see MotionController.go_home).
        self.motion.go_home(include_gripper=False)

    def lift_in_place(self) -> bool:
        """Rise vertically to hover height at the current xy, if below it.

        Solves for the measured wrist_roll so the jaws (and any held block)
        do not turn while still close to other blocks. Returns True when it moved.
        """
        joints = self.robot.read_joints()
        x, y, z = self.ik.forward_position_mm(joints)
        safe_z = self.grasp_z_mm + self.cfg.motion.hover_min_clearance_mm
        if z >= safe_z - 1.0:
            return False
        yaw = self.ik.yaw_for_wrist_roll_deg(x, y, z, joints["wrist_roll"])
        target_z = highest_reachable_hover(self.ik, x, y, self.grasp_z_mm, self.cfg, yaw_deg=yaw)
        result = self.ik.solve_holding_wrist_roll(x, y, target_z, joints["wrist_roll"])
        if over_ik_gate(result, self.cfg):
            logger.warning("vertical lift at x=%.0f y=%.0f misses the IK gate; homing directly", x, y)
            return False
        self.player.move_to(result.joints, max_step=1.0, tol=self.cfg.motion.transit_arrival_tol)
        return True

    def return_home_safely(self) -> tuple[bool, bool]:
        """Lift clear of the blocks if low, then home. Returns (lifted, at_home).

        Uses the measured pose, never a remembered one: after a STOP the arm
        may be anywhere between two waypoints.
        """
        lifted = False
        if self.arm_at_home():
            # home's tool frame sits near table height by design (measured FK
            # z ~8mm at x ~157mm); "lifting" there would only unfold the arm
            return False, True
        try:
            lifted = self.lift_in_place()
        except (Cancelled, TimeoutError):
            raise
        except Exception as exc:  # noqa: BLE001 - FK/IK trouble must not block homing
            logger.warning("vertical lift before homing skipped: %s", exc)
        self.go_home()
        return lifted, self.arm_at_home()

    def carry_and_release(self, slot: Task1SlotPlan) -> None:
        """Fly the held block to ``slot`` and release it (Task 1's own motions)."""
        if self.held is None:
            raise RuntimeError("carry_and_release without a held block")
        origin = replace(
            self.held.attempt, xy_mm=self.held.over_xy_mm, grasp_z_mm=self.grasp_z_mm
        )
        plan = Task1TransportPlan(slot=slot, carry=carry_waypoints(self.ik, self.cfg, origin, slot))
        fly_carry(self.player, self.cfg, plan)
        release_at(self.player, self.motion, self.cfg, slot)
        self.last_block_color = self.held.color
        self.held = None

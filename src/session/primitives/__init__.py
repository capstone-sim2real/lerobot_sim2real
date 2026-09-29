"""Bounded, guarded arm actions for the web agent.

All motion goes through the session's cancellable IO. FK is model-derived
feedback, not a camera measurement.
"""
from __future__ import annotations

from session.skills import Skills

from session.primitives.calibrated import CalibratedPickMixin
from session.primitives.episode import EpisodeMixin
from session.primitives.observe import ObserveMixin
from session.primitives.slot_mission import SlotMissionMixin
from session.primitives.stack import StackMixin
from session.primitives.motion import MotionMixin
from session.primitives.gripper import GripperMixin


class PrimitiveSkills(
    CalibratedPickMixin,
    EpisodeMixin,
    ObserveMixin,
    SlotMissionMixin,
    StackMixin,
    MotionMixin,
    GripperMixin,
    Skills,
):
    def __init__(self, session, *, collection_factory=None):
        super().__init__(session)
        self.observation_id = 0
        self._observed_at = 0.0
        self._objects = {}
        self._target = None
        self._contact = False
        self._stack_drop_ready = False
        self._zone_drop_ready = False
        self._zone_drop_xy = None
        self._zone_retreat_joints = None
        self._zone_home_pending = False
        self._grasp_failed = False
        self._pick_calibration = None
        self._pick_ready = False
        self._held_radial_tilt_deg = 0.0
        self._held_block_angle_deg = None
        self._held_pick_yaw_deg = None
        self._place_yaw_deg = None
        self._place_yaw_explicit = False
        self._place_radial_tilt_deg = None
        self._place_command_xy = None
        self._observed_scene = None
        self._pending_placement = None
        self._task2_placed_floors = {}
        self._mission_slot_ledger = None
        self._stack_target = None
        self._transfer_source = None
        self._tower_histories = {}
        self._recovery_target_xy = None
        from session.collection import Collection
        from control.trajectory import TrajectoryPlayer
        from control.motion import MotionController
        options = {} if collection_factory is None else {"resource_factory": collection_factory}
        self.collection = Collection(session, **options)
        session.robot = self.collection.io
        session.player = TrajectoryPlayer(session.robot, self.cfg.motion)
        session.motion = MotionController(session.robot, session.poses, self.cfg.motion, self.cfg.sensing)

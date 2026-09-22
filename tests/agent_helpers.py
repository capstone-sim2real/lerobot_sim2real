"""Hardware-free fixtures for the session/agent tests.

``FakeIk`` encodes a Cartesian target directly in the joints
(shoulder_pan=x, shoulder_lift=y, elbow_flex=z, wrist_roll=yaw), so forward
kinematics is exact and ``SimRobotIO`` can tell where the jaws are without
placo. Everything else -- grasp planning, TrajectoryPlayer, CvIkPickState,
check_grasp, the Task 1 FSM -- is the production code.
"""

from __future__ import annotations

import math

import numpy as np

from agent.sim import SimRobotIO, SimWorld
from config import AppConfig
from control.ik import IkResult
from control.poses import PoseRegistry
from perception.homography import PlaneCalibration
from session.arm_session import ArmSession
from session.skills import Skills

HOME = {"shoulder_pan": 150.0, "shoulder_lift": 0.0, "elbow_flex": 8.0,
        "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 1.0}


class FakeIk:
    def __init__(self, reach_mm: float = 330.0):
        self.reach_mm = reach_mm
        self.solves = 0

    def solve(self, x_mm, y_mm, z_mm, yaw_deg=None, radial_tilt_deg=0.0):
        self.solves += 1
        error = 0.2 if math.hypot(x_mm, y_mm) <= self.reach_mm else 80.0
        joints = {"shoulder_pan": float(x_mm), "shoulder_lift": float(y_mm), "elbow_flex": float(z_mm),
                  "wrist_flex": 0.0, "wrist_roll": float(yaw_deg or 0.0)}
        return IkResult(joints, error, 0.1)

    def neutral_yaw_deg(self, x_mm, y_mm, z_mm):
        return 0.0

    def yaw_for_wrist_roll_deg(self, x_mm, y_mm, z_mm, wrist_roll):
        return wrist_roll

    def solve_holding_wrist_roll(self, x_mm, y_mm, z_mm, wrist_roll_deg, *, radial_tilt_deg=0.0, **_kwargs):
        return self.solve(x_mm, y_mm, z_mm, yaw_deg=wrist_roll_deg, radial_tilt_deg=radial_tilt_deg)

    def grasp_yaw_and_rotation_deg(self, x_mm, y_mm, z_mm, block_angle_deg):
        return 0.0, 0.0

    def forward_position_mm(self, joints):
        return float(joints["shoulder_pan"]), float(joints["shoulder_lift"]), float(joints["elbow_flex"])


def calibration() -> PlaneCalibration:
    """A top-down view of the table, 1mm per pixel.

    The camera sits behind the robot looking down, so image right is -y and
    image up is +x, exactly like the real rig. It is not the identity: code
    that asks whether a point is *visible* needs a frame the workspace
    actually maps into, and an identity H puts the whole right half of the
    table at negative pixels.
    """
    width, height = 700, 500
    H = np.array([[0.0, -1.0, float(height)], [-1.0, 0.0, width / 2.0], [0.0, 0.0, 1.0]])
    return PlaneCalibration(
        H=H,
        image_size=(width, height),
        square_mm=1.0,
        base_xy_mm=(0.0, 0.0),
        zone_polygon_mm=[(300.0, 100.0), (300.0, -100.0), (200.0, -100.0), (200.0, 100.0)],
        meta={"grasp_z_mm_mean": 10.0, "rms_mm": 5.0},
    )


def fast_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.agent.primitives.calibrated_pick = False  # Synthetic IK tests opt out of rig calibration.
    cfg.motion.fps = 0.0
    cfg.motion.gripper_action_wait_s = 0.0
    cfg.motion.place_settle_s = 0.0
    cfg.motion.grasp_hover_settle_s = 0.0
    cfg.motion.descent_settle_s = 0.0
    cfg.motion.max_step_per_tick = 50.0
    cfg.motion.descent_step_per_tick = 50.0
    cfg.sensing.grasp_settle_s = 0.0
    cfg.sensing.sample_interval_s = 0.0
    cfg.sensing.grasp_samples = 1
    cfg.task1.scan_interval_s = 0.0
    cfg.logging.save_transitions = False
    cfg.agent.transcript_dir = ""
    return cfg


def make_skills(blocks: dict[str, tuple[float, float]], *, cfg: AppConfig | None = None,
                ik: FakeIk | None = None, grab_radius_mm: float = 25.0):
    cfg = cfg or fast_cfg()
    calib = calibration()
    ik = ik or FakeIk()
    world = SimWorld(blocks)
    robot = SimRobotIO(world, ik.forward_position_mm, grasp_z_mm=10.0,
                       initial_joints=HOME, grab_radius_mm=grab_radius_mm)
    holder = {}

    def scene():
        session = holder["session"]
        return world.scene(session.calib, session.slot_centres, cfg.agent.slot_snap_radius_mm)

    session = ArmSession(cfg, calib=calib, poses=PoseRegistry({"home": dict(HOME)}), robot=robot,
                         ik=ik, scene_fn=scene)
    holder["session"] = session
    return Skills(session), world, robot

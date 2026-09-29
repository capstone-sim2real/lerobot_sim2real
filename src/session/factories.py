"""Factories shared by ``so101-run``, ``so101-collect`` and ``so101-agent``."""

from __future__ import annotations

from camera.client import fetch_snapshot_with_metadata
from config import AppConfig
from control import MotionController
from control.ik import TopDownIK
from control.robot_io import BaseRobotIO
from fsm.ik_handler import CvIkPickState
from fsm.task1 import Task1Perception
from perception import PlaneCalibration, detect_blocks


def calibration_grasp_z_mm(calib: PlaneCalibration) -> float:
    """The block-top grasp plane recorded by base-frame calibration."""
    try:
        return float(calib.meta["grasp_z_mm_mean"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Calibration metadata is missing grasp_z_mm_mean; rerun base-frame calibration "
            "with the gripper on the block top plane."
        ) from exc


def make_task1_perceive(calib: PlaneCalibration, cfg: AppConfig):
    def perceive() -> Task1Perception:
        snapshot = fetch_snapshot_with_metadata(cfg.perception.snapshot_url)
        detections = detect_blocks(snapshot.frame, calib, cfg.perception, is_rgb=False)
        return Task1Perception(detections, snapshot.frame_seq, snapshot.captured_at)

    return perceive


def make_pick_state(
    *,
    robot: BaseRobotIO,
    motion: MotionController,
    cfg: AppConfig,
    calib: PlaneCalibration,
    retreat_after_grasp: bool = True,
    radial_tilt_extra_key: str | None = None,
    max_grasp_attempts: int | None = None,
    ik: TopDownIK | None = None,
) -> CvIkPickState:
    return CvIkPickState(
        robot=robot,
        motion=motion,
        cfg=cfg,
        grasp_z_mm=calibration_grasp_z_mm(calib),
        retreat_after_grasp=retreat_after_grasp,
        radial_tilt_extra_key=radial_tilt_extra_key,
        max_grasp_attempts=max_grasp_attempts,
        ik=ik,
    )

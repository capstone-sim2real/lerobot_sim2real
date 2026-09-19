"""Factories shared by ``so101-run``, ``so101-collect`` and ``so101-agent``.

Moved out of ``runners.run_task`` so ``runners.run_task3`` no longer imports
a sibling runner (which itself lazily imports ``run_task3``). The old names
stay importable from ``runners.run_task``; ``tests/test_runner_contracts.py``
pins them to these objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from camera.client import fetch_snapshot_with_metadata
from config import AppConfig
from control import MotionController
from control.ik import TopDownIK
from control.robot_io import BaseRobotIO
from fsm.act_handler import ActPickState
from fsm.ik_handler import CvIkPickState
from fsm.task1 import Task1Perception
from perception import PlaneCalibration, detect_blocks

if TYPE_CHECKING:
    from policy import ActPolicyClient


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
    pick_mode: str,
    *,
    robot: BaseRobotIO,
    motion: MotionController,
    cfg: AppConfig,
    calib: PlaneCalibration,
    retreat_pose,
    retreat_after_grasp: bool = True,
    radial_tilt_extra_key: str | None = None,
    max_grasp_attempts: int | None = None,
    client: "ActPolicyClient | None" = None,
    ik: TopDownIK | None = None,
):
    """Build the PICK implementation without coupling common FSM states to ACT."""
    if pick_mode == "cv_ik":
        grasp_z_mm = calibration_grasp_z_mm(calib)
        return CvIkPickState(
            robot=robot,
            motion=motion,
            cfg=cfg,
            grasp_z_mm=grasp_z_mm,
            retreat_pose=retreat_pose,
            retreat_after_grasp=retreat_after_grasp,
            radial_tilt_extra_key=radial_tilt_extra_key,
            max_grasp_attempts=max_grasp_attempts,
            ik=ik,
        )
    if pick_mode == "act":
        if client is None:
            raise ValueError("ACT pick mode requires a connected policy client")
        return ActPickState(client, motion, retreat_pose)
    raise ValueError(f"Unknown pick mode: {pick_mode!r}")

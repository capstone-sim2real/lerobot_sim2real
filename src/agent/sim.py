"""Offline stand-ins for the arm and the camera (``so101-agent --sim``).

Motion is the real code path: real TopDownIK, TrajectoryPlayer, grasp
planning and FSM. Only the bus and the photo are faked: ``SimRobotIO``
teleports joints like ``MockRobotIO`` and grasps a simulated block when the
jaws close near it (so ``check_grasp`` sees a realistic held reading), and
``SimWorld.scene`` reports the blocks where the simulated releases left them.
For UI and tool-flow rehearsal only -- it says nothing about real accuracy.
"""

from __future__ import annotations

import math
import threading
from typing import Callable

from config import AppConfig
from control.robot_io import MockRobotIO
from perception.detector import BlockDetection
from perception.homography import PlaneCalibration
from perception.scene import Scene, build_scene
from perception.zone import point_in_zone

HELD_GRIPPER_POS = 44.0
HELD_LOAD = 500
EMPTY_LOAD = 40


class SimWorld:
    def __init__(self, blocks: dict[str, tuple[float, float]]):
        self._lock = threading.Lock()
        self.blocks = dict(blocks)
        self.held: str | None = None

    @classmethod
    def default(cls, cfg: AppConfig, calib: PlaneCalibration) -> "SimWorld":
        del cfg, calib
        return cls(
            {
                "yellow": (215.0, 150.0),
                "green": (180.0, -140.0),
                "blue": (140.0, 95.0),
                "red": (250.0, -175.0),
                "wood": (160.0, 20.0),
            }
        )

    def scene(self, calib: PlaneCalibration, slot_xy, snap_radius_mm: float) -> Scene:
        with self._lock:
            detections = [
                BlockDetection(color, xy, 1600.0, 1.0, 0.95, 0.9, [], angle_deg=0.0)
                for color, xy in self.blocks.items()
                if color != self.held
            ]
        outside = [d for d in detections if not point_in_zone(d.center_mm, calib)]
        inside = [d for d in detections if point_in_zone(d.center_mm, calib)]
        return build_scene(outside, inside, calib, slot_xy, snap_radius_mm=snap_radius_mm)


class SimRobotIO(MockRobotIO):
    def __init__(
        self,
        world: SimWorld,
        fk: Callable[[dict[str, float]], tuple[float, float, float]],
        *,
        grasp_z_mm: float,
        initial_joints: dict[str, float] | None = None,
        grab_radius_mm: float = 25.0,
        grab_height_mm: float = 30.0,
    ):
        super().__init__(initial_joints)
        self._world = world
        self._fk = fk
        self._grasp_z = grasp_z_mm
        self._grab_radius = grab_radius_mm
        self._grab_height = grab_height_mm
        self.loads["gripper"] = EMPTY_LOAD

    def send_joints(self, positions: dict[str, float]) -> dict[str, float]:
        positions = dict(positions)
        if "gripper" in positions:
            target = positions["gripper"]
            closing = target < self.joints["gripper"]
            if self._world.held is None and closing and target < HELD_GRIPPER_POS:
                x, y, z = self._fk(self.joints)
                if z <= self._grasp_z + self._grab_height:
                    with self._world._lock:
                        near = min(
                            self._world.blocks.items(),
                            key=lambda item: math.dist(item[1], (x, y)),
                            default=None,
                        )
                        if near is not None and math.dist(near[1], (x, y)) <= self._grab_radius:
                            self._world.held = near[0]
            if self._world.held is not None:
                if target < HELD_GRIPPER_POS:
                    positions["gripper"] = HELD_GRIPPER_POS
                    self.loads["gripper"] = HELD_LOAD
                elif target > HELD_GRIPPER_POS + 10:
                    x, y, _z = self._fk(self.joints)
                    with self._world._lock:
                        self._world.blocks[self._world.held] = (x, y)
                        self._world.held = None
                    self.loads["gripper"] = EMPTY_LOAD
            else:
                self.loads["gripper"] = EMPTY_LOAD
        return super().send_joints(positions)

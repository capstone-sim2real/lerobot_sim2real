"""Cartesian top-down IK for the CV+IK pick path (AGENTS.md §7).

Placo's IK is seed-sensitive: seeded from the current/actual pose it can
converge 200-350mm off target for a lateral move, because copying the
current orientation demands a pose the 5-DOF arm cannot reach. Seeded from
a pre-computed top-down configuration instead, the same solver converges to
millimeter-level error (in the URDF model — AGENTS.md §6 documents the
larger, position-dependent error the real arm's FK carries beyond that).

``lerobot``/``placo`` are imported lazily inside methods so that importing
this module (and therefore ``pick_stack``) never requires them (AGENTS.md
§2/§14) — only constructing a ``TopDownIK`` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pick_stack.config import IkConfig

ARM_JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


@dataclass
class IkResult:
    joints: dict[str, float]
    position_error_mm: float
    tilt_error_deg: float

    @property
    def ok(self) -> bool:
        return True  # gate applied by the caller against cfg thresholds


def _topdown_pose(x_mm: float, y_mm: float, z_mm: float, yaw_deg: float) -> np.ndarray:
    """4x4 pose: approach axis straight down, jaw plane rotated by yaw_deg."""
    T = np.eye(4)
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    T[:3, 0] = [-s, c, 0.0]
    T[:3, 1] = [c, s, 0.0]
    T[:3, 2] = [0.0, 0.0, -1.0]
    T[:3, 3] = [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0]
    return T


def _tilt_deg(T: np.ndarray) -> float:
    """Angle between the gripper's approach axis and straight down."""
    return float(np.degrees(np.arccos(np.clip(-T[2, 2], -1.0, 1.0))))


class TopDownIK:
    """Solves (x_mm, y_mm, z_mm, yaw_deg) -> arm joint angles (degrees).

    ``yaw_deg`` is the desired jaw-plane rotation in the robot base frame
    (e.g. the detected block angle, folded mod 90 for a square block —
    AGENTS.md §9). Excludes the gripper joint; callers set that separately.
    """

    def __init__(self, cfg: IkConfig, project_root: Path | str = "."):
        self._cfg = cfg
        self._project_root = Path(project_root)
        self._kinematics = None
        self._seed_table: np.ndarray | None = None  # columns: r_mm, z_mm, lift, elbow, wf

    def _load_kinematics(self):
        if self._kinematics is not None:
            return self._kinematics
        from lerobot.model.kinematics import RobotKinematics

        urdf_path = (self._project_root / self._cfg.urdf_path).resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        old_cwd = Path.cwd()
        os_chdir_target = urdf_path.parent
        import os

        os.chdir(os_chdir_target)
        try:
            self._kinematics = RobotKinematics(
                str(urdf_path), target_frame_name=self._cfg.target_frame, joint_names=list(ARM_JOINTS)
            )
        finally:
            os.chdir(old_cwd)
        return self._kinematics

    def _build_seed_table(self) -> np.ndarray:
        k = self._load_kinematics()
        step = self._cfg.seed_step_deg
        rng = self._cfg.seed_range_deg
        values = np.arange(-rng, rng + step, step)
        rows: list[tuple[float, float, float, float, float]] = []
        q = np.zeros(5)
        for lift in values:
            q[1] = lift
            for elbow in values:
                q[2] = elbow
                for wf in values:
                    q[3] = wf
                    T = k.forward_kinematics(q)
                    if _tilt_deg(T) <= self._cfg.seed_tilt_max_deg:
                        r_mm = float(np.hypot(T[0, 3], T[1, 3]) * 1000.0)
                        z_mm = float(T[2, 3] * 1000.0)
                        rows.append((r_mm, z_mm, float(lift), float(elbow), float(wf)))
        if not rows:
            raise RuntimeError(
                f"No top-down seed configurations found within {self._cfg.seed_tilt_max_deg} deg tilt "
                f"— widen seed_tilt_max_deg or seed_range_deg."
            )
        return np.array(rows, dtype=np.float64)

    def _seeds(self) -> np.ndarray:
        if self._seed_table is not None:
            return self._seed_table
        cache_path = (self._project_root / self._cfg.seed_cache_path).resolve()
        urdf_path = (self._project_root / self._cfg.urdf_path).resolve()
        if cache_path.is_file():
            cached = np.load(cache_path)
            same_params = (
                cached["step"] == self._cfg.seed_step_deg
                and cached["range"] == self._cfg.seed_range_deg
                and cached["tilt_max"] == self._cfg.seed_tilt_max_deg
                and cached["urdf_mtime"] == urdf_path.stat().st_mtime
            )
            if same_params:
                self._seed_table = cached["table"]
                return self._seed_table
        table = self._build_seed_table()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            table=table,
            step=self._cfg.seed_step_deg,
            range=self._cfg.seed_range_deg,
            tilt_max=self._cfg.seed_tilt_max_deg,
            urdf_mtime=urdf_path.stat().st_mtime,
        )
        self._seed_table = table
        return table

    def _nearest_seeds(self, r_mm: float, z_mm: float, n: int = 3) -> np.ndarray:
        table = self._seeds()
        d2 = (table[:, 0] - r_mm) ** 2 + (table[:, 1] - z_mm) ** 2
        idx = np.argsort(d2)[:n]
        return table[idx]

    def solve(self, x_mm: float, y_mm: float, z_mm: float, yaw_deg: float = 0.0) -> IkResult:
        """Best-effort top-down IK solve. Check ``position_error_mm`` /
        ``tilt_error_deg`` against config thresholds before trusting the
        result (AGENTS.md §6/§7 — this can fail gracefully out-of-reach)."""
        k = self._load_kinematics()
        target = _topdown_pose(x_mm, y_mm, z_mm, yaw_deg)
        r_mm = float(np.hypot(x_mm, y_mm))
        pan0 = -float(np.degrees(np.arctan2(y_mm, x_mm)))
        seeds = self._nearest_seeds(r_mm, z_mm)

        best_q: np.ndarray | None = None
        best_pos_err = float("inf")
        best_tilt_err = float("inf")
        for lift, elbow, wf in seeds[:, 2:5]:
            for dpan in self._cfg.pan_offset_candidates_deg:
                q = np.array([pan0 + dpan, lift, elbow, wf, 0.0])
                for _ in range(self._cfg.ik_iters):
                    q = k.inverse_kinematics(q, target, orientation_weight=1.0)
                achieved = k.forward_kinematics(q)
                pos_err = float(np.linalg.norm(achieved[:3, 3] * 1000.0 - [x_mm, y_mm, z_mm]))
                tilt_err = _tilt_deg(achieved)
                if pos_err < best_pos_err:
                    best_q, best_pos_err, best_tilt_err = q, pos_err, tilt_err
                    if pos_err < 1.0 and tilt_err < 1.0:
                        break
            if best_pos_err < 1.0 and best_tilt_err < 1.0:
                break

        assert best_q is not None
        joints = {name: float(val) for name, val in zip(ARM_JOINTS, best_q)}
        return IkResult(joints=joints, position_error_mm=best_pos_err, tilt_error_deg=best_tilt_err)

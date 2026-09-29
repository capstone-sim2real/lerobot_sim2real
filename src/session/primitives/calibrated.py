"""Rig-calibrated pick targets and step-wise hover correction."""
from __future__ import annotations

from dataclasses import replace



class CalibratedPickMixin:
    def _lateral_clearance_ready(self, z_mm: float) -> bool:
        required_z = self.s.grasp_z_mm + self.limits.lateral_clearance_mm
        return z_mm + self.limits.lateral_clearance_tolerance_mm >= required_z

    def _calibration(self):
        if self._pick_calibration is None:
            from pathlib import Path
            from session.calibration_motion import CalibrationMotion
            # This helper shares the existing session/IO; it never opens a bus
            # or installs another recording wrapper.
            output = Path(self.cfg.agent.collection.root) / "calibration-motion"
            self._pick_calibration = CalibrationMotion(self.s, output)
            self._pick_calibration.primitive_wrist_limit = self.limits.wrist_roll_limit_deg
        return self._pick_calibration

    def _invalidate_pick(self):
        self._pick_ready = False
        if self._pick_calibration is not None:
            self._pick_calibration.attempt = None
            self._pick_calibration.descent_ready = False

    def _calibrated_target(self, block, phase, route_guard=None):
        cal = self._calibration()
        self._pick_ready = False
        if phase == "pregrasp":
            result = cal.calibration_prepare(block.color, _scene=self._observed_scene,
                                             _open_gripper=False, route_guard=route_guard)
            if not result.ok and result.data.get("stop_reason") == "hover_not_settled":
                result = cal.calibration_correct_hover(dry_run=False)
        else:
            result = cal.calibration_descend_guarded()
            self._pick_ready = result.ok
        result = replace(result, action="move_to_target")
        result.data["calibrated_pick"] = True
        if cal.baseline is not None:
            result.data["calibrated_xy_mm"] = list(cal.baseline.xy_mm)
        return result

    def inspect_motion(self):
        """Read feedback without moving or invalidating an approach."""
        q = self.s.robot.read_joints()
        cal = self._pick_calibration
        a = (cal.attempt or cal.baseline) if cal is not None else None
        nominal = dict(a.hover.joints) if a is not None else None
        return self._result(True, "inspect_motion", "ok", measured_joints=q,
                            measured_fk_mm=list(self.s.ik.forward_position_mm(q)),
                            loads=self.s.robot.read_loads(), nominal_hover_joints=nominal,
                            hover_error_deg={j: v-q[j] for j,v in nominal.items()} if nominal else None,
                            descent_ready=self._pick_ready,
                            reference_valid=bool(cal is not None and cal.attempt is not None),
                            position_source="joint_feedback_and_URDF_not_visual_TCP")

    def correct_hover(self, joint="all", gain=1.0, dry_run=True):
        action = "correct_hover"
        cal = self._pick_calibration
        if (not self.limits.calibrated_pick or cal is None or self.s.held is not None
                or not self._target or self._target[-1] != "pregrasp" or cal.attempt is None):
            return self._fail(action, "A valid calibrated pregrasp is required")
        self._object(self._target[1], self._target[2])
        self._pick_ready = False
        attempt = cal.attempt
        old_baseline = cal.baseline
        cal.baseline = attempt  # retain any bounded XY adjustment
        cal.attempt = None
        try:
            result = cal.calibration_correct_hover(dry_run=dry_run, joint=joint, gain=gain)
        finally:
            cal.baseline = old_baseline
            if dry_run:
                cal.attempt = attempt
        return replace(result, action=action)

    def descend_step(self, down_mm):
        action = "descend_step"
        cal = self._pick_calibration
        if (not self.limits.calibrated_pick or cal is None or self.s.held is not None
                or not self._target or self._target[-1] != "pregrasp" or cal.attempt is None):
            return self._fail(action, "A valid calibrated pregrasp is required")
        self._object(self._target[1], self._target[2])
        self._pick_ready = False
        result = cal.calibration_descend_step(down_mm)
        self._pick_ready = result.ok and cal.descent_ready
        if self._pick_ready:
            self._target = (*self._target[:-1], "grasp")
        return replace(result, action=action)

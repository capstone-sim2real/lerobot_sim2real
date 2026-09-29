"""Task 1: move a block into a named zone slot, with recovery."""
from __future__ import annotations

import math
import time



class SlotMissionMixin:
    def _slot_occupant(self, index: int):
        if self._mission_slot_ledger is not None:
            return self._mission_slot_ledger[index]
        scene = self._observed_scene
        return scene.slot_occupancy.get(index) if scene is not None else None

    def _free_slot_labels(self):
        labels = list(self.cfg.agent.zone_slots.labels)
        return [label for index, label in enumerate(labels)
                if self._slot_occupant(index) is None]

    def _record_mission_release(self, slot: str, color: str):
        if self._mission_slot_ledger is not None:
            index = list(self.cfg.agent.zone_slots.labels).index(slot)
            self._mission_slot_ledger[index] = color

    def _release_held_at_slot_from_recovery(self, color: str, slot: str):
        """Use a verified, already reached slot pose after a short loaded lift."""
        if self.s.held is None or self.s.held.color != color:
            return None
        labels = list(self.cfg.agent.zone_slots.labels)
        index = labels.index(slot)
        xyz = self.s.arm_position_mm()
        target_xy = self.s.slot_centres[index]
        if (not self.s.in_zone(xyz[:2])
                or math.dist(xyz[:2], target_xy) > self.limits.alignment_tolerance_mm
                or not self.s.grasp_z_mm + self.limits.zone_release_floor_margin_mm
                <= xyz[2] <= self.s.drop_z_mm + 10):
            return None
        observed = self.observe_scene()
        if not observed.ok or not self._held_check():
            return None
        scene = self._observed_scene
        if scene is None or self._slot_occupant(index) not in (None, color):
            return None
        reason, _ = self.placement_verdict(xyz[:2], allow_zone=True,
                                           ignore_color=color, check_ik=False)
        if reason is not None:
            return None
        self._target = ("slot", None, None, slot, None, None, "preplace")
        self._zone_drop_xy = tuple(xyz[:2])
        self._zone_drop_ready = True
        self._place_yaw_deg = None
        self._place_yaw_explicit = False
        released = self.open_gripper()
        if not released.ok:
            return released
        self._record_mission_release(slot, color)
        home = self.return_to_home()
        if not home.ok:
            return home
        return self._result(True, "move_block_to_slot", "released",
                            color=color, slot=slot, recovery="release_at_reached_slot",
                            placement_verified=False, slot_source="commanded")

    def _place_held_in_other_slot(self, color: str, requested_slot: str):
        """Keep a verified grasp when only the requested slot fails IK."""
        if self.s.held is None or self.s.held.color != color or not self._held_check():
            return None
        scene = self._observed_scene
        if scene is None:
            return None
        labels = list(self.cfg.agent.zone_slots.labels)
        requested_xy = self.s.slot_centres[labels.index(requested_slot)]
        alternatives = sorted(
            (label for index, label in enumerate(labels)
             if label != requested_slot and self._slot_occupant(index) is None),
            key=lambda label: math.dist(
                self.s.slot_centres[labels.index(label)], requested_xy),
        )
        for alternate in alternatives:
            reached = self.move_to_target("slot", "preplace", slot=alternate)
            if not reached.ok:
                if reached.reason == "ik_gate":
                    continue
                return reached
            for stage, action in (("drop", self.drop_at_zone_target),
                                  ("release", self.open_gripper),
                                  ("home", self.return_to_home)):
                result = action()
                if not result.ok:
                    return result
                if stage == "release":
                    self._record_mission_release(alternate, color)
            return self._result(True, "move_block_to_slot", "released",
                                color=color, slot=alternate, requested_slot=requested_slot,
                                recovery="alternate_slot_while_held", placement_verified=False)
        return None

    def move_block_to_slot(self, color=None, slot=None, source=None):
        """Retry a recoverable transfer once, after measured set-down or home."""
        if source is not None:
            if color is not None:
                return self._fail("move_block_to_slot", "Specify color or source, not both", "invalid_arguments")
            return self._run_with_source(source, "move_block_to_slot",
                                         lambda selected: self.move_block_to_slot(selected, slot))
        attempts = []
        recoverable_stages = {
            "lift_empty", "pregrasp", "align", "grasp", "close_verify",
            "lift_held", "prezone_clearance", "preplace", "drop_approach",
        }
        for attempt_number in (1, 2):
            result = self._move_block_to_slot_once(color, slot)
            attempts.append({"attempt": attempt_number, "reason": result.reason,
                             "failed_stage": result.data.get("failed_stage")})
            if result.ok or result.robot_fault or self.s.cancel.is_set():
                result.data["attempts"] = attempts
                return result
            if result.data.get("failed_stage") not in recoverable_stages:
                result.data["attempts"] = attempts
                return result
            if self.s.held is not None:
                if (result.data.get("failed_stage") == "preplace"
                        and result.reason == "ik_gate"):
                    alternate = self._place_held_in_other_slot(color, slot)
                    if alternate is not None:
                        attempts[-1]["recovery_reason"] = alternate.reason
                        alternate.data["attempts"] = attempts
                        if alternate.ok or self.s.held is None or alternate.robot_fault:
                            return alternate
                if result.data.get("failed_stage") == "lift_held":
                    placed = self._release_held_at_slot_from_recovery(color, slot)
                    if placed is not None:
                        attempts[-1]["recovery_reason"] = placed.reason
                        placed.data["attempts"] = attempts
                        if placed.ok or self.s.held is None or placed.robot_fault:
                            return placed
                recovery = self._put_held_block_on_table(self.s.held.color)
            elif not self.s.arm_at_home():
                recovery = self.return_to_home()
            else:
                recovery = None
            if recovery is not None:
                attempts[-1]["recovery_reason"] = recovery.reason
                if not recovery.ok:
                    return self._result(
                        False, "move_block_to_slot", recovery.reason,
                        "Transfer failed and bounded recovery could not finish: " + recovery.detail,
                        retry_advice="ask_operator", failed_stage="recovery",
                        original_failed_stage=result.data.get("failed_stage"),
                        original_reason=result.reason, original_steps=result.data.get("steps"),
                        attempts=attempts, holding=self.s.held.color if self.s.held else None,
                    )
            if attempt_number == 2:
                result.data.update(attempts=attempts, holding=None,
                                   recovery_completed=True)
                result.retry_advice = "try_other_target"
                return result
        raise AssertionError("bounded Task 1 attempt loop exhausted")

    def _move_block_to_slot_once(self, color: str, slot: str):
        """Run one Task 1 transfer with measured gates between every primitive."""
        action, t0 = "move_block_to_slot", time.monotonic()
        labels = list(self.cfg.agent.zone_slots.labels)
        if (color not in self.cfg.perception.color_prototypes and not (color == "selected" and self._transfer_source)) or slot not in labels:
            return self._fail(action, "Unknown block colour or slot", "invalid_arguments")
        if self.s.held is not None:
            return self._fail(action, "Place the held block before starting another transfer",
                              "already_holding")
        if self.collection.status().get("episode_open") and not self.collection.sequence_mode:
            return self._fail(action, "Finish the recording episode before a Task 1 transfer")

        steps = []

        def run(stage, fn):
            self.s.cancel.raise_if_set()
            result = fn()
            step = {"stage": stage, "reason": result.reason}
            if stage in ("lift_held", "prezone_clearance"):
                step["measured_z_mm"] = round(self.s.arm_position_mm()[2], 1)
            if result.ok and stage in ("close_verify", "preplace", "drop_approach", "release"):
                try:
                    measured = self.s.robot.read_joints()
                    step["wrist_roll_deg"] = round(measured["wrist_roll"], 1)
                    step["jaw_yaw_deg"] = round(self.s.ik.forward_yaw_deg(measured), 1)
                except (KeyError, OSError, RuntimeError):
                    pass  # diagnostics must never turn a successful motion into a failure
            if result.ok and stage == "close_verify":
                step["observed_block_angle_deg"] = self._held_block_angle_deg
            if result.ok and stage == "preplace":
                step["commanded_yaw_deg"] = result.data.get("place_yaw_deg")
                step["zone_alignment_fallback"] = result.data.get("zone_alignment_fallback")
            steps.append(step)
            if result.ok:
                self.collection.settle_for_sequence_pause()
                return None
            failed = self._result(
                False, action, result.reason, result.detail,
                retry_advice=result.retry_advice, t0=t0,
                failed_stage=stage, steps=steps, color=color, slot=slot,
                holding=self.s.held.color if self.s.held else None,
                failure_measurements=result.data or None,
            )
            failed.images = result.images
            return failed

        observed = self._observe_target(color)
        steps.append({"stage": "observe_before", "reason": observed.reason})
        if not observed.ok:
            failed = self._result(False, action, observed.reason, observed.detail,
                                  retry_advice=observed.retry_advice, t0=t0,
                                  failed_stage="observe_before", steps=steps)
            failed.images = observed.images
            return failed
        index = labels.index(slot)
        scene = self._observed_scene
        assert scene is not None
        block = scene.find(color)
        if block is None:
            reason = "not_in_zone" if color in scene.inside else "not_detected"
            failed = self._result(
                False, action, reason, f"{color} block is not detected in the scene.",
                retry_advice="try_other_target", t0=t0, failed_stage="select",
                steps=steps, visible_outside=list(scene.outside), color=color, slot=slot,
            )
            failed.images = observed.images
            return failed
        occupant = self._slot_occupant(index)
        if occupant is not None:
            return self._result(
                False, action, "slot_occupied", f"{slot} is occupied by {occupant}.",
                retry_advice="retry_ok", t0=t0, failed_stage="select", steps=steps,
                color=color, slot=slot, free_slots=self._free_slot_labels(),
            )

        object_id, observation_id = f"{color}_1", self.observation_id
        failed = run("open", self.open_gripper)
        if failed is not None:
            return failed

        def lift_until_clear(stage, clearance_mm):
            required_z = self.s.grasp_z_mm + clearance_mm
            def ready(z):
                return z + self.limits.lateral_clearance_tolerance_mm >= required_z
            correction_mm = 0.0
            for _ in range(self.limits.max_lift_attempts):
                x, y, actual_z = self.s.arm_position_mm()
                if ready(actual_z):
                    return None
                # The selected tilted pick already preflighted a reverse path.
                # Use it before trying fixed-XY vertical IK at the far reach.
                held = self.s.held
                reverse_available = (
                    held is not None and abs(self._held_radial_tilt_deg) > self.cfg.task1.near_vertical_pick_max_deg
                    and held.attempt.hover_xy_mm is not None
                    and held.attempt.hover_z_mm > actual_z
                )
                if (held is not None and not reverse_available
                        and abs(held.attempt.radial_tilt_deg)
                            > self.cfg.task1.near_vertical_pick_max_deg):
                    if self._recover_loaded_reverse_lift(
                            required_z - self.limits.lateral_clearance_tolerance_mm,
                            max_command_z_mm=required_z if stage == "lift_held" else None):
                        steps.append({"stage": stage, "reason": "moved",
                                      "measured_z_mm": round(self.s.arm_position_mm()[2], 1)})
                        return None
                    return run(stage, lambda: self._result(
                        False, "move_relative", "grasp_blocked",
                        "Loaded inward/upward lift did not reach clearance: "
                        + self._loaded_lift_diagnostic))
                if reverse_available:
                    rise = min(held.attempt.hover_z_mm - actual_z,
                               self.cfg.agent.relative.max_jog_mm)
                    if rise > 0:
                        failed = run(stage, lambda rise=rise: self.move_relative(up_mm=rise))
                        if failed is not None:
                            return failed
                        continue
                # Target measured clearance, then compensate only for the
                # undershoot observed on the preceding lift. A full max_jog
                # from a far-edge grasp can be unreachable even though the
                # clearance pose is reachable.
                target_z = min(
                    required_z + correction_mm,
                    actual_z + self.cfg.agent.relative.max_jog_mm,
                )
                if target_z > required_z:
                    try:
                        self._solve((x, y, target_z))
                    except ValueError:
                        # Keep feedback below the IK gate. Never weaken the
                        # clearance requirement or try lateral motion low.
                        low, high = required_z, target_z
                        while high - low > self.limits.contact_step_mm:
                            middle = (low + high) / 2
                            try:
                                self._solve((x, y, middle))
                            except ValueError:
                                high = middle
                            else:
                                low = middle
                        target_z = low
                failed = run(stage, lambda: self.move_relative(
                    up_mm=target_z - actual_z))
                if failed is not None:
                    return failed
                actual_z = self.s.arm_position_mm()[2]
                error_limit = (self.limits.loaded_arrival_error_mm if self.s.held is not None
                               else self.limits.arrival_error_mm)
                correction_mm = min(max(0.0, target_z - actual_z), error_limit)
            # The last measured FK sample can precede the motor's final settle.
            if not ready(self.s.arm_position_mm()[2]):
                time.sleep(self.cfg.motion.descent_settle_s)
            measured_z = self.s.arm_position_mm()[2]
            if ready(measured_z):
                return None
            return self._result(
                False, action, "limit_exceeded",
                f"Arm did not reach lateral clearance: {measured_z:.1f}/{required_z:.1f}mm.",
                retry_advice="ask_operator", t0=t0, failed_stage=stage, steps=steps,
                color=color, slot=slot, holding=self.s.held.color if self.s.held else None,
            )

        failed = lift_until_clear("lift_empty", self.limits.lateral_clearance_mm)
        if failed is not None:
            return failed
        for stage, fn in (
            ("pregrasp", lambda: self.move_to_target(
                "object", "pregrasp", object_id=object_id, observation_id=observation_id)),
            ("align", lambda: self.align_gripper(object_id, observation_id)),
            ("grasp", lambda: self.move_to_target(
                "object", "grasp", object_id=object_id, observation_id=observation_id)),
            ("close_verify", self.close_gripper),
        ):
            failed = run(stage, fn)
            if failed is not None:
                return failed

        if self.s.held is None or self.s.held.color != color:
            return self._result(
                False, action, "precondition",
                "Grasp identity was not confirmed; do not transport.",
                retry_advice="ask_operator", t0=t0, failed_stage="close_verify",
                steps=steps, color=color, slot=slot,
                holding=self.s.held.color if self.s.held else None,
            )
        # A near-vertical grasp keeps the original measured, fixed-XY lift.
        # Oblique far grasps use the cached reverse path for the first 30 mm.
        if abs(self.s.held.attempt.radial_tilt_deg) <= self.cfg.task1.near_vertical_pick_max_deg:
            failed = lift_until_clear("lift_held", self.limits.lateral_clearance_mm)
        else:
            failed = run("lift_held", self._lift_held_joint_space)
        if failed is not None:
            return failed

        def raise_before_zone():
            required_z = self.s.grasp_z_mm + self.limits.lateral_clearance_mm
            if self._lateral_clearance_ready(self.s.arm_position_mm()[2]):
                return self._result(True, "prezone_clearance", "moved")
            if self._recover_loaded_reverse_lift(
                    required_z - self.limits.lateral_clearance_tolerance_mm):
                return self._result(True, "prezone_clearance", "moved",
                                    measured_fk_mm=list(self.s.arm_position_mm()))
            return self._result(
                False, "prezone_clearance", "grasp_blocked",
                "Could not raise held block before entering zone: "
                + self._loaded_lift_diagnostic,
                retry_advice="try_other_target")

        failed = run("prezone_clearance", raise_before_zone)
        if failed is not None:
            return failed

        for stage, fn in (
            ("preplace", lambda: self.move_to_target("slot", "preplace", slot=slot)),
            ("drop_approach", self.drop_at_zone_target),
            ("release", self.open_gripper),
            ("home", self.return_to_home),
        ):
            failed = run(stage, fn)
            if failed is not None:
                return failed
            if stage == "release":
                self._record_mission_release(slot, color)

        return self._result(True, action, "released",
                            f"{color} release and home return completed at {slot}.",
                            t0=t0, color=color, slot=slot, steps=steps,
                            placement_verified=False, slot_source="commanded")

    def _task1_near_low_zone_block(self, point) -> bool:
        """Model-FK low-path check around observed in-zone blocks only.

        Task 1 does not stack, so the observed block top is the calibrated
        grasp plane. A carried block extends below the gripper frame.
        """
        scene = self._observed_scene
        if scene is None:
            return False
        radius = self.cfg.agent.place_clear_radius_mm
        clear_z = self.s.grasp_z_mm + self.cfg.task1.zone_path_clearance_mm
        if self.s.held is not None:
            clear_z += self.cfg.agent.calibration_clearance.obstacle_height_mm
        return point[2] < clear_z and any(
            block.color != (self.s.held.color if self.s.held else None)
            and math.dist(point[:2], block.center_mm) < radius
            for block in scene.inside.values()
        )

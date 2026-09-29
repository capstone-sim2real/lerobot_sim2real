"""Experimental, bounded actions; perception never moves the arm.

All motion uses the session's cancellable IO. FK is model-derived feedback,
not camera-measured TCP. Limits are configurable assumptions, not validation.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, replace

from control.grasp import GraspAttempt
from control.task1_transport import angle_error_deg, square_angle_error_deg, zone_axis_yaw_deg
from control.sensing import ContactMonitor, check_grasp
from control.trajectory import interpolate
from session.arm_session import CameraError, HeldBlock
from session.results import ObservationImage
from session.skills import Skills


class PrimitiveSkills(Skills):
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
        self._grasp_failed = False
        self._pick_calibration = None
        self._pick_ready = False
        self._held_radial_tilt_deg = 0.0
        self._held_block_angle_deg = None
        self._held_pick_yaw_deg = None
        self._place_yaw_deg = None
        self._observed_scene = None
        self._pending_placement = None
        self._task2_placed_floors = {}
        self._stack_target = None
        self._recovery_target_xy = None
        from session.collection import Collection
        from control.trajectory import TrajectoryPlayer
        from control.motion import MotionController
        options = {} if collection_factory is None else {"resource_factory": collection_factory}
        self.collection = Collection(session, **options)
        session.robot = self.collection.io
        session.player = TrajectoryPlayer(session.robot, self.cfg.motion)
        session.motion = MotionController(session.robot, session.poses, self.cfg.motion, self.cfg.sensing)

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

    def idle_tick(self):
        self.collection.idle_tick()

    @property
    def idle_poll_s(self):
        return self.cfg.agent.collection.idle_poll_s if self.collection.recording else None

    def end_command(self):
        # Never leave background recording running after the owning chat/direct command.
        self.collection.discard("turn_ended")

    def on_tool_result(self, action, result):
        self.collection.note(action, result)

    def close(self):
        try:
            self.collection.close()
        finally:
            super().close()

    def record_tool_sequence(self, task, color, steps):
        """Run a model-planned tool program without model waits between steps."""
        from agent.primitive_tools import RECORDABLE_TOOLS
        from agent.tools import build_tools, result_from_exception, validate_arguments

        action, t0 = "record_tool_sequence", time.monotonic()
        allowed = set(RECORDABLE_TOOLS)
        if (not isinstance(task, str) or not task.strip() or len(task) > self.cfg.agent.collection.max_task_text_chars
                or not isinstance(color, str) or color not in self.cfg.task3.task_templates
                or not isinstance(steps, list)
                or not 1 <= len(steps) <= self.cfg.agent.collection.max_steps):
            return self._fail(action, "A task sentence, known color and bounded steps are required",
                              "invalid_arguments")
        definitions = {tool.spec.name: tool for tool in build_tools(self.cfg)}
        planned = []
        for index, step in enumerate(steps):
            if not isinstance(step, dict) or set(step) != {"name", "arguments"}:
                return self._fail(action, f"Invalid step {index + 1}", "invalid_arguments")
            name, args = step["name"], step["arguments"]
            if not isinstance(name, str) or name not in allowed or name not in definitions:
                return self._fail(action, f"Tool {name!r} is not recordable", "invalid_arguments")
            error = validate_arguments(definitions[name].spec.input_schema, args)
            if error is not None:
                return self._fail(action, f"Step {index + 1}: {error}", "invalid_arguments")
            planned.append((name, dict(args), definitions[name]))

        try:
            self.collection.begin(color, task_text=task.strip(), sequence_mode=True)
        except ImportError as exc:
            return self._fail(action, f"Dataset dependencies missing: {exc}", "disabled")
        except ValueError as exc:
            return self._fail(action, str(exc))

        results = []
        try:
            for index, (name, args, definition) in enumerate(planned, 1):
                try:
                    result = definition.run(self, args)
                except Exception as exc:  # return the failed step, then stop the program
                    result = result_from_exception(name, exc)
                self.collection.note(name, result)
                results.append({"step": index, "tool": name, **result.to_envelope()})
                if not result.ok or not self.collection.recording:
                    return self._result(False, action, result.reason if not result.ok else "task_incomplete",
                                        f"Step {index} ({name}) stopped: {result.detail}",
                                        t0=t0, step_results=results,
                                        collection=self.collection.status())
            try:
                saved = self.collection.save_sequence()
            except ValueError as exc:
                return self._result(False, action, "task_incomplete", str(exc), t0=t0,
                                    step_results=results, collection=self.collection.status())
            return self._result(saved, action, "ok" if saved else "task_incomplete",
                                "Recorded tool sequence saved." if saved else "Recorded take was discarded.",
                                t0=t0, step_results=results, collection=self.collection.status())
        finally:
            if self.collection.recorder and self.collection.recorder.is_open:
                self.collection.discard("sequence_interrupted")

    def begin_episode(self, object_id, observation_id):
        try:
            block = self._object(object_id, observation_id)
            if block.in_zone:
                return self._fail("begin_episode", "Collection target must start outside the zone")
            self.collection.begin(block.color)
        except ImportError as exc:
            return self._fail("begin_episode", f"Dataset dependencies missing: {exc}", "disabled")
        except ValueError as exc:
            return self._fail("begin_episode", str(exc))
        return self._result(True, "begin_episode", "ok", collection=self.collection.status())

    def save_episode(self):
        try:
            saved = self.collection.save()
        except ValueError as exc:
            return self._fail("save_episode", str(exc))
        return self._result(saved, "save_episode", "ok" if saved else "precondition", collection=self.collection.status())

    def discard_episode(self, reason):
        self.collection.discard(reason)
        return self._result(True, "discard_episode", "ok", collection=self.collection.status())

    def collection_status(self):
        return self._result(True, "collection_status", "ok", collection=self.collection.status())

    def finish_dataset(self):
        try:
            status = self.collection.finish()
        except ValueError as exc:
            return self._fail("finish_dataset", str(exc))
        return self._result(True, "finish_dataset", "ok", collection=status)

    @property
    def limits(self):
        return self.cfg.agent.primitives

    def state_dict(self, *, read_robot=True):
        state = super().state_dict(read_robot=read_robot)
        state.update(observation_id=self.observation_id,
                     joints=self.s.robot.read_joints() if read_robot else None,
                     arm_position_source="joint_feedback_and_URDF_FK_not_visual_TCP",
                     contact_confirmed=self._contact,
                     grasp_failed=self._grasp_failed, collection=self.collection.status())
        return state

    def _fail(self, action, detail, reason="precondition"):
        return self._result(False, action, reason, detail, retry_advice="retry_ok")

    def observe_scene(self):
        try:
            scene = self.s.observe(after=self.s._clock())
        except CameraError as exc:
            return self._fail("observe_scene", str(exc), "camera_stale" if exc.stale else "camera_unreachable")
        self._invalidate_pick()
        self._observed_scene = scene
        self.observation_id += 1
        self._observed_at = time.monotonic()
        self._objects = {f"{b.color}_1": b for b in [*scene.outside.values(), *scene.inside.values()]}
        learned_placement = None
        if self._pending_placement is not None and self.s.arm_at_home():
            color, target_xy, expected_slot = self._pending_placement
            landed = scene.find(color)
            cfg = self.cfg.agent.place_correction
            if (landed is not None and landed.in_zone
                    and (expected_slot is None or landed.slot_index == expected_slot)
                    and cfg.enabled and cfg.learn):
                measured_xy = landed.center_mm
                accepted = self.place_correction.observe(
                    target_xy, measured_xy,
                    base_xy_mm=self.s.base_xy,
                    frame=self.cfg.agent.relative.frame,
                )
                learned_placement = {
                    "accepted": accepted,
                    "color": color,
                    "target_xy_mm": list(target_xy),
                    "measured_xy_mm": list(measured_xy),
                    "miss_mm": round(math.dist(target_xy, measured_xy), 1),
                    "correction": self.place_correction.as_dict(),
                }
            self._pending_placement = None
        images = ()
        snapshot = self.s.last_snapshot
        if snapshot is not None:
            import cv2  # optional runtime dependency
            frame = snapshot.frame
            if frame.shape[1] > self.limits.image_max_width:
                scale = self.limits.image_max_width / frame.shape[1]
                frame = cv2.resize(frame, (self.limits.image_max_width, round(frame.shape[0] * scale)))
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.limits.image_jpeg_quality])
            if not ok:
                return self._fail("observe_scene", "JPEG encoding failed", "camera_unreachable")
            images = (ObservationImage(encoded.tobytes(), "shoulder", snapshot.frame_seq, snapshot.captured_at),)
        result = self._result(True, "observe_scene", "ok", observation_id=self.observation_id,
                              frame_seq=scene.frame_seq, captured_at=scene.captured_at,
                              objects=[dict(object_id=key, **self._block_dict(b)) for key, b in self._objects.items()],
                              image_available=bool(images), arm_occlusion_possible=not self.s.arm_at_home(),
                              learned_placement=learned_placement,
                              limitations=["one detection per colour; IDs are observation-scoped",
                                            "planar CV does not measure stack height or visual TCP",
                                            "missing detection is not proof of absence"])
        return replace(result, images=images)

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
        if scene is None or scene.slot_occupancy.get(index) not in (None, color):
            return None
        reason, _ = self.placement_verdict(xyz[:2], allow_zone=True,
                                           ignore_color=color, check_ik=False)
        if reason is not None:
            return None
        self._target = ("slot", None, None, slot, None, None, "preplace")
        self._zone_drop_xy = tuple(xyz[:2])
        self._zone_drop_ready = True
        self._place_yaw_deg = None
        released = self.open_gripper()
        if not released.ok:
            return released
        home = self.return_to_home()
        if not home.ok:
            return home
        verified = self.observe_scene()
        landed = self._observed_scene.find(color) if verified.ok else None
        actual_slot = labels[landed.slot_index] if landed and landed.slot_index is not None else None
        if actual_slot == slot:
            return self._result(True, "move_block_to_slot", "released",
                                f"{color} block observed in {slot} after short-lift recovery.",
                                color=color, slot=slot, recovery="release_at_reached_slot",
                                measured_xy_mm=list(landed.center_mm))
        return self._result(False, "move_block_to_slot", "task_incomplete",
                            "Block released at the reached slot, but placement was not confirmed.",
                            retry_advice="retry_ok", color=color, slot=slot,
                            recovery="release_at_reached_slot", actual_slot=actual_slot)

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
             if label != requested_slot and scene.slot_occupancy.get(index) is None),
            key=lambda label: math.dist(
                self.s.slot_centres[labels.index(label)], requested_xy),
        )
        for alternate in alternatives:
            reached = self.move_to_target("slot", "preplace", slot=alternate)
            if not reached.ok:
                if reached.reason == "ik_gate":
                    continue
                return reached
            for action in (self.drop_at_zone_target, self.open_gripper,
                           self.return_to_home):
                result = action()
                if not result.ok:
                    return result
            observed = self.observe_scene()
            landed = self._observed_scene.find(color) if observed.ok else None
            actual_slot = (labels[landed.slot_index]
                           if landed and landed.slot_index is not None else None)
            if actual_slot == alternate:
                return self._result(
                    True, "move_block_to_slot", "released",
                    f"{color} block observed in alternate slot {alternate}.",
                    color=color, slot=alternate, requested_slot=requested_slot,
                    recovery="alternate_slot_while_held",
                    measured_xy_mm=list(landed.center_mm))
            return self._result(
                False, "move_block_to_slot", "task_incomplete",
                f"Released {color} at alternate slot {alternate}, but placement was not confirmed.",
                retry_advice="retry_ok", color=color, slot=alternate,
                requested_slot=requested_slot, actual_slot=actual_slot)
        return None

    def move_block_to_slot(self, color: str, slot: str):
        """Retry a recoverable transfer once, after measured set-down or home."""
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
        if color not in self.cfg.perception.color_prototypes or slot not in labels:
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

        observed = self.observe_scene()
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
        block = scene.outside.get(color)
        if block is None:
            reason = "not_in_zone" if color in scene.inside else "not_detected"
            failed = self._result(
                False, action, reason, f"{color} block is not detected outside the zone.",
                retry_advice="try_other_target", t0=t0, failed_stage="select",
                steps=steps, visible_outside=list(scene.outside), color=color, slot=slot,
            )
            failed.images = observed.images
            return failed
        occupant = scene.slot_occupancy.get(index)
        if occupant is not None:
            return self._result(
                False, action, "slot_occupied", f"{slot} is occupied by {occupant}.",
                retry_advice="retry_ok", t0=t0, failed_stage="select", steps=steps,
                color=color, slot=slot, free_slots=[
                    labels[i] for i, value in scene.slot_occupancy.items() if value is None
                ],
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
                    held is not None and abs(self._held_radial_tilt_deg) >= 5.0
                    and held.attempt.hover_xy_mm is not None
                    and held.attempt.hover_z_mm > actual_z
                )
                if held is not None and not reverse_available:
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
        # Retreat only 30 mm from the grasp. Reach the zone-crossing height
        # in a separate inward/upward move, before entering the zone.
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

        verified = self.observe_scene()
        steps.append({"stage": "observe_after", "reason": verified.reason})
        if not verified.ok:
            failed = self._result(
                False, action, verified.reason,
                f"Block released, but placement could not be observed: {verified.detail}",
                retry_advice="ask_operator", t0=t0, failed_stage="observe_after",
                steps=steps, color=color, slot=slot,
            )
            failed.images = verified.images
            return failed
        landed = self._observed_scene.find(color)
        actual_slot = labels[landed.slot_index] if landed and landed.slot_index is not None else None
        miss = math.dist(landed.center_mm, self.s.slot_centres[index]) if landed else None
        if actual_slot != slot:
            still_at_source = bool(
                landed and not landed.in_zone
                and math.dist(landed.center_mm, block.center_mm) <= self.cfg.agent.calibration_clearance.block_side_mm / 2
            )
            detail = (
                f"{color} is still near its starting position; {slot} placement was not verified."
                if still_at_source else
                f"{color} was released, but the camera did not confirm {slot}."
            )
            failed = self._result(
                False, action, "task_incomplete", detail,
                retry_advice="try_other_target" if still_at_source else
                             "retry_ok" if landed else "ask_operator",
                t0=t0, failed_stage="verify", steps=steps, color=color, slot=slot,
                actual_slot=actual_slot, measured_xy_mm=list(landed.center_mm) if landed else None,
                still_at_source=still_at_source,
                miss_mm=round(miss, 1) if miss is not None else None,
            )
            failed.images = verified.images
            return failed
        yaw_error = square_angle_error_deg(
            landed.angle_deg, zone_axis_yaw_deg(self.s.calib.zone_polygon_mm)
        )
        return self._result(
            True, action, "released", f"{color} block observed in {slot}.", t0=t0,
            color=color, slot=slot, measured_xy_mm=list(landed.center_mm),
            miss_mm=round(miss, 1), frame_seq=verified.data.get("frame_seq"),
            place_correction=self.place_correction.as_dict(),
            observed_block_yaw_deg=round(landed.angle_deg, 1),
            zone_yaw_error_deg=round(yaw_error, 1),
            placement_aligned=abs(yaw_error) <= self.cfg.task1.place_yaw_tolerance_deg,
        )

    def stack_block_to_floor(self, color: str, floor: int):
        """Try the requested 0-based floor, with one guarded table recovery."""
        attempts = []
        for attempt_number in (1, 2):
            result = self._stack_block_to_floor_once(color, floor)
            attempts.append({"attempt": attempt_number, "reason": result.reason,
                             "failed_stage": result.data.get("failed_stage")})
            if result.ok:
                result.data["attempts"] = attempts
                return result
            if result.robot_fault or self.s.cancel.is_set() or self.s.held is None:
                result.data["attempts"] = attempts
                return result
            recovered = self._put_held_block_on_table(color)
            attempts[-1]["fallback_reason"] = recovered.reason
            attempts[-1]["fallback_verified"] = recovered.data.get("verified")
            if not recovered.ok:
                result.data.update(attempts=attempts,
                                   failed_detail=result.detail,
                                   fallback_error=recovered.detail,
                                   holding=self.s.held.color if self.s.held else None)
                result.detail = "적층이 막혔고 임시 배치도 완료하지 못했습니다: " + recovered.detail
                result.retry_advice = "ask_operator"
                return result
            if attempt_number == 2:
                result.data.update(attempts=attempts, fallback_released=True,
                                   failed_detail=result.detail,
                                   holding=None, fallback_placement=recovered.data)
                result.detail = "적층 재시도도 실패해 블록을 빈 테이블에 내려놓았습니다."
                result.retry_advice = "try_other_target"
                return result
        raise AssertionError("bounded Task 2 attempt loop exhausted")

    def _put_held_block_on_table(self, color: str):
        """Set down a held block on a clear table point before a bounded retry."""
        action = "stack_recovery"
        if self.s.held is None or self.s.held.color != color or not self._held_check():
            return self._fail(action, "Cannot verify the held block", "no_block_held")
        observed = self.observe_scene()
        if not observed.ok:
            return self._fail(action, "Camera unavailable; cannot find a free table point",
                              observed.reason)
        xyz = self.s.arm_position_mm()
        self.s.held.over_xy_mm = tuple(xyz[:2])
        self._target = None
        self._stack_target = None
        self._contact = False
        self._stack_drop_ready = False
        required_z = self.s.grasp_z_mm + self.limits.lateral_clearance_mm
        if not self._lateral_clearance_ready(xyz[2]):
            # After a failed lift the block may still be right above its
            # source. Set it down there without a lateral sweep if that point
            # is clear. A blocked/inside-zone point needs a guarded lift.
            reason, _ = self.placement_verdict(xyz[:2], allow_zone=False,
                                               ignore_color=color, check_ik=False)
            if reason is None:
                self._recovery_target_xy = tuple(xyz[:2])
                self._target = ("recovery", None, None, None, None, None, "preplace")
                landed = self.descend_until_contact(self.limits.contact_max_descent_mm)
                if not landed.ok:
                    return landed
                released = self.open_gripper()
                if not released.ok:
                    return released
                home = self.return_to_home()
                if not home.ok:
                    return home
                seen = self.observe_scene()
                block = self._observed_scene.find(color) if seen.ok else None
                if block is None or block.in_zone:
                    return self._result(False, action, "task_incomplete",
                                        "Set-down at the source was not confirmed",
                                        retry_advice="ask_operator")
                return self._result(True, action, "released", verified=True,
                                    in_zone=False, measured=self._block_dict(block),
                                    target_xy_mm=list(xyz[:2]), mode="vertical_put_back")
            lift = self.move_relative(up_mm=min(required_z - xyz[2],
                                                self.cfg.agent.relative.max_jog_mm))
            if not lift.ok or not self._lateral_clearance_ready(self.s.arm_position_mm()[2]):
                return self._result(False, action, "limit_exceeded",
                                    "No safe nearby set-down: current point is blocked and lift failed",
                                    retry_advice="ask_operator", lift_reason=lift.reason,
                                    blocked_reason=reason)
        self.s.held.over_xy_mm = tuple(self.s.arm_position_mm()[:2])
        placed = self.place_on_table()
        if not placed.ok:
            return placed
        measured = placed.data.get("measured") or {}
        if (self.s.held is not None or not placed.data.get("verified")
                or placed.data.get("in_zone") is not False
                or measured.get("color") != color):
            return self._result(False, action, "task_incomplete",
                                "Temporary table release was not confirmed outside the zone",
                                retry_advice="ask_operator", placement=placed.data)
        return placed

    def _task2_path_outside_zone(self, start, waypoints) -> bool:
        """Model-FK preflight of commanded joint sweeps before tower entry."""
        from perception.zone import point_in_zone
        previous = start
        left_zone = not point_in_zone(
            self.s.ik.forward_position_mm(previous)[:2], self.s.calib
        )
        for goal in waypoints:
            for command in interpolate(previous, goal, self.cfg.motion.max_step_per_tick):
                pose = {**previous, **command}
                inside = point_in_zone(
                    self.s.ik.forward_position_mm(pose)[:2], self.s.calib
                )
                if left_zone and inside:
                    return False
                if not inside:
                    left_zone = True
            previous = {**previous, **goal}
        return True

    def _stack_block_to_floor_once(self, color: str, floor: int):
        """Transfer one block to an explicitly requested tower floor."""
        action, t0 = "stack_block_to_floor", time.monotonic()
        if color not in self.cfg.perception.color_prototypes:
            return self._fail(action, "Unknown block colour", "invalid_arguments")
        if self.s.held is not None:
            return self._fail(action, "A block is already held", "already_holding")
        if self.collection.status().get("episode_open") and not self.collection.sequence_mode:
            return self._fail(action, "Finish the recording episode before stacking")
        if type(floor) is not int or not 0 <= floor <= 4:
            return self._fail(action, "floor must be an integer from 0 to 4", "invalid_arguments")
        level_number = floor + 1
        try:
            planner = self.s.stack
        except ValueError as exc:
            return self._fail(action, str(exc), "ik_gate")
        steps = []

        def run(stage, fn):
            self.s.cancel.raise_if_set()
            result = fn()
            steps.append({"stage": stage, "reason": result.reason})
            if result.ok:
                self.collection.settle_for_sequence_pause()
                return None
            failed = self._result(False, action, result.reason, result.detail,
                                  retry_advice=result.retry_advice, t0=t0,
                                  failed_stage=stage, steps=steps, color=color,
                                  floor=floor, level=level_number,
                                  holding=self.s.held.color if self.s.held else None)
            failed.images = result.images
            return failed

        observed = self.observe_scene()
        steps.append({"stage": "observe_before", "reason": observed.reason})
        if not observed.ok:
            failed = self._result(False, action, observed.reason, observed.detail,
                                  retry_advice=observed.retry_advice, t0=t0,
                                  failed_stage="observe_before", steps=steps)
            failed.images = observed.images
            return failed
        scene = self._observed_scene
        assert scene is not None
        radius = self.cfg.agent.place_clear_radius_mm
        # A known block moved back outside cannot still occupy its old floor.
        # This history helps with arm occlusion but never selects the floor.
        for placed_floor, placed_color in list(self._task2_placed_floors.items()):
            if placed_color in scene.outside:
                del self._task2_placed_floors[placed_floor]
        if floor >= len(planner.levels):
            return self._fail(action, "Requested floor is not configured", "limit_exceeded")
        level = planner.levels[floor]
        if not level.reachable:
            return self._result(False, action, "ik_gate",
                                f"Floor {floor} is outside the IK gate: {level.reason}",
                                retry_advice="ask_operator", floor=floor, level=level_number)
        block = scene.find(color)
        if block is None:
            return self._result(False, action, "not_detected",
                                f"{color} is not visible in the current scene",
                                retry_advice="try_other_target", t0=t0,
                                failed_stage="select", steps=steps, color=color,
                                floor=floor, level=level_number)
        nearby = [b for b in scene.inside.values()
                  if math.dist(b.center_mm, level.xy_mm) < radius]
        if nearby:
            support_evidence = "visible_near_tower_height_unverified"
        elif self._task2_placed_floors.get(floor - 1) is not None:
            support_evidence = "prior_release_unseen_height_unverified"
        else:
            support_evidence = "requested_floor_unverified"
        # A fallen block can be in the zone, and a collapsed tower makes
        # earlier floor records unreliable. These are observations for the
        # caller, not reasons to refuse the requested bounded attempt.

        object_id, observation_id = f"{color}_1", self.observation_id
        failed = run("open", self.open_gripper)
        if failed is not None:
            return failed

        def lift_until_clear(stage):
            required_z = self.s.grasp_z_mm + self.limits.lateral_clearance_mm
            correction_mm = 0.0
            for _ in range(self.limits.max_lift_attempts):
                x, y, actual_z = self.s.arm_position_mm()
                if self._lateral_clearance_ready(actual_z):
                    return None
                held = self.s.held
                if (held is not None and abs(self._held_radial_tilt_deg) >= 5.0
                        and held.attempt.hover_xy_mm is not None
                        and held.attempt.hover_z_mm >= required_z):
                    rise = min(held.attempt.hover_z_mm - actual_z,
                               self.cfg.agent.relative.max_jog_mm)
                    if rise > 0:
                        failed = run(stage, lambda rise=rise: self.move_relative(up_mm=rise))
                        if failed is not None:
                            return failed
                        continue
                target_z = min(required_z + correction_mm,
                               actual_z + self.cfg.agent.relative.max_jog_mm)
                if target_z > required_z:
                    try:
                        self._solve((x, y, target_z))
                    except ValueError:
                        target_z = required_z
                failed = run(stage, lambda: self.move_relative(up_mm=target_z - actual_z))
                if failed is not None:
                    return failed
                actual_z = self.s.arm_position_mm()[2]
                error_limit = (self.limits.loaded_arrival_error_mm if self.s.held
                               else self.limits.arrival_error_mm)
                correction_mm = min(max(0.0, target_z - actual_z), error_limit)
            if self._lateral_clearance_ready(self.s.arm_position_mm()[2]):
                return None
            return self._result(False, action, "limit_exceeded",
                                "Arm did not reach lateral clearance",
                                retry_advice="ask_operator", t0=t0,
                                failed_stage=stage, steps=steps, color=color,
                                floor=floor, level=level_number,
                                holding=self.s.held.color if self.s.held else None)

        failed = lift_until_clear("lift_empty")
        if failed is not None:
            return failed
        for stage, fn in (
            ("pregrasp", lambda: self.move_to_target(
                "object", "pregrasp", object_id=object_id,
                observation_id=observation_id,
                _route_guard=(self._task2_path_outside_zone
                              if not block.in_zone else None))),
            ("align", lambda: self.align_gripper(object_id, observation_id)),
            ("grasp", lambda: self.move_to_target(
                "object", "grasp", object_id=object_id,
                observation_id=observation_id)),
            ("close_verify", self.close_gripper),
        ):
            failed = run(stage, fn)
            if failed is not None:
                return failed
        if self.s.held is None or self.s.held.color != color:
            return self._result(False, action, "precondition",
                                "Grasp identity was not confirmed; do not transport",
                                retry_advice="ask_operator", t0=t0,
                                failed_stage="close_verify", steps=steps,
                                color=color, floor=floor, level=level_number)
        failed = lift_until_clear("lift_held")
        if failed is not None:
            return failed

        from control.task1_transport import over_ik_gate
        transfer = planner.plan(self.s.held.attempt, level_number - 1)
        try:
            entry = planner.entry_pose(level)
            outside_stage = planner.outside_stage(level)
        except ValueError as exc:
            return self._result(False, action, "ik_gate", str(exc),
                                retry_advice="ask_operator", t0=t0,
                                failed_stage="transport", holding=color)
        # The prior apex_place sat inside the zone. Stay outside until the
        # final stage-to-hover placement entry, and preflight every joint sweep.
        carry = tuple((name, waypoint) for name, waypoint in transfer.carry
                      if name != "apex_place")
        if any(over_ik_gate(waypoint, self.cfg) for _, waypoint in carry):
            return self._result(False, action, "ik_gate",
                                "Tower carry apex is outside the IK gate; block remains held",
                                retry_advice="ask_operator", t0=t0,
                                failed_stage="transport", steps=steps,
                                color=color, floor=floor, level=level_number, holding=color)
        current = self.s.robot.read_joints()
        route = tuple(waypoint.joints for _, waypoint in carry) + (outside_stage.joints,)
        if not self._task2_path_outside_zone(current, route):
            # The optional pick apex can curve across the zone even when a
            # direct joint sweep to the outside stage stays on the near side.
            if self._task2_path_outside_zone(current, (outside_stage.joints,)):
                carry = ()
            else:
                return self._result(False, action, "limit_exceeded",
                                    "Task 2 carry would cross the occupied target zone",
                                    retry_advice="ask_operator", t0=t0,
                                    failed_stage="transport", holding=color)


        entry_z = (level.place_z_mm + self.cfg.task2.upper_entry_clearance_mm
                   if level.level >= self.cfg.task2.upper_entry_level
                   else level.hover_z_mm)

        def transport():
            for name, waypoint in (*carry, ("outside_stage", outside_stage),
                                   ("tower_hover", entry)):
                self.s.cancel.raise_if_set()
                self.s.player.move_to(waypoint.joints,
                                      tol=self.cfg.motion.transit_arrival_tol)
                if not self._held_check():
                    return self._result(False, "transport", "grasp_empty",
                                        f"Grasp verification failed at {name}",
                                        retry_advice="ask_operator")
            actual = self.s.arm_position_mm()
            if math.dist(actual, (*level.xy_mm, entry_z)) > self.limits.loaded_arrival_error_mm:
                raise TimeoutError("Measured FK did not reach tower hover")
            self._stack_target = level
            self._target = ("stack", None, None, None, None, None, "preplace")
            return self._result(True, "transport", "moved",
                                hover_z_mm=entry_z)

        failed = run("transport", transport)
        if failed is not None:
            return failed
        def approach_drop():
            if self.s.held is None or not self._held_check():
                return self._fail("stack_drop", "Verified held block required", "no_block_held")
            self._stack_drop_ready = False
            # No load/contact decision here: the Task 2 primitive releases
            # above the nominal level. The clearance is an unmeasured tuneable.
            drop_z = min(entry_z,
                         level.place_z_mm + self.cfg.task2.drop_clearance_mm)
            approach_tilt = (self.cfg.task2.upper_entry_radial_tilt_deg
                             if level.level >= self.cfg.task2.upper_entry_level
                             else level.radial_tilt_deg)
            try:
                pose = self._solve(
                    (*level.xy_mm, drop_z),
                    radial_tilt_deg=approach_tilt,
                    max_position_error_mm=self.cfg.ik.max_position_error_mm,
                    max_tilt_error_deg=self.cfg.task1.place_level_tolerance_deg,
                )
            except ValueError as exc:
                return self._fail("stack_drop", str(exc), "ik_gate")
            self.s.player.move_to(
                pose.joints, max_step=self.cfg.motion.descent_step_per_tick,
                tol=self.cfg.motion.arrival_tol,
            )
            actual = self.s.arm_position_mm()
            if (math.dist(actual[:2], level.xy_mm) > self.limits.alignment_tolerance_mm
                    or not level.place_z_mm + 3 <= actual[2]
                    <= level.place_z_mm + self.cfg.task2.drop_clearance_mm + 10):
                return self._fail("stack_drop", "Measured FK missed the release window",
                                  "motion_timeout")
            if not self._held_check():
                return self._fail("stack_drop", "Grasp verification failed", "grasp_empty")
            self._stack_drop_ready = True
            return self._result(True, "stack_drop", "ok",
                                release_fk_mm=list(actual), drop_z_mm=drop_z)

        def retreat_outside():
            # Rise over the tower before any lateral motion. After crossing
            # the near edge, no joint-interpolated segment may re-enter it.
            current = self.s.robot.read_joints()
            home_joints = {joint: value for joint, value in
                           self.s.poses.get(self.cfg.motion.home_pose).items()
                           if joint != "gripper"}
            if (not self._task2_path_outside_zone(entry.joints, (outside_stage.joints,))
                    or not self._task2_path_outside_zone(outside_stage.joints,
                                                         (home_joints,))):
                return self._fail("stack_retreat", "Retreat re-enters the target zone",
                                  "limit_exceeded")
            self.s.player.move_to(entry.joints,
                                  tol=self.cfg.motion.transit_arrival_tol)
            self.s.player.move_to(outside_stage.joints,
                                  tol=self.cfg.motion.transit_arrival_tol)
            return self._result(True, "stack_retreat", "moved")

        for stage, fn in (
            ("drop_approach", approach_drop),
            ("release", self.open_gripper),
            ("retreat_outside", retreat_outside),
            ("home", self.return_to_home),
        ):
            failed = run(stage, fn)
            if failed is not None:
                return failed
        verified = self.observe_scene()
        steps.append({"stage": "observe_after", "reason": verified.reason})
        if not verified.ok:
            return self._result(False, action, verified.reason,
                                "Block was released, but the camera could not reobserve the tower",
                                retry_advice="ask_operator", t0=t0,
                                failed_stage="observe_after", steps=steps,
                                color=color, floor=floor, level=level_number)
        landed = self._observed_scene.find(color)
        if landed is None or math.dist(landed.center_mm, level.xy_mm) >= radius:
            return self._result(False, action, "task_incomplete",
                                "Released block was not observed near the tower point",
                                retry_advice="ask_operator", t0=t0,
                                failed_stage="verify", steps=steps, color=color,
                                floor=floor, level=level_number, stack_verified=False)
        self._task2_placed_floors[floor] = color
        return self._result(True, action, "released",
                            f"Floor {floor} released near tower point; tower height remains unverified",
                            t0=t0, color=color, floor=floor, level=level_number,
                            contact_confirmed=False,
                            release_mode="height_drop",
                            drop_clearance_mm=self.cfg.task2.drop_clearance_mm,
                            placement_observed=True, stack_verified=False,
                            support_evidence=support_evidence,
                            expected_place_z_mm=level.place_z_mm,
                            measured_xy_mm=list(landed.center_mm))

    def _object(self, object_id, observation_id):
        if observation_id != self.observation_id or time.monotonic() - self._observed_at > self.limits.target_max_age_s:
            raise ValueError("Target observation expired; observe_scene again")
        if object_id not in self._objects:
            raise ValueError("Object is not in this observation")
        return self._objects[object_id]

    def _held_check(self):
        if self.s.held is not None:
            reading = check_grasp(self.s.robot, self.cfg.sensing, settle=False)
            # VERIFY required both position and load when the block was first
            # picked. During transport the gripper load can relax with arm
            # posture even while the block remains visibly between the jaws.
            # A fully closed jaw position still detects an actual loss.
            if not reading.pos_says_held:
                self._grasp_failed = True
                self._contact = False
                self._stack_drop_ready = False
                self._zone_drop_ready = False
                return False
        return not self._grasp_failed

    def _choose_place_yaw(self, xyz, place_tilt):
        """Keep the held jaw heading when reachable, otherwise use a safe yaw."""
        if self._held_block_angle_deg is None or self._held_pick_yaw_deg is None:
            raise ValueError("Held block orientation was not recorded at grasp")
        axis = zone_axis_yaw_deg(self.s.calib.zone_polygon_mm)
        base = self._held_pick_yaw_deg + axis - self._held_block_angle_deg
        neutral = self.s.ik.neutral_yaw_deg(*xyz)
        current_yaw = self.s.ik.forward_yaw_deg(self.s.robot.read_joints())
        # A square does not need a quarter-turn to satisfy Task 1. Keep the
        # held jaw heading when it is reachable at both hover and release.
        aligned = sorted(
            {(base + 90.0 * k + 180.0) % 360.0 - 180.0 for k in range(-3, 4)},
            key=lambda yaw: abs(angle_error_deg(yaw, current_yaw)),
        )
        fallback = [neutral, *(neutral + offset
                               for offset in self.cfg.task1.place_yaw_fallback_offsets_deg)]
        safe_z = (self.s.grasp_z_mm + self.limits.lateral_clearance_mm
                  - self.limits.lateral_clearance_tolerance_mm)

        def viable(yaws):
            plans = []
            for yaw in yaws:
                try:
                    hover = self._solve(
                        xyz, radial_tilt_deg=place_tilt,
                        max_position_error_mm=self.cfg.ik.max_position_error_mm,
                        max_tilt_error_deg=self.cfg.task1.place_level_tolerance_deg,
                        yaw_deg=yaw,
                        max_yaw_error_deg=self.cfg.task1.place_yaw_tolerance_deg,
                    )
                    # Release is above the grasp plane; no need to prove the
                    # wrist can reach the table while holding a block.
                    self._solve(
                        (*xyz[:2], self.s.drop_z_mm), radial_tilt_deg=place_tilt,
                        max_position_error_mm=self.cfg.ik.max_position_error_mm,
                        max_tilt_error_deg=self.cfg.task1.place_level_tolerance_deg,
                        yaw_deg=yaw,
                        max_yaw_error_deg=self.cfg.task1.place_yaw_tolerance_deg,
                    )
                except ValueError:
                    continue
                planned_z = self.s.ik.forward_position_mm(hover.joints)[2]
                if planned_z >= safe_z:
                    plans.append((abs(angle_error_deg(yaw, current_yaw)),
                                  abs(angle_error_deg(yaw, neutral)),
                                  hover.position_error_mm, yaw))
            return plans

        unchanged_plans = viable([current_yaw])
        if unchanged_plans:
            return unchanged_plans[0][3], False
        aligned_plans = viable(aligned)
        if aligned_plans:
            return min(aligned_plans)[3], True
        fallback_plans = viable(fallback)
        if fallback_plans:
            return min(fallback_plans)[3], False
        raise ValueError("No placement yaw reaches IK and clearance gates")

    def _solve(self, xyz, *, radial_tilt_deg=None, max_position_error_mm=None,
               max_tilt_error_deg=None, yaw_deg=None, max_yaw_error_deg=None):
        if not self.s.in_workspace(xyz[:2]):
            raise ValueError("Waypoint outside workspace")
        joints = self.s.robot.read_joints()
        tilt = (self._held_radial_tilt_deg if self.s.held is not None else 0.0) if radial_tilt_deg is None else radial_tilt_deg
        error_limit = (self.cfg.agent.relative.jog_max_ik_error_mm
                       if max_position_error_mm is None else max_position_error_mm)
        solved = (self.s.ik.solve_holding_wrist_roll(
            *xyz, wrist_roll_deg=joints["wrist_roll"], radial_tilt_deg=tilt
        ) if yaw_deg is None else self.s.ik.solve(
            *xyz, yaw_deg=yaw_deg, radial_tilt_deg=tilt
        ))
        if not math.isfinite(solved.position_error_mm) or solved.position_error_mm > error_limit:
            raise ValueError(
                f"Waypoint failed IK gate: target=({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f})mm "
                f"error={solved.position_error_mm:.1f}mm limit={error_limit:.1f}mm "
                f"radial_tilt={tilt:.1f}deg"
            )
        if (max_tilt_error_deg is not None
                and (not math.isfinite(solved.tilt_error_deg)
                     or abs(solved.tilt_error_deg - abs(tilt)) > max_tilt_error_deg)):
            raise ValueError(
                f"Waypoint missed placement level: achieved={solved.tilt_error_deg:.1f}deg "
                f"target={abs(tilt):.1f}deg limit={max_tilt_error_deg:.1f}deg"
            )
        if (max_yaw_error_deg is not None
                and abs(angle_error_deg(
                    self.s.ik.forward_yaw_deg(solved.joints), yaw_deg)) > max_yaw_error_deg):
            raise ValueError("Waypoint missed placement jaw yaw")
        if (abs(solved.joints["wrist_roll"]) > self.limits.wrist_roll_limit_deg
                or (self.limits.calibrated_pick and solved.joints["wrist_roll"] < self.cfg.agent.calibration_clearance.wrist_roll_min_deg)):
            raise ValueError("Wrist exceeds primitive neutral limit")
        return solved

    def _move(self, action, xyz, *, radial_tilt_deg=None, max_ik_error_mm=None,
              max_tilt_error_deg=None, level_during_carry=False, place_yaw_deg=None,
              vertical_only=False):
        self._invalidate_pick()
        start = self.s.arm_position_mm()
        if vertical_only:
            # A vertical jog has no commanded XY motion; repeated encoder/FK
            # reads can differ slightly while building and checking its goal.
            xyz = (*start[:2], xyz[2])
        lateral = math.dist(start[:2], xyz[:2]) > 1e-6
        clear_z = self.s.grasp_z_mm + self.limits.lateral_clearance_mm
        if lateral and not self._lateral_clearance_ready(min(start[2], xyz[2])):
            return self._fail(action, "Lift vertically above clearance before lateral movement")
        if not self._held_check() and (lateral or xyz[2] < start[2]):
            return self._fail(action, "Grasp verification failed; open/retry or lift vertically")
        if not self.s.grasp_z_mm <= xyz[2] <= self.cfg.agent.relative.jog_max_z_mm:
            return self._fail(action, "Target outside configured vertical window")
        count = max(1, math.ceil(math.dist(start, xyz) / self.limits.cartesian_step_mm))
        points = [tuple(a + (b-a)*i/count for a,b in zip(start,xyz)) for i in range(1,count+1)]
        inward_carry = (
            level_during_carry
            and math.dist(start[:2], self.s.base_xy) > math.dist(xyz[:2], self.s.base_xy)
        )
        if inward_carry:
            from control.task1_transport import carry_level_tilt_deg
            # Level the actual grasp tilt while moving inward, above obstacles.
            held_tilt = self.s.held.attempt.radial_tilt_deg if self.s.held else None
            tilts = [carry_level_tilt_deg(
                point[:2], self.s.base_xy, self.cfg, held_tilt_deg=held_tilt)
                     for point in points]
            points.append(xyz)
            tilts.append(radial_tilt_deg)
        else:
            # On an outward trip, the final placement orientation is reachable
            # throughout the path. Avoid a same-position tilt flip at the slot.
            tilts = [radial_tilt_deg] * len(points)
        if place_yaw_deg is not None:
            start_yaw = self.s.ik.forward_yaw_deg(self.s.robot.read_joints())
            delta_yaw = angle_error_deg(place_yaw_deg, start_yaw)
            yaws = [start_yaw + delta_yaw * i / count for i in range(1, count + 1)]
            if inward_carry:
                yaws.append(place_yaw_deg)
        else:
            yaws = [None] * len(points)
        try:
            plans = [
                self._solve(
                    point,
                    radial_tilt_deg=tilt,
                    max_position_error_mm=max_ik_error_mm,
                    max_tilt_error_deg=max_tilt_error_deg,
                    yaw_deg=yaw,
                    max_yaw_error_deg=(self.cfg.task1.place_yaw_tolerance_deg
                                       if yaw is not None else None),
                )
                for point, tilt, yaw in zip(points, tilts, yaws, strict=True)
            ]
        except ValueError as exc:
            return self._fail(action, str(exc), "ik_gate")
        if self.s.held is not None and level_during_carry:
            # A permissive placement IK error can solve a hover several mm
            # below its requested Z. Reject that before moving a held block
            # into the zone; the subsequent motor lag can only make it worse.
            safe_z = clear_z - self.limits.lateral_clearance_tolerance_mm
            for point, plan in zip(points, plans, strict=True):
                planned = self.s.ik.forward_position_mm(plan.joints)
                if (self.s.in_zone(point[:2]) or self.s.in_zone(planned[:2])) and planned[2] < safe_z:
                    return self._result(
                        False, action, "ik_gate",
                        f"Planned carry enters zone below clearance: "
                        f"planned_z={planned[2]:.1f}mm required_z={safe_z:.1f}mm",
                        retry_advice="try_other_target",
                        target_mm=list(point), planned_fk_mm=list(planned),
                    )
        self._contact = False
        self._zone_drop_ready = False
        deadline = time.monotonic() + self.cfg.motion.move_timeout_s
        descending_empty = xyz[2] < start[2] and self.s.held is None
        if descending_empty:
            # Near-table descent retains its existing guarded, bounded steps.
            deadline = time.monotonic() + self.cfg.motion.move_timeout_s
            for point, plan in zip(points, plans):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Primitive move deadline reached")
                _, blocked = self.s.player.descend(plan.joints)
                if blocked:
                    return self._fail(action, "Empty approach descent stopped short; reobserve before closing", "grasp_blocked")
                if math.dist(self.s.arm_position_mm(), point) > self.limits.arrival_error_mm:
                    raise TimeoutError("Measured FK did not reach primitive waypoint")
        else:
            # The complete joint path was solved and gated before motion.
            # Its measured FK trace is not a Cartesian straight line, so a
            # straight-corridor or mid-flight Z test rejects valid diagonal
            # carries. Keep motor tick limits and verify the endpoint below.
            self.s.player.move_through(
                [plan.joints for plan in plans],
                tol=(self.cfg.motion.transit_arrival_tol if self.limits.calibrated_pick
                     else self.cfg.motion.arrival_tol),
                timeout_s=deadline-time.monotonic(),
            )
            if self.limits.calibrated_pick:
                self.s.player.settle(plans[-1].joints, tol=self.cfg.motion.arrival_tol,
                                     timeout_s=min(self.cfg.motion.grasp_hover_settle_s,max(0.0,deadline-time.monotonic())))
            if self.s.held is not None and not self._held_check():
                return self._fail(
                    action,
                    "Grasp verification failed after transit; do not descend",
                    "grasp_lost",
                )
            correction = {}
            if self.limits.calibrated_pick and self.s.held is None:
                # Empty-arm hover correction is bounded and measured against
                # the intended grasp pose. Under payload, the same joint-space
                # feedforward can move Cartesian Z in the wrong direction near
                # the far-workspace singularity; loaded moves keep the gated
                # path and endpoint check without this experimental correction.
                nominal = plans[-1].joints
                measured = self.s.robot.read_joints()
                errors = {j: nominal[j]-measured[j] for j in nominal}
                if max(map(abs, errors.values())) > self.cfg.motion.descent_max_lag:
                    raise TimeoutError("Transit tracking error exceeds correction bound")
                if max(map(abs, errors.values())) > self.cfg.motion.grasp_hover_arrival_tol:
                    bound = self.cfg.agent.calibration_clearance.hover_correction_max_deg
                    correction = {j: max(-bound,min(bound,e)) for j,e in errors.items()}
                    corrected = {j: nominal[j]+correction[j] for j in nominal}
                    if (corrected["wrist_roll"] < self.cfg.agent.calibration_clearance.wrist_roll_min_deg
                            or abs(corrected["wrist_roll"]) > self.limits.wrist_roll_limit_deg):
                        raise TimeoutError("Transit correction exceeds wrist limit")
                    # Reading all motor loads takes about 0.22 s on this bus.
                    # This correction gates on joint lag, not load; skip the
                    # diagnostic read while a 30 Hz episode is being recorded.
                    recording = self.collection.recording
                    baseline_load = (None if recording else
                                     ContactMonitor(self.s.robot, self.cfg.sensing).start())
                    def check_correction():
                        loads = None if recording else self.s.robot.read_loads()
                        q = self.s.robot.read_joints()
                        # Loaded upward/transit motion naturally raises holding
                        # torque. A descent's contact delta is not a jam test
                        # here; retain bounded following error and path guards.
                        if max(abs(q[j]-corrected[j]) for j in corrected) > self.cfg.motion.descent_max_lag:
                            self._calibration()._record("transit_correction_stop", correction_deg=correction,
                                baseline_load=baseline_load, loads=loads, nominal=nominal, corrected=corrected)
                            self.s.robot.send_joints({j:q[j] for j in corrected})
                            raise TimeoutError("Transit correction tracking lag")
                    self.s.player.move_through([corrected], tol=self.cfg.motion.transit_arrival_tol,
                                               check_progress=check_correction,
                                               timeout_s=deadline-time.monotonic(),
                                               max_step=self.cfg.motion.descent_step_per_tick)
                    # A single bounded correction, never an accumulating loop.
                    self.s.player.settle(corrected, tol=self.cfg.motion.arrival_tol,
                                         timeout_s=min(self.cfg.motion.grasp_hover_settle_s,max(0.0,deadline-time.monotonic())),
                                         check_progress=check_correction)
                    check_correction()
                    if not recording:
                        self._calibration()._record("transit_tracking_corrected", correction_deg=correction,
                                                    nominal=nominal, corrected=corrected, target_mm=list(xyz))
            endpoint_error_mm = math.dist(self.s.arm_position_mm(), xyz)
            endpoint_limit_mm = max(
                self.limits.arrival_error_mm,
                max_ik_error_mm if max_ik_error_mm is not None else 0.0,
                self.limits.loaded_arrival_error_mm if self.s.held is not None else 0.0,
            )
            if endpoint_error_mm > endpoint_limit_mm:
                detail = (
                    f"Measured FK missed primitive endpoint: "
                    f"error={endpoint_error_mm:.1f}mm limit={endpoint_limit_mm:.1f}mm"
                )
                if self.s.held is not None:
                    # Motion completed and the grasp is still verified. This is
                    # a failed placement, not a bus/trajectory fault: let the
                    # composite set the block down and try another candidate.
                    return self._result(
                        False, action, "grasp_blocked", detail,
                        retry_advice="try_other_target",
                        target_mm=list(xyz),
                        planned_fk_mm=list(self.s.ik.forward_position_mm(plans[-1].joints)),
                        measured_fk_mm=list(self.s.arm_position_mm()),
                    )
                raise TimeoutError(detail)
            if place_yaw_deg is not None:
                actual_yaw = self.s.ik.forward_yaw_deg(self.s.robot.read_joints())
                if abs(angle_error_deg(actual_yaw, place_yaw_deg)) > self.cfg.task1.place_yaw_tolerance_deg:
                    return self._fail(action, "Measured jaw yaw missed placement alignment; still holding", "grasp_blocked")
        if self.s.held is not None:
            self.s.held.over_xy_mm = tuple(self.s.arm_position_mm()[:2])
        return self._result(True, action, "moved", commanded_mm=list(xyz),
                            measured_fk_mm=list(self.s.arm_position_mm()),
                            tracking_correction_deg={} if descending_empty else correction)

    def move_to_target(self, target_type, phase, object_id=None, observation_id=None, slot=None, x=None, y=None,
                       _route_guard=None):
        action = "move_to_target"
        try:
            if target_type == "object":
                if slot is not None or x is not None or y is not None:
                    raise ValueError("Object targets accept only object_id and observation_id")
                block = self._object(object_id, observation_id)
                xy = block.center_mm
                if self.s.held is not None and self.s.held.color == block.color:
                    raise ValueError("Cannot place onto the held object")
            elif target_type == "cell":
                if object_id is not None or observation_id is not None or slot is not None or x is None or y is None:
                    raise ValueError("Cell targets require x,y only")
                xy = self.cells.get((x, y))
                if xy is None:
                    raise ValueError("Cell is not addressable")
            elif target_type == "slot":
                if object_id is not None or observation_id is not None or x is not None or y is not None:
                    raise ValueError("Slot targets require slot only")
                slot_index = list(self.cfg.agent.zone_slots.labels).index(slot)
                xy = self.s.slot_centres[slot_index]
            else:
                raise ValueError("Unknown target type")
        except ValueError as exc:
            return self._fail(action, str(exc), "invalid_arguments")
        if phase not in ("pregrasp", "grasp", "preplace", "hover"):
            return self._fail(action, "Unknown phase", "invalid_arguments")
        if phase == "preplace" and self.s.held is None:
            return self._fail(action, "Close and verify grasp before preplace", "no_block_held")
        if phase == "preplace" and target_type == "slot" and self._observed_scene is not None:
            occupant = self._observed_scene.slot_occupancy.get(slot_index)
            held_color = self.s.held.color
            if occupant is not None and occupant != held_color:
                free_slots = [
                    self.cfg.agent.zone_slots.labels[i]
                    for i, color in sorted(self._observed_scene.slot_occupancy.items())
                    if color is None
                ]
                return self._result(
                    False, action, "slot_occupied",
                    f"Slot {slot} is occupied by {occupant}",
                    retry_advice="retry_ok", free_slots=free_slots,
                )
        if phase in ("pregrasp", "grasp") and (self.s.held is not None or target_type != "object"):
            return self._fail(action, "Grasp phases require an object target and empty gripper")
        xyz = self.s.arm_position_mm()
        nominal_xy = xy
        if phase == "grasp":
            if self._target != (target_type, object_id, observation_id, slot, x, y, "pregrasp"):
                return self._fail(action, "Approach this target first")
            if self.limits.calibrated_pick:
                # A tilted approach deliberately hovers inward of the block.
                # Check drift from that preflighted hover, not from its grasp
                # centre; the latter can differ by more than the offset limit.
                attempt = self._pick_calibration.attempt if self._pick_calibration else None
                if attempt is None:
                    return self._fail(action, "Calibrated pregrasp is no longer valid")
                hover_xy = attempt.hover_xy_mm or attempt.xy_mm
                if math.dist(xyz[:2], hover_xy) > self.cfg.agent.relative.max_pick_offset_mm:
                    return self._fail(action, "Measured hover drift exceeds grasp offset limit")
            elif math.dist(xyz[:2], xy) > self.cfg.agent.relative.max_pick_offset_mm:
                return self._fail(action, "Correction exceeds grasp offset limit")
            goal = (*xyz[:2], self.s.grasp_z_mm)  # preserve intentional relative correction
        else:
            goal = (*xy, self.s.grasp_z_mm + self.limits.approach_clearance_mm)
        if self.limits.calibrated_pick and phase in ("pregrasp", "grasp"):
            result = self._calibrated_target(block, phase, route_guard=_route_guard)
        elif phase == "preplace":
            from control.task1_transport import place_tilt_deg
            from session.relative import offset_xy
            place_tilt = place_tilt_deg(nominal_xy, self.s.base_xy, self.cfg)
            enabled = self.cfg.agent.place_correction.enabled
            scales = (1.0, 0.75, 0.5, 0.25, 0.0) if enabled else (0.0,)
            attempts = []
            for correction_scale in scales:
                if enabled:
                    xy = offset_xy(
                        nominal_xy,
                        self.place_correction.forward_mm * correction_scale,
                        self.place_correction.left_mm * correction_scale,
                        frame=self.cfg.agent.relative.frame,
                        base_xy_mm=self.s.base_xy,
                    )
                else:
                    xy = nominal_xy
                goal = (*xy, self.s.grasp_z_mm + self.limits.approach_clearance_mm)
                try:
                    place_yaw, zone_aligned = self._choose_place_yaw(goal, place_tilt)
                except ValueError as exc:
                    result = self._fail(action, str(exc), "ik_gate")
                else:
                    result = self._move(
                        action,
                        goal,
                        radial_tilt_deg=place_tilt,
                        max_ik_error_mm=self.cfg.ik.max_position_error_mm,
                        max_tilt_error_deg=self.cfg.task1.place_level_tolerance_deg,
                        level_during_carry=True,
                        place_yaw_deg=place_yaw,
                    )
                attempts.append({
                    "scale": correction_scale,
                    "xy_mm": [round(value, 1) for value in xy],
                    "reason": result.reason,
                })
                # An IK-gate failure happens before motion because _move
                # solves the complete path first. Other failures may follow
                # physical motion and must never be retried automatically.
                if result.ok or result.reason != "ik_gate":
                    break
            result.data["radial_tilt_deg"] = place_tilt
            if result.ok:
                self._place_yaw_deg = place_yaw
                result.data["place_yaw_deg"] = round(place_yaw, 1)
                result.data["zone_alignment_fallback"] = not zone_aligned
                if zone_aligned:
                    result.data["zone_aligned_yaw_deg"] = round(place_yaw, 1)
            result.data["nominal_xy_mm"] = list(nominal_xy)
            result.data["command_xy_mm"] = list(xy)
            result.data["place_correction"] = self.place_correction.as_dict()
            result.data["place_correction_scale"] = correction_scale
            result.data["place_correction_attempts"] = attempts
        else:
            result = self._move(action, goal)
        if result.ok:
            self._target = (target_type, object_id, observation_id, slot, x, y, phase)
            result.data["collection_zone_destination"] = phase == "preplace" and target_type != "object" and self.s.in_zone(xy)
            if phase == "grasp":
                result.data["gripper_intentionally_open"] = True
                result.data["next_required_action"] = "close_gripper"
        return result

    def _lift_held_joint_space(self):
        """Follow the cached grasp path only until the first 30 mm lift."""
        action = "lift_held"
        held = self.s.held
        if held is None or not self._held_check():
            return self._fail(action, "Verified held block required", "no_block_held")
        joints = self.s.robot.read_joints()
        origin = self.s.ik.forward_position_mm(joints)
        hover_goal = held.attempt.hover.joints
        target_z = self.s.grasp_z_mm + self.cfg.task1.tilted_pick_hover_clearance_mm
        tolerance = self.limits.lateral_clearance_tolerance_mm
        if held.attempt.hover_z_mm <= origin[2] + self.limits.contact_step_mm:
            # Plain primitive picks have no cached descent path. Use the
            # existing bounded vertical jog for their first 30 mm retreat.
            moved = self.move_relative(
                up_mm=self.cfg.task1.tilted_pick_hover_clearance_mm)
            if not moved.ok:
                return moved
            actual = self.s.arm_position_mm()
            held.over_xy_mm = actual[:2]
            return self._result(
                True, action, "moved", measured_fk_mm=list(actual),
                initial_lift_mm=round(actual[2] - self.s.grasp_z_mm, 1),
                requested_initial_lift_mm=self.cfg.task1.tilted_pick_hover_clearance_mm)
        zone_safe_z = (self.s.grasp_z_mm + self.limits.lateral_clearance_mm
                       - tolerance)
        trace = []
        goal = None
        for step in interpolate(joints, hover_goal,
                                self.cfg.motion.descent_step_per_tick):
            point = self.s.ik.forward_position_mm({**joints, **step})
            trace.append(point)
            goal = step
            if point[2] >= target_z - tolerance:
                break
        previous_z = origin[2]
        for point in trace:
            if (point[2] < previous_z - self.limits.contact_step_mm
                    or point[2] > target_z + tolerance
                    or (self.s.in_zone(point[:2]) and point[2] < zone_safe_z)
                    or not self.s.in_workspace(point[:2])):
                return self._result(
                    False, action, "grasp_blocked",
                    "Cached pick retreat failed measured-path preflight",
                    retry_advice="try_other_target")
            previous_z = point[2]
        if not trace or goal is None:
            return self._result(
                False, action, "grasp_blocked", "Cached pick retreat is empty",
                retry_advice="try_other_target")
        self.s.player.move_to(goal,
                              max_step=self.cfg.motion.descent_step_per_tick,
                              tol=self.cfg.motion.transit_arrival_tol)
        time.sleep(self.cfg.motion.descent_settle_s)
        actual = self.s.arm_position_mm()
        if not self._held_check():
            return self._result(
                False, action, "grasp_blocked",
                "Gripper position closed after retreat; grasp may be lost",
                retry_advice="ask_operator", measured_fk_mm=list(actual))
        if actual[2] < origin[2] + self.limits.contact_step_mm:
            return self._result(
                False, action, "grasp_blocked",
                f"Retreat did not raise measured FK: {origin[2]:.1f}->{actual[2]:.1f}mm",
                retry_advice="try_other_target", measured_fk_mm=list(actual))
        held.over_xy_mm = actual[:2]
        # The next stage raises to 38 mm before entering the zone. Do not
        # reject a grasp just because motor lag left this first retreat short.
        return self._result(
            True, action, "moved", measured_fk_mm=list(actual),
            initial_lift_mm=round(actual[2] - self.s.grasp_z_mm, 1),
            requested_initial_lift_mm=self.cfg.task1.tilted_pick_hover_clearance_mm)

    def _recover_loaded_reverse_lift(self, required_z, *, max_command_z_mm=None):
        """Continue a short loaded retreat inward and upward as one IK path."""
        self._loaded_lift_diagnostic = ""
        if self.s.held is None or not self._held_check():
            self._loaded_lift_diagnostic = "grasp no longer verified"
            return False
        measured = self.s.robot.read_joints()
        origin = self.s.ik.forward_position_mm(measured)
        radius = math.dist(origin[:2], self.s.base_xy)
        if radius <= 0 or origin[2] >= required_z:
            return origin[2] >= required_z
        inward = tuple((base - value) / radius
                       for value, base in zip(origin[:2], self.s.base_xy))
        max_inward = self.cfg.agent.relative.max_jog_mm
        target_z = min(self.cfg.agent.relative.jog_max_z_mm,
                       required_z + min(required_z - origin[2],
                                        self.limits.loaded_arrival_error_mm))
        if max_command_z_mm is not None:
            target_z = min(target_z, max_command_z_mm)
        heights = (target_z, (target_z + required_z) / 2, required_z)
        zone_safe_z = (self.s.grasp_z_mm + self.limits.lateral_clearance_mm
                       - self.limits.lateral_clearance_tolerance_mm)
        scene = self._observed_scene
        obstacles = ([block for block in scene.all() if block.color != self.s.held.color]
                     if scene is not None else [])
        plan = None
        rejected_ik = rejected_trace = 0
        self._loaded_lift_diagnostic = ""
        for distance in (max_inward, max_inward / 2, 0.0):
            xy = tuple(value + axis * distance
                       for value, axis in zip(origin[:2], inward))
            for height in heights:
                try:
                    candidate = self._solve((*xy, height))
                except ValueError:
                    rejected_ik += 1
                    continue
                trace = [self.s.ik.forward_position_mm({**measured, **step})
                         for step in interpolate(measured, candidate.joints,
                                                 self.cfg.motion.descent_step_per_tick)]
                previous_z = origin[2]
                safe = bool(trace)
                for point in trace:
                    if (point[2] < previous_z - self.limits.contact_step_mm
                            or point[2] > self.cfg.agent.relative.jog_max_z_mm
                            or (max_command_z_mm is not None
                                and point[2] > max_command_z_mm
                                    + self.limits.lateral_clearance_tolerance_mm)
                            or (self.s.in_zone(point[:2]) and point[2] < zone_safe_z)
                            or not self.s.in_workspace(point[:2])
                            or math.dist(point[:2], origin[:2]) > max_inward):
                        safe = False
                        break
                    if point[2] < required_z and any(
                        math.dist(point[:2], block.center_mm) < self.cfg.agent.place_clear_radius_mm
                        and math.dist(point[:2], block.center_mm)
                            < math.dist(origin[:2], block.center_mm) - self.limits.contact_step_mm
                        for block in obstacles
                    ):
                        safe = False
                        break
                    previous_z = point[2]
                if safe and trace[-1][2] >= required_z:
                    plan = candidate
                    break
                rejected_trace += 1
            if plan is not None:
                break
        if plan is None:
            self._loaded_lift_diagnostic = (
                f"no safe inward/upward IK path; start_z={origin[2]:.1f}mm "
                f"required_z={required_z:.1f}mm ik_rejected={rejected_ik} "
                f"trace_rejected={rejected_trace}")
            return False
        self.s.player.move_to(plan.joints,
                              max_step=self.cfg.motion.descent_step_per_tick,
                              tol=self.cfg.motion.transit_arrival_tol)
        time.sleep(self.cfg.motion.descent_settle_s)
        actual = self.s.arm_position_mm()
        held_ok = self._held_check()
        inward_distance = math.dist(actual[:2], origin[:2])
        if (not held_ok
                or actual[2] < origin[2] - self.limits.contact_step_mm
                or not self.s.in_workspace(actual[:2])
                or (self.s.in_zone(actual[:2]) and actual[2] < zone_safe_z)):
            # The command path already has a bounded inward sweep. After
            # motion, judge the measured pose by clearance and workspace;
            # servo lag can change the displacement by a few millimetres.
            # Shoulder load changes with posture during an upward carry.
            # It is not contact evidence here; the measured FK and verified
            # grasp still bound this recovery motion.
            self._loaded_lift_diagnostic = (
                f"held={held_ok} start_z={origin[2]:.1f}mm "
                f"measured_z={actual[2]:.1f}mm inward={inward_distance:.1f}mm "
                f"limit={max_inward:.1f}mm"
            )
            return False
        if actual[2] >= required_z:
            self.s.held.over_xy_mm = actual[:2]
            return True
        self._loaded_lift_diagnostic = (
            f"measured_z={actual[2]:.1f}mm required_z={required_z:.1f}mm")
        return False

    def move_relative(self, forward_mm=0.0, left_mm=0.0, up_mm=0.0):
        from session.relative import offset_xy
        if not all(math.isfinite(v) for v in (forward_mm,left_mm,up_mm)) or math.sqrt(forward_mm**2+left_mm**2+up_mm**2) > self.cfg.agent.relative.max_jog_mm:
            return self._fail("move_relative", "Relative vector exceeds limit", "invalid_arguments")
        if self.s.held is not None and up_mm < 0:
            return self._fail("move_relative", "Use the guarded placement descent while holding")
        if (self.limits.calibrated_pick and self._target and self._target[-1] == "pregrasp"
                and up_mm == 0 and self._pick_calibration is not None):
            self._pick_ready = False
            return replace(self._pick_calibration.calibration_adjust(forward_mm, left_mm),
                           action="move_relative")
        xyz = self.s.arm_position_mm()
        # A calibrated tilted grasp has a preflighted reverse approach path.
        # Use it to clear the block before an ordinary upward jog; fixed-XY
        # lift at the far reach can miss the strict 5 mm jog IK gate.
        if (self.s.held is not None and up_mm > 0 and forward_mm == left_mm == 0
                and abs(self._held_radial_tilt_deg) >= 5.0):
            retreat = self.s.held.attempt
            clear_z = self.s.grasp_z_mm + self.limits.lateral_clearance_mm
            first_retreat_z = (self.s.grasp_z_mm
                               + self.cfg.task1.tilted_pick_hover_clearance_mm)
            hover_xy = retreat.hover_xy_mm
            if (hover_xy is not None and xyz[2] < clear_z
                    and retreat.hover_z_mm > xyz[2]
                    and retreat.hover_z_mm <= xyz[2] + up_mm + 2.0
                    and math.dist(xyz, (*retreat.xy_mm, retreat.grasp_z_mm))
                    <= self.cfg.agent.relative.max_pick_offset_mm):
                if not self._held_check():
                    return self._fail("move_relative", "Grasp verification failed", "grasp_lost")
                measured_joints = self.s.robot.read_joints()
                steps = interpolate(measured_joints, retreat.hover.joints,
                                    self.cfg.motion.descent_step_per_tick)
                trace = [self.s.ik.forward_position_mm({**measured_joints, **step})
                         for step in steps]
                if (not trace or any(point[2] < xyz[2] - 2.0
                                     or point[2] > self.cfg.agent.relative.jog_max_z_mm
                                     or not self.s.in_workspace(point[:2])
                                     for point in trace)):
                    return self._fail("move_relative", "Reverse grasp path failed clearance preflight", "ik_gate")
                self._invalidate_pick()
                self.s.player.move_to(retreat.hover.joints,
                                      max_step=self.cfg.motion.descent_step_per_tick,
                                      tol=self.cfg.motion.transit_arrival_tol)
                if not self._held_check():
                    return self._fail("move_relative", "Grasp lost during reverse approach", "grasp_lost")
                reached = self.s.arm_position_mm()
                if reached[2] + self.limits.lateral_clearance_tolerance_mm < first_retreat_z:
                    if not self._recover_loaded_reverse_lift(
                            first_retreat_z - self.limits.lateral_clearance_tolerance_mm,
                            max_command_z_mm=first_retreat_z):
                        actual_z = self.s.arm_position_mm()[2]
                        return self._fail(
                            "move_relative",
                            f"Initial loaded retreat stayed below 30 mm clearance: "
                            f"{actual_z:.1f}/{first_retreat_z:.1f}mm; "
                            + self._loaded_lift_diagnostic,
                            "grasp_blocked")
                    reached = self.s.arm_position_mm()
                self.s.held.over_xy_mm = reached[:2]
                if xyz[2] + up_mm <= reached[2] + 1.0:
                    return self._result(True, "move_relative", "moved",
                                        measured_fk_mm=list(reached), reverse_pick_retreat=True,
                                        lateral_clearance_ready=self._lateral_clearance_ready(reached[2]),
                                        next_required_action=(
                                            "move_to_placement_target"
                                            if self._lateral_clearance_ready(reached[2])
                                            else "raise_before_zone"))
                result = self._move("move_relative", (*reached[:2], xyz[2] + up_mm),
                                    vertical_only=True)
                result.data["reverse_pick_retreat"] = True
                result.data["lateral_clearance_ready"] = self._lateral_clearance_ready(
                    self.s.arm_position_mm()[2])
                result.data["next_required_action"] = (
                    "move_to_placement_target" if result.data["lateral_clearance_ready"]
                    else "lift_vertically_again"
                )
                return result
        xy = offset_xy(xyz[:2], forward_mm, left_mm, frame=self.cfg.agent.relative.frame, base_xy_mm=self.s.base_xy)
        result = self._move("move_relative", (*xy, xyz[2]+up_mm),
                            vertical_only=(forward_mm == 0 and left_mm == 0))
        if self.s.held is not None and up_mm > 0:
            measured_z = self.s.arm_position_mm()[2]
            required_z = self.s.grasp_z_mm + self.limits.lateral_clearance_mm
            ready = self._lateral_clearance_ready(measured_z)
            result.data["required_lateral_clearance_z_mm"] = round(required_z, 1)
            result.data["lateral_clearance_ready"] = ready
            result.data["next_required_action"] = (
                "move_to_placement_target" if ready else "lift_vertically_again"
            )
        return result

    def align_gripper(self, object_id, observation_id):
        try:
            block = self._object(object_id, observation_id)
        except ValueError as exc:
            return self._fail("align_gripper", str(exc))
        if self.s.held is not None:
            return self._fail("align_gripper", "Alignment requires empty gripper")
        if self.limits.calibrated_pick:
            cal = self._pick_calibration
            if (self._target and self._target[1:3] == (object_id, observation_id)
                    and self._target[-1] == "pregrasp" and cal is not None and cal.attempt is not None):
                # The selected collision-checked grasp already includes alignment.
                return self._result(True, "align_gripper", "moved", calibrated_pick=True,
                                    alignment_already_applied=True)
            return self._fail("align_gripper", "Approach the calibrated object first")
        xyz = self.s.arm_position_mm()
        if not self._lateral_clearance_ready(xyz[2]):
            return self._fail("align_gripper", "Lift before alignment")
        yaw, _ = self.s.ik.grasp_yaw_and_rotation_deg(*xyz, block.angle_deg)
        plan = self.s.ik.solve(*xyz, yaw_deg=yaw)
        if plan.position_error_mm > self.cfg.agent.relative.jog_max_ik_error_mm or abs(plan.joints["wrist_roll"]) > self.limits.wrist_roll_limit_deg:
            return self._fail("align_gripper", "Alignment failed IK/wrist gate", "ik_gate")
        self._contact = False
        self.s.player.move_to(plan.joints)
        return self._result(True, "align_gripper", "moved")

    def close_gripper(self):
        if self.s.held is not None:
            return self._fail("close_gripper", "Already holding", "already_holding")
        if self.limits.calibrated_pick and not self._pick_ready:
            return self._fail("close_gripper", "Complete calibrated grasp descent before closing")
        self._pick_ready = False
        self._contact = False
        self.s.motion.close_gripper()
        check = check_grasp(self.s.robot, self.cfg.sensing)
        self._grasp_failed = not check.grasped
        if not check.grasped:
            return self._result(False, "close_gripper", "grasp_empty", grasp=asdict(check), retry_advice="retry_ok")
        xyz = self.s.arm_position_mm()
        plan = self.s.ik.solve_holding_wrist_roll(*xyz, wrist_roll_deg=self.s.robot.read_joints()["wrist_roll"])
        color = None
        block_angle = None
        if self._target and self._target[-1] == "grasp":
            try:
                block = self._object(self._target[1], self._target[2])
                # A calibrated grasp intentionally offsets the URDF TCP from
                # the physical jaw centre. The gated target/observation pair
                # remains authoritative until close_gripper; comparing TCP XY
                # with the visual centre incorrectly discards that identity.
                if self.limits.calibrated_pick or math.dist(
                    xyz[:2], block.center_mm
                ) <= self.cfg.agent.relative.max_pick_offset_mm:
                    color = block.color
                    block_angle = block.angle_deg
            except ValueError:
                pass
        cal = self._pick_calibration
        attempt = (cal.attempt if self.limits.calibrated_pick and cal is not None
                   and cal.attempt is not None else
                   GraspAttempt("primitive", (0.,0.), xyz[:2], plan, plan, True, xyz[2], xyz[2]))
        # Carry the calibrated pick's approach tilt into lift/transport IK.
        # Resetting to top-down can make a reachable lift fail its IK gate.
        cal = self._pick_calibration
        self._held_radial_tilt_deg = (cal.plan.radial_tilt_deg
            if self.limits.calibrated_pick and cal is not None and cal.plan is not None else 0.0)
        self.s.held = HeldBlock(color, attempt, xyz[:2], self.s.in_zone(xyz[:2]), xyz[:2])
        self._held_block_angle_deg = block_angle
        # Pair the observed block angle with measured jaw yaw at the instant
        # the grasp closes; commanded yaw can differ under load.
        self._held_pick_yaw_deg = self.s.ik.forward_yaw_deg(self.s.robot.read_joints())
        self._place_yaw_deg = None
        self._target = None
        return self._result(
            True, "close_gripper", "held", grasp=asdict(check),
            identity_confirmed=color is not None, associated_color=color,
            next_required_action="move_relative_up",
        )

    def _zone_drop_target_xy(self):
        target = self._target
        if target is None or target[-1] != "preplace":
            return None
        if target[0] == "slot":
            return self.s.slot_centres[list(self.cfg.agent.zone_slots.labels).index(target[3])]
        if target[0] == "cell":
            point = self.cells.get((target[4], target[5]))
            return point if point is not None and self.s.in_zone(point) else None
        return None

    def drop_at_zone_target(self):
        """Release-ready descent above a zone slot/cell, without contact sensing."""
        action = "drop_at_zone_target"
        if self.s.held is None or not self._held_check():
            return self._fail(action, "Verified held block required", "no_block_held")
        if self._zone_drop_target_xy() is None:
            return self._fail(action, "Approach a zone slot or cell first")
        if self._place_yaw_deg is None:
            return self._fail(action, "Placement yaw was not aligned at hover")
        start = self.s.arm_position_mm()
        retreat_joints = self.s.robot.read_joints()
        drop_z = self.s.drop_z_mm
        if start[2] < drop_z:
            return self._fail(action, "Arm is already below the zone drop height", "grasp_blocked")
        if start[2] - drop_z > self.limits.contact_max_descent_mm:
            return self._fail(action, "Zone drop exceeds the bounded descent", "limit_exceeded")
        from control.task1_transport import place_tilt_deg
        tilt = place_tilt_deg(start[:2], self.s.base_xy, self.cfg)
        try:
            pose = self._solve(
                (*start[:2], drop_z),
                radial_tilt_deg=tilt,
                max_position_error_mm=self.cfg.ik.max_position_error_mm,
                max_tilt_error_deg=self.cfg.task1.place_level_tolerance_deg,
                yaw_deg=self._place_yaw_deg,
                max_yaw_error_deg=self.cfg.task1.place_yaw_tolerance_deg,
            )
        except ValueError as exc:
            return self._fail(action, str(exc), "ik_gate")
        self._zone_drop_ready = False
        def in_release_window(pose):
            return (math.dist(pose[:2], start[:2]) <= self.limits.alignment_tolerance_mm
                    and self.s.grasp_z_mm + self.limits.zone_release_floor_margin_mm
                    <= pose[2] <= drop_z + 10)
        try:
            self.s.player.move_to(
                pose.joints, max_step=self.cfg.motion.descent_step_per_tick,
                tol=self.cfg.motion.transit_arrival_tol,
            )
        except TimeoutError:
            # A loaded joint can stay outside angle tolerance after the
            # block has reached the measured release column. Check the pose.
            if not in_release_window(self.s.arm_position_mm()):
                raise
        actual = self.s.arm_position_mm()
        first_release_fk = actual
        # The joint tolerance can accept several degrees of loaded shoulder
        # sag. If that consumes over half the intended drop clearance, make
        # one measured upward correction before opening the gripper.
        if (actual[2] < self.s.grasp_z_mm + self.cfg.task1.release_clearance_mm / 2
                and math.dist(actual[:2], start[:2]) <= self.limits.alignment_tolerance_mm):
            correction_z = min(start[2], drop_z + max(0.0, drop_z - actual[2]))
            try:
                correction = self._solve(
                    (*start[:2], correction_z), radial_tilt_deg=tilt,
                    max_position_error_mm=self.cfg.ik.max_position_error_mm,
                    max_tilt_error_deg=self.cfg.task1.place_level_tolerance_deg,
                    yaw_deg=self._place_yaw_deg,
                    max_yaw_error_deg=self.cfg.task1.place_yaw_tolerance_deg,
                )
                joints = self.s.robot.read_joints()
                trace = [self.s.ik.forward_position_mm({**joints, **step})
                         for step in interpolate(joints, correction.joints,
                                                 self.cfg.motion.descent_step_per_tick)]
                if (not trace or min(point[2] for point in trace) < actual[2] - 1
                        or any(math.dist(point[:2], start[:2]) > self.limits.alignment_tolerance_mm
                               for point in trace)):
                    raise ValueError("upward correction path leaves release column")
            except ValueError:
                pass  # still allow a bounded release if the measured pose is safe
            else:
                try:
                    self.s.player.move_to(
                        correction.joints, max_step=self.cfg.motion.descent_step_per_tick,
                        tol=self.cfg.motion.transit_arrival_tol,
                    )
                except TimeoutError:
                    if not in_release_window(self.s.arm_position_mm()):
                        raise
                actual = self.s.arm_position_mm()

        if not in_release_window(actual):
            time.sleep(self.cfg.motion.descent_settle_s)
            actual = self.s.arm_position_mm()
        if not in_release_window(actual):
            # The motor command finished; only the release pose is unverified.
            # Keep holding so a later recovery can lift or set down the block.
            return self._result(False, action, "grasp_blocked",
                                "Measured FK missed the zone release window",
                                retry_advice="retry_ok", release_target_z_mm=drop_z,
                                measured_fk_mm=list(actual))
        if not self._held_check():
            return self._fail(action, "Grasp verification failed", "grasp_empty")
        self._zone_drop_xy = tuple(start[:2])
        self._zone_retreat_joints = retreat_joints
        self._zone_drop_ready = True
        return self._result(True, action, "ok", release_mode="height_drop",
                            release_fk_mm=list(actual), drop_z_mm=drop_z,
                            first_release_fk_mm=list(first_release_fk))

    def descend_until_contact(self, max_descent_mm):
        action = "descend_until_contact"
        if not math.isfinite(max_descent_mm) or not 0 < max_descent_mm <= self.limits.contact_max_descent_mm:
            return self._fail(action, "Descent bound invalid", "invalid_arguments")
        if self._zone_drop_target_xy() is not None:
            if self.s.arm_position_mm()[2] - self.s.drop_z_mm > max_descent_mm:
                return self._fail(action, "Zone drop exceeds requested descent bound", "limit_exceeded")
            return replace(self.drop_at_zone_target(), action=action)
        if self.s.held is None or not self._held_check():
            return self._fail(action, "Verified held block required", "no_block_held")
        if not self._target or self._target[-1] != "preplace":
            return self._fail(action, "Approach a placement target first")
        try:
            if self._target[0] == "object":
                block = self._object(self._target[1], self._target[2])
                target_xy = block.center_mm
            elif self._target[0] == "slot":
                target_xy = self.s.slot_centres[list(self.cfg.agent.zone_slots.labels).index(self._target[3])]
            elif self._target[0] == "stack" and self._stack_target is not None:
                target_xy = self._stack_target.xy_mm
            elif self._target[0] == "recovery" and self._recovery_target_xy is not None:
                target_xy = self._recovery_target_xy
            else:
                target_xy = self.cells[(self._target[4], self._target[5])]
        except (ValueError, KeyError) as exc:
            return self._fail(action, str(exc))
        self._contact = False
        place_yaw = (self._place_yaw_deg if self._target[0] in ("slot", "cell", "object") else None)
        if self._target[0] in ("slot", "cell", "object") and place_yaw is None:
            return self._fail(action, "Placement yaw was not aligned at hover", "precondition")
        start = self.s.arm_position_mm()
        if math.dist(start[:2], target_xy) > self.limits.alignment_tolerance_mm:
            return self._fail(action, "Gripper is outside target alignment tolerance")
        stack_level = self._stack_target if self._target[0] == "stack" else None
        floor_z = (max(self.s.grasp_z_mm, stack_level.floor_z_mm)
                   if stack_level is not None else self.s.grasp_z_mm)
        distance = min(max_descent_mm, max(0.0, start[2] - floor_z))
        count = max(1, math.ceil(distance/self.limits.contact_step_mm))
        from control.task1_transport import place_tilt_deg
        place_tilt = (stack_level.radial_tilt_deg if stack_level is not None
                      else place_tilt_deg(target_xy, self.s.base_xy, self.cfg))
        try:
            plans = [
                self._solve(
                    (*start[:2], start[2] - distance * i / count),
                    radial_tilt_deg=place_tilt,
                    max_position_error_mm=self.cfg.ik.max_position_error_mm,
                    max_tilt_error_deg=self.cfg.task1.place_level_tolerance_deg,
                    yaw_deg=place_yaw,
                    max_yaw_error_deg=(self.cfg.task1.place_yaw_tolerance_deg
                                       if place_yaw is not None else None),
                )
                for i in range(1, count + 1)
            ]
        except ValueError as exc:
            return self._fail(action, str(exc), "ik_gate")
        monitor = ContactMonitor(self.s.robot, self.cfg.sensing, magnitude_increase=True)
        baseline_loads = monitor.start()
        max_load_deltas = {joint: 0.0 for joint in self.cfg.sensing.contact_joints}
        contact_samples = 0
        deadline = time.monotonic() + self.limits.contact_timeout_s
        for plan in plans:
            for joints in interpolate(self.s.robot.read_joints(), plan.joints, self.cfg.motion.descent_step_per_tick):
                self.s.cancel.raise_if_set()
                if time.monotonic() >= deadline:
                    measured = self.s.robot.read_joints()
                    self.s.robot.send_joints({j: measured[j] for j in plan.joints})
                    return self._fail(action, "Contact deadline reached; still holding")
                self.s.robot.send_joints(joints)
                if self.cfg.motion.fps > 0:
                    time.sleep(1/self.cfg.motion.fps)
                measured = self.s.robot.read_joints()
                reading = monitor.check()
                contact_samples += 1
                for joint, delta in reading.deltas.items():
                    max_load_deltas[joint] = max(max_load_deltas[joint], delta)
                tracking_error = max(abs(measured[j]-joints[j]) for j in joints)
                contact_source = (
                    "load" if reading.contact
                    else "lag" if tracking_error > self.cfg.motion.descent_max_lag
                    else None
                )
                if contact_source is None:
                    # Gravity load changes continuously with arm posture. Keep
                    # the reference local so that cumulative free-motion load
                    # is not mistaken for a collision later in the descent.
                    if all(joint in reading.loads for joint in self.cfg.sensing.contact_joints):
                        monitor.rebase(reading.loads)
                    continue
                self.s.robot.send_joints({j: measured[j] for j in plan.joints})
                actual = self.s.arm_position_mm()
                if stack_level is not None:
                    half_band = self.cfg.task2.block_height_mm / 2
                    if abs(actual[2] - stack_level.place_z_mm) > half_band:
                        return self._result(
                            False, action, "grasp_blocked",
                            "Contact was outside the planned tower level; still holding",
                            retry_advice="ask_operator", contact_source=contact_source,
                            level=stack_level.level, expected_contact_z_mm=stack_level.place_z_mm,
                            measured_fk_mm=list(actual), tracking_error_deg=tracking_error,
                            contact=asdict(reading),
                        )
                if (self._target[0] in ("slot", "cell", "recovery") and actual[2] > self.s.grasp_z_mm
                        + self.cfg.agent.calibration_clearance.obstacle_height_mm):
                    if contact_source == "load" and all(
                        joint in reading.loads for joint in self.cfg.sensing.contact_joints
                    ):
                        # The arm's gravity load can jump at intermediate poses.
                        # Ignore it only when complete samples let us move the
                        # baseline locally. An incomplete high-contact reading
                        # cannot be distinguished from an obstacle, so fail safe.
                        monitor.rebase(reading.loads)
                        continue
                    threshold = self.cfg.sensing.contact_load_delta
                    triggered = [joint for joint, delta in reading.deltas.items() if delta >= threshold]
                    return self._result(
                        False, action, "grasp_blocked",
                        "Tracking stopped above table; still holding. Reobserve before retry",
                        retry_advice="retry_ok", contact_source=contact_source,
                        contact=asdict(reading), baseline_loads=baseline_loads,
                        trigger_joints=triggered, contact_load_delta=threshold,
                        max_load_deltas=max_load_deltas, contact_samples=contact_samples,
                        measured_fk_mm=list(actual), tracking_error_deg=tracking_error,
                    )
                backoff = self._solve(
                    (*actual[:2], min(start[2], actual[2] + self.limits.contact_backoff_mm)),
                    radial_tilt_deg=place_tilt,
                    max_position_error_mm=self.cfg.ik.max_position_error_mm,
                )
                self.s.player.move_to(backoff.joints, tol=self.cfg.motion.transit_arrival_tol)
                self._contact = True
                return self._result(
                    True, action, "ok", contact=asdict(reading),
                    contact_source=contact_source, baseline_loads=baseline_loads,
                    trigger_joints=[joint for joint, delta in reading.deltas.items()
                                    if delta >= self.cfg.sensing.contact_load_delta],
                    contact_load_delta=self.cfg.sensing.contact_load_delta,
                    max_load_deltas=max_load_deltas, contact_samples=contact_samples,
                    contact_fk_mm=list(actual),
                    measured_fk_mm=list(self.s.arm_position_mm()),
                    tracking_error_deg=tracking_error, stack_verified=False,
                )
        measured = self.s.robot.read_joints()
        self.s.robot.send_joints({j: measured[j] for j in plans[-1].joints})
        actual = self.s.arm_position_mm()
        if (self._target[0] in ("slot", "cell", "recovery") or
                (stack_level is not None and stack_level.level == 1)) and (
                actual[2] <= self.s.grasp_z_mm + self.limits.arrival_error_mm):
            # Task 1 places on the known table plane. The load signal is noisy
            # enough to miss contact at z~=7mm and fire later at z~=15mm; once
            # the bounded descent reaches the calibrated floor band, authorise
            # release without repeating the same descent.
            backoff = self._solve(
                (*actual[:2], min(start[2], actual[2] + self.limits.contact_backoff_mm)),
                radial_tilt_deg=place_tilt,
                max_position_error_mm=self.cfg.ik.max_position_error_mm,
            )
            self.s.player.move_to(backoff.joints, tol=self.cfg.motion.transit_arrival_tol)
            self._contact = True
            return self._result(
                True, action, "ok", contact_source="calibrated_floor_bound",
                baseline_loads=baseline_loads, max_load_deltas=max_load_deltas,
                contact_samples=contact_samples, contact_fk_mm=list(actual),
                measured_fk_mm=list(self.s.arm_position_mm()),
                tracking_error_deg=0.0, stack_verified=False,
            )
        return self._fail(action, "No contact within bound; still holding. Do not release")

    def open_gripper(self):
        if self.s.held is not None and not (self._contact or self._stack_drop_ready or self._zone_drop_ready):
            return self._fail("open_gripper", "Held block needs contact or a verified drop pose")
        if self.s.held is not None and self._stack_drop_ready:
            level = self._stack_target
            actual = self.s.arm_position_mm()
            if (level is None or self._target is None or self._target[0] != "stack"
                    or math.dist(actual[:2], level.xy_mm) > self.limits.alignment_tolerance_mm
                    or not level.place_z_mm + 3 <= actual[2]
                    <= level.place_z_mm + self.cfg.task2.drop_clearance_mm + 10):
                self._stack_drop_ready = False
                return self._fail("open_gripper", "Arm left the verified stack drop pose")
        if self.s.held is not None and self._zone_drop_ready:
            actual = self.s.arm_position_mm()
            if (self._zone_drop_target_xy() is None or self._zone_drop_xy is None
                    or math.dist(actual[:2], self._zone_drop_xy) > self.limits.alignment_tolerance_mm
                    or not self.s.grasp_z_mm + self.limits.zone_release_floor_margin_mm <= actual[2] <= self.s.drop_z_mm + 10):
                self._zone_drop_ready = False
                return self._fail("open_gripper", "Arm left the verified zone drop pose")
        if (self.s.held is not None and self._place_yaw_deg is not None
                and self._target is not None and self._target[0] in ("slot", "cell", "object")):
            measured_yaw = self.s.ik.forward_yaw_deg(self.s.robot.read_joints())
            if abs(angle_error_deg(measured_yaw, self._place_yaw_deg)) > self.cfg.task1.place_yaw_tolerance_deg:
                return self._fail("open_gripper", "Jaw yaw drifted before release; still holding", "grasp_blocked")
        pending = None
        if self.s.held is not None and (self._contact or self._zone_drop_ready) and self._target and self._target[-1] == "preplace":
            if self._target[0] == "slot":
                target_xy = self.s.slot_centres[list(self.cfg.agent.zone_slots.labels).index(self._target[3])]
                pending = (self.s.held.color, target_xy,
                           list(self.cfg.agent.zone_slots.labels).index(self._target[3]))
            elif self._target[0] == "cell":
                pending = (self.s.held.color, self.cells[(self._target[4], self._target[5])], None)
        self._invalidate_pick()
        self.s.motion.open_gripper()
        self.s.held = None
        self._pending_placement = pending
        self._grasp_failed = False
        self._contact = False
        self._stack_drop_ready = False
        self._zone_drop_ready = False
        self._zone_drop_xy = None
        self._target = None
        self._stack_target = None
        self._recovery_target_xy = None
        self._place_yaw_deg = None
        self._held_block_angle_deg = None
        self._held_pick_yaw_deg = None
        return self._result(True, "open_gripper", "released", stack_verified=False)

    def recover_and_home(self):
        self._invalidate_pick()
        self._place_yaw_deg = None
        self._held_block_angle_deg = None
        self._held_pick_yaw_deg = None
        self._stack_target = None
        self._recovery_target_xy = None
        self.collection.discard("operator_recovery")
        return super().recover_and_home()

    def return_to_home(self):
        self._invalidate_pick()
        if self.s.held is not None:
            return self._fail("return_to_home", "Place held block before returning home")
        if self._zone_retreat_joints is not None:
            target = self._zone_retreat_joints
            current = self.s.robot.read_joints()
            start = self.s.ik.forward_position_mm(current)
            trace = [self.s.ik.forward_position_mm({**current, **step})
                     for step in interpolate(current, target,
                                             self.cfg.motion.descent_step_per_tick)]
            if not trace or min(point[2] for point in trace) < start[2] - 1.0:
                return self._fail("return_to_home", "Release-column retreat is not clear",
                                  "limit_exceeded")
            self.s.player.move_to(target, max_step=self.cfg.motion.descent_step_per_tick,
                                  tol=self.cfg.motion.transit_arrival_tol)
            self._zone_retreat_joints = None
        self._target = None
        self._contact = False
        return super().return_to_home()

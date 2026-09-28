"""Experimental, bounded actions; perception never moves the arm.

All motion uses the session's cancellable IO. FK is model-derived feedback,
not camera-measured TCP. Limits are configurable assumptions, not validation.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, replace

from control.grasp import GraspAttempt
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
        self._grasp_failed = False
        self._pick_calibration = None
        self._pick_ready = False
        self._held_radial_tilt_deg = 0.0
        self._observed_scene = None
        self._pending_placement = None
        from session.collection import Collection
        from control.trajectory import TrajectoryPlayer
        from control.motion import MotionController
        options = {} if collection_factory is None else {"resource_factory": collection_factory}
        self.collection = Collection(session, **options)
        session.robot = self.collection.io
        session.player = TrajectoryPlayer(session.robot, self.cfg.motion)
        session.motion = MotionController(session.robot, session.poses, self.cfg.motion, self.cfg.sensing)

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

    def _calibrated_target(self, block, phase):
        cal = self._calibration()
        self._pick_ready = False
        if phase == "pregrasp":
            result = cal.calibration_prepare(block.color, _scene=self._observed_scene,
                                             _open_gripper=False)
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

    def move_block_to_slot(self, color: str, slot: str):
        """Run one Task 1 transfer with measured gates between every primitive."""
        action, t0 = "move_block_to_slot", time.monotonic()
        labels = list(self.cfg.agent.zone_slots.labels)
        if color not in self.cfg.perception.color_prototypes or slot not in labels:
            return self._fail(action, "Unknown block colour or slot", "invalid_arguments")
        if self.s.held is not None:
            return self._fail(action, "Place the held block before starting another transfer",
                              "already_holding")
        if self.collection.status().get("episode_open"):
            return self._fail(action, "Finish the recording episode before a Task 1 transfer")

        steps = []

        def run(stage, fn):
            self.s.cancel.raise_if_set()
            result = fn()
            steps.append({"stage": stage, "reason": result.reason})
            if result.ok:
                return None
            failed = self._result(
                False, action, result.reason, result.detail,
                retry_advice=result.retry_advice, t0=t0,
                failed_stage=stage, steps=steps, color=color, slot=slot,
                holding=self.s.held.color if self.s.held else None,
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

        def lift_until_clear(stage):
            required_z = self.s.grasp_z_mm + self.limits.lateral_clearance_mm
            correction_mm = 0.0
            for _ in range(self.limits.max_lift_attempts):
                x, y, actual_z = self.s.arm_position_mm()
                if actual_z >= required_z:
                    return None
                # The selected tilted pick already preflighted a reverse path.
                # Use it before trying fixed-XY vertical IK at the far reach.
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
            if self.s.arm_position_mm()[2] >= required_z:
                return None
            return self._result(
                False, action, "limit_exceeded", "Arm did not reach lateral clearance.",
                retry_advice="ask_operator", t0=t0, failed_stage=stage, steps=steps,
                color=color, slot=slot, holding=self.s.held.color if self.s.held else None,
            )

        failed = lift_until_clear("lift_empty")
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
        failed = lift_until_clear("lift_held")
        if failed is not None:
            return failed

        for stage, fn in (
            ("preplace", lambda: self.move_to_target("slot", "preplace", slot=slot)),
            ("contact", lambda: self.descend_until_contact(
                self.limits.contact_max_descent_mm)),
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
        return self._result(
            True, action, "released", f"{color} block observed in {slot}.", t0=t0,
            color=color, slot=slot, measured_xy_mm=list(landed.center_mm),
            miss_mm=round(miss, 1), frame_seq=verified.data.get("frame_seq"),
            place_correction=self.place_correction.as_dict(),
        )

    def _object(self, object_id, observation_id):
        if observation_id != self.observation_id or time.monotonic() - self._observed_at > self.limits.target_max_age_s:
            raise ValueError("Target observation expired; observe_scene again")
        if object_id not in self._objects:
            raise ValueError("Object is not in this observation")
        return self._objects[object_id]

    def _held_check(self):
        if self.s.held is not None and not check_grasp(self.s.robot, self.cfg.sensing, settle=False).grasped:
            self._grasp_failed = True
            self._contact = False
            return False
        return not self._grasp_failed

    def _solve(self, xyz, *, radial_tilt_deg=None, max_position_error_mm=None):
        if not self.s.in_workspace(xyz[:2]):
            raise ValueError("Waypoint outside workspace")
        joints = self.s.robot.read_joints()
        tilt = (self._held_radial_tilt_deg if self.s.held is not None else 0.0) if radial_tilt_deg is None else radial_tilt_deg
        error_limit = (self.cfg.agent.relative.jog_max_ik_error_mm
                       if max_position_error_mm is None else max_position_error_mm)
        solved = self.s.ik.solve_holding_wrist_roll(
            *xyz, wrist_roll_deg=joints["wrist_roll"], radial_tilt_deg=tilt
        )
        if not math.isfinite(solved.position_error_mm) or solved.position_error_mm > error_limit:
            raise ValueError(
                f"Waypoint failed IK gate: target=({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f})mm "
                f"error={solved.position_error_mm:.1f}mm limit={error_limit:.1f}mm "
                f"radial_tilt={tilt:.1f}deg"
            )
        if (abs(solved.joints["wrist_roll"]) > self.limits.wrist_roll_limit_deg
                or (self.limits.calibrated_pick and solved.joints["wrist_roll"] < self.cfg.agent.calibration_clearance.wrist_roll_min_deg)):
            raise ValueError("Wrist exceeds primitive neutral limit")
        return solved

    def _move(self, action, xyz, *, radial_tilt_deg=None, max_ik_error_mm=None):
        self._invalidate_pick()
        start = self.s.arm_position_mm()
        lateral = math.dist(start[:2], xyz[:2]) > 1e-6
        clear_z = self.s.grasp_z_mm + self.limits.lateral_clearance_mm
        if lateral and min(start[2], xyz[2]) < clear_z:
            return self._fail(action, "Lift vertically above clearance before lateral movement")
        if not self._held_check() and (lateral or xyz[2] < start[2]):
            return self._fail(action, "Grasp verification failed; open/retry or lift vertically")
        if not self.s.grasp_z_mm <= xyz[2] <= self.cfg.agent.relative.jog_max_z_mm:
            return self._fail(action, "Target outside configured vertical window")
        count = max(1, math.ceil(math.dist(start, xyz) / self.limits.cartesian_step_mm))
        points = [tuple(a + (b-a)*i/count for a,b in zip(start,xyz)) for i in range(1,count+1)]
        try:
            plans = [
                self._solve(
                    point,
                    radial_tilt_deg=radial_tilt_deg,
                    max_position_error_mm=max_ik_error_mm,
                )
                for point in points
            ]
        except ValueError as exc:
            return self._fail(action, str(exc), "ik_gate")
        self._contact = False
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
                    baseline_load = ContactMonitor(self.s.robot, self.cfg.sensing).start()
                    def check_correction():
                        loads = self.s.robot.read_loads()
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
                    self._calibration()._record("transit_tracking_corrected", correction_deg=correction,
                                                nominal=nominal, corrected=corrected, target_mm=list(xyz))
            endpoint_error_mm = math.dist(self.s.arm_position_mm(), xyz)
            endpoint_limit_mm = max(
                self.limits.arrival_error_mm,
                max_ik_error_mm if max_ik_error_mm is not None else 0.0,
                self.limits.loaded_arrival_error_mm if self.s.held is not None else 0.0,
            )
            if endpoint_error_mm > endpoint_limit_mm:
                raise TimeoutError(
                    f"Measured FK did not reach primitive endpoint: "
                    f"error={endpoint_error_mm:.1f}mm limit={endpoint_limit_mm:.1f}mm"
                )
        if self.s.held is not None:
            self.s.held.over_xy_mm = tuple(self.s.arm_position_mm()[:2])
        return self._result(True, action, "moved", commanded_mm=list(xyz),
                            measured_fk_mm=list(self.s.arm_position_mm()),
                            tracking_correction_deg={} if descending_empty else correction)

    def move_to_target(self, target_type, phase, object_id=None, observation_id=None, slot=None, x=None, y=None):
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
            if math.dist(xyz[:2], xy) > self.cfg.agent.relative.max_pick_offset_mm:
                return self._fail(action, "Correction exceeds grasp offset limit")
            goal = (*xyz[:2], self.s.grasp_z_mm)  # preserve intentional relative correction
        else:
            goal = (*xy, self.s.grasp_z_mm + self.limits.approach_clearance_mm)
        if self.limits.calibrated_pick and phase in ("pregrasp", "grasp"):
            result = self._calibrated_target(block, phase)
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
                result = self._move(
                    action,
                    goal,
                    radial_tilt_deg=place_tilt,
                    max_ik_error_mm=self.cfg.ik.max_position_error_mm,
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

    def move_relative(self, forward_mm=0.0, left_mm=0.0, up_mm=0.0):
        from session.relative import offset_xy
        if not all(math.isfinite(v) for v in (forward_mm,left_mm,up_mm)) or math.sqrt(forward_mm**2+left_mm**2+up_mm**2) > self.cfg.agent.relative.max_jog_mm:
            return self._fail("move_relative", "Relative vector exceeds limit", "invalid_arguments")
        if self.s.held is not None and up_mm < 0:
            return self._fail("move_relative", "Use descend_until_contact while holding")
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
            hover_xy = retreat.hover_xy_mm
            if (hover_xy is not None and xyz[2] < clear_z
                    and retreat.hover_z_mm >= clear_z
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
                if reached[2] < clear_z - 2.0:
                    return self._fail("move_relative", "Reverse approach stopped below lateral clearance", "grasp_blocked")
                self.s.held.over_xy_mm = reached[:2]
                if xyz[2] + up_mm <= reached[2] + 1.0:
                    return self._result(True, "move_relative", "moved",
                                        measured_fk_mm=list(reached), reverse_pick_retreat=True,
                                        lateral_clearance_ready=True,
                                        next_required_action="move_to_placement_target")
                result = self._move("move_relative", (*reached[:2], xyz[2] + up_mm))
                result.data["reverse_pick_retreat"] = True
                result.data["lateral_clearance_ready"] = self.s.arm_position_mm()[2] >= clear_z - 2.0
                result.data["next_required_action"] = (
                    "move_to_placement_target" if result.data["lateral_clearance_ready"]
                    else "lift_vertically_again"
                )
                return result
        xy = offset_xy(xyz[:2], forward_mm, left_mm, frame=self.cfg.agent.relative.frame, base_xy_mm=self.s.base_xy)
        result = self._move("move_relative", (*xy, xyz[2]+up_mm))
        if self.s.held is not None and up_mm > 0:
            measured_z = self.s.arm_position_mm()[2]
            required_z = self.s.grasp_z_mm + self.limits.lateral_clearance_mm
            ready = measured_z >= required_z
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
        if xyz[2] < self.s.grasp_z_mm + self.limits.lateral_clearance_mm:
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
        self._target = None
        return self._result(
            True, "close_gripper", "held", grasp=asdict(check),
            identity_confirmed=color is not None, associated_color=color,
            next_required_action="move_relative_up",
        )

    def descend_until_contact(self, max_descent_mm):
        action = "descend_until_contact"
        if not math.isfinite(max_descent_mm) or not 0 < max_descent_mm <= self.limits.contact_max_descent_mm:
            return self._fail(action, "Descent bound invalid", "invalid_arguments")
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
            else:
                target_xy = self.cells[(self._target[4], self._target[5])]
        except (ValueError, KeyError) as exc:
            return self._fail(action, str(exc))
        self._contact = False
        start = self.s.arm_position_mm()
        if math.dist(start[:2], target_xy) > self.limits.alignment_tolerance_mm:
            return self._fail(action, "Gripper is outside target alignment tolerance")
        distance = min(max_descent_mm, max(0.0, start[2]-self.s.grasp_z_mm))
        count = max(1, math.ceil(distance/self.limits.contact_step_mm))
        from control.task1_transport import place_tilt_deg
        place_tilt = place_tilt_deg(target_xy, self.s.base_xy, self.cfg)
        try:
            plans = [
                self._solve(
                    (*start[:2], start[2] - distance * i / count),
                    radial_tilt_deg=place_tilt,
                    max_position_error_mm=self.cfg.ik.max_position_error_mm,
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
                if (self._target[0] != "object" and actual[2] > self.s.grasp_z_mm
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
                    measured_fk_mm=list(self.s.arm_position_mm()),
                    tracking_error_deg=tracking_error, stack_verified=False,
                )
        measured = self.s.robot.read_joints()
        self.s.robot.send_joints({j: measured[j] for j in plans[-1].joints})
        actual = self.s.arm_position_mm()
        if (self._target[0] != "object"
                and actual[2] <= self.s.grasp_z_mm + self.limits.arrival_error_mm):
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
                contact_samples=contact_samples,
                measured_fk_mm=list(self.s.arm_position_mm()),
                tracking_error_deg=0.0, stack_verified=False,
            )
        return self._fail(action, "No contact within bound; still holding. Do not release")

    def open_gripper(self):
        if self.s.held is not None and not self._contact:
            return self._fail("open_gripper", "Held block may only be released after contact")
        pending = None
        if self.s.held is not None and self._contact and self._target and self._target[-1] == "preplace":
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
        self._target = None
        return self._result(True, "open_gripper", "released", stack_verified=False)

    def recover_and_home(self):
        self._invalidate_pick()
        self.collection.discard("operator_recovery")
        return super().recover_and_home()

    def return_to_home(self):
        self._invalidate_pick()
        if self.s.held is not None:
            return self._fail("return_to_home", "Place held block before returning home")
        self._target = None
        self._contact = False
        return super().return_to_home()

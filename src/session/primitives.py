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
                              limitations=["one detection per colour; IDs are observation-scoped",
                                            "planar CV does not measure stack height or visual TCP",
                                            "missing detection is not proof of absence"])
        return replace(result, images=images)

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

    def _solve(self, xyz):
        if not self.s.in_workspace(xyz[:2]):
            raise ValueError("Waypoint outside workspace")
        joints = self.s.robot.read_joints()
        solved = self.s.ik.solve_holding_wrist_roll(*xyz, wrist_roll_deg=joints["wrist_roll"],
            radial_tilt_deg=(self._held_radial_tilt_deg if self.s.held is not None else 0.0))
        if not math.isfinite(solved.position_error_mm) or solved.position_error_mm > self.cfg.agent.relative.jog_max_ik_error_mm:
            raise ValueError("Waypoint failed IK gate")
        if (abs(solved.joints["wrist_roll"]) > self.limits.wrist_roll_limit_deg
                or (self.limits.calibrated_pick and solved.joints["wrist_roll"] < self.cfg.agent.calibration_clearance.wrist_roll_min_deg)):
            raise ValueError("Wrist exceeds primitive neutral limit")
        return solved

    def _move(self, action, xyz):
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
            plans = [self._solve(point) for point in points]
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
            # Check the measured path corridor without stopping at IK knots.
            # Longitudinal servo lag is allowed while moving; final arrival is
            # still checked against the requested endpoint below.
            delta = tuple(b-a for a,b in zip(start, xyz))
            length2 = sum(d*d for d in delta)
            def check_progress():
                actual = self.s.arm_position_mm()
                t = (sum((v-a)*d for v,a,d in zip(actual,start,delta)) / length2
                     if length2 else 0.0)
                t = max(0.0, min(1.0, t))
                nearest = tuple(a+t*d for a,d in zip(start,delta))
                if math.dist(actual, nearest) > self.limits.arrival_error_mm:
                    raise TimeoutError("Measured FK left primitive path corridor")
            self.s.player.move_through([plan.joints for plan in plans],
                tol=(self.cfg.motion.transit_arrival_tol if self.limits.calibrated_pick
                     else self.cfg.motion.arrival_tol), check_progress=check_progress,
                timeout_s=deadline-time.monotonic())
            if self.limits.calibrated_pick:
                self.s.player.settle(plans[-1].joints, tol=self.cfg.motion.arrival_tol,
                                     timeout_s=min(self.cfg.motion.grasp_hover_settle_s,max(0.0,deadline-time.monotonic())))
            correction = {}
            if self.limits.calibrated_pick and self.s.held is not None:
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
                        check_progress()
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
            if math.dist(self.s.arm_position_mm(), xyz) > self.limits.arrival_error_mm:
                raise TimeoutError("Measured FK did not reach primitive endpoint")
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
                xy = self.s.slot_centres[list(self.cfg.agent.zone_slots.labels).index(slot)]
            else:
                raise ValueError("Unknown target type")
        except ValueError as exc:
            return self._fail(action, str(exc), "invalid_arguments")
        if phase not in ("pregrasp", "grasp", "preplace", "hover"):
            return self._fail(action, "Unknown phase", "invalid_arguments")
        if phase == "preplace" and self.s.held is None:
            return self._fail(action, "Close and verify grasp before preplace", "no_block_held")
        if phase in ("pregrasp", "grasp") and (self.s.held is not None or target_type != "object"):
            return self._fail(action, "Grasp phases require an object target and empty gripper")
        xyz = self.s.arm_position_mm()
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
        else:
            result = self._move(action, goal)
        if result.ok:
            self._target = (target_type, object_id, observation_id, slot, x, y, phase)
            result.data["collection_zone_destination"] = phase == "preplace" and target_type != "object" and self.s.in_zone(xy)
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
        xy = offset_xy(xyz[:2], forward_mm, left_mm, frame=self.cfg.agent.relative.frame, base_xy_mm=self.s.base_xy)
        return self._move("move_relative", (*xy, xyz[2]+up_mm))

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
                if math.dist(xyz[:2], block.center_mm) <= self.cfg.agent.relative.max_pick_offset_mm:
                    color = block.color
            except ValueError:
                pass
        attempt = GraspAttempt("primitive", (0.,0.), xyz[:2], plan, plan, True, xyz[2], xyz[2])
        # Carry the calibrated pick's approach tilt into lift/transport IK.
        # Resetting to top-down can make a reachable lift fail its IK gate.
        cal = self._pick_calibration
        self._held_radial_tilt_deg = (cal.plan.radial_tilt_deg
            if self.limits.calibrated_pick and cal is not None and cal.plan is not None else 0.0)
        self.s.held = HeldBlock(color, attempt, xyz[:2], self.s.in_zone(xyz[:2]), xyz[:2])
        self._target = None
        return self._result(True, "close_gripper", "held", grasp=asdict(check), identity_confirmed=False,
                            associated_color=color)

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
        try:
            plans = [self._solve((*start[:2], start[2]-distance*i/count)) for i in range(1,count+1)]
        except ValueError as exc:
            return self._fail(action, str(exc), "ik_gate")
        monitor = ContactMonitor(self.s.robot, self.cfg.sensing, magnitude_increase=True)
        monitor.start()
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
                if not reading.contact and max(abs(measured[j]-joints[j]) for j in joints) > self.cfg.motion.descent_max_lag:
                    self.s.robot.send_joints({j: measured[j] for j in plan.joints})
                    raise TimeoutError("Contact descent following error; stopped without release")
                if reading.contact:
                    self.s.robot.send_joints({j: measured[j] for j in plan.joints})
                    actual = self.s.arm_position_mm()
                    if (self._target[0] != "object" and actual[2] > self.s.grasp_z_mm
                            + self.cfg.agent.calibration_clearance.obstacle_height_mm):
                        return self._fail(action, "Unexpected contact above table; still holding. Reobserve before retry",
                                          "grasp_blocked")
                    backoff = self._solve((*actual[:2], min(start[2], actual[2]+self.limits.contact_backoff_mm)))
                    self.s.player.move_to(backoff.joints)
                    self._contact = True
                    return self._result(True, action, "ok", contact=asdict(reading),
                                        measured_fk_mm=list(self.s.arm_position_mm()), stack_verified=False)
        measured = self.s.robot.read_joints()
        self.s.robot.send_joints({j: measured[j] for j in plans[-1].joints})
        return self._fail(action, "No contact within bound; still holding. Do not release")

    def open_gripper(self):
        if self.s.held is not None and not self._contact:
            return self._fail("open_gripper", "Held block may only be released after contact")
        self._invalidate_pick()
        self.s.motion.open_gripper()
        self.s.held = None
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

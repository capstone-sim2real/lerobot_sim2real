"""Gripper, zone drop, contact descent and homing."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, replace

from control.grasp import GraspAttempt
from control.task1_transport import zone_axis_yaw_deg
from control.sensing import ContactMonitor, check_grasp
from control.trajectory import interpolate
from session.arm_session import HeldBlock


class GripperMixin:
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
        self._place_yaw_explicit = False
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
        target_xy = self._place_command_xy or self._zone_drop_target_xy()
        tilt = (self._place_radial_tilt_deg if self._place_radial_tilt_deg is not None
                else place_tilt_deg(target_xy, self.s.base_xy, self.cfg))
        try:
            pose = self._solve(
                (*target_xy, drop_z),
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
            return (math.dist(pose[:2], target_xy) <= self.limits.alignment_tolerance_mm
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
        xy_error = math.dist(actual[:2], target_xy)
        if ((actual[2] < self.s.grasp_z_mm + self.cfg.task1.release_clearance_mm / 2
             or xy_error > self.cfg.task1.place_ik_error_mm)
                and xy_error <= self.limits.alignment_tolerance_mm):
            correction_z = min(start[2], max(actual[2], drop_z + max(0.0, drop_z - actual[2])))
            correction_xy = tuple(2 * target - measured for target, measured in zip(target_xy, actual[:2]))
            try:
                correction = self._solve(
                    (*correction_xy, correction_z), radial_tilt_deg=tilt,
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
        self._zone_drop_xy = tuple(target_xy)
        self._zone_retreat_joints = retreat_joints
        self._zone_drop_ready = True
        return self._result(True, action, "ok", release_mode="height_drop",
                            release_fk_mm=list(actual), drop_z_mm=drop_z, target_xy_mm=list(target_xy),
                            release_xy_error_mm=math.dist(actual[:2], target_xy),
                            radial_tilt_deg=tilt,
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
            axis = (self._place_yaw_deg if self._place_yaw_explicit else
                    zone_axis_yaw_deg(self.s.calib.zone_polygon_mm))
            zone_error = (measured_yaw - axis + 90.0) % 180.0 - 90.0
            if abs(zone_error) > self.cfg.task1.place_yaw_tolerance_deg:
                return self._fail("open_gripper",
                                  f"Jaw line misses requested placement yaw by {zone_error:.1f}deg; still holding",
                                  "grasp_blocked")
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
        self._place_yaw_explicit = False
        self._held_block_angle_deg = None
        self._held_pick_yaw_deg = None
        return self._result(True, "open_gripper", "released", stack_verified=False)

    def recover_and_home(self):
        self._invalidate_pick()
        self._place_yaw_deg = None
        self._place_yaw_explicit = False
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
        post_release = self._zone_retreat_joints is not None or self._zone_home_pending
        if self._zone_retreat_joints is not None:
            # The retreat was recorded while holding the block. Preserve the
            # open jaws after release; only the arm should revisit that pose.
            target = {joint: value for joint, value in self._zone_retreat_joints.items()
                      if joint != "gripper"}
            self.s.player.move_to(target, max_step=self.cfg.motion.descent_step_per_tick,
                                  tol=self.cfg.motion.transit_arrival_tol)
            self._zone_retreat_joints = None
            self._zone_home_pending = True
        self._target = None
        self._contact = False
        result = super().return_to_home(post_release=post_release)
        if result.ok:
            self._zone_home_pending = False
        return result

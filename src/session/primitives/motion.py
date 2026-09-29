"""IK-planned arm moves: targets, lifts, relative moves and wrist alignment."""
from __future__ import annotations

import math
import time
from dataclasses import replace

from control.task1_transport import angle_error_deg, zone_axis_yaw_deg
from control.sensing import ContactMonitor
from control.trajectory import interpolate


class MotionMixin:
    def _choose_place_yaw(self, xyz, place_tilt):
        """Align the jaw line to the zone long edge, with 180-degree symmetry."""
        axis = zone_axis_yaw_deg(self.s.calib.zone_polygon_mm)
        current_yaw = self.s.ik.forward_yaw_deg(self.s.robot.read_joints())
        aligned = [(axis + 180 * k + 180) % 360 - 180 for k in (0, 1)]
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
                    release = self._solve(
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
                    plans.append((max(hover.position_error_mm, release.position_error_mm),
                                  max(abs(hover.joints["wrist_roll"]),
                                      abs(release.joints["wrist_roll"])),
                                  abs(angle_error_deg(yaw, current_yaw)),
                                  hover.position_error_mm, yaw))
            return plans

        aligned_plans = viable(aligned)
        if aligned_plans:
            return min(aligned_plans)[4], True
        raise ValueError("No zone-aligned placement yaw reaches IK and clearance gates")

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
            # below its requested Z. Check only observed placed blocks;
            # the rest of the zone is not an obstacle.
            for point, plan in zip(points, plans, strict=True):
                planned = self.s.ik.forward_position_mm(plan.joints)
                if (self._task1_near_low_zone_block(point)
                        or self._task1_near_low_zone_block(planned)):
                    return self._result(
                        False, action, "ik_gate",
                        f"Planned carry passes a placed block below clearance: "
                        f"planned_z={planned[2]:.1f}mm",
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
        if phase == "preplace" and target_type == "slot":
            occupant = self._slot_occupant(slot_index)
            held_color = self.s.held.color
            if occupant is not None and occupant != held_color:
                free_slots = self._free_slot_labels()
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
                options = []
                for candidate_tilt in self.cfg.task1.place_tilt_candidates_deg:
                    try:
                        yaw, _ = self._choose_place_yaw(goal, candidate_tilt)
                        poses = [self._solve(
                            (*xy, height), radial_tilt_deg=candidate_tilt,
                            max_position_error_mm=self.cfg.ik.max_position_error_mm,
                            max_tilt_error_deg=self.cfg.task1.place_level_tolerance_deg,
                            yaw_deg=yaw, max_yaw_error_deg=self.cfg.task1.place_yaw_tolerance_deg,
                        ) for height in (goal[2], self.s.drop_z_mm)]
                    except ValueError:
                        continue
                    error = max(p.position_error_mm for p in poses)
                    accurate = error <= self.cfg.task1.place_ik_error_mm
                    # Prefer a small tilt when accurate; otherwise the closest
                    # FK solution to the unchanged requested coordinates.
                    options.append((not accurate, abs(candidate_tilt) if accurate else error,
                                    error, candidate_tilt, yaw))
                result = self._fail(action, "No horizontal-jaw placement pose within IK limits", "ik_gate")
                place_tilt = 0.0
                for _, _, planned_error, place_tilt, place_yaw in sorted(options):
                    zone_aligned = True
                    result = self._move(
                        action, goal, radial_tilt_deg=place_tilt,
                        max_ik_error_mm=self.cfg.ik.max_position_error_mm,
                        max_tilt_error_deg=self.cfg.task1.place_level_tolerance_deg,
                        level_during_carry=True, place_yaw_deg=place_yaw,
                    )
                    result.data["placement_plan_error_mm"] = planned_error
                    result.data["placement_approximate"] = planned_error > self.cfg.task1.place_ik_error_mm
                    attempts.append({"scale": correction_scale, "xy_mm": list(xy),
                                     "tilt_deg": place_tilt, "plan_error_mm": planned_error,
                                     "reason": result.reason})
                    if result.ok or result.reason != "ik_gate":
                        break
                if result.ok or result.reason != "ik_gate":
                    break
            result.data["radial_tilt_deg"] = place_tilt
            if result.ok:
                self._place_yaw_deg = place_yaw
                self._place_yaw_explicit = False
                self._place_radial_tilt_deg = place_tilt
                self._place_command_xy = tuple(xy)
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
                    or self._task1_near_low_zone_block(point)
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
                            or self._task1_near_low_zone_block(point)
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
                or self._task1_near_low_zone_block(actual)):
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
                and abs(self._held_radial_tilt_deg) > self.cfg.task1.near_vertical_pick_max_deg):
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

    def align_gripper(self, object_id=None, observation_id=None, yaw_deg=None):
        if yaw_deg is not None:
            if object_id is not None or observation_id is not None or not math.isfinite(yaw_deg) or not -180 <= yaw_deg <= 180:
                return self._fail("align_gripper", "Use yaw_deg alone in [-180, 180]", "invalid_arguments")
            xyz = self.s.arm_position_mm()
            if not self._lateral_clearance_ready(xyz[2]):
                return self._fail("align_gripper", "Lift before rotation")
            if self.s.held is not None and not self._held_check():
                return self._fail("align_gripper", "Verified grasp required", "grasp_empty")
            result = self._move("align_gripper", xyz, place_yaw_deg=yaw_deg)
            if result.ok:
                self._invalidate_pick()
                self._place_yaw_deg = yaw_deg if self._target and self._target[-1] == "preplace" else None
                self._place_yaw_explicit = self._place_yaw_deg is not None
            return result
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

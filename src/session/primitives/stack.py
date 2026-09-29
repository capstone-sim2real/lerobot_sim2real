"""Task 2: stack a block on an explicit tower floor."""
from __future__ import annotations

import math
import time

from control.trajectory import interpolate


class StackMixin:
    def stack_block_to_floor(self, color=None, floor=0, source=None, destination=None):
        """Try the requested 0-based floor, with one guarded table recovery."""
        if destination is not None:
            if not isinstance(destination, dict) or set(destination) != {"u", "v", "calibration_id"}:
                return self._fail("stack_block_to_floor", "destination requires u,v,calibration_id", "invalid_arguments")
            target, failure = self._pixel_point("stack_block_to_floor", time.monotonic(), **destination)
            if failure is not None:
                return failure
            from control.task2_stack import Task2StackPlanner
            previous, history = self.s._stack, self._task2_placed_floors
            self.s._stack = Task2StackPlanner(self.s.calib, self.cfg, self.s.ik,
                                            target_xy_mm=(target["x_mm"], target["y_mm"]))
            key = (round(target["x_mm"], 1), round(target["y_mm"], 1))
            self._task2_placed_floors = self._tower_histories.setdefault(key, {})
            if type(floor) is int and 0 < floor <= 4:
                self._task2_placed_floors.setdefault(floor - 1, "requested_support")
            try:
                return self.stack_block_to_floor(color, floor, source=source)
            finally:
                self.s._stack, self._task2_placed_floors = previous, history
        if source is not None:
            if color is not None:
                return self._fail("stack_block_to_floor", "Specify color or source, not both", "invalid_arguments")
            return self._run_with_source(source, "stack_block_to_floor",
                                         lambda selected: self.stack_block_to_floor(selected, floor))
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
                return self._result(True, action, "released", verified=False,
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
        if self.s.held is not None:
            return self._fail(action, "Temporary release did not clear the held state", "grasp_blocked")
        return placed

    def _task2_path_clear_of_tower(self, start, waypoints) -> bool:
        """Preflight the local tower footprint, allowing travel above its estimated top.

        The top camera only locates blocks in XY. A visible block contributes one
        floor; prior releases contribute their *planned* floor, not a measured
        tower height. An empty tower does not obstruct a far-side pick.
        """
        tower_xy = self.s.stack.stack_xy_mm
        radius = self.cfg.agent.place_clear_radius_mm
        visible = (self._observed_scene is not None and any(
            math.dist(block.center_mm, tower_xy) < radius
            for block in self._observed_scene.inside.values()
        ))
        occupied_levels = max(
            (floor + 1 for floor in self._task2_placed_floors), default=0
        )
        if visible:
            occupied_levels = max(occupied_levels, 1)
        if not occupied_levels:
            return True

        # grasp_z is the first block's top-face plane. The carried block hangs
        # below the gripper frame, so include its height in the clearance.
        top_z = self.s.grasp_z_mm + (occupied_levels - 1) * self.cfg.task2.block_height_mm
        clear_z = top_z + self.cfg.task2.tower_path_clearance_mm
        if self.s.held is not None:
            clear_z += self.cfg.task2.block_height_mm

        def blocked(joints):
            x, y, z = self.s.ik.forward_position_mm(joints)
            return math.dist((x, y), tower_xy) < radius and z < clear_z

        previous = start
        left_tower = not blocked(previous)
        for goal in waypoints:
            for command in interpolate(previous, goal, self.cfg.motion.max_step_per_tick):
                pose = {**previous, **command}
                inside = blocked(pose)
                if left_tower and inside:
                    return False
                if not inside:
                    left_tower = True
            previous = {**previous, **goal}
        return True

    def _stack_block_to_floor_once(self, color: str, floor: int):
        """Transfer one block to an explicitly requested tower floor."""
        action, t0 = "stack_block_to_floor", time.monotonic()
        if color not in self.cfg.perception.color_prototypes and not (color == "selected" and self._transfer_source):
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

        observed = self._observe_target(color)
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
                if (held is not None and abs(self._held_radial_tilt_deg) > self.cfg.task1.near_vertical_pick_max_deg
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
                _route_guard=(self._task2_path_clear_of_tower
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
        # Skip the old inside-zone apex. Only low sweeps through the local
        # tower footprint are blocked before the final placement entry.
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
        if not self._task2_path_clear_of_tower(current, route):
            # The optional pick apex can sweep low through the tower even
            # when direct travel to the outside stage stays clear.
            if self._task2_path_clear_of_tower(current, (outside_stage.joints,)):
                carry = ()
            else:
                return self._result(False, action, "limit_exceeded",
                                    "Task 2 carry would cross the tower below clearance",
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
            # Rise over the tower before lateral travel; reject low re-entry.
            home_joints = {joint: value for joint, value in
                           self.s.poses.get(self.cfg.motion.home_pose).items()
                           if joint != "gripper"}
            if (not self._task2_path_clear_of_tower(entry.joints, (outside_stage.joints,))
                    or not self._task2_path_clear_of_tower(outside_stage.joints,
                                                         (home_joints,))):
                return self._fail("stack_retreat", "Retreat re-enters the tower below clearance",
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
        self._task2_placed_floors[floor] = color
        return self._result(True, action, "released",
                            f"Floor {floor} release and retreat completed; no visual placement check",
                            t0=t0, color=color, floor=floor, level=level_number,
                            contact_confirmed=False,
                            release_mode="height_drop",
                            drop_clearance_mm=self.cfg.task2.drop_clearance_mm,
                            placement_observed=False, stack_verified=False,
                            support_evidence=support_evidence,
                            expected_place_z_mm=level.place_z_mm)

"""Continuous jog execution; input state stays in :mod:`keyboard_jog`."""
import time

from .jog_planner import JogPathRejected, plan_next
from .jog_ramp import JogRamp
from .keyboard_jog import JogReleased


class JogExecutor:
    """Schedule incremental IK planning between fixed-rate motor writes."""

    def __init__(self, stream, skills, cancel):
        self.stream = stream
        self.skills = skills
        self.cancel = cancel
        self.cfg = stream.cfg
        self.session = skills.s
        self.period = 1.0 / self.cfg.keyboard_tick_hz
        self.ramp = JogRamp(self.cfg)
        self.path = None
        self.following = None
        self.planner = None
        self.direction = None
        self.last = None
        self.sent = False
        self.failure = None
        self.failure_reason = "out_of_workspace"
        self.next_tick = stream.clock()
        self.next_grasp_check = self.next_tick

    def _clearance(self):
        if (self.session.held is not None and hasattr(self.skills, "_held_check")
                and (self.direction[0] or self.direction[1])):
            return self.session.grasp_z_mm + self.skills.limits.lateral_clearance_mm
        return None

    def _begin_plan(self):
        self.failure = None
        start = self.path.goal if self.path is not None else self.session.robot.read_joints()
        self.planner = plan_next(self.session, start, self.direction, clearance=self._clearance())

    def _advance_plan(self):
        try:
            next(self.planner)
        except StopIteration as done:
            self.planner = None
            if self.path is None:
                self.path = done.value
                self.last = dict(self.path.start)
            else:
                self.following = done.value
        except JogPathRejected as exc:
            self.planner = None
            self.failure = str(exc)

    def _prepare_direction(self, desired):
        self.path = self.following = self.planner = None
        self.failure = None
        self.direction = desired
        if not any(desired):
            return False
        if hasattr(self.skills, "_held_check") and self.session.held is not None:
            xyz = self.session.arm_position_mm()
            lateral = bool(desired[0] or desired[1])
            unsafe = (desired[2] < 0 or (lateral and xyz[2] < self._clearance())
                      or (lateral and not self.skills._held_check()))
            if unsafe:
                self.failure_reason = "precondition"
                self.failure = "블록을 든 상태입니다. 먼저 위로 들어 올리고 파지를 확인하세요."
                return False
        if hasattr(self.skills, "_held_check"):
            self.skills._contact = False
            self.skills._target = None
        self._begin_plan()
        self.next_tick = self.stream.clock()
        return True

    def _braking(self, stopping, remaining, available):
        reserve = self.ramp.braking_distance() + 2 * self.cfg.keyboard_speed_mm_s * self.period
        crossing_ready = (
            self.following is not None
            and self.ramp.input.current_velocity[0]
            + max(0.0, self.ramp.input.current_acceleration[0]) ** 2
            / (2 * self.cfg.keyboard_jerk_mm_s3)
            <= self.following.speed + 1e-8
        )
        brake = stopping or available <= reserve or (not crossing_ready and remaining <= reserve)
        return brake, reserve

    def _write_tick(self, stopping):
        remaining = self.path.length - self.path.progress
        available = remaining + (self.following.length if self.following is not None else 0.0)
        brake, reserve = self._braking(stopping, remaining, available)
        if brake and self.ramp.stopped:
            if stopping:
                return "direction_stopped"
            if self.failure:
                return "finished"
            if self.planner is not None:
                self._advance_plan()
                return "waiting"
            return "finished"
        speed = self.path.speed
        if self.following is not None:
            speed = min(speed, self.following.speed)
        elif remaining <= reserve:
            brake = True
        delta = self.ramp.advance(0.0 if brake else speed)
        if delta > available + 1e-8:
            raise RuntimeError("keyboard braking path exhausted")
        progress = self.path.progress + delta
        if progress > self.path.length and self.following is not None:
            progress -= self.path.length
            self.path, self.following = self.following, None
        if progress > self.path.length + 1e-8:
            raise RuntimeError("keyboard path exhausted")
        step = self.path.pose(min(self.path.length, progress))
        limit = min(1.0, self.session.cfg.motion.max_step_per_tick,
                    self.cfg.keyboard_joint_speed_deg_s * self.period)
        if any(abs(value - self.last[joint]) > limit + 1e-8 for joint, value in step.items()):
            raise RuntimeError("keyboard joint step limit exceeded")
        self.session.robot.send_joints(step)
        self.last = step
        self.path.progress = progress
        self.sent = True
        return "written"

    def _check_grasp(self):
        now = self.stream.clock()
        if now < self.next_grasp_check:
            return
        self.next_grasp_check = now + self.cfg.keyboard_segment_s
        if (self.session.held is not None and hasattr(self.skills, "_held_check")
                and not self.skills._held_check()):
            self.stream.release()
            self.failure = "파지가 확인되지 않아 키보드 이동을 멈췄습니다."

    def _hold_measured(self):
        if not self.sent or self.cancel.is_set():
            return
        joints = self.session.robot.read_joints()
        self.session.robot.send_joints({j: value for j, value in joints.items() if j != "gripper"})
        if self.session.held is not None:
            self.session.held.over_xy_mm = tuple(self.session.ik.forward_position_mm(joints)[:2])

    def run(self):
        if hasattr(self.skills, "_invalidate_pick"):
            self.skills._invalidate_pick()
        if hasattr(self.skills, "attempt"):
            self.skills.attempt = None
            self.skills.descent_ready = False
        try:
            while True:
                self.cancel.raise_if_set()
                desired = self.stream.current()
                if not any(desired) and self.ramp.stopped:
                    if self.stream.releasing:
                        break
                    self.direction = None
                    self.cancel.event.wait(self.period)
                    continue
                turning = self.direction is not None and desired != self.direction
                stopping = self.stream.releasing or turning
                if stopping:
                    self.planner = None
                if self.ramp.stopped and (stopping or self.direction is None):
                    if self.stream.releasing:
                        break
                    if not self._prepare_direction(desired):
                        if self.failure:
                            break
                        self.cancel.event.wait(self.period)
                        continue
                    stopping = False
                if self.path is None:
                    if self.failure:
                        break
                    self._advance_plan()
                    continue
                if not stopping and self.following is None and self.planner is None and self.failure is None:
                    self._begin_plan()
                now = self.stream.clock()
                if now < self.next_tick:
                    if self.planner is not None:
                        self._advance_plan()
                    else:
                        self.cancel.event.wait(self.next_tick - now)
                    continue
                with self.stream.lock:
                    self.cancel.raise_if_set()
                    desired = self.stream.current()
                    stopping = self.stream.releasing or desired != self.direction
                    outcome = self._write_tick(stopping)
                if outcome in ("direction_stopped", "waiting"):
                    continue
                if outcome == "finished":
                    break
                self.next_tick = max(self.next_tick + self.period, now + self.period)
                self._check_grasp()
        except JogReleased:
            pass
        finally:
            self.stream.close()
            self._hold_measured()
        if self.failure:
            return self.skills._result(False, "move_arm", self.failure_reason,
                                       self.failure, t0=time.monotonic())
        return None

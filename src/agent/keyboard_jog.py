"""Latest-input keyboard stream. No queued moves; all bus work stays on RobotWorker."""
import math
import threading
import time
import secrets

from .jog_ramp import JogRamp
from control.trajectory import interpolate


class JogReleased(Exception):
    pass


class JogDirectionChanged(Exception):
    pass


class JogPathRejected(Exception):
    pass


class KeyboardJog:
    def __init__(self, cfg, *, clock=time.monotonic):
        self.cfg = cfg
        self.clock = clock
        self.id = secrets.token_urlsafe(18)
        self.lock = threading.RLock()
        self.seq = -1
        self.vector = (0., 0., 0.)
        self.expires = clock() + cfg.keyboard_timeout_s
        self.closed = False
        self.releasing = False

    def update(self, seq, vector):
        with self.lock:
            if self.closed or self.releasing:
                return False
            if self.clock() >= self.expires:
                self.closed = True
                return False
            if seq <= self.seq:
                return False
            self.seq = seq
            norm = math.hypot(*vector)
            self.vector = tuple(v / max(1., norm) for v in vector)
            self.expires = self.clock() + self.cfg.keyboard_timeout_s
            if norm == 0:
                self.release()
            return True

    def release(self):
        with self.lock:
            if not self.closed and not self.releasing:
                self.releasing = True
                self.vector = (0., 0., 0.)
                self.expires = self.clock() + self.cfg.keyboard_release_timeout_s

    def close(self):
        with self.lock:
            self.closed = True

    def current(self):
        with self.lock:
            if self.closed or self.clock() >= self.expires:
                self.closed = True
                raise JogReleased()
            return self.vector

    def run(self, skills, cancel):
        cfg, session = self.cfg, skills.s
        period = 1. / cfg.keyboard_tick_hz
        ramp = JogRamp(cfg)  # Missing optional dependency fails before any motion.
        if hasattr(skills, "_invalidate_pick"):
            skills._invalidate_pick()
        sent = False
        path = None
        direction = None
        if hasattr(skills, 'attempt'):
            skills.attempt = None
            skills.descent_ready = False

        def travel(*, braking=False):
            nonlocal sent
            ticks = max(1, math.ceil(cfg.keyboard_segment_s / period))
            while True:
                with self.lock:
                    cancel.raise_if_set()
                    desired = self.current()
                    turning = desired != direction
                    brake = braking or turning or self.releasing
                    remaining = path['length'] - path['progress']
                    # Keep the complete stop inside the already gated IK path.
                    reserve = ramp.braking_distance() + 2 * cfg.keyboard_speed_mm_s * period
                    brake = brake or remaining <= reserve
                    if brake and ramp.stopped:
                        if not braking and not turning and not self.releasing:
                            raise JogReleased()  # No room to restart without a new key press.
                        return
                    delta = ramp.advance(0. if brake else path['speed'])
                    progress = path['progress'] + delta
                    if progress > path['length'] + 1e-8:
                        raise RuntimeError('keyboard braking path exhausted')
                    fraction = min(1., progress / path['length'])
                    step = {j: path['start'][j] + (v-path['start'][j])*fraction for j,v in path['goal'].items()}
                    previous = path['last']
                    limit = min(1., session.cfg.motion.max_step_per_tick, cfg.keyboard_joint_speed_deg_s*period)
                    if any(abs(v-previous[j]) > limit+1e-8 for j,v in step.items()):
                        raise RuntimeError('keyboard joint step limit exceeded')
                    session.robot.send_joints(step)
                    path['last'] = step
                    sent = True
                    path['progress'] = progress
                cancel.event.wait(period)
                ticks -= 1
                if not brake and ticks <= 0:
                    return

        def playback(goal):
            nonlocal path
            cancel.raise_if_set()
            start = session.robot.read_joints()
            a, b = session.ik.forward_position_mm(start), session.ik.forward_position_mm(goal)
            length = math.dist(a, b)
            if length <= 1e-8:
                raise RuntimeError('keyboard IK returned a zero-length path')
            # Check the interpolated path, including its reserved braking portion.
            # Endpoints alone do not cover a curved Cartesian path from joint interpolation.
            slack = cfg.jog_max_ik_error_mm
            lo, hi = min(a[2], cfg.jog_min_z_mm-slack), max(a[2], cfg.jog_max_z_mm+slack)
            if hasattr(skills, '_held_check') and (direction[0] or direction[1]):
                lo = max(lo, session.grasp_z_mm + skills.limits.lateral_clearance_mm)
            for pose in interpolate(start, goal, min(1., session.cfg.motion.max_step_per_tick,
                                                    cfg.keyboard_joint_speed_deg_s*period)):
                x,y,z = session.ik.forward_position_mm(pose)
                if not all(math.isfinite(v) for v in (x,y,z)) or not session.in_workspace((x,y)) or not lo-1e-8 <= z <= hi+1e-8:
                    raise JogPathRejected('이동·감속 경로가 작업 범위를 벗어납니다.')
            largest = max(abs(goal[j] - start[j]) for j in goal)
            joint_speed = min(cfg.keyboard_joint_speed_deg_s,
                              min(1., session.cfg.motion.max_step_per_tick) / period)
            speed = min(cfg.keyboard_speed_mm_s, joint_speed*length/largest) if largest else cfg.keyboard_speed_mm_s
            # A changed IK slope must not cause a step-limit violation while braking.
            required_speed = ramp.input.current_velocity[0] + max(0., ramp.input.current_acceleration[0])**2/(2*cfg.keyboard_jerk_mm_s3)
            if required_speed > speed + 1e-8:
                if path is not None:travel(braking=True)
                raise JogDirectionChanged()
            path = dict(start=start, last=start, goal=goal, length=length, progress=0., speed=speed)
            travel()

        result = None
        try:
            while True:
                cancel.raise_if_set()
                desired = self.current()
                if direction is None or ramp.stopped:
                    direction = desired
                if not any(desired) and ramp.stopped:
                    if self.releasing:break
                    cancel.event.wait(period)
                    continue
                if desired != direction and path is not None:
                    travel(braking=True)
                    continue
                # Look ahead far enough to include braking, not only the next tick.
                stop_time = cfg.keyboard_speed_mm_s/cfg.keyboard_acceleration_mm_s2 + 2*cfg.keyboard_acceleration_mm_s2/cfg.keyboard_jerk_mm_s3
                distance = min(cfg.keyboard_speed_mm_s*(cfg.keyboard_segment_s+stop_time)+2*cfg.keyboard_speed_mm_s*period, cfg.max_jog_mm)
                # Primitive contact/held state must not be bypassed by keyboard playback.
                if hasattr(skills, '_held_check'):
                    xyz = session.arm_position_mm()
                    lateral = bool(direction[0] or direction[1])
                    clear_z = session.grasp_z_mm + skills.limits.lateral_clearance_mm
                    if ((session.held is not None and direction[2] < 0)
                            or (lateral and xyz[2] < clear_z)
                            or (not skills._held_check() and (lateral or direction[2] < 0))):
                        result = skills._fail('move_relative', 'Lift/verify grasp before keyboard motion')
                        if path is not None and not ramp.stopped:travel(braking=True)
                        break
                    skills._contact = False
                    skills._target = None
                try:
                    result = skills.move_arm(*(v*distance for v in direction), _playback=playback)
                except JogDirectionChanged:
                    continue
                except JogPathRejected as exc:
                    result = skills._result(False, 'move_arm', 'out_of_workspace', str(exc), t0=time.monotonic())
                if not result.ok:
                    if path is not None and not ramp.stopped:travel(braking=True)
                    break
        except JogReleased:
            pass
        finally:
            self.close()
            if sent and not cancel.is_set():
                joints = session.robot.read_joints()
                session.robot.send_joints({j: v for j, v in joints.items() if j != 'gripper'})
                if session.held is not None:
                    session.held.over_xy_mm = tuple(session.ik.forward_position_mm(joints)[:2])
        return result

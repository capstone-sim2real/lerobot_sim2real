"""Latest-input keyboard stream. No queued moves; all bus work stays on RobotWorker."""
import math
import threading
import time
import secrets

from .jog_ramp import JogRamp


class JogReleased(Exception):
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
        from .jog_planner import plan_next, JogPathRejected

        cfg, session = self.cfg, skills.s
        period = 1. / cfg.keyboard_tick_hz
        ramp = JogRamp(cfg)
        if hasattr(skills, "_invalidate_pick"):
            skills._invalidate_pick()
        if hasattr(skills, 'attempt'):
            skills.attempt = None
            skills.descent_ready = False
        sent = False
        path = following = planner = None
        direction = None
        last = None
        failure = None
        failure_reason = 'out_of_workspace'
        next_tick = self.clock()
        next_grasp_check = next_tick

        def clearance():
            if session.held is not None and hasattr(skills, '_held_check') and (direction[0] or direction[1]):
                return session.grasp_z_mm + skills.limits.lateral_clearance_mm
            return None

        def begin():
            nonlocal planner, failure
            failure = None
            start = path.goal if path is not None else session.robot.read_joints()
            planner = plan_next(session, start, direction, clearance=clearance())

        def advance_plan():
            nonlocal planner, path, following, last, failure
            try:
                next(planner)
            except StopIteration as done:
                planner = None
                if path is None:
                    path = done.value
                    last = dict(path.start)
                else:
                    following = done.value
            except JogPathRejected as exc:
                planner = None
                failure = str(exc)

        try:
            while True:
                cancel.raise_if_set()
                desired = self.current()
                if not any(desired) and ramp.stopped:
                    if self.releasing:
                        break
                    direction = None
                    cancel.event.wait(period)
                    continue
                turning = direction is not None and desired != direction
                stopping = self.releasing or turning
                if stopping:
                    # Discard unfinished work immediately; an already validated
                    # continuation may still be needed for braking.
                    planner = None
                if ramp.stopped and (stopping or direction is None):
                    if self.releasing:
                        break
                    path = following = planner = None
                    failure = None
                    direction = desired
                    if not any(direction):
                        cancel.event.wait(period)
                        continue
                    if hasattr(skills, '_held_check') and session.held is not None:
                        xyz = session.arm_position_mm()
                        lateral = bool(direction[0] or direction[1])
                        if (direction[2] < 0 or (lateral and xyz[2] < clearance())
                                or (lateral and not skills._held_check())):
                            failure_reason = 'precondition'
                            failure = '블록을 든 상태입니다. 먼저 위로 들어 올리고 파지를 확인하세요.'
                            break
                    if hasattr(skills, '_held_check'):
                        skills._contact = False
                        skills._target = None
                    begin()
                    next_tick = self.clock()
                    stopping = False
                if path is None:
                    if failure:
                        break
                    advance_plan()
                    continue
                if not stopping and following is None and planner is None and failure is None:
                    begin()
                now = self.clock()
                if now < next_tick:
                    if planner is not None:
                        # One bounded numerical step, on RobotWorker. No second
                        # bus owner, no background IK thread, no queued commands.
                        advance_plan()
                    else:
                        cancel.event.wait(next_tick-now)
                    continue
                with self.lock:
                    cancel.raise_if_set()
                    desired = self.current()
                    stopping = self.releasing or desired != direction
                    remaining = path.length-path.progress
                    available = remaining + (following.length if following is not None else 0.)
                    reserve = ramp.braking_distance() + 2*cfg.keyboard_speed_mm_s*period
                    crossing_ready = (following is not None and
                        ramp.input.current_velocity[0] + max(0., ramp.input.current_acceleration[0])**2 /
                        (2*cfg.keyboard_jerk_mm_s3) <= following.speed + 1e-8)
                    brake = stopping or available <= reserve or (not crossing_ready and remaining <= reserve)
                    if brake and ramp.stopped:
                        if stopping:
                            continue
                        if failure:
                            break
                        # A slow planner may exhaust the braking reserve. Stay
                        # inside approved space and resume only when ready.
                        if planner is not None:
                            advance_plan()
                            continue
                        break
                    speed = path.speed
                    if following is not None:
                        # Respect both segment slopes before crossing the knot.
                        speed = min(speed, following.speed)
                    elif remaining <= reserve:
                        brake = True
                    delta = ramp.advance(0. if brake else speed)
                    if delta > available + 1e-8:
                        raise RuntimeError('keyboard braking path exhausted')
                    progress = path.progress + delta
                    if progress > path.length and following is not None:
                        progress -= path.length
                        path, following = following, None
                    if progress > path.length + 1e-8:
                        raise RuntimeError('keyboard path exhausted')
                    step = path.pose(min(path.length, progress))
                    limit = min(1., session.cfg.motion.max_step_per_tick, cfg.keyboard_joint_speed_deg_s*period)
                    if any(abs(v-last[j]) > limit+1e-8 for j,v in step.items()):
                        raise RuntimeError('keyboard joint step limit exceeded')
                    session.robot.send_joints(step)
                    last = step
                    path.progress = progress
                    sent = True
                # Never burst to catch up after an overrun.
                next_tick = max(next_tick+period, now+period)
                if self.clock() >= next_grasp_check:
                    next_grasp_check = self.clock()+cfg.keyboard_segment_s
                    if session.held is not None and hasattr(skills, '_held_check') and not skills._held_check():
                        self.release()
                        failure = '파지가 확인되지 않아 키보드 이동을 멈췄습니다.'
        except JogReleased:
            pass
        finally:
            self.close()
            if sent and not cancel.is_set():
                joints = session.robot.read_joints()
                session.robot.send_joints({j: v for j, v in joints.items() if j != 'gripper'})
                if session.held is not None:
                    session.held.over_xy_mm = tuple(session.ik.forward_position_mm(joints)[:2])
        if failure:
            return skills._result(False, 'move_arm', failure_reason, failure, t0=time.monotonic())
        return None

"""Incremental, bus-free planning on the same worker as keyboard playback."""
import math
from dataclasses import dataclass

from control.task1_transport import over_ik_gate, place_tilt_deg
from control.trajectory import interpolate
from session.relative import offset_xy


class JogPathRejected(Exception):
    pass


@dataclass
class JogPath:
    start: dict
    goal: dict
    length: float
    speed: float
    progress: float = 0.

    def pose(self, progress):
        fraction = progress / self.length
        return {j: a + (self.goal[j]-a)*fraction for j, a in self.start.items()}


def plan_span(session, start, direction, distance, *, clearance=None):
    """Yield after each solver iteration/FK sample; return a fully gated span.

    Only the first span starts from measured joints. Subsequent spans extend
    that same trajectory from its previous endpoint, never from lagging feedback.
    """
    cfg = session.cfg.agent.relative
    start = {j: v for j, v in start.items() if j != 'gripper'}
    a = session.ik.forward_position_mm(start)
    xy = offset_xy(a[:2], direction[0]*distance, direction[1]*distance,
                   frame=cfg.frame, base_xy_mm=session.base_xy)
    z = a[2] + direction[2]*distance
    slack = cfg.jog_max_ik_error_mm
    lo, hi = cfg.jog_min_z_mm-slack, cfg.jog_max_z_mm+slack
    if not lo <= a[2] <= hi:
        z = min(max(z, cfg.jog_min_z_mm), cfg.jog_max_z_mm)
    elif not lo <= z <= hi:
        raise JogPathRejected('키보드 이동 높이 한계에 도달했습니다.')
    if not session.in_workspace(xy):
        raise JogPathRejected('이동·감속 경로가 작업 범위를 벗어납니다.')
    yield
    kwargs = dict(radial_tilt_deg=place_tilt_deg(xy, session.base_xy, session.cfg))
    incremental = getattr(session.ik, 'solve_holding_wrist_roll_steps', None)
    if incremental is None:  # Lightweight test/simulation solvers only.
        solved = session.ik.solve_holding_wrist_roll(*xy, z, start['wrist_roll'], **kwargs)
    else:
        solved = yield from incremental(*xy, z, start['wrist_roll'], **kwargs)
    if (not math.isfinite(solved.position_error_mm) or over_ik_gate(solved, session.cfg)
            or solved.position_error_mm > slack):
        raise JogPathRejected('다음 키보드 경로가 IK 허용 오차를 벗어났습니다.')
    goal = {j: solved.joints[j] for j in start}
    b = session.ik.forward_position_mm(goal)
    length = math.dist(a, b)
    if length <= 1e-8:
        raise JogPathRejected('이동 가능한 경로가 없습니다.')
    lo, hi = min(a[2], lo), max(a[2], hi)
    if clearance is not None:
        lo = max(lo, clearance)
    step = min(1., session.cfg.motion.max_step_per_tick,
               cfg.keyboard_joint_speed_deg_s / cfg.keyboard_tick_hz)
    # Calibrated wrist bounds are checked before playback as well as by RobotIO.
    bounds = session.cfg.agent.calibration_clearance
    for pose in interpolate(start, goal, step):
        if not bounds.wrist_roll_min_deg <= pose['wrist_roll'] <= session.cfg.agent.primitives.wrist_roll_limit_deg:
            raise JogPathRejected('손목 회전 한계에 도달했습니다.')
        x, y, z = session.ik.forward_position_mm(pose)
        if (not all(math.isfinite(v) for v in (x,y,z)) or not session.in_workspace((x,y))
                or not lo-1e-8 <= z <= hi+1e-8):
            raise JogPathRejected('이동·감속 경로가 작업 범위를 벗어납니다.')
        yield
    largest = max(abs(goal[j]-start[j]) for j in start)
    speed = min(cfg.keyboard_speed_mm_s, step*cfg.keyboard_tick_hz*length/largest)
    return JogPath(start, goal, length, speed)


def plan_next(session, start, direction, *, clearance=None):
    cfg = session.cfg.agent.relative
    stop_time = cfg.keyboard_speed_mm_s/cfg.keyboard_acceleration_mm_s2 + 2*cfg.keyboard_acceleration_mm_s2/cfg.keyboard_jerk_mm_s3
    shorter = min(cfg.max_jog_mm, cfg.keyboard_speed_mm_s*(cfg.keyboard_segment_s+stop_time+2/cfg.keyboard_tick_hz))
    try:
        return (yield from plan_span(session, start, direction, cfg.max_jog_mm, clearance=clearance))
    except JogPathRejected:
        if shorter >= cfg.max_jog_mm:
            raise
        return (yield from plan_span(session, start, direction, shorter, clearance=clearance))

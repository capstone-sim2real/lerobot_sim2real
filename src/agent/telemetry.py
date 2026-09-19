"""Read-only diagnostics. Called exclusively by the existing robot worker."""
import math
import time
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import asdict


def collect(skills):
    s = skills.s
    robot = getattr(s, '_inner_robot', None)
    hardware = getattr(robot, 'robot', None)
    if hardware is None:
        return {'sampled_at': time.time(), 'error': '실제 모터 진단을 지원하지 않는 실행 환경', 'motors': []}
    bus = hardware.bus
    errors = []
    def read(register, normalize=False):
        try:
            return bus.sync_read(register, normalize=normalize)
        except Exception as exc:
            errors.append(f'{register}: {type(exc).__name__}')
            return {}
    raw = read('Present_Position')
    current = read('Present_Position', True)
    goal = read('Goal_Position', True)
    goal_raw = read('Goal_Position')
    loads = read('Present_Load')
    temp = read('Present_Temperature')
    volts = read('Present_Voltage')
    torque = read('Torque_Enable')
    try:
        live = bus.read_calibration()
        match = all(asdict(live[n]) == asdict(c) for n, c in bus.calibration.items())
    except Exception as exc:
        match = None
        errors.append(f'calibration: {type(exc).__name__}')
    limits = {}
    try:
        root = ET.parse(Path(__file__).resolve().parents[2] / s.cfg.ik.urdf_path)
        for joint in root.findall("joint"):
            limit = joint.find("limit")
            if limit is not None and "lower" in limit.attrib:
                limits[joint.attrib["name"]] = [math.degrees(float(limit.attrib[k])) for k in ("lower", "upper")]
    except (OSError, ValueError, ET.ParseError) as exc:
        errors.append(f'URDF: {type(exc).__name__}')
    motors = []
    for name, motor in bus.motors.items():
        c = bus.calibration[name]
        mode = motor.norm_mode.name
        span = (c.range_max-c.range_min)*360/(bus.model_resolution_table[motor.model]-1)
        lo, hi = (-span/2, span/2) if mode == 'DEGREES' else ((0,100) if mode=='RANGE_0_100' else (-100,100))
        now, target = current.get(name), goal.get(name)
        motors.append(dict(name=name, id=motor.id, unit='°' if mode=='DEGREES' else '% (정규화)',
            current=now, target=target, error=None if now is None or target is None else target-now,
            minimum=lo, maximum=hi, encoder=raw.get(name), goal_encoder=goal_raw.get(name),
            load=loads.get(name), temperature=temp.get(name), voltage_raw=volts.get(name),
            torque=torque.get(name), urdf_limits_deg=limits.get(name), calibration=asdict(c)))
    fk = target_fk = delta = None
    try:
        fk = list(s.ik.forward_position_mm(current))
        target_fk = list(s.ik.forward_position_mm(goal))
        delta = [b-a for a,b in zip(fk,target_fk)]
    except Exception as exc:
        errors.append(f'FK: {type(exc).__name__}')
    return dict(sampled_at=time.time(), motors=motors, fk=fk, target_fk=target_fk,
        fk_delta=delta, fk_distance=None if delta is None else math.sqrt(sum(x*x for x in delta)),
        calibration_file=str(hardware.calibration_fpath), calibration_match=match,
        max_relative_target=s.cfg.robot.max_relative_target,
        grasp={'held': s.held is not None, 'mode':s.cfg.sensing.grasp_check_mode, 'position_threshold':s.cfg.sensing.gripper_empty_closed_max,
               'load_threshold':s.cfg.sensing.gripper_load_min,
               'note':'파지 판정은 닫기 후 VERIFY 결과를 확인하세요. 열린 그리퍼의 위치·부하만으로 파지 여부를 판단하지 않습니다.'},
        errors=errors)

"""Continuous input safety using fake IK and mock motor IO only."""
import math
import time

import pytest
pytest.importorskip("ruckig")
from agent.keyboard_jog import KeyboardJog, JogReleased
from agent_helpers import make_skills
from session.cancel import Cancelled


def setup():
    sk, _, robot = make_skills({})
    robot.joints['elbow_flex'] = 60.
    sk.cfg.agent.relative.keyboard_tick_hz = 200.
    return sk, robot


def test_latest_sequence_normalization_and_terminal_stop():
    sk, _ = setup()
    now = [1.]
    stream = KeyboardJog(sk.cfg.agent.relative, clock=lambda:now[0])
    assert stream.update(1, [1, 1, 1])
    assert math.hypot(*stream.current()) == pytest.approx(1)
    assert not stream.update(0, [-1, 0, 0])
    now[0] += sk.cfg.agent.relative.keyboard_timeout_s
    assert not stream.update(2, [1, 0, 0])
    with pytest.raises(JogReleased):stream.current()
    stopped = KeyboardJog(sk.cfg.agent.relative)
    assert stopped.update(0, [0, 0, 0])
    assert not stopped.update(1, [1, 0, 0])


def test_release_interrupts_playback_and_holds_without_gripper_or_settle(monkeypatch):
    sk, robot = setup()
    stream = KeyboardJog(sk.cfg.agent.relative)
    stream.update(0, [1, 0, 0])
    sk.attempt = object();sk.descent_ready = True
    def forbidden(*a, **kw):raise AssertionError('continuous jog must not settle at each step')
    monkeypatch.setattr(sk.s.player, 'move_to', forbidden)
    original = sk.s.robot.send_joints
    poses = [dict(robot.joints)]
    def send(pose):
        poses.append(dict(pose))
        result = original(pose)
        if len(poses) == 5:stream.close()
        return result
    monkeypatch.setattr(sk.s.robot, 'send_joints', send)
    stream.run(sk, sk.s.cancel)
    assert sk.attempt is None and not sk.descent_ready
    assert len(poses) == 6  # four motion ticks then measured-pose hold
    limit = min(sk.cfg.motion.max_step_per_tick, sk.cfg.agent.relative.keyboard_joint_speed_deg_s / sk.cfg.agent.relative.keyboard_tick_hz)
    for before, after in zip(poses, poses[1:]):
        assert 'gripper' not in after
        assert max(abs(v-before[j]) for j,v in after.items()) <= limit + 1e-8
    assert stream.closed


def test_deadman_during_slow_ik_prevents_late_motion(monkeypatch):
    sk, robot = setup()
    sk.cfg.agent.relative.keyboard_timeout_s = .02
    stream = KeyboardJog(sk.cfg.agent.relative)
    stream.update(0, [1, 0, 0])
    original = sk.s.ik.solve_holding_wrist_roll
    def slow(*a, **kw):
        time.sleep(.04)
        return original(*a, **kw)
    monkeypatch.setattr(sk.s.ik, 'solve_holding_wrist_roll', slow)
    stream.run(sk, sk.s.cancel)
    assert not robot.sent_actions


def test_stop_cancels_before_motor_write():
    sk, robot = setup()
    stream = KeyboardJog(sk.cfg.agent.relative)
    stream.update(0, [1, 0, 0]);sk.s.cancel.set()
    with pytest.raises(Cancelled):stream.run(sk, sk.s.cancel)
    assert not robot.sent_actions


def test_workspace_and_ik_gate_still_apply(monkeypatch):
    sk, robot = setup()
    stream = KeyboardJog(sk.cfg.agent.relative)
    stream.update(0, [1, 0, 0])
    monkeypatch.setattr(sk.s, 'in_workspace', lambda p:False)
    result = stream.run(sk, sk.s.cancel)
    assert not result.ok and result.reason == 'out_of_workspace'
    assert not robot.sent_actions


def test_start_reserves_worker_without_moving_and_http_contract():
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from agent.provider.fake import ScriptedProvider
    from agent.server import EventHub, create_app, CONTROL_UI_VERSION
    from agent.service import AgentService
    sk, robot = setup()
    holder = {}
    def builder(publish):
        svc = AgentService(sk.cfg, provider=ScriptedProvider([]), skills_factory=lambda:sk,
                           cancel=sk.s.cancel, publish=publish, system_prompt='', transcript_dir='')
        svc.start = lambda: (svc._worker.start(), setattr(svc, 'started', True))
        holder['service'] = svc
        return svc
    with TestClient(create_app(sk.cfg, builder, EventHub())) as client:
        headers={'x-so101-control-version':CONTROL_UI_VERSION}
        assert client.post('/api/keyboard/start',headers=headers).status_code==403
        headers['x-operator-token']=client.post('/api/lease').json()['token']
        assert client.post('/api/keyboard/start',headers={'x-operator-token':headers['x-operator-token']}).status_code==428
        start=client.post('/api/keyboard/start',headers=headers)
        assert start.status_code==202
        sid=start.json()['session_id']
        assert not robot.sent_actions
        assert client.post('/api/jog',headers=headers,json={'forward_mm':1}).status_code==409
        def update(seq, vector, session_id=sid):
            return client.post('/api/keyboard/update',headers=headers,json={'session_id':session_id,'seq':seq,'vector':vector})
        assert update(0,[2,0,0]).status_code==400
        assert update(0,[1,0]).status_code==400
        assert update(0,[True,0,0]).status_code==400
        assert update(0,[1,0,0],'wrong').status_code==409
        assert update(0,[1,0,0]).status_code==200
        assert update(0,[-1,0,0]).status_code==409
        assert client.post('/api/keyboard/stop',headers=headers,json={'session_id':sid}).status_code==200
        assert update(1,[1,0,0]).status_code==409
        holder['service'].wait_idle()
        assert holder['service'].gate.snapshot()['state']=='idle'


def test_changed_direction_during_ik_discards_old_plan(monkeypatch):
    sk, robot = setup()
    stream = KeyboardJog(sk.cfg.agent.relative)
    stream.update(0, [1, 0, 0])
    initial = robot.joints['shoulder_pan']
    solve = sk.s.ik.solve_holding_wrist_roll
    def change(*args, **kwargs):
        result = solve(*args, **kwargs)
        if sk.s.ik.solves == 1:stream.update(1, [-1, 0, 0])
        return result
    monkeypatch.setattr(sk.s.ik, 'solve_holding_wrist_roll', change)
    send = sk.s.robot.send_joints
    def release(pose):
        result = send(pose)
        stream.close()
        return result
    monkeypatch.setattr(sk.s.robot, 'send_joints', release)
    stream.run(sk, sk.s.cancel)
    assert sk.s.ik.solves == 2
    assert robot.sent_actions[0]['shoulder_pan'] < initial


def test_continuous_path_keeps_shared_wrist_limit(monkeypatch):
    sk, robot = setup()
    from session.calibration_joint_limit import CalibrationJointLimitIO
    cfg = sk.cfg.agent.calibration_clearance
    cfg.wrist_roll_min_deg = -65.
    sk.s.robot = CalibrationJointLimitIO(sk.s.robot, cfg)
    stream = KeyboardJog(sk.cfg.agent.relative)
    stream.update(0, [1, 0, 0])
    robot.joints['wrist_roll'] = -65.
    solve = sk.s.ik.solve_holding_wrist_roll
    def outside(*args, **kwargs):
        result = solve(*args, **kwargs)
        result.joints['wrist_roll'] = -66.
        return result
    monkeypatch.setattr(sk.s.ik, 'solve_holding_wrist_roll', outside)
    with pytest.raises(ValueError, match='below measured limit'):
        stream.run(sk, sk.s.cancel)
    assert not robot.sent_actions


def test_normal_release_decelerates_but_remains_terminal(monkeypatch):
    sk, robot = setup()
    sk.cfg.agent.relative.keyboard_tick_hz = 30.
    stream = KeyboardJog(sk.cfg.agent.relative)
    stream.update(0,[1,0,0])
    send = sk.s.robot.send_joints
    samples=[]
    def release_after_acceleration(pose):
        result=send(pose)
        samples.append(pose['shoulder_pan'])
        if len(samples)==8:
            stream.release()
            assert not stream.update(100,[1,0,0])
        elif len(samples)<8:
            stream.update(len(samples),[1,0,0])
        return result
    monkeypatch.setattr(sk.s.robot,'send_joints',release_after_acceleration)
    stream.run(sk,sk.s.cancel)
    assert len(samples)>10
    deltas=[b-a for a,b in zip(samples,samples[1:])]
    assert max(deltas[:7])>deltas[-2]
    assert deltas[-1]==pytest.approx(0,abs=1e-8)
    assert stream.closed


def test_failed_next_plan_brakes_inside_previously_approved_path(monkeypatch):
    sk,robot=setup()
    stream=KeyboardJog(sk.cfg.agent.relative)
    stream.update(0,[1,0,0])
    move=sk.move_arm
    calls=[]
    def only_first(*args,**kwargs):
        calls.append(args)
        if len(calls)>1:
            return sk._result(False,'move_arm','out_of_workspace','boundary',t0=time.monotonic())
        return move(*args,**kwargs)
    monkeypatch.setattr(sk,'move_arm',only_first)
    initial=robot.joints['shoulder_pan']
    result=stream.run(sk,sk.s.cancel)
    assert result.reason=='out_of_workspace'
    assert len(robot.sent_actions)>1
    assert max(p['shoulder_pan'] for p in robot.sent_actions)<=initial+calls[0][0]+1e-8


def test_emergency_stop_interrupts_normal_deceleration(monkeypatch):
    sk,robot=setup()
    stream=KeyboardJog(sk.cfg.agent.relative)
    stream.update(0,[1,0,0])
    send=sk.s.robot.send_joints
    count=[0]
    def stop_during_brake(pose):
        result=send(pose);count[0]+=1
        if count[0]==5:stream.release()
        if count[0]==6:sk.s.cancel.set()
        return result
    monkeypatch.setattr(sk.s.robot,'send_joints',stop_during_brake)
    with pytest.raises(Cancelled):stream.run(sk,sk.s.cancel)
    assert count[0]==6


def test_intermediate_braking_path_is_checked_before_motion(monkeypatch):
    sk,robot=setup()
    stream=KeyboardJog(sk.cfg.agent.relative)
    stream.update(0,[1,0,0])
    # Endpoint is allowed, but an interior band is excluded.
    monkeypatch.setattr(sk.s,'in_workspace',lambda xy:not 151.<xy[0]<152.)
    result=stream.run(sk,sk.s.cancel)
    assert not result.ok and result.reason=='out_of_workspace'
    assert not robot.sent_actions


def test_primitive_keyboard_rejects_low_lateral_motion():
    from session.primitives import PrimitiveSkills
    sk, robot = setup()
    sk = PrimitiveSkills(sk.s)
    sk.cfg.agent.primitives.lateral_clearance_mm = 200.
    stream = KeyboardJog(sk.cfg.agent.relative)
    stream.update(0, [1, 0, 0])
    result = stream.run(sk, sk.s.cancel)
    assert not result.ok and result.reason == 'precondition'
    assert not robot.sent_actions

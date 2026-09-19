"""Calibration web compatibility without physical hardware or live HTTP servers."""
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from agent.provider.fake import ScriptedProvider
from agent.provider.types import ToolCall
from agent.server import EventHub, create_app, CONTROL_UI_VERSION
from agent.service import AgentService
from agent.tools import ToolRegistry
from agent_helpers import make_skills
from session.calibration_joint_limit import CalibrationJointLimitIO
from session.cancel import Cancelled
from tools.astra_calibration_server import configure_manual_tools, definitions, WEB_MANUAL_TOOLS


def configured(sk):
    service=SimpleNamespace(registry=ToolRegistry(sk.cfg,lambda job:job(sk)))
    configure_manual_tools(service,sk.cfg)
    return service


def test_web_buttons_have_an_explicit_calibration_policy():
    sk,_,_=make_skills({})
    service=configured(sk)
    script=(Path(__file__).resolve().parents[1]/'src/agent/web/app.js').read_text()
    web=set(re.findall(r"manual\([\"']([a-z_]+)[\"']",script))|{'move_arm'}
    assert web-service.MANUAL_TOOLS <= {'pick_here','move_to_pixel','rotate_gripper'}
    assert WEB_MANUAL_TOOLS <= service.MANUAL_TOOLS
    assert {d.spec.name for d in definitions(sk.cfg)}<=service.MANUAL_TOOLS
    assert service.MANUAL_TOOLS==service.registry._tools.keys()


def test_jog_preserves_standard_gates_and_invalidates_descent_authorization():
    sk,_,robot=make_skills({})
    sk.attempt=object();sk.descent_ready=True
    registry=configured(sk).registry
    result=registry.execute(ToolCall('jog','move_arm',{'forward_mm':5.0}))
    assert result.content['ok'],result.content
    assert sk.attempt is None and sk.descent_ready is False
    sent=len(robot.sent_actions)
    result=registry.execute(ToolCall('large','move_arm',{'forward_mm':9999}))
    assert result.content['reason']=='invalid_arguments'
    assert len(robot.sent_actions)==sent
    sk.s.cancel.set()
    result=registry.execute(ToolCall('cancel','move_arm',{'forward_mm':5.0}))
    assert result.content['reason']=='cancelled'
    assert len(robot.sent_actions)==sent


def test_unsupported_descent_pick_and_rotation_never_execute():
    sk,_,robot=make_skills({})
    registry=configured(sk).registry
    for name in ('pick_here','move_to_pixel','rotate_gripper'):
        result=registry.execute(ToolCall(name,name,{}))
        assert result.is_error
    assert not robot.sent_actions


def test_shared_minus_65_wrist_guard_remains_on_mock_write_path():
    sk,_,robot=make_skills({})
    cfg=sk.cfg.agent.calibration_clearance
    cfg.wrist_roll_min_deg=-65.0
    guard=CalibrationJointLimitIO(robot,cfg)
    robot.joints['wrist_roll']=-60.0
    before=len(robot.sent_actions)
    with pytest.raises(ValueError,match='below measured limit'):
        guard.send_joints({'wrist_roll':-66.0})
    assert len(robot.sent_actions)==before
    guard.send_joints({'wrist_roll':-65.0})
    assert robot.sent_actions[-1]['wrist_roll']==-65.0


def test_http_config_jog_lease_and_denied_manual_tools():
    pytest.importorskip('fastapi');pytest.importorskip('httpx')
    from fastapi.testclient import TestClient
    sk,_,_=make_skills({})
    from session.primitives import PrimitiveSkills
    sk=PrimitiveSkills(sk.s)
    cfg=sk.cfg
    hub=EventHub();holder={};events=[]
    def builder(publish):
        svc=AgentService(cfg,provider=ScriptedProvider([]),skills_factory=lambda:sk,
                         cancel=sk.s.cancel,publish=lambda e:(events.append(e),publish(e)),system_prompt='',transcript_dir='')
        configure_manual_tools(svc,cfg)
        svc.start=lambda: (svc._worker.start(),setattr(svc,'started',True))
        holder['svc']=svc
        return svc
    with TestClient(create_app(cfg,builder,hub)) as client:
        allowed=client.get('/api/config').json()['manual_tools']
        assert 'move_arm' in allowed and 'pick_here' not in allowed
        version={'x-so101-control-version':CONTROL_UI_VERSION}
        assert client.post('/api/jog',json={'forward_mm':5},headers=version).status_code==403
        token=client.post('/api/lease').json()['token']
        headers={**version,'x-operator-token':token}
        assert client.post('/api/jog',json={'forward_mm':5},headers=headers).status_code==202
        holder['svc'].wait_idle()
        result=next(e for e in events if e['type']=='tool_result')
        assert result['name']=='move_arm' and result['result']['ok'],result
        assert client.post('/api/manual',json={'tool':'pick_here'},headers=headers).status_code==400
        assert client.post('/api/manual',json={'tool':'move_to_pixel'},headers=headers).status_code==400

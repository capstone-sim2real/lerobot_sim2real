"""Calibration web compatibility without physical hardware or live HTTP servers."""
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
from agent.panel import CalibrationSkills, configure_manual_tools


def configured(sk):
    service=SimpleNamespace(registry=ToolRegistry(sk.cfg,lambda job:job(sk)))
    configure_manual_tools(service,sk.cfg)
    return service


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
        assert client.post('/api/manual',json={'tool':'move_to_pixel'},headers=headers).status_code==202
        holder['svc'].wait_idle()
        missing = [e for e in events if e['type']=='tool_result' and e['name']=='move_to_pixel'][-1]
        assert missing['result']['reason']=='invalid_arguments'


def test_manual_registration_preserves_pixel_calibration_argument(monkeypatch):
    from unittest.mock import Mock
    from session.results import SkillResult
    sk, _, _ = make_skills({})
    place = Mock(return_value=SkillResult(True, "place_at_pixel", "released"))
    monkeypatch.setattr(sk, "place_at_pixel", place)
    svc = configured(sk)
    spec = next(s for s in svc.registry.specs() if s.name == "place_at_pixel")
    assert "calibration_id" in spec.input_schema["required"]
    bad = svc.registry.execute(ToolCall("missing", "place_at_pixel", {"u": 354, "v": 462}))
    assert bad.content["reason"] == "invalid_arguments"
    place.assert_not_called()
    args = {"u": 354, "v": 462, "calibration_id": "8a8be3855592f93e"}
    result = svc.registry.execute(ToolCall("valid", "place_at_pixel", args))
    assert not result.is_error
    place.assert_called_once_with(**args)


def test_calibration_prepare_adjust_stop_and_grasp_in_simulation(tmp_path):
    sk, _, _ = make_skills({"green": (180, 0)})
    cal = CalibrationSkills(sk.s, tmp_path)
    assert cal.calibration_prepare("green").ok
    assert cal.calibration_adjust(left_mm=2).ok
    assert cal.calibration_adjust(left_mm=999).reason == "limit_exceeded"
    cal.s.cancel.set()
    with pytest.raises(Cancelled):
        cal.calibration_adjust(left_mm=1)
    assert cal.attempt is None
    cal.s.cancel.clear()
    assert cal.calibration_prepare("green").ok
    assert cal.calibration_grasp().ok

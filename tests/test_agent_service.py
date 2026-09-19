"""AgentService threads: chat, busy lock, STOP during a slow skill, home, jog."""

import threading
import time

from agent.control import ControlState
from agent.provider.fake import ScriptedProvider
from agent.provider.types import TextDelta, ToolCall, ToolCallEvent, TurnEnd
from agent.service import AgentService, needs_fresh_scene
from config import AppConfig
from session.cancel import Cancelled, CancelToken
from session.results import SkillResult


class Skills:
    """Stand-in for session.skills.Skills; the slow skill honours the token."""

    def __init__(self, cancel):
        self.cancel = cancel
        self.started = threading.Event()
        self.homed = 0
        self.at_home = True
        self.s = self

    @property
    def slot_centres(self):
        return []

    def state_dict(self):
        return {}

    def close(self):
        pass

    def run_task(self, task):
        self.started.set()
        for _ in range(200):
            if self.cancel.is_set():
                raise Cancelled("stop")
            time.sleep(0.01)
        return SkillResult(True, f"run_task{task}", "ok")

    def get_state(self):
        return SkillResult(True, "get_state", "ok")

    def observe_scene(self, include_zone=True):
        return SkillResult(
            True,
            "observe_scene",
            "ok",
            data={
                "blocks_outside": [{"color": "red", "x_mm": 100.0, "y_mm": 0.0}],
                "blocks_inside": [{"color": "blue", "slot": "top-left"}] if include_zone else None,
            },
        )

    def move_arm(self, forward_mm, left_mm, up_mm):
        return SkillResult(True, "move_arm", "moved")

    def rotate_gripper(self, delta_deg):
        return SkillResult(True, "rotate_gripper", "moved")

    def pick_here(self):
        return SkillResult(True, "pick_here", "held")

    def place_here(self):
        return SkillResult(True, "place_here", "released")

    def open_gripper(self):
        return SkillResult(True, "open_gripper", "ok")

    def recover_and_home(self):
        self.homed += 1
        return SkillResult(self.at_home, "recover_and_home", "ok" if self.at_home else "motion_timeout",
                           data={"arm_at_home": self.at_home})


def _service(script, cfg=None):
    cfg = cfg or AppConfig()
    cancel = CancelToken()
    holder = {}
    events = []

    def factory():
        holder["skills"] = Skills(cancel)
        return holder["skills"]

    service = AgentService(cfg, provider=ScriptedProvider(script), skills_factory=factory, cancel=cancel,
                           publish=events.append, system_prompt="sys", transcript_dir="")
    service._worker.start()
    service.started = True
    return service, holder["skills"], events


def test_chat_runs_and_returns_to_idle():
    service, _skills, events = _service([
        [ToolCallEvent(ToolCall("a", "get_state", {})), TurnEnd("tool_use")],
        [TextDelta("ok"), TurnEnd("end_turn")],
    ])
    token = service.acquire_lease(None)
    assert service.chat(None, "hi")[0] == 403
    assert service.chat(token, "상태")[0] == 202
    service.wait_idle()
    assert service.gate.state is ControlState.IDLE
    assert any(e["type"] == "tool_result" for e in events)
    service.shutdown()


def test_scene_request_injects_fresh_cv_before_the_llm_turn():
    service, _skills, events = _service([[TextDelta("확인했어요"), TurnEnd("end_turn")]])
    token = service.acquire_lease(None)

    assert service.chat(token, "안에 있는 블록을 알려줘")[0] == 202
    service.wait_idle()

    sent = service.provider.seen_messages[0][-1].text
    assert '<current_cv source="server_preflight">' in sent
    assert '"blocks_inside": [{"color": "blue", "slot": "top-left"}]' in sent
    automatic = [e for e in events if e.get("automatic")]
    assert [e["type"] for e in automatic] == ["tool_call", "tool_result"]
    assert automatic[1]["result"]["ok"] is True
    service.shutdown()


def test_manual_jog_and_gripper_language_do_not_force_home_observation():
    assert not needs_fresh_scene("오른쪽으로 20mm 이동해")
    assert not needs_fresh_scene("그리퍼를 45도 돌려줘")
    assert needs_fresh_scene("안에 있는 거 전부 밖으로 꺼내줘")
    assert needs_fresh_scene("빨간 블록을 오른쪽으로 20mm 옮겨줘")


def test_stop_during_a_long_skill_auto_recovers_and_can_retry_manually():
    service, skills, events = _service([[ToolCallEvent(ToolCall("a", "run_task1", {})), TurnEnd("tool_use")]])
    token = service.acquire_lease(None)
    assert service.chat(token, "미션 1")[0] == 202
    assert skills.started.wait(5)
    assert service.chat(token, "상태")[0] == 409
    assert service.jog(token, 0, 10, 0)[0] == 409
    skills.at_home = False  # first, automatic recovery attempt does not reach home
    t0 = time.monotonic()
    assert service.stop()["stopped"] is True
    assert time.monotonic() - t0 < 0.05
    service.wait_idle()
    # stop_auto_home is on by default: recovery (drop + home + close) already ran once, unasked
    assert service.gate.state is ControlState.STOPPED and skills.homed == 1
    assert service.chat(token, "상태")[0] == 409
    assert service.home(None)[0] == 403

    assert service.home(token)[0] == 202  # manual retry, still not home
    service.wait_idle()
    assert service.gate.state is ControlState.STOPPED and skills.homed == 2

    skills.at_home = True
    assert service.home(token)[0] == 202
    service.wait_idle()
    assert service.gate.state is ControlState.IDLE and skills.homed == 3
    assert not service.cancel.is_set()
    result = next(e for e in events if e["type"] == "tool_result" and e["name"] == "run_task1")
    assert result["result"]["reason"] == "cancelled"
    service.shutdown()


def test_stop_auto_home_can_be_disabled_for_a_manual_only_recovery():
    cfg = AppConfig()
    cfg.agent.stop_auto_home = False
    service, skills, _events = _service([[ToolCallEvent(ToolCall("a", "run_task1", {})), TurnEnd("tool_use")]], cfg)
    token = service.acquire_lease(None)
    service.chat(token, "미션 1")
    assert skills.started.wait(5)
    service.stop()
    service.wait_idle()
    # no automatic recovery: stays STOPPED until the operator clicks it
    assert service.gate.state is ControlState.STOPPED and skills.homed == 0
    assert service.home(token)[0] == 202
    service.wait_idle()
    assert service.gate.state is ControlState.IDLE and skills.homed == 1
    service.shutdown()


def test_direct_jog_goes_through_the_same_tool_boundary():
    service, _skills, events = _service([])
    token = service.acquire_lease(None)
    assert service.jog(token, 0.0, 10.0, 0.0)[0] == 202
    service.wait_idle()
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["name"] == "move_arm" and result["result"]["ok"]
    service.jog(token, 0.0, 500.0, 0.0)
    service.wait_idle()
    last = [e for e in events if e["type"] == "tool_result"][-1]
    assert last["result"]["reason"] == "invalid_arguments"
    assert service.gate.state is ControlState.IDLE
    service.shutdown()


def test_direct_covers_the_manual_control_panel_actions():
    service, _skills, events = _service([])
    token = service.acquire_lease(None)
    actions = [("rotate_gripper", {"delta_deg": 30}), ("pick_here", {}), ("place_here", {}), ("open_gripper", {})]
    for tool, arguments in actions:
        assert service.direct(token, tool, arguments)[0] == 202
        service.wait_idle()
        result = next(e for e in reversed(events) if e["type"] == "tool_result" and e["name"] == tool)
        assert result["result"]["ok"], result["result"]
    service.shutdown()


def test_direct_refuses_tools_outside_the_manual_allowlist():
    service, skills, _events = _service([])
    token = service.acquire_lease(None)
    status, body = service.direct(token, "run_task1", {})
    assert status == 400 and "error" in body
    service.wait_idle()
    assert not skills.started.is_set()  # nothing was ever dispatched to the robot thread
    assert service.gate.state is ControlState.IDLE
    service.shutdown()


def test_new_operator_joins_shared_conversation_and_new_session_resets_it():
    service, _skills, events = _service([[TextDelta("hi"), TurnEnd("end_turn")]])
    token = service.acquire_lease(None)
    service.chat(token, "안녕")
    service.wait_idle()
    assert service.runner.history
    joined = service.acquire_lease(None)
    assert joined == token and service.runner.history
    assert service.gate.release_lease(token)
    service.acquire_lease(None)
    assert service.runner.history == [] and events[-1]["type"] == "reset"
    service.shutdown()

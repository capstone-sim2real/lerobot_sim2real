"""The LLM loop, driven by scripted providers (no network, no API key)."""

from agent.provider.fake import RuleBasedFakeProvider, ScriptedProvider
from agent.provider.types import Message, TextDelta, ToolCall, ToolCallEvent, TurnEnd
from agent.runner import AgentRunner
from agent.tools import ToolRegistry
from config import AppConfig
from session.results import SkillResult


class Skills:
    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    def state_dict(self):
        return {"holding": None}

    def __getattr__(self, name):
        def skill(*args, **kwargs):
            self.calls.append((name, kwargs))
            return self.results.get(name, SkillResult(True, name, "ok", "done"))
        return skill


def _runner(script, skills=None, **kwargs):
    cfg = AppConfig()
    skills = skills or Skills()
    events = []
    provider = ScriptedProvider(script)
    runner = AgentRunner(provider, ToolRegistry(cfg, lambda job: job(skills)), "system", cfg.agent,
                         emit=events.append, **kwargs)
    return runner, provider, skills, events


def _call(name, args, cid="t1"):
    return ToolCallEvent(ToolCall(cid, name, args))


def test_happy_path_runs_one_skill_and_feeds_the_result_back():
    runner, provider, skills, events = _runner([
        [TextDelta("옮길게요"), _call("move_to_target", {"target_type": "slot", "phase": "preplace", "slot": "top-left"}), TurnEnd("tool_use")],
        [TextDelta("완료했어요"), TurnEnd("end_turn")],
    ])
    outcome = runner.run_turn("노란 블록 좌상단으로")
    assert not outcome.robot_fault and outcome.error is None
    assert skills.calls == [("move_to_target", {"target_type": "slot", "phase": "preplace", "slot": "top-left"})]
    fed_back = provider.seen_messages[1][-1]
    assert fed_back.tool_results[0].content["ok"] is True
    kinds = [e["type"] for e in events if e["type"] != "text_delta"]
    assert kinds == ["assistant_text", "tool_call", "tool_result", "assistant_text", "turn_end"]


def test_robot_fault_halts_remaining_calls_and_the_conversation():
    cancelled = SkillResult(False, "close_gripper", "cancelled", "정지", "do_not_retry")
    runner, provider, skills, events = _runner([
        [_call("close_gripper", {}, "a"), TurnEnd("tool_use")],
        [TextDelta("never requested"), TurnEnd("end_turn")],
    ], skills=Skills({"close_gripper": cancelled}))
    outcome = runner.run_turn("빨간 블록 옮겨")
    assert outcome.robot_fault and outcome.stopped
    assert [c[0] for c in skills.calls] == ["close_gripper"]
    assert len(provider.seen_messages) == 1
    assert runner.history[-1].role == "assistant"
    assert events[-1] == {"type": "turn_end", "robot_fault": True}


def test_stop_pressed_while_the_model_talks_prevents_the_motion():
    runner, _provider, skills, _ = _runner(
        [[_call("close_gripper", {}), TurnEnd("tool_use")]], should_stop=lambda: True
    )
    outcome = runner.run_turn("집어")
    assert outcome.robot_fault and skills.calls == []

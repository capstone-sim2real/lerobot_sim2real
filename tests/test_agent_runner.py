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
        def skill(*args):
            self.calls.append((name, args))
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
        [TextDelta("옮길게요"), _call("move_block_to_slot", {"color": "yellow", "slot": "top-left"}), TurnEnd("tool_use")],
        [TextDelta("완료했어요"), TurnEnd("end_turn")],
    ])
    outcome = runner.run_turn("노란 블록 좌상단으로")
    assert not outcome.robot_fault and outcome.error is None
    assert skills.calls == [("move_block_to_slot", ("yellow", 0))]
    fed_back = provider.seen_messages[1][-1]
    assert fed_back.tool_results[0].content["ok"] is True
    kinds = [e["type"] for e in events if e["type"] != "text_delta"]
    assert kinds == ["assistant_text", "tool_call", "tool_result", "assistant_text", "turn_end"]


def test_failure_reaches_the_model_verbatim_and_is_not_retried_by_the_loop():
    failed = SkillResult(False, "pick_block", "grasp_empty", "못 집음", "ask_operator")
    runner, provider, skills, _ = _runner([
        [_call("pick_block", {"color": "red"}), TurnEnd("tool_use")],
        [TextDelta("집지 못했어요"), TurnEnd("end_turn")],
    ], skills=Skills({"pick_block": failed}))
    runner.run_turn("빨간 블록 집어")
    assert len(skills.calls) == 1
    assert provider.seen_messages[1][-1].tool_results[0].content["retry_advice"] == "ask_operator"


def test_parallel_calls_come_back_in_one_message():
    runner, provider, skills, _ = _runner([
        [_call("get_state", {}, "a"), _call("describe_places", {}, "b"), TurnEnd("tool_use")],
        [TurnEnd("end_turn")],
    ])
    runner.run_turn("상태와 장소")
    last = provider.seen_messages[1][-1]
    assert [r.call_id for r in last.tool_results] == ["a", "b"]


def test_robot_fault_halts_remaining_calls_and_the_conversation():
    cancelled = SkillResult(False, "pick_block", "cancelled", "정지", "do_not_retry")
    runner, provider, skills, events = _runner([
        [_call("pick_block", {"color": "red"}, "a"), _call("place_on_table", {}, "b"), TurnEnd("tool_use")],
        [TextDelta("never requested"), TurnEnd("end_turn")],
    ], skills=Skills({"pick_block": cancelled}))
    outcome = runner.run_turn("빨간 블록 옮겨")
    assert outcome.robot_fault and outcome.stopped
    assert [c[0] for c in skills.calls] == ["pick_block"]
    assert len(provider.seen_messages) == 1
    assert runner.history[-1].role == "assistant"
    assert events[-1] == {"type": "turn_end", "robot_fault": True}


def test_stop_pressed_while_the_model_talks_prevents_the_motion():
    runner, _provider, skills, _ = _runner(
        [[_call("pick_block", {"color": "red"}), TurnEnd("tool_use")]], should_stop=lambda: True
    )
    outcome = runner.run_turn("집어")
    assert outcome.robot_fault and skills.calls == []


def test_endless_tool_calls_stop_at_the_turn_limit():
    cfg_turns = AppConfig().agent.max_tool_turns
    runner, provider, _skills, events = _runner(
        [[_call("get_state", {}, f"c{i}"), TurnEnd("tool_use")] for i in range(cfg_turns + 5)]
    )
    outcome = runner.run_turn("loop")
    assert outcome.error == "max_tool_turns exceeded"
    assert len(provider.seen_messages) == cfg_turns


def test_llm_error_leaves_history_retryable():
    def boom(_messages):
        raise ConnectionError("network down")

    runner, _provider, _skills, events = _runner([boom])
    outcome = runner.run_turn("안녕")
    assert outcome.error and runner.history == []
    assert any(e["type"] == "error" for e in events)


def test_history_trim_never_starts_with_orphan_tool_results():
    runner, _provider, _skills, _ = _runner([])
    runner.cfg.max_history_messages = 4
    runner.history = [
        Message("user", text="1"), Message("assistant", tool_calls=(ToolCall("a", "get_state", {}),)),
        Message("user", tool_results=()), Message("assistant", text="ok"),
        Message("user", text="2"), Message("assistant", text="ok"),
    ]
    runner.history[2].tool_results = (object(),)
    runner._trim_history()
    assert runner.history[0].role == "user" and not runner.history[0].tool_results


def test_rule_based_fake_provider_maps_demo_phrases():
    rule = RuleBasedFakeProvider._rule
    assert rule("노란블록을적재구역좌상단으로옮겨줘").arguments == {"color": "yellow", "slot": "top-left"}
    assert rule("미션1해줘").name == "run_task1"
    assert rule("빨간블록5mm더멀리집어줘").arguments == {"color": "red", "forward_mm": 5.0, "relative_to_last": True}

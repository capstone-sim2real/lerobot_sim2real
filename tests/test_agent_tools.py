"""Tool schemas, argument validation, zone names, and exception envelopes."""

import concurrent.futures
import subprocess
import sys

import pytest

from agent.provider.schema import sanitize_for_gemini, to_anthropic_tools, to_gemini_function_declarations, to_openai_tools
from agent.provider.types import ToolCall
from agent.tools import ToolRegistry, build_tools, result_from_exception, validate_arguments
from agent.zone_names import resolve_slot
from config import AppConfig, load_config
from session.cancel import Cancelled
from session.results import SkillResult


def _never(_job):
    raise AssertionError("executor must not run")


def test_tool_list_and_enums_follow_config():
    cfg = AppConfig()
    names = [t.spec.name for t in build_tools(cfg)]
    assert len(names) == 18 and not any(n.startswith("run_task") for n in names)
    assert "begin_episode" in names and "save_episode" in names

    cfg.perception.color_prototypes["purple"] = [[140, 120]]
    specs = {t.spec.name: t.spec for t in build_tools(cfg)}
    assert "purple_1" in specs["move_to_target"].input_schema["properties"]["object_id"]["enum"]
    assert specs["move_to_target"].input_schema["properties"]["slot"]["enum"] == cfg.agent.zone_slots.labels
    jog = specs["move_relative"].input_schema["properties"]["left_mm"]
    assert jog["maximum"] == cfg.agent.relative.max_jog_mm
    for spec in specs.values():
        schema = spec.input_schema
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert set(schema["required"]) <= set(schema["properties"])


def test_bad_arguments_never_reach_the_robot():
    registry = ToolRegistry(AppConfig(), _never)
    for call in (
        ToolCall("1", "fly_away", {}),
        ToolCall("2", "pick_block", {}),
        ToolCall("3", "pick_block", {"color": "purple"}),
        ToolCall("4", "move_arm", {"left_mm": 500}),
        ToolCall("5", "move_arm", {"left_mm": "10"}),
        ToolCall("6", "get_state", {"extra": 1}),
        # a board cell is an address, so it must be a whole number in range
        ToolCall("7", "move_to_cell", {"x": 3}),
        ToolCall("8", "move_to_cell", {"x": 3, "y": 2.5}),
        ToolCall("9", "move_to_cell", {"x": 9999, "y": 0}),
        ToolCall("10", "move_block_to_cell", {"color": "red", "x": 0}),
    ):
        result = registry.execute(call)
        assert result.is_error and result.content["reason"] == "invalid_arguments"


def test_validate_arguments_messages():
    schema = next(t.spec.input_schema for t in build_tools(AppConfig()) if t.spec.name == "move_relative")
    assert validate_arguments(schema, {"left_mm": 10}) is None
    assert "unknown" in validate_arguments(schema, {"left_mm": 10, "relative_to_last": 1})


def test_exceptions_map_to_fault_reasons():
    assert result_from_exception("x", Cancelled()).reason == "cancelled"
    assert result_from_exception("x", TimeoutError()).reason == "motion_timeout"
    assert result_from_exception("x", concurrent.futures.TimeoutError()).reason == "motion_timeout"
    assert result_from_exception("x", ConnectionError()).reason == "bus_lost"
    assert result_from_exception("x", KeyError("boom")).reason == "internal_error"
    for reason in ("cancelled", "motion_timeout", "bus_lost", "internal_error"):
        assert SkillResult(False, "x", reason).robot_fault


def test_registry_attaches_state_and_survives_state_errors():
    class Skills:
        def get_state(self):
            return SkillResult(True, "get_state", "ok")

        def state_dict(self):
            raise RuntimeError("bus read failed")

    registry = ToolRegistry(AppConfig(), lambda job: job(Skills()))
    result = registry.execute(ToolCall("1", "get_state", {}))
    assert result.content["ok"] and result.content["state"] == {"state_error": "bus read failed"}


def test_unknown_reason_is_rejected():
    with pytest.raises(ValueError):
        SkillResult(False, "x", "made_up")


@pytest.mark.parametrize("name,index", [
    ("좌상단", 0), ("좌 상단", 0), ("top-left", 0), ("Top Left", 0), ("우하단", 4),
    ("오른쪽 아래", 4), ("상단 중앙", 1), ("3번", 2), ("3", 2), ("far-right", 2), ("6번", None), ("부엌", None),
])
def test_zone_names_resolve(name, index):
    assert resolve_slot(name, AppConfig().agent.zone_slots) == index


def test_config_validation_rejects_broken_agent_settings(tmp_path):
    cases = {
        "agent:\n  provider: llama\n": "agent.provider",
        "agent:\n  fallback_provider: llama\n": "agent.fallback_provider",
        "agent:\n  provider: openai\n  fallback_provider: openai\n": "must differ",
        "agent:\n  zone_slots:\n    aliases:\n      좌상단: 7\n": "slot index",
        "agent:\n  zone_slots:\n    labels: [a, b]\n": "one name per",
        "agent:\n  relative:\n    frame: camera\n": "relative.frame",
        "agent:\n  relative:\n    jog_min_z_mm: 200.0\n": "jog_min_z_mm",
        "agent:\n  table_regions:\n    rows_fraction:\n      near: 1.5\n      middle: 0.5\n      far: 0.9\n": "rows_fraction",
    }
    for text, message in cases.items():
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ValueError, match=message):
            load_config(path)


def test_vendor_tool_shapes():
    specs = [t.spec for t in build_tools(AppConfig())]
    anthropic = to_anthropic_tools(specs)
    assert set(anthropic[0]) == {"name", "description", "input_schema"}
    openai = to_openai_tools(specs)
    assert openai[0]["type"] == "function" and "parameters" in openai[0]["function"]
    gemini = to_gemini_function_declarations(specs)

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    assert "additionalProperties" not in set(keys(gemini))
    assert "parameters" not in next(d for d in gemini if d["name"] == "get_state")
    assert sanitize_for_gemini({"type": "object", "properties": {}, "required": []}) == {"type": "object"}


def test_agent_core_imports_without_sdks_or_web_framework():
    code = (
        "import sys, agent.runner, agent.tools, agent.control, agent.service, agent.provider.schema, "
        "session.skills; bad = [m for m in ('anthropic', 'openai', 'google.genai', 'fastapi', 'uvicorn', "
        "'placo', 'lerobot') if m in sys.modules]; assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd="src")

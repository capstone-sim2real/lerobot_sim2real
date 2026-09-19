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


def test_agent_core_imports_without_sdks_or_web_framework():
    code = (
        "import sys, agent.runner, agent.tools, agent.control, agent.service, agent.provider.schema, "
        "session.skills; bad = [m for m in ('anthropic', 'openai', 'google.genai', 'fastapi', 'uvicorn', "
        "'placo', 'lerobot') if m in sys.modules]; assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd="src")

"""The tools the LLM sees, and the one boundary where they execute.

No raw joint or free-form coordinate command is exposed. Relative moves take
bounded millimetre vectors whose limits are written into the schema from
config, and every enum (colour, zone cell, table region) is generated from
config, so adding a colour or renaming a cell needs no code change.

Legacy absolute addresses include a *chessboard cell* (``x``/``y`` integers,
AGENTS.md §16.3): it is discrete, it resolves through the same workspace
sector and IK gates every other target passes, and it exists because an
operator pointing at the camera page needs to say "there" without a name
for it. Millimetres still never arrive from the LLM.

``ToolRegistry.execute`` never raises: bad arguments, STOP, motion timeouts
and unexpected errors all come back as a result envelope.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable

from config import AppConfig
from session.results import SkillResult

from .provider.types import ToolCall, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

# The executor runs ``fn(skills)`` on the robot thread and returns its value.
Executor = Callable[[Callable[[Any], SkillResult]], SkillResult]


@dataclass(frozen=True)
class ToolDef:
    spec: ToolSpec
    run: Callable[[Any, dict[str, Any]], SkillResult]
    moves_arm: bool = True


def _obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _mm(description: str, limit: float) -> dict[str, Any]:
    return {"type": "number", "description": description, "minimum": -limit, "maximum": limit}


def _cell_span(cfg: AppConfig) -> int:
    """A generous index bound for board cells, from config alone.

    The exact set of addressable cells depends on the measured lattice in
    the calibration file, which this module deliberately does not load. So
    the schema carries a bounding box and the skill does the real
    membership check -- the same split as a blocked table region, where the
    schema accepts the name and the skill reports there is no room.
    """
    reach = float(cfg.perception.workspace_radius_mm)
    return int(reach / max(1e-6, float(cfg.agent.board_grid.cell_mm))) + 1


def build_tools(cfg: AppConfig) -> list[ToolDef]:
    from .primitive_tools import build_primitive_tools
    return build_primitive_tools(cfg)


# ── argument validation (the subset of JSON Schema the tools use) ──────


def validate_arguments(schema: dict[str, Any], args: Any) -> str | None:
    if not isinstance(args, dict):
        return "arguments must be an object"
    properties = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            return f"missing required argument '{name}'"
    for name, value in args.items():
        if name not in properties:
            if schema.get("additionalProperties") is False:
                return f"unknown argument '{name}'"
            continue
        prop = properties[name]
        kind = prop.get("type")
        if kind == "string" and not isinstance(value, str):
            return f"'{name}' must be a string"
        if kind == "boolean" and not isinstance(value, bool):
            return f"'{name}' must be true or false"
        if kind in ("number", "integer"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"'{name}' must be a number"
            if not math.isfinite(value):
                return f"'{name}' must be finite"
            if kind == "integer" and int(value) != value:
                return f"'{name}' must be a whole number"
            if "minimum" in prop and value < prop["minimum"]:
                return f"'{name}' must be >= {prop['minimum']:g}"
            if "maximum" in prop and value > prop["maximum"]:
                return f"'{name}' must be <= {prop['maximum']:g}"
        if "enum" in prop and value not in prop["enum"]:
            return f"'{name}' must be one of {prop['enum']}"
    return None


def result_from_exception(action: str, exc: BaseException) -> SkillResult:
    """Map anything a skill raised to a fault/failure envelope."""
    from session.cancel import Cancelled

    name = type(exc).__name__
    if isinstance(exc, Cancelled) or name == "StopRecording":
        return SkillResult(False, action, "cancelled", "비상정지로 동작을 멈췄습니다.", "do_not_retry")
    if isinstance(exc, (TimeoutError, concurrent.futures.TimeoutError)):
        return SkillResult(False, action, "motion_timeout",
                           f"팔이 제시간에 동작을 끝내지 못했습니다: {exc}", "ask_operator")
    if isinstance(exc, ConnectionError) or "serial" in name.lower():
        return SkillResult(False, action, "bus_lost", f"로봇 통신 오류: {exc}", "ask_operator")
    logger.exception("tool %s failed", action, exc_info=exc)
    return SkillResult(False, action, "internal_error", f"예상하지 못한 오류: {name}: {exc}", "ask_operator")


class ToolRegistry:
    def __init__(self, cfg: AppConfig, executor: Executor):
        self._cfg = cfg
        self._executor = executor
        self._tools = {tool.spec.name: tool for tool in build_tools(cfg)}

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def moves_arm(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool is not None and tool.moves_arm

    def run_skill(self, action: str, fn: Callable[[Any], SkillResult]) -> SkillResult:
        """Run ``fn(skills)`` on the robot thread with state attached; never raises."""

        def job(skills: Any) -> SkillResult:
            try:
                result = fn(skills)
            except BaseException as exc:  # noqa: BLE001 - converted to an envelope
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                result = result_from_exception(action, exc)
            try:
                callback = getattr(type(skills), "on_tool_result", None)
                if callback is not None:
                    callback(skills, action, result)
            except Exception as exc:
                result = result_from_exception(action, exc)
                collection = getattr(skills, "collection", None)
                if collection is not None:
                    collection.discard("recording_error")
            try:
                result.state = skills.state_dict()
            except BaseException as exc:  # noqa: BLE001 - state is best effort
                result.state = {"state_error": str(exc)}
            return result

        try:
            return self._executor(job)
        except BaseException as exc:  # noqa: BLE001 - executor/timeouts
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return result_from_exception(action, exc)

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            result = SkillResult(False, call.name, "invalid_arguments",
                                 f"unknown tool '{call.name}'", "do_not_retry")
            return ToolResult(call.id, call.name, result.to_envelope(), is_error=True)
        error = validate_arguments(tool.spec.input_schema, call.arguments)
        if error is not None:
            result = SkillResult(False, call.name, "invalid_arguments", error, "retry_ok")
            return ToolResult(call.id, call.name, result.to_envelope(), is_error=True)
        args = dict(call.arguments)
        result = self.run_skill(call.name, lambda skills: tool.run(skills, args))
        return ToolResult(call.id, call.name, result.to_envelope(), is_error=not result.ok, images=result.images)

    def last_fault(self, results: list[ToolResult]) -> bool:
        from session.results import ROBOT_FAULT_REASONS

        return any(r.content.get("reason") in ROBOT_FAULT_REASONS for r in results)

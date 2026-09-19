"""The tools the LLM sees, and the one boundary where they execute.

No raw joint or free-form coordinate command is exposed. Relative moves take
bounded millimetre vectors whose limits are written into the schema from
config, and every enum (colour, zone cell, table region) is generated from
config, so adding a colour or renaming a cell needs no code change.

The one absolute address is a *chessboard cell* (``x``/``y`` integers,
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
    agent = cfg.agent
    rel = agent.relative
    colors = sorted(cfg.perception.color_prototypes)
    labels = list(agent.zone_slots.labels)
    columns = list(agent.table_regions.columns_deg)
    rows = list(agent.table_regions.rows_fraction)
    slot_index = {label: i for i, label in enumerate(labels)}

    color = {"type": "string", "enum": colors, "description": "Block colour."}
    slot = {
        "type": "string",
        "enum": labels,
        "description": (
            "Target-zone cell as seen on the operator's camera page: 'top' is the row of three "
            "FARTHEST from the robot, 'bottom' the row of two nearest; 'left' is the image left. "
            f"Korean: {', '.join(f'{k}={e}' for k, e in zip(agent.zone_slots.korean_labels, labels))}."
        ),
    }
    column = {
        "type": "string",
        "enum": columns,
        "description": (
            "Column of the fan-shaped table workspace (camera view, left = image left). "
            "'leftmost'/'rightmost' are the extreme columns ('가장 왼쪽/오른쪽')."
        ),
    }
    row = {
        "type": "string",
        "enum": rows,
        "description": (
            "Row of the workspace: 'near' = closest to the robot = bottom of the camera image "
            "('아래'), 'far' = top of the image ('위')."
        ),
    }
    forward_desc = "mm away from the robot base (negative = toward the base / '가까이')."
    left_desc = "mm to the left as seen on the camera page (negative = right)."
    span = _cell_span(cfg)
    cell_frame = (
        "Chessboard cell of the table, as the operator sees it on the camera page: "
        "x grows to the IMAGE RIGHT, y grows AWAY from the robot (image up). (0, 0) is the "
        "reference square at the bottom centre of the reachable band, and negative values are "
        "normal. One cell is one chessboard square. The operator page writes clicks as "
        "'격자 (3, 4)'; plain '3,4' in a sentence means the same. Only cells inside the "
        "fan-shaped workspace exist - call describe_places for the real range, and a cell "
        "outside it comes back as invalid_arguments naming the range."
    )
    cell_x = {"type": "integer", "description": "Cell x " + cell_frame, "minimum": -span, "maximum": span}
    cell_y = {"type": "integer", "description": "Cell y " + cell_frame, "minimum": -span, "maximum": span}

    def slot_arg(args: dict[str, Any]) -> int:
        return slot_index[args["slot"]]

    tools = [
        ToolDef(
            ToolSpec(
                "get_state",
                "Current robot and world state from memory: what is held, whether the arm is home, "
                "the last camera scene. Does not move the arm or take a new photo. Cheap.",
                _obj({}),
            ),
            lambda sk, a: sk.get_state(),
            moves_arm=False,
        ),
        ToolDef(
            ToolSpec(
                "observe_scene",
                "Return the arm home (to clear the camera view), take a fresh camera frame and list "
                "every detected block, whether it is inside the target zone, and which zone cells "
                "are occupied.",
                _obj({"include_zone": {"type": "boolean", "description": "Also list blocks already in the zone. Default true."}}),
            ),
            lambda sk, a: sk.observe_scene(a.get("include_zone", True)),
        ),
        ToolDef(
            ToolSpec(
                "describe_places",
                "List the exact names accepted for zone cells (with occupancy) and table regions, "
                "plus which direction words mean what. Call this when unsure of a place name.",
                _obj({}),
            ),
            lambda sk, a: sk.describe_places(),
            moves_arm=False,
        ),
        ToolDef(
            ToolSpec(
                "pick_block",
                "Find one block by colour (inside or outside the zone) and grasp it. INTERNALLY "
                "already retries with the gripper rotated 90 degrees and re-approaches from home up "
                "to the configured limit - never call it again for the same colour unless "
                "retry_advice is retry_ok or the user asks for an adjusted grasp. The block stays "
                "held afterwards. forward_mm/left_mm shift the grasp point (e.g. '5mm 더 멀리 집어줘' "
                "-> forward_mm=5, relative_to_last=true). If that same block is currently held it is "
                "put back first.",
                _obj(
                    {
                        "color": color,
                        "forward_mm": _mm("Grasp point shift, " + forward_desc, rel.max_pick_offset_mm),
                        "left_mm": _mm("Grasp point shift, " + left_desc, rel.max_pick_offset_mm),
                        "relative_to_last": {
                            "type": "boolean",
                            "description": "Add the shift to the previous grasp shift for this colour ('더').",
                        },
                    },
                    ["color"],
                ),
            ),
            lambda sk, a: sk.pick_block(
                a["color"], a.get("forward_mm", 0.0), a.get("left_mm", 0.0), a.get("relative_to_last", False)
            ),
        ),
        ToolDef(
            ToolSpec(
                "place_at_slot",
                "Put the currently HELD block into a named target-zone cell and release it, then "
                "return home and verify with the camera. Optional forward_mm/left_mm offset the drop "
                "point from the cell centre.",
                _obj(
                    {
                        "slot": slot,
                        "forward_mm": _mm("Drop offset, " + forward_desc, rel.max_shift_mm),
                        "left_mm": _mm("Drop offset, " + left_desc, rel.max_shift_mm),
                    },
                    ["slot"],
                ),
            ),
            lambda sk, a: sk.place_at_slot(slot_arg(a), a.get("forward_mm", 0.0), a.get("left_mm", 0.0)),
        ),
        ToolDef(
            ToolSpec(
                "place_on_table",
                "Put the HELD block down on the table (outside the zone) at a named region of the "
                "fan-shaped workspace. Omit column/row to let the robot choose the nearest free spot. "
                "If the named spot is taken a nearby free spot is used and reported.",
                _obj({"column": column, "row": row}),
            ),
            lambda sk, a: sk.place_on_table(a.get("column"), a.get("row")),
        ),
        ToolDef(
            ToolSpec(
                "place_at_cell",
                "Put the HELD block down on one chessboard cell of the table ('(3, 4)에 놓아줘'). "
                "Use this when the operator names coordinates; use place_on_table when they name a "
                "region ('가장 왼쪽 아래'). If the cell is taken a nearby free spot is used and "
                "reported. Cells inside the target zone are refused - those have names.",
                _obj({"x": cell_x, "y": cell_y}, ["x", "y"]),
            ),
            lambda sk, a: sk.place_at_cell(a["x"], a["y"]),
        ),
        ToolDef(
            ToolSpec(
                "place_here",
                "Lower and release the HELD block right below the gripper's current position "
                "('여기 내려놔'), typically after move_arm.",
                _obj({}),
            ),
            lambda sk, a: sk.place_here(),
        ),
        ToolDef(
            ToolSpec(
                "move_block_to_slot",
                "Pick a block by colour and place it into a named zone cell in one call. Preferred for "
                "'X 블록을 적재 구역 Y로 옮겨줘'. Refuses without moving if the cell is occupied.",
                _obj({"color": color, "slot": slot}, ["color", "slot"]),
            ),
            lambda sk, a: sk.move_block_to_slot(a["color"], slot_arg(a)),
        ),
        ToolDef(
            ToolSpec(
                "move_block_to_table",
                "Pick a block by colour (also one already inside the zone - '다시 밖으로 꺼내줘') and put "
                "it on the table at a named workspace region, or the nearest free spot when "
                "column/row are omitted.",
                _obj({"color": color, "column": column, "row": row}, ["color"]),
            ),
            lambda sk, a: sk.move_block_to_table(a["color"], a.get("column"), a.get("row")),
        ),
        ToolDef(
            ToolSpec(
                "move_block_to_cell",
                "Pick a block by colour and put it on one chessboard cell in one call. Preferred for "
                "'X 블록을 (3, 4)로 옮겨줘'. Refuses without moving if that is not a cell of the "
                "workspace.",
                _obj({"color": color, "x": cell_x, "y": cell_y}, ["color", "x", "y"]),
            ),
            lambda sk, a: sk.move_block_to_cell(a["color"], a["x"], a["y"]),
        ),
        ToolDef(
            ToolSpec(
                "shift_block",
                "Move a block that is lying somewhere (table or zone) by a relative offset, e.g. "
                "'그 블록 20mm만 더 왼쪽으로' -> left_mm=20. Checks the destination before grasping, "
                "then re-measures with the camera and reports the actually measured displacement.",
                _obj(
                    {
                        "color": color,
                        "forward_mm": _mm(forward_desc, rel.max_shift_mm),
                        "left_mm": _mm(left_desc, rel.max_shift_mm),
                    },
                    ["color", "forward_mm", "left_mm"],
                ),
            ),
            lambda sk, a: sk.shift_block(a["color"], a["forward_mm"], a["left_mm"]),
        ),
        ToolDef(
            ToolSpec(
                "move_arm",
                f"Jog the gripper by a small relative vector (max {rel.max_jog_mm:g}mm per call; larger "
                "requests are refused, so split them). up_mm is HEIGHT. If the arm is at home it "
                "first rises to working height. Does not open or close the gripper.",
                _obj(
                    {
                        "forward_mm": _mm(forward_desc, rel.max_jog_mm),
                        "left_mm": _mm(left_desc, rel.max_jog_mm),
                        "up_mm": _mm("mm upward (negative = down).", rel.max_jog_mm),
                    }
                ),
            ),
            lambda sk, a: sk.move_arm(a.get("forward_mm", 0.0), a.get("left_mm", 0.0), a.get("up_mm", 0.0)),
        ),
        ToolDef(
            ToolSpec(
                "move_to_cell",
                "Fly the EMPTY-or-holding gripper over one chessboard cell at its current height, in "
                "one move ('(3, 4) 위로 가줘'). Unlike move_arm there is no per-call distance cap, "
                "because a cell is an address rather than a nudge. Does not open or close the jaws; "
                "follow with pick_here or place_here.",
                _obj({"x": cell_x, "y": cell_y}, ["x", "y"]),
            ),
            lambda sk, a: sk.move_to_cell(a["x"], a["y"]),
        ),
        ToolDef(
            ToolSpec(
                "rotate_gripper",
                f"Spin the jaws in place (max {rel.max_gripper_roll_deg:g} degrees per call; larger "
                "requests are refused, so split them). Does not move x/y/z. Positive turns one way, "
                "negative the other - direction is not calibrated to a compass, so ask the student "
                "which way if it matters.",
                _obj({"delta_deg": _mm("Degrees to rotate.", rel.max_gripper_roll_deg)}, ["delta_deg"]),
            ),
            lambda sk, a: sk.rotate_gripper(a["delta_deg"]),
        ),
        ToolDef(
            ToolSpec(
                "pick_here",
                "Manual 'claw machine' grab: close the gripper on whatever is directly below the arm's "
                "current position, with no colour lookup (position the arm with move_arm first). "
                "Internally retries with a rotated gripper like pick_block. The held item's colour is "
                "unknown afterwards.",
                _obj({}),
            ),
            lambda sk, a: sk.pick_here(),
        ),
        ToolDef(
            ToolSpec(
                "return_to_home",
                "Return the arm to its home pose (rising first if it is low). Keeps a held block held.",
                _obj({}),
            ),
            lambda sk, a: sk.return_to_home(),
        ),
        ToolDef(
            ToolSpec(
                "open_gripper",
                "Open the jaws where the arm is now, dropping anything held. Recovery only - prefer "
                "the place_* tools to put a block down.",
                _obj({}),
            ),
            lambda sk, a: sk.open_gripper(),
        ),
        ToolDef(
            ToolSpec(
                "run_task1",
                "Run rule-based mission 1 unchanged: gather every block outside the zone into the five "
                "zone cells until the outside has been empty for the hold time. Takes minutes. Cells "
                "already occupied are kept.",
                _obj({}),
            ),
            lambda sk, a: sk.run_task(1),
        ),
        ToolDef(
            ToolSpec(
                "run_task2",
                "Run rule-based mission 2 unchanged: stack every outside block into one tower in the "
                "zone. Takes minutes.",
                _obj({}),
            ),
            lambda sk, a: sk.run_task(2),
        ),
    ]
    if agent.enable_task3_tool:
        tools.append(
            ToolDef(
                ToolSpec(
                    "run_task3",
                    "Run one round of mission 3 (ACT demonstration dataset collection): gather every "
                    "outside block while recording. Requires an empty zone. Takes minutes.",
                    _obj({}),
                ),
                lambda sk, a: sk.run_task(3),
            )
        )
    return tools


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
        return ToolResult(call.id, call.name, result.to_envelope(), is_error=not result.ok)

    def last_fault(self, results: list[ToolResult]) -> bool:
        from session.results import ROBOT_FAULT_REASONS

        return any(r.content.get("reason") in ROBOT_FAULT_REASONS for r in results)

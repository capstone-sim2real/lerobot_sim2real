"""Guarded robot primitives and deterministic one-block mission transfers."""
from .provider.types import ToolSpec
from .tools import ToolDef, _obj, _mm, _cell_span


RECORDABLE_TOOLS = (
    "observe_scene", "get_state", "inspect_motion", "correct_hover",
    "descend_step", "move_to_target", "move_relative", "align_gripper",
    "close_gripper", "drop_at_zone_target", "descend_until_contact",
    "open_gripper", "return_to_home", "move_block_to_slot",
    "stack_block_to_floor", "place_at_pixel", "select_pixel_target", "move_to_pixel",
)


def build_primitive_tools(cfg):
    tools = []
    def add(name, description, properties=None, required=None, moves=True):
        tools.append(ToolDef(ToolSpec(name, description, _obj(properties or {}, required)),
                             lambda skills, args, method=name: getattr(skills, method)(**args), moves))
    source_fields = {
        "type": "object", "description": "Either object_id+observation_id, or exact user pixel u+v+calibration_id (+optional angle_deg)",
        "properties": {"object_id": {"type": "string"}, "observation_id": {"type": "integer"},
                       "u": {"type": "integer", "minimum": 0}, "v": {"type": "integer", "minimum": 0},
                       "calibration_id": {"type": "string"}, "angle_deg": {"type": "number"}},
        "additionalProperties": False}
    add("move_block_to_slot",
        "Transfer a block from inside or outside the zone to a free slot. Specify color OR source. The server "
        "observes before picking, checks grasp and clearance after each primitive, tries "
        "another free slot while still holding if the requested slot fails IK, makes "
        "one bounded retry after safe recovery, then homes. Release completion needs no camera verification. "
        "Prefer this over issuing the "
        "same primitive sequence one call at a time.",
        {"color": {"type": "string", "enum": sorted(cfg.perception.color_prototypes)},
         "slot": {"type": "string", "enum": list(cfg.agent.zone_slots.labels)},
         "source": source_fields},
        ["slot"])
    pixel_fields = {"u": {"type": "integer", "minimum": 0},
                    "v": {"type": "integer", "minimum": 0},
                    "calibration_id": {"type": "string"}}
    add("select_pixel_target",
        "Register the user's exact head-camera pixel as an object target for existing "
        "move_to_target, align_gripper and grasp/place primitives. Does not move. "
        "Works even when CV misses the block; returns selected_1 and observation_id. "
        "Optional angle_deg is a robot-plane block edge angle, not image angle. "
        "Unmatched targets assume the configured flat block size. Reobservation expires "
        "the selection. Never substitute a nearby slot or demand a colour.",
        {**pixel_fields, "angle_deg": {"type": "number", "minimum": -180, "maximum": 180}},
        ["u", "v", "calibration_id"], moves=False)
    add("move_to_pixel", "Move an empty gripper to the user's exact calibrated pixel on "
        "the block-top plane. Server preplans clearance and IK; does not grasp or release.",
        pixel_fields, ["u", "v", "calibration_id"])
    add("place_at_pixel",
        "Place an already verified held block at the user's exact selected head-camera "
        "pixel. Copy u, v and calibration_id verbatim from the request. The server "
        "validates calibration and workspace and solves IK. For a pixel destination, "
        "pick the requested block with primitives, lift, then use this tool; never "
        "substitute move_block_to_slot or guess a slot/cell. If rejected, report the "
        "reason without changing the destination.",
        {"u": {"type": "integer", "minimum": 0},
         "v": {"type": "integer", "minimum": 0},
         "calibration_id": {"type": "string"}},
        ["u", "v", "calibration_id"])
    add("stack_block_to_floor",
        "Task 2: restack any visible block, including one fallen inside the zone, "
        "at the requested floor (0 is the table, 4 is the fifth block). Use color OR source; optional destination is an exact user pixel, otherwise the configured tower. The caller "
        "chooses the floor; tower visibility and old floor records never block a retry. "
        "The server checks grasp and "
        "carry IK and releases from a bounded height without post-release camera verification. "
        "If a held placement fails, it sets the block on free table and retries once; "
        "STOP and robot faults never trigger automatic recovery. A successful call "
        "does not prove physical floor or five-second stability.",
        {"color": {"type": "string", "enum": sorted(cfg.perception.color_prototypes)},
         "floor": {"type": "integer", "minimum": 0,
                   "maximum": min(4, cfg.task2.max_levels - 1)},
         "source": source_fields, "destination": _obj(pixel_fields, ["u", "v", "calibration_id"])},
        ["floor"])
    object_fields = {
        "object_id": {"type": "string", "description": "ID returned by observe_scene or select_pixel_target; never invent one"},
        "observation_id": {"type": "integer", "minimum": 1},
    }
    add("observe_scene", "Fresh image and CV objects without moving. Missing objects may be occluded; planar CV cannot verify a stack.", moves=False)
    add("inspect_motion", "Read measured joints, loads, FK and nominal hover error; no motion, reference preserved.", moves=False)
    add("correct_hover", "One bounded measured-error correction above the block. Select joint(s) and fraction; no arbitrary joint target. Preview with dry_run first. Fails on clearance/load/lag limits.", {
        "joint": {"type": "string", "enum": ["all", "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]},
        "gain": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        "dry_run": {"type": "boolean"}})
    add("descend_step", "One bounded empty-gripper approach segment then stop. Load increase/lag abort. Never closes; only final depth authorizes close_gripper.", {
        "down_mm": {"type": "number", "exclusiveMinimum": 0, "maximum": cfg.agent.relative.max_jog_mm}}, ["down_mm"])
    add("get_state", "Measured joints and model FK, held state, contact state; no movement.", moves=False)
    add("describe_places", "List addressable cells and named slots; no movement.", moves=False)
    span = _cell_span(cfg)
    add("move_to_target", "Move to an observed object, named slot or discrete cell. pregrasp/preplace are hover only; grasp descends empty gripper at its corrected current XY. Clear the block-height band before ordinary lateral motion.", {
        "target_type": {"type": "string", "enum": ["object", "slot", "cell"]},
        "phase": {"type": "string", "enum": ["pregrasp", "grasp", "preplace", "hover"]},
        **object_fields,
        "slot": {"type": "string", "enum": list(cfg.agent.zone_slots.labels)},
        "x": {"type": "integer", "minimum": -span, "maximum": span},
        "y": {"type": "integer", "minimum": -span, "maximum": span},
    }, ["target_type", "phase"])
    add("move_relative", f"Bounded correction in {cfg.agent.relative.frame} frame. up is robot-base vertical. While holding, downward moves require a guarded placement action. After a calibrated tilted grasp, up first reverses the pick approach and may shift XY inward. A loaded upward result reports lateral_clearance_ready; when true, do not lift again and proceed to the placement target.", {
        key: _mm(key, cfg.agent.relative.max_jog_mm) for key in ("forward_mm", "left_mm", "up_mm")})
    add("align_gripper", "Align an empty gripper to object_id+observation_id, OR rotate to explicit robot-base yaw_deg at clearance (empty or verified held). IK and wrist limits apply.", {**object_fields, "yaw_deg": {"type": "number", "minimum": -180, "maximum": 180}})
    add("close_gripper", "Required immediately after a successful grasp descent with stop_reason=depth_reached: the jaws are intentionally open then. Close in place and verify grasp with position/load sensing. Does not approach, lift or transport.")
    add("drop_at_zone_target", "At a named zone slot or in-zone cell preplace, move to the configured release height without contact sensing. Verify the pose before open_gripper.")
    add("descend_until_contact", "At a zone slot/cell this uses the configured height drop without contact sensing. Other placement targets use bounded contact descent. Never releases.", {
        "max_descent_mm": {"type": "number", "minimum": 0,
                           "maximum": cfg.agent.primitives.contact_max_descent_mm}}, ["max_descent_mm"])
    add("open_gripper", "Open in place. A held block requires confirmed placement contact or a verified zone/stack drop pose. Release does not prove stacking success.")
    add("return_to_home", "Return empty gripper home. Refuses while holding.")
    add("record_tool_sequence",
        "Execute a preplanned list of existing safe tools on the robot worker while recording "
        "one episode. Step arguments may reference an earlier result with {\"$ref\": \"step1.observation_id\"} (1-based). The LLM supplies the whole list before motion; the server stops on the "
        "first failure and returns every step result. Start empty-handed at home and include "
        "a home return in the plan. Only a completed home-to-home run is saved; physical task "
        "success beyond the tool results is not inferred.",
        {"task": {"type": "string", "description": "Dataset task sentence matching the requested behavior"},
         "color": {"type": "string", "description": "Episode label; may be an arbitrary nonempty label for pixel-selected objects"},
         "steps": {"type": "array", "minItems": 1,
                   "maxItems": cfg.agent.collection.max_steps,
                   "items": {"type": "object", "properties": {
                       "name": {"type": "string", "enum": list(RECORDABLE_TOOLS)},
                       "arguments": {"type": "object"}},
                       "required": ["name", "arguments"], "additionalProperties": False}}},
        ["task", "color", "steps"])
    add("begin_episode", "Begin a home-to-home demonstration for an observed or selected block anywhere. Optional task supplies the dataset sentence. Does not move.", {**object_fields, "task": {"type": "string"}}, list(object_fields), moves=False)
    add("save_episode", "Save after verified grasp, release completion and home return. No post-release camera gate. Backend checks evidence, frame count and timing; no success argument.", moves=False)
    add("discard_episode", "Discard the current demonstration buffer, preserving saved episodes and recording the reason.", {
        "reason": {"type": "string", "enum": ["operator_requested", "pick_failed", "placement_failed", "bad_demonstration", "scene_changed", "interrupted"]}}, ["reason"], moves=False)
    add("collection_status", "Dataset location, active episode, frames, saved counts, rejection reasons and validation evidence. Does not train or upload.", moves=False)
    add("finish_dataset", "Finalize local dataset/video writers after saving or discarding the last episode. Does not run training or move the arm.", moves=False)
    return tools

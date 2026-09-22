"""Unified primitive tool surface; no composite mission tools are exposed."""
from .provider.types import ToolSpec
from .tools import ToolDef, _obj, _mm, _cell_span


def build_primitive_tools(cfg):
    tools = []
    def add(name, description, properties=None, required=None, moves=True):
        tools.append(ToolDef(ToolSpec(name, description, _obj(properties or {}, required)),
                             lambda skills, args, method=name: getattr(skills, method)(**args), moves))
    object_fields = {
        "object_id": {"type": "string", "enum": [f"{c}_1" for c in sorted(cfg.perception.color_prototypes)]},
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
    add("move_to_target", "Move to an observed object, named slot or discrete cell. pregrasp/preplace are hover only; grasp descends empty gripper at its corrected current XY. Lift vertically before lateral motion at low height.", {
        "target_type": {"type": "string", "enum": ["object", "slot", "cell"]},
        "phase": {"type": "string", "enum": ["pregrasp", "grasp", "preplace", "hover"]},
        **object_fields,
        "slot": {"type": "string", "enum": list(cfg.agent.zone_slots.labels)},
        "x": {"type": "integer", "minimum": -span, "maximum": span},
        "y": {"type": "integer", "minimum": -span, "maximum": span},
    }, ["target_type", "phase"])
    add("move_relative", f"Bounded correction in {cfg.agent.relative.frame} frame. up is robot-base vertical. While holding, downward moves require contact descent.", {
        key: _mm(key, cfg.agent.relative.max_jog_mm) for key in ("forward_mm", "left_mm", "up_mm")})
    add("align_gripper", "Align to observed block using nearest neutral symmetric yaw at clearance. Empty gripper only.", object_fields, list(object_fields))
    add("close_gripper", "Close in place and verify grasp with existing position/load sensing. Does not approach, lift or transport.")
    add("descend_until_contact", "Bounded vertical descent at placement target; load contact stops and backs off. Never releases; failure leaves block held.", {
        "max_descent_mm": {"type": "number", "minimum": 0,
                           "maximum": cfg.agent.primitives.contact_max_descent_mm}}, ["max_descent_mm"])
    add("open_gripper", "Open in place. A held block requires confirmed contact from the immediately preceding descent. Release does not prove stacking success.")
    add("return_to_home", "Return empty gripper home. Refuses while holding.")
    add("begin_episode", "Begin a home-to-home demonstration for an observed outside-zone block. Uses configured local dataset and camera streams. Does not pick or run a task.", object_fields, list(object_fields), moves=False)
    add("save_episode", "Save only after verified grasp, zone delivery, home return and fresh observed placement. Backend checks evidence, frame count and timing; no success argument.", moves=False)
    add("discard_episode", "Discard the current demonstration buffer, preserving saved episodes and recording the reason.", {
        "reason": {"type": "string", "enum": ["operator_requested", "pick_failed", "placement_failed", "bad_demonstration", "scene_changed", "interrupted"]}}, ["reason"], moves=False)
    add("collection_status", "Dataset location, active episode, frames, saved counts, rejection reasons and validation evidence. Does not train or upload.", moves=False)
    add("finish_dataset", "Finalize local dataset/video writers after saving or discarding the last episode. Does not run training or move the arm.", moves=False)
    return tools

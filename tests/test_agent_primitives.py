"""Offline control-contract tests; these do not validate real stacking."""
import time
from dataclasses import replace
from unittest.mock import Mock

import pytest

from agent_helpers import make_skills, fast_cfg
from agent.tools import ToolRegistry, build_tools
from agent.provider.types import Message, ToolResult, ToolCall, ToolCallEvent, TurnEnd
from agent.provider.fake import ScriptedProvider
from agent.runner import AgentRunner
from session.primitives import PrimitiveSkills
from session.results import ObservationImage, SkillResult
from control.sensing import ContactMonitor, ContactReading


def fixture():
    cfg = fast_cfg()
    cfg.agent.relative.frame = "base"
    sk, world, robot = make_skills({"yellow": (160., 40.), "red": (230., -40.)}, cfg=cfg)
    return PrimitiveSkills(sk.s), world, robot


def pick(sk):
    assert sk.observe_scene().ok
    assert sk.open_gripper().ok
    assert sk.move_relative(up_mm=50).ok
    assert sk.move_to_target("object", "pregrasp", "yellow_1", sk.observation_id).ok
    assert sk.align_gripper("yellow_1", sk.observation_id).ok
    assert sk.move_to_target("object", "grasp", "yellow_1", sk.observation_id).ok
    result = sk.close_gripper()
    assert result.ok, result


def place_hover(sk):
    pick(sk)
    assert sk.move_relative(up_mm=50).ok
    assert sk.move_to_target("object", "preplace", "red_1", sk.observation_id).ok


def test_observation_does_not_move_and_has_scoped_ids(monkeypatch):
    sk, _, robot = fixture()
    send = Mock(side_effect=AssertionError("observation must not move"))
    monkeypatch.setattr(robot, "send_joints", send)
    first = sk.observe_scene()
    assert first.ok and first.data["observation_id"] == 1
    assert {o["object_id"] for o in first.data["objects"]} == {"yellow_1", "red_1"}
    assert sk.observe_scene().data["observation_id"] == 2
    assert not sk.move_to_target("object", "pregrasp", "yellow_1", 1).ok
    send.assert_not_called()


def test_stale_targets_refused():
    sk, _, _ = fixture()
    sk.observe_scene()
    sk._observed_at = time.monotonic() - sk.limits.target_max_age_s - 1
    assert not sk.move_to_target("object", "pregrasp", "yellow_1", 1).ok


def test_low_lateral_move_and_unverified_carry_refused():
    sk, _, _ = fixture()
    assert not sk.move_relative(forward_mm=40).ok
    assert not sk.close_gripper().ok
    assert sk.move_relative(up_mm=50).ok
    assert not sk.move_relative(forward_mm=40).ok
    assert sk.open_gripper().ok
    assert sk.move_relative(forward_mm=20).ok


def test_vertical_lift_ignores_xy_feedback_jitter(monkeypatch):
    sk, _, _ = fixture()
    measured = sk.s.arm_position_mm
    calls = 0

    def jittered_position():
        nonlocal calls
        calls += 1
        x, y, z = measured()
        return (x + 0.2, y, z) if calls == 1 else (x, y, z)

    monkeypatch.setattr(sk.s, "arm_position_mm", jittered_position)
    assert sk.move_relative(up_mm=50).ok


def test_calibrated_tilted_hover_checks_planned_hover_not_block_centre(monkeypatch):
    from types import SimpleNamespace

    sk, _, _ = fixture()
    assert sk.observe_scene().ok
    sk.limits.calibrated_pick = True
    sk._target = ("object", "yellow_1", sk.observation_id, None, None, None, "pregrasp")
    sk._pick_calibration = SimpleNamespace(attempt=SimpleNamespace(
        hover_xy_mm=(100.0, 40.0), xy_mm=(160.0, 40.0)))
    monkeypatch.setattr(sk._pick_calibration, "baseline", None, raising=False)
    monkeypatch.setattr(sk, "_calibrated_target",
                        lambda *_args, **_kwargs: SkillResult(True, "move_to_target", "ok"))
    monkeypatch.setattr(sk.s, "arm_position_mm", lambda: (100.0, 40.0, 39.0))
    assert sk.move_to_target("object", "grasp", "yellow_1", sk.observation_id).ok

    sk._target = ("object", "yellow_1", sk.observation_id, None, None, None, "pregrasp")
    monkeypatch.setattr(sk.s, "arm_position_mm", lambda: (130.0, 40.0, 39.0))
    result = sk.move_to_target("object", "grasp", "yellow_1", sk.observation_id)
    assert not result.ok
    assert "hover drift" in result.detail


def test_loaded_reverse_lift_uses_inward_upward_ik_path(monkeypatch):
    from types import SimpleNamespace

    sk, _, robot = fixture()
    robot.joints.update(shoulder_pan=320.0, shoulder_lift=0.0, elbow_flex=20.0)
    sk.s.held = SimpleNamespace(color="yellow", over_xy_mm=None)
    monkeypatch.setattr(sk, "_held_check", lambda: True)
    monkeypatch.setattr(sk, "_observed_scene", None)
    assert sk._recover_loaded_reverse_lift(48.0)
    reached = sk.s.arm_position_mm()
    assert reached[0] < 320.0 and reached[2] >= 48.0
    assert any("shoulder_pan" in action and "elbow_flex" in action
               for action in robot.sent_actions)
    assert sk.s.held.over_xy_mm == reached[:2]


def test_primitive_preplace_rejects_slot_occupied_by_another_block():
    sk, _, _ = fixture()
    pick(sk)
    assert sk.move_relative(up_mm=50).ok
    occupied_index = next(
        index for index, color in sk._observed_scene.slot_occupancy.items()
        if color == "red"
    )
    slot = sk.cfg.agent.zone_slots.labels[occupied_index]
    result = sk.move_to_target("slot", "preplace", slot=slot)
    assert not result.ok
    assert result.reason == "slot_occupied"
    assert slot not in result.data["free_slots"]


def test_no_contact_does_not_release():
    sk, world, _ = fixture()
    place_hover(sk)
    assert not sk.descend_until_contact(20).ok
    assert world.held == "yellow"
    assert not sk.open_gripper().ok
    assert not sk.move_relative(up_mm=-5).ok
    assert not sk.return_to_home().ok


def test_moving_invalidates_contact(monkeypatch):
    sk, _, _ = fixture()
    place_hover(sk)
    monkeypatch.setattr("session.primitives.ContactMonitor.check", lambda self: ContactReading(True))
    assert sk.descend_until_contact(20).ok
    assert sk.move_relative(up_mm=10).ok
    assert not sk.open_gripper().ok


def test_multiple_calls_execute_none():
    sk, _, robot = fixture()
    provider = ScriptedProvider([
        [ToolCallEvent(ToolCall("1", "open_gripper", {})), ToolCallEvent(ToolCall("2", "close_gripper", {})), TurnEnd("tool_use")],
        [TurnEnd("end_turn")],
    ])
    registry = ToolRegistry(sk.cfg, lambda fn: fn(sk))
    registry.execute = Mock(side_effect=AssertionError("must reject entire batch"))
    runner = AgentRunner(provider, registry, "system", sk.cfg.agent)
    assert runner.run_turn("pick").error is None
    registry.execute.assert_not_called()
    assert all(r.is_error for r in runner.history[2].tool_results)


def test_images_serialized_for_openai_and_anthropic():
    from agent.provider.openai_provider import OpenAIProvider
    from agent.provider.anthropic_provider import AnthropicProvider
    image = ObservationImage(b"jpeg-test", "shoulder", 4, 123.0)
    result = ToolResult("1", "observe_scene", {"ok": True}, images=(image,))
    messages = [Message("user", tool_results=(result,))]
    oa = OpenAIProvider("fake", max_tokens=100, client=object()).to_messages("system", messages)
    assert oa[1]["role"] == "tool"
    assert oa[2]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    an = AnthropicProvider("fake", max_tokens=100, client=object()).to_messages(messages)
    assert an[0]["content"][0]["content"][1]["source"]["media_type"] == "image/jpeg"


def test_frame_bytes_match_observation_and_downscale(monkeypatch):
    import cv2
    import numpy as np
    from camera.client import CameraSnapshot
    sk, _, _ = fixture()
    scene = sk.s.observe()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :, 2] = 255
    def observe(**kwargs):
        sk.s.last_snapshot = CameraSnapshot(frame, 42, 123.5)
        return replace(scene, frame_seq=42, captured_at=123.5)
    monkeypatch.setattr(sk.s, "observe", observe)
    result = sk.observe_scene()
    assert result.images[0].frame_seq == result.data["frame_seq"] == 42
    decoded = cv2.imdecode(np.frombuffer(result.images[0].jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == sk.limits.image_max_width
    assert decoded[0, 0, 2] > 240 and decoded[0, 0, 0] < 10


def test_block_transfer_uses_gated_primitives_and_verifies_actual_slot(monkeypatch):
    sk, world, _ = fixture()
    monkeypatch.setattr(
        ContactMonitor, "check",
        lambda self: (_ for _ in ()).throw(AssertionError("zone placement must not seek contact")),
    )
    lift_commands = []
    retreat_heights = []
    joint_lift_heights = []
    move_relative = sk.move_relative
    recover_lift = sk._recover_loaded_reverse_lift
    joint_lift = sk._lift_held_joint_space

    def record_lift(**kwargs):
        lift_commands.append(kwargs["up_mm"])
        return move_relative(**kwargs)

    def record_retreat(required_z, **kwargs):
        retreat_heights.append((required_z, kwargs.get("max_command_z_mm")))
        return recover_lift(required_z, **kwargs)

    def record_joint_lift():
        result = joint_lift()
        joint_lift_heights.append(sk.s.arm_position_mm()[2])
        return result

    monkeypatch.setattr(sk, "move_relative", record_lift)
    monkeypatch.setattr(sk, "_recover_loaded_reverse_lift", record_retreat)
    monkeypatch.setattr(sk, "_lift_held_joint_space", record_joint_lift)
    result = sk.move_block_to_slot("yellow", "top-left")

    assert lift_commands and max(lift_commands) < sk.cfg.agent.relative.max_jog_mm
    assert retreat_heights == [(
        sk.s.grasp_z_mm + sk.limits.lateral_clearance_mm
        - sk.limits.lateral_clearance_tolerance_mm, None)]
    assert joint_lift_heights
    assert joint_lift_heights[0] <= (
        sk.s.grasp_z_mm + sk.cfg.task1.tilted_pick_hover_clearance_mm
        + sk.limits.lateral_clearance_tolerance_mm)
    assert result.ok and result.reason == "released"
    assert result.data["slot"] == "top-left"
    assert result.data["miss_mm"] < sk.cfg.agent.slot_snap_radius_mm
    assert world.held is None
    assert sk.s.arm_at_home()


def test_task1_preplace_can_use_unaligned_yaw_when_aligned_yaw_fails(monkeypatch):
    sk, _, _ = fixture()
    pick(sk)
    sk._held_pick_yaw_deg = 0.0
    sk._held_block_angle_deg = 30.0
    monkeypatch.setattr(sk.s.ik, "neutral_yaw_deg", lambda *_args: 0.0)
    solve = sk._solve

    def only_fallback_yaw(xyz, **kwargs):
        if kwargs.get("yaw_deg") != -60.0:
            raise ValueError("aligned yaw misses clearance")
        return solve(xyz, **kwargs)

    monkeypatch.setattr(sk, "_solve", only_fallback_yaw)
    xy = sk.s.slot_centres[0]
    yaw, aligned = sk._choose_place_yaw(
        (*xy, sk.s.grasp_z_mm + sk.limits.approach_clearance_mm), 0.0)
    assert yaw == -60.0 and not aligned


def test_unreachable_slot_keeps_block_held_for_another_slot(monkeypatch):
    sk, world, _ = fixture()
    move_to_target = sk.move_to_target

    def unreachable_top_right(target_type, phase, *args, **kwargs):
        if target_type == "slot" and phase == "preplace" and kwargs.get("slot") == "top-right":
            return SkillResult(False, "move_to_target", "ik_gate",
                               "No reachable placement yaw")
        return move_to_target(target_type, phase, *args, **kwargs)

    monkeypatch.setattr(sk, "move_to_target", unreachable_top_right)
    monkeypatch.setattr(sk, "_put_held_block_on_table",
                        lambda *_args: (_ for _ in ()).throw(
                            AssertionError("do not set down a held block for one unreachable slot")))
    result = sk.move_block_to_slot("yellow", "top-right")
    assert result.ok and result.data["recovery"] == "alternate_slot_while_held"
    assert result.data["requested_slot"] == "top-right"
    assert result.data["slot"] != "top-right"
    assert world.held is None


def test_cached_high_pregrasp_still_lifts_only_30mm_first():
    sk, _, _ = fixture()
    sk.cfg.motion.descent_step_per_tick = 0.3
    pick(sk)
    attempt = sk.s.held.attempt
    hover = replace(attempt.hover, joints={**attempt.hover.joints,
                                           "elbow_flex": sk.s.grasp_z_mm + 50.0})
    sk.s.held.attempt = replace(attempt, hover=hover,
                                hover_z_mm=sk.s.grasp_z_mm + 50.0)
    result = sk._lift_held_joint_space()
    assert result.ok, result
    assert sk.s.arm_position_mm()[2] <= (
        sk.s.grasp_z_mm + sk.cfg.task1.tilted_pick_hover_clearance_mm
        + sk.limits.lateral_clearance_tolerance_mm)


def test_verified_grasp_survives_transport_load_relaxation(monkeypatch):
    sk, _, robot = fixture()
    pick(sk)
    read_loads = robot.read_loads

    def relaxed_load():
        loads = read_loads()
        loads["gripper"] = 56
        return loads

    monkeypatch.setattr(robot, "read_loads", relaxed_load)
    result = sk._lift_held_joint_space()
    assert result.ok and result.data["initial_lift_mm"] >= (
        sk.cfg.task1.tilted_pick_hover_clearance_mm
        - sk.limits.lateral_clearance_tolerance_mm)
    robot.joints["gripper"] = 3.4
    assert not sk._held_check()


def test_loaded_endpoint_miss_is_recoverable(monkeypatch):
    sk, _, _ = fixture()
    pick(sk)
    assert sk.move_relative(up_mm=50).ok
    start = sk.s.arm_position_mm()
    monkeypatch.setattr(sk.s.player, "move_through", lambda *_args, **_kwargs: None)
    sk.limits.calibrated_pick = False
    result = sk._move("move_to_target", (start[0] + 40.0, start[1], start[2]))
    assert not result.ok and result.reason == "grasp_blocked"
    assert result.data["target_mm"] and result.data["measured_fk_mm"]
    assert sk.s.held is not None


def test_loaded_zone_carry_rejects_low_planned_fk_before_motion(monkeypatch):
    from agent_helpers import FakeIk
    from control.ik import IkResult

    class LowZoneIk(FakeIk):
        def solve(self, x_mm, y_mm, z_mm, yaw_deg=None, radial_tilt_deg=0.0):
            plan = super().solve(x_mm, y_mm, z_mm, yaw_deg, radial_tilt_deg)
            if x_mm >= 200.0:
                joints = {**plan.joints, "elbow_flex": z_mm - 12.0}
                return IkResult(joints, 12.0, plan.tilt_error_deg)
            return plan

    sk, _, robot = fixture()
    pick(sk)
    assert sk.move_relative(up_mm=50).ok
    sk.s._ik = LowZoneIk()
    sent_before = len(robot.sent_actions)
    result = sk._move("move_to_target", (250.0, 0.0, 48.0),
                      max_ik_error_mm=20.0, level_during_carry=True)
    assert not result.ok and result.reason == "ik_gate"
    assert "below clearance" in result.detail
    assert len(robot.sent_actions) == sent_before


def test_zone_drop_requires_verified_height_before_release():
    sk, world, _ = fixture()
    pick(sk)
    assert sk.move_relative(up_mm=50).ok
    assert sk.move_to_target("slot", "preplace", slot="top-left").ok
    assert not sk.open_gripper().ok
    dropped = sk.drop_at_zone_target()
    assert dropped.ok and dropped.data["release_mode"] == "height_drop"
    assert dropped.data["drop_z_mm"] == sk.s.drop_z_mm
    assert sk.open_gripper().ok
    assert world.held is None


def test_zone_drop_accepts_small_loaded_fk_undershoot(monkeypatch):
    sk, _, _ = fixture()
    pick(sk)
    assert sk.move_relative(up_mm=50).ok
    assert sk.move_to_target("slot", "preplace", slot="top-left").ok
    start = sk.s.arm_position_mm()
    measured = [start]
    monkeypatch.setattr(sk.s, "arm_position_mm", lambda: measured[0])
    monkeypatch.setattr(
        sk.s.player, "move_to",
        lambda *_args, **_kwargs: measured.__setitem__(
            0, (*start[:2], sk.s.grasp_z_mm + sk.limits.zone_release_floor_margin_mm + 0.3)),
    )
    result = sk.drop_at_zone_target()
    assert result.ok and result.data["release_mode"] == "height_drop"
    assert sk.open_gripper().ok


def test_block_transfer_retries_after_safe_home(monkeypatch):
    sk, _, _ = fixture()
    calls = []
    results = iter((
        SkillResult(False, "move_block_to_slot", "limit_exceeded", data={"failed_stage": "lift_empty"}),
        SkillResult(True, "move_block_to_slot", "released"),
    ))
    monkeypatch.setattr(sk, "_move_block_to_slot_once", lambda *_args: next(results))
    monkeypatch.setattr(sk.s, "arm_at_home", lambda: False)
    monkeypatch.setattr(sk, "return_to_home",
                        lambda: calls.append("home") or SkillResult(True, "return_to_home", "ok"))
    result = sk.move_block_to_slot("yellow", "top-left")
    assert result.ok and calls == ["home"]
    assert len(result.data["attempts"]) == 2


def test_block_transfer_stops_before_transport_when_grasp_fails(monkeypatch):
    sk, world, _ = fixture()
    monkeypatch.setattr(
        sk, "close_gripper",
        lambda: SkillResult(False, "close_gripper", "grasp_empty", "empty"),
    )

    result = sk.move_block_to_slot("yellow", "top-left")

    assert not result.ok and result.reason == "grasp_empty"
    assert result.data["failed_stage"] == "close_verify"
    assert world.held is None
    assert "preplace" not in [step["stage"] for step in result.data["steps"]]


def test_unverified_transfer_at_source_is_recoverable(monkeypatch):
    sk, world, _ = fixture()
    monkeypatch.setattr(
        ContactMonitor, "check",
        lambda self: ContactReading(
            True, loads={joint: 0.0 for joint in sk.cfg.sensing.contact_joints}
        ),
    )
    home = sk.return_to_home
    source = world.blocks["yellow"]

    def home_after_simulated_drop():
        result = home()
        world.blocks["yellow"] = source
        return result

    monkeypatch.setattr(sk, "return_to_home", home_after_simulated_drop)
    result = sk.move_block_to_slot("yellow", "top-left")

    assert not result.ok and result.reason == "task_incomplete"
    assert result.data["still_at_source"] is True
    assert result.retry_advice == "try_other_target"
    assert result.to_envelope()["severity"] == "warning"
    assert SkillResult(False, "move", "motion_timeout").to_envelope()["severity"] == "error"


def test_failed_transfer_while_holding_is_error_not_warning():
    blocked = SkillResult(False, "move_block_to_slot", "ik_gate", "lift blocked",
                          data={"holding": "red", "failed_stage": "lift_held"},
                          state={"holding": "red", "arm_at_home": False})
    skipped = SkillResult(False, "move_block_to_slot", "neighbour_clearance",
                          data={"holding": None}, state={"holding": None})
    assert blocked.to_envelope()["severity"] == "error"
    assert skipped.to_envelope()["severity"] == "warning"


def test_task2_explicit_floors_use_requested_height_without_contact(monkeypatch):
    cfg = fast_cfg()
    cfg.agent.relative.frame = "base"
    base, world, _ = make_skills({"yellow": (160., 40.), "red": (140., -120.)}, cfg=cfg)
    sk = PrimitiveSkills(base.s)

    def contact_must_not_run(_monitor):
        raise AssertionError("height drop must not read contact")

    monkeypatch.setattr(ContactMonitor, "check", contact_must_not_run)
    first = sk.stack_block_to_floor("yellow", 0)
    second = sk.stack_block_to_floor("red", 1)
    assert first.ok and second.ok
    assert (first.data["floor"], second.data["floor"]) == (0, 1)
    assert second.data["expected_place_z_mm"] - first.data["expected_place_z_mm"] == cfg.task2.block_height_mm
    assert first.data["release_mode"] == second.data["release_mode"] == "height_drop"
    assert first.data["contact_confirmed"] is second.data["contact_confirmed"] is False
    assert sk._task2_placed_floors == {0: "yellow", 1: "red"}
    assert world.held is None


def test_task2_explicit_floor_uses_visible_support_without_prior_history():
    cfg = fast_cfg()
    cfg.agent.relative.frame = "base"
    base, world, _ = make_skills({"red": (140., -120.)}, cfg=cfg)
    sk = PrimitiveSkills(base.s)
    world.blocks["wood"] = sk.s.stack.stack_xy_mm
    result = sk.stack_block_to_floor("red", 2)
    assert result.ok and result.data["floor"] == 2
    assert result.data["support_evidence"] == "visible_near_tower_height_unverified"


def test_task2_uses_prior_confirmed_support_when_arm_occludes_it(monkeypatch):
    from dataclasses import replace

    cfg = fast_cfg()
    cfg.agent.relative.frame = "base"
    base, _, _ = make_skills({"yellow": (160., 40.), "red": (140., -120.)}, cfg=cfg)
    sk = PrimitiveSkills(base.s)
    assert sk.stack_block_to_floor("yellow", 0).ok
    assert sk.move_relative(up_mm=40).ok
    assert not sk.s.arm_at_home()

    observe = sk.observe_scene
    def occluded_once():
        result = observe()
        scene = sk._observed_scene
        sk._observed_scene = replace(scene, inside={
            color: block for color, block in scene.inside.items()
            if color != "yellow"
        })
        return result

    monkeypatch.setattr(sk, "observe_scene", occluded_once)
    result = sk.stack_block_to_floor("red", 1)
    assert result.ok
    assert result.data["support_evidence"] == "prior_release_unseen_height_unverified"


def test_task2_floor_zero_can_be_reused_after_block_moves_outside():
    cfg = fast_cfg()
    cfg.agent.relative.frame = "base"
    base, world, _ = make_skills({"yellow": (160., 40.), "red": (140., -120.)}, cfg=cfg)
    sk = PrimitiveSkills(base.s)
    assert sk.stack_block_to_floor("yellow", 0).ok
    world.blocks["yellow"] = (120., 110.)
    result = sk.stack_block_to_floor("red", 0)
    assert result.ok and result.data["floor"] == 0
    assert sk._task2_placed_floors == {0: "red"}


def test_task2_retries_requested_floor_without_visible_support():
    cfg = fast_cfg()
    cfg.agent.relative.frame = "base"
    base, world, _ = make_skills({"red": (140., -120.)}, cfg=cfg)
    sk = PrimitiveSkills(base.s)
    result = sk.stack_block_to_floor("red", 1)
    assert result.ok and result.data["floor"] == 1
    assert result.data["support_evidence"] == "requested_floor_unverified"
    assert world.held is None
    assert sk.stack_block_to_floor("red", -1).reason == "invalid_arguments"


def test_task2_can_repick_fallen_block_inside_zone():
    cfg = fast_cfg()
    cfg.agent.relative.frame = "base"
    base, world, _ = make_skills({"yellow": (160., 40.)}, cfg=cfg)
    sk = PrimitiveSkills(base.s)
    world.blocks["yellow"] = sk.s.stack.stack_xy_mm
    result = sk.stack_block_to_floor("yellow", 0)
    assert result.ok and result.data["floor"] == 0
    assert world.held is None


def test_zone_retreat_keeps_gripper_open_after_release(monkeypatch):
    sk, _, robot = fixture()
    sk._zone_retreat_joints = {
        **robot.read_joints(),
        "shoulder_pan": robot.joints["shoulder_pan"] + 1.0,
        "gripper": 5.0,
    }
    monkeypatch.setattr(sk.s.ik, "forward_position_mm", lambda _joints: (0.0, 0.0, 50.0))
    move = Mock()
    monkeypatch.setattr(sk.s.player, "move_to", move)
    monkeypatch.setattr(sk.s, "return_home_safely", lambda: (False, True))

    assert sk.return_to_home().ok
    assert "gripper" not in move.call_args.args[0]


def test_home_does_not_sweep_low_when_vertical_lift_is_unreachable(monkeypatch):
    sk, _, robot = fixture()
    robot.joints.update(shoulder_pan=230.0, elbow_flex=15.0)
    monkeypatch.setattr(sk.s, "lift_in_place", lambda: False)
    home = Mock(side_effect=AssertionError("unsafe home motion"))
    monkeypatch.setattr(sk.s, "go_home", home)
    with pytest.raises(TimeoutError, match="hover clearance"):
        sk.s.return_home_safely()
    home.assert_not_called()


def test_home_can_fold_when_low_but_already_in_home_column(monkeypatch):
    sk, _, robot = fixture()
    robot.joints.update(shoulder_pan=170.0, shoulder_lift=25.0, elbow_flex=30.0)
    monkeypatch.setattr(sk.s, "lift_in_place", lambda: False)
    home = Mock()
    monkeypatch.setattr(sk.s, "go_home", home)
    lifted, at_home = sk.s.return_home_safely()
    assert not lifted and not at_home
    home.assert_called_once_with()


def test_low_home_path_requires_rise_before_lateral_sweep():
    from session.arm_session import _low_home_path_clear

    common = dict(safe_z=35.0, home_radius=40.0,
                  low_lateral_limit=20.0, tolerance=2.0)
    assert _low_home_path_clear(
        (280.0, 0.0, 17.0),
        [(275.0, 0.0, 27.0), (265.0, 0.0, 37.0),
         (200.0, 0.0, 65.0), (158.0, 0.0, 8.0)],
        (157.0, 0.0), **common)
    assert not _low_home_path_clear(
        (280.0, 0.0, 17.0),
        [(250.0, 0.0, 20.0), (220.0, 0.0, 40.0),
         (158.0, 0.0, 8.0)],
        (157.0, 0.0), **common)


def test_stop_recovery_uses_direct_home_when_low_path_gate_would_refuse(monkeypatch):
    sk, _, _ = fixture()
    monkeypatch.setattr(sk.s, "return_home_safely",
                        Mock(side_effect=AssertionError("low path gate must not block STOP home")))
    home = Mock()
    monkeypatch.setattr(sk.s.motion, "go_home", home)
    monkeypatch.setattr(sk.s, "arm_at_home", lambda: True)
    result = sk.recover_and_home()
    assert result.ok and result.data["arm_at_home"]
    home.assert_called_once_with(include_gripper=False)


def test_task2_route_guard_only_blocks_low_sweep_through_local_tower(monkeypatch):
    sk, _, robot = fixture()
    tower_x, tower_y = sk.s.stack.stack_xy_mm
    grasp_z = sk.s.grasp_z_mm
    start = {**robot.read_joints(), "shoulder_pan": 0.0}
    goal = {**start, "shoulder_pan": 100.0}
    height = grasp_z + 10.0
    y_offset = 0.0

    def fake_fk(joints):
        return (tower_x + (joints["shoulder_pan"] - 50.0) * 2.0,
                tower_y + y_offset, height)

    monkeypatch.setattr(sk.s.ik, "forward_position_mm", fake_fk)
    guard = sk._task2_path_clear_of_tower
    assert guard(start, (goal,))  # Empty zone: far-side pick is allowed.

    sk._task2_placed_floors[0] = "yellow"
    assert not guard(start, (goal,))
    y_offset = sk.cfg.agent.place_clear_radius_mm + 1.0
    assert guard(start, (goal,))  # Other parts of the target zone stay open.
    y_offset = 0.0
    height = grasp_z + sk.cfg.task2.tower_path_clearance_mm + 1.0
    assert guard(start, (goal,))  # Cross above one block.

    sk._task2_placed_floors[1] = "red"
    assert not guard(start, (goal,))  # Second recorded floor raises the wall.
    height += sk.cfg.task2.block_height_mm
    assert guard(start, (goal,))


def test_task1_zone_guard_is_local_and_height_aware():
    from types import SimpleNamespace

    sk, world, _ = fixture()
    assert sk.observe_scene().ok
    point = (*sk.s.slot_centres[0], sk.s.grasp_z_mm + 10.0)
    assert not sk._task1_near_low_zone_block(point)

    world.blocks["red"] = sk.s.slot_centres[0]
    assert sk.observe_scene().ok
    block_xy = sk._observed_scene.inside["red"].center_mm
    low = (*block_xy, sk.s.grasp_z_mm + 10.0)
    assert sk._task1_near_low_zone_block(low)
    assert not sk._task1_near_low_zone_block(
        (block_xy[0] + sk.cfg.agent.place_clear_radius_mm + 1.0,
         block_xy[1], low[2]))
    assert not sk._task1_near_low_zone_block(
        (*block_xy, sk.s.grasp_z_mm + sk.cfg.task1.zone_path_clearance_mm + 1.0))

    sk.s.held = SimpleNamespace(color="yellow")
    assert sk._task1_near_low_zone_block(
        (*block_xy, sk.s.grasp_z_mm + sk.cfg.task1.zone_path_clearance_mm + 1.0))
    assert not sk._task1_near_low_zone_block(
        (*block_xy, sk.s.grasp_z_mm + sk.cfg.task1.zone_path_clearance_mm
         + sk.cfg.agent.calibration_clearance.obstacle_height_mm + 1.0))

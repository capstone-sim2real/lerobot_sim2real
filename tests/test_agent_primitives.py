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
from control.sensing import ContactReading


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

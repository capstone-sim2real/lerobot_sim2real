"""Composed demonstrations with real recorder/decorator and simulated IO only."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from agent_helpers import make_skills, fast_cfg
from agent.tools import ToolRegistry
from agent.provider.types import ToolCall
from data.episode_recorder import EpisodeRecorder, RecordingRobotIO
from session.collection import CollectionResources
from session.primitives import PrimitiveSkills
from control.sensing import ContactReading
from test_task3 import _Sink


@pytest.fixture
def rig(tmp_path, monkeypatch):
    # FakeIk is linear Cartesian geometry, not the real arm folding corridor.
    monkeypatch.setattr("session.arm_session._low_home_path_clear", lambda *_a, **_k: True)
    cfg = fast_cfg()
    cfg.agent.relative.frame = "base"
    cfg.task3.min_episode_frames = 2
    cfg.task3.max_episode_frames = 3000
    cfg.task3.motion_fps_override = 0.0  # fake IO has no wall-clock transport latency
    sk, world, robot = make_skills({"yellow": (160., 40.)}, cfg=cfg)
    sink = _Sink()
    source = SimpleNamespace(latest=lambda: SimpleNamespace(image=np.zeros((2,2,3),dtype=np.uint8), received_at=time.monotonic()))
    finish = Mock()
    resources = CollectionResources(EpisodeRecorder(sink, {"top": source}, cfg.task3), tmp_path / "dataset", finish)
    skills = PrimitiveSkills(sk.s, collection_factory=lambda cfg: resources)
    # Virtual control clock advances at exactly record_fps per recorded tick.
    mono = [0.0]
    skills.collection.clock = lambda: mono[0]
    original = skills.collection.check_tick
    def tick():
        mono[0] += 1/cfg.task3.record_fps
        original()
    monkeypatch.setattr(skills.collection, "check_tick", tick)
    monkeypatch.setattr(RecordingRobotIO, "_pace", lambda self: None)
    registry = ToolRegistry(cfg, lambda job: job(skills))
    def call(name, **args):
        return registry.execute(ToolCall("call", name, args)).content
    return SimpleNamespace(sk=skills, sink=sink, world=world, robot=robot, call=call, mono=mono,
                           resources=resources, finish=finish, registry=registry)


def begin(rig):
    assert rig.call("observe_scene")["ok"]
    assert rig.call("begin_episode", object_id="yellow_1", observation_id=1)["ok"]


def deliver(rig, monkeypatch):
    begin(rig)
    assert rig.call("open_gripper")["ok"]
    assert rig.call("move_relative", up_mm=50)["ok"]
    assert rig.call("move_to_target", target_type="object", phase="pregrasp", object_id="yellow_1", observation_id=1)["ok"]
    assert rig.call("move_to_target", target_type="object", phase="grasp", object_id="yellow_1", observation_id=1)["ok"]
    assert rig.call("close_gripper")["ok"]
    assert rig.call("move_relative", up_mm=50)["ok"]
    assert rig.call("move_to_target", target_type="slot", phase="preplace", slot="top-left")["ok"]
    monkeypatch.setattr("control.sensing.ContactMonitor.check",
                        lambda self: ContactReading(rig.sk.s.arm_position_mm()[2] <= rig.sk.s.grasp_z_mm))
    assert rig.call("descend_until_contact", max_descent_mm=60)["ok"]
    assert rig.call("open_gripper")["ok"]
    assert rig.call("return_to_home")["ok"]


def test_task3_outcome_is_composed_without_running_task3(rig, monkeypatch):
    deliver(rig, monkeypatch)
    assert rig.call("save_episode")["ok"]
    assert rig.sink.save_count == 1
    assert all("observation.images.top" in f and "action" in f for f in rig.sink.saved[0])
    assert not rig.sk.collection.recording
    assert rig.call("collection_status")["collection"]["episodes_saved"] == 1
    assert rig.call("finish_dataset")["ok"]
    rig.finish.assert_called_once()


def test_llm_cannot_claim_success_or_save_before_verified_motion(rig):
    begin(rig)
    assert not rig.call("save_episode", success=True)["ok"]
    assert not rig.call("save_episode")["ok"]
    assert rig.sink.save_count == 0
    assert not rig.call("finish_dataset")["ok"]


def test_cancel_during_model_wait_discards_without_motion(rig, monkeypatch):
    begin(rig)
    rig.sk.s.cancel.set()
    send = Mock(side_effect=AssertionError("must not write after STOP"))
    monkeypatch.setattr(rig.robot, "send_joints", send)
    rig.sk.idle_tick()
    assert not rig.sk.collection.recorder.is_open
    send.assert_not_called()


def test_long_tick_gap_discards_instead_of_compressing_time(rig):
    begin(rig)
    rig.mono[0] += 1
    rig.sk.idle_tick()
    assert not rig.sk.collection.recorder.is_open
    assert rig.sk.collection.recorder.discard_reasons["timing_gap"] == 1
    assert rig.sink.save_count == 0


def test_worker_idle_recording_stays_on_owner_thread():
    from agent.worker import RobotWorker
    ready = threading.Event()
    ids = []
    class Resource:
        idle_poll_s = 0.001
        def idle_tick(self):
            ids.append(threading.get_ident())
            ready.set()
        def close(self): pass
    worker = RobotWorker(Resource)
    worker.start()
    try:
        owner_id = worker.run(lambda _: threading.get_ident())
        assert ready.wait(1)
        assert set(ids) == {owner_id}
    finally:
        worker.stop()


def test_failed_finalization_blocks_new_recording_until_retry(rig):
    begin(rig)
    assert rig.call("discard_episode", reason="operator_requested")["ok"]
    rig.finish.side_effect = OSError("disk unavailable")
    assert not rig.call("finish_dataset")["ok"]
    assert not rig.call("begin_episode", object_id="yellow_1", observation_id=1)["ok"]
    assert rig.sk.collection.finalizing
    rig.finish.side_effect = None
    assert rig.call("finish_dataset")["ok"]


def test_preplanned_tool_sequence_records_and_returns_each_result(rig):
    plan = [
        {"name": "open_gripper", "arguments": {}},
        {"name": "return_to_home", "arguments": {}},
    ]
    result = rig.call("record_tool_sequence", task="Move the yellow block as requested.",
                      color="yellow", steps=plan)
    assert result["ok"], result
    assert [item["tool"] for item in result["step_results"]] == ["open_gripper", "return_to_home"]
    assert result["collection"]["episodes_saved"] == 1
    assert rig.sink.save_count == 1
    assert rig.sink.saved[0][0]["task"] == "Move the yellow block as requested."


def test_standalone_episode_still_rejects_composite_stack(rig):
    begin(rig)
    result = rig.call("stack_block_to_floor", color="yellow", floor=0)
    assert not result["ok"]
    assert result["detail"] == "Finish the recording episode before stacking"
    assert rig.sink.save_count == 0


def test_recorded_task2_stack_sequence_runs_and_saves(rig):
    result = rig.call("record_tool_sequence", task="Place yellow at stack floor zero and return home.",
                      color="yellow", steps=[
                          {"name": "stack_block_to_floor", "arguments": {"color": "yellow", "floor": 0}},
                          {"name": "return_to_home", "arguments": {}},
                      ])
    assert result["ok"], result
    assert [item["tool"] for item in result["step_results"]] == [
        "stack_block_to_floor", "return_to_home"]
    assert result["collection"]["episodes_saved"] == 1
    assert rig.sink.save_count == 1


def test_recorded_task1_transfer_sequence_runs_and_saves(rig):
    result = rig.call("record_tool_sequence", task="Move yellow to the top-left zone slot.",
                      color="yellow", steps=[
                          {"name": "move_block_to_slot", "arguments": {"color": "yellow", "slot": "top-left"}},
                      ])
    assert result["ok"], result
    assert result["collection"]["episodes_saved"] == 1
    assert rig.sink.save_count == 1


def test_invalid_sequence_is_rejected_before_recording(rig):
    result = rig.call("record_tool_sequence", task="Unsafe plan", color="yellow",
                      steps=[{"name": "move_relative", "arguments": {"up_mm": 9999}}])
    assert not result["ok"]
    assert result["reason"] == "invalid_arguments"
    assert rig.sk.collection.resources is None


def test_recorded_sequence_stops_and_discards_on_first_failed_step(rig):
    result = rig.call("record_tool_sequence", task="Pick up the yellow block.", color="yellow",
                      steps=[
                          {"name": "move_to_target", "arguments": {
                              "target_type": "object", "phase": "pregrasp",
                              "object_id": "yellow_1", "observation_id": 999}},
                          {"name": "open_gripper", "arguments": {}},
                      ])
    assert not result["ok"]
    assert [item["tool"] for item in result["step_results"]] == ["move_to_target"]
    assert rig.sink.save_count == 0
    assert not rig.sk.collection.recording


def test_sequence_records_servo_tail_before_compressing_plan_gap(rig, monkeypatch):
    rig.sk.collection.begin("yellow", task_text="Move yellow.", sequence_mode=True)
    read = rig.robot.read_joints
    samples = iter([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
    last = [2.0]

    def coasting_read():
        joints = read()
        last[0] = next(samples, last[0])
        joints["gripper"] = last[0]
        return joints

    monkeypatch.setattr(rig.robot, "read_joints", coasting_read)
    before = rig.sk.collection.recorder.frames
    rig.sk.collection.settle_for_sequence_pause()
    assert rig.sk.collection.recorder.frames > before + 5
    rig.mono[0] += 1.0  # IK planning is safe only after the tail was recorded.
    rig.sk.collection.io.send_joints(dict(rig.sk.collection.io.last_command))
    assert rig.sk.collection.recording
    assert rig.sk.collection.stationary_pauses[-1]["max_joint_drift"] == 0
    rig.sk.collection.discard("operator_requested")


def test_sequence_compresses_only_measured_stationary_pause(rig):
    rig.sk.collection.begin("yellow", task_text="Move yellow.", sequence_mode=True)
    rig.mono[0] += 1.0
    rig.sk.collection.check_tick()
    assert rig.sk.collection.recording
    assert len(rig.sk.collection.stationary_pauses) == 1
    assert not rig.sk.collection.intervals or max(rig.sk.collection.intervals) < 0.1
    rig.sk.collection.discard("operator_requested")


def test_sequence_discards_pause_with_unrecorded_motion(rig):
    rig.sk.collection.begin("yellow", task_text="Move yellow.", sequence_mode=True)
    rig.robot.joints["shoulder_pan"] += 5.0
    rig.mono[0] += 1.0
    with pytest.raises(ValueError, match="gap budget exceeded"):
        rig.sk.collection.check_tick()
    assert rig.sk.collection.recorder.discard_reasons["moving_timing_gap"] == 1


def test_recorded_sequence_resolves_prior_result_and_accepts_label(rig, monkeypatch):
    source = Mock(return_value=rig.sk._result(True, "observe_scene", "ok",
                                            object_id="selected_1", observation_id=72))
    align = Mock(return_value=rig.sk._result(True, "align_gripper", "moved"))
    monkeypatch.setattr(rig.sk, "observe_scene", source)
    monkeypatch.setattr(rig.sk, "align_gripper", align)
    result = rig.call("record_tool_sequence", task="Inspect the selected target", color="selected",
                      steps=[{"name":"observe_scene", "arguments":{}},
                             {"name":"align_gripper", "arguments":{
                                 "object_id":{"$ref":"step1.object_id"},
                                 "observation_id":{"$ref":"step1.observation_id"}}}])
    assert result["ok"], result
    align.assert_called_once_with(object_id="selected_1", observation_id=72)


@pytest.mark.parametrize("gap", [2/30, 3/30, 0.3])
def test_brief_missing_motion_is_logged_without_discard(rig, gap):
    c = rig.sk.collection
    c.begin("yellow", task_text="Move yellow.", sequence_mode=True)
    rig.robot.joints["shoulder_pan"] += 5
    rig.mono[0] += gap
    c.check_tick()
    assert c.recording and c.recorder.is_open
    assert c.recording_gaps[-1]["accepted"]
    assert c.missing_motion_s > 0
    assert not c.stationary_pauses
    assert c.intervals[-1] >= gap


def test_repeated_motion_gaps_exceed_cumulative_budget(rig):
    c = rig.sk.collection
    c.begin("yellow", task_text="Move yellow.", sequence_mode=True)
    rig.robot.joints["shoulder_pan"] += 5
    for _ in range(3):
        rig.mono[0] += 0.3
        c.check_tick()
    assert c.recording
    rig.mono[0] += 0.3
    with pytest.raises(ValueError, match="missing_total"):
        c.check_tick()
    assert not c.recording and not c.recorder.is_open
    assert not c.recording_gaps[-1]["accepted"]


def test_observation_records_recovery_tail_before_camera_wait(rig, monkeypatch):
    rig.sk.collection.begin("yellow", task_text="Move yellow.", sequence_mode=True)
    settle = Mock(wraps=rig.sk.collection.settle_for_sequence_pause)
    monkeypatch.setattr(rig.sk.collection, "settle_for_sequence_pause", settle)
    observe = rig.sk.s.observe
    def delayed_observe(**kwargs):
        settle.assert_called_once()
        rig.mono[0] += 0.857
        return observe(**kwargs)
    monkeypatch.setattr(rig.sk.s, "observe", delayed_observe)
    assert rig.sk.observe_scene().ok
    rig.sk.collection.io.send_joints(dict(rig.sk.collection.io.last_command))
    assert rig.sk.collection.recording
    assert rig.sk.collection.stationary_pauses[-1]["duration_s"] >= 0.857
    rig.sk.collection.discard("test_done")


def test_recording_quality_failure_is_not_a_robot_fault():
    from session.collection import RecordingQualityError
    from agent.tools import result_from_exception
    result = result_from_exception("record_tool_sequence", RecordingQualityError("gap budget exceeded"))
    assert result.reason == "task_incomplete"
    assert not result.robot_fault
    assert result.severity == "warning"


def test_tolerated_jitter_does_not_exhaust_long_gap_budget(rig):
    c = rig.sk.collection
    c.begin("yellow", task_text="Move yellow.", sequence_mode=True)
    for _ in range(40):
        rig.mono[0] += 1/30
        c.io.send_joints(dict(c.io.last_command))
    assert c.recording
    assert c.missing_motion_s > 1.0
    c.discard("test_done")


def test_gap_discards_recording_without_interrupting_current_motion(rig):
    c = rig.sk.collection
    c.begin("yellow", task_text="Move yellow.", sequence_mode=True)
    c.writer.last_measured["shoulder_pan"] -= 5
    rig.mono[0] += 1.0
    sent = c.io.send_joints(dict(c.io.last_command))
    assert sent
    assert not c.recording
    assert c.last_error == "moving_timing_gap"
    assert not rig.sk.s.cancel.is_set()


def test_transport_hold_check_does_not_wait_for_unused_load_samples(rig, monkeypatch):
    rig.sk.s.held = SimpleNamespace(color="yellow")
    joints = rig.robot.read_joints()
    joints["gripper"] = rig.sk.cfg.sensing.gripper_empty_closed_max + 5
    monkeypatch.setattr(rig.robot, "read_joints", lambda: joints)
    monkeypatch.setattr("session.primitives.gripper.check_grasp", Mock(side_effect=AssertionError("transport must not average unused loads")))
    assert rig.sk._held_check()
    joints["gripper"] = rig.sk.cfg.sensing.gripper_empty_closed_max - 1
    assert not rig.sk._held_check()

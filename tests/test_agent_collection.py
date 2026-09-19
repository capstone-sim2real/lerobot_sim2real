"""Composed demonstrations with real recorder/decorator and simulated IO only."""
import json
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
    monkeypatch.setattr("session.primitives.ContactMonitor.check",
                        lambda self: ContactReading(rig.sk.s.arm_position_mm()[2] <= rig.sk.s.grasp_z_mm))
    assert rig.call("descend_until_contact", max_descent_mm=60)["ok"]
    assert rig.call("open_gripper")["ok"]
    assert rig.call("return_to_home")["ok"]


def test_task3_outcome_is_composed_without_running_task3(rig, monkeypatch):
    deliver(rig, monkeypatch)
    assert rig.call("save_episode")["ok"] is False  # no visual evidence yet
    assert rig.call("observe_scene")["ok"]
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

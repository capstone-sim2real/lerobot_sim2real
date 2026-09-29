"""LLM-free mission choice and stopping behavior with fake composite tools."""

import numpy as np

from agent.primitive_mission import PrimitiveMission
from config import AppConfig
from perception.homography import PlaneCalibration


def calibration():
    return PlaneCalibration(np.eye(3), (500, 400), 1.0, base_xy_mm=(0.0, 0.0))


def scene(outside, inside):
    objects = [
        {"color": color, "x_mm": xy[0], "y_mm": xy[1]}
        for color, xy in outside.items()
    ] + [
        {"color": color, "x_mm": xy[0], "y_mm": xy[1], "slot": "top-left"}
        for color, xy in inside.items()
    ]
    state = {"holding": None, "arm_at_home": True,
             "zone_slots": {"top-left": None, "top-center": None,
                            "top-right": None, "bottom-left": None,
                            "bottom-right": None}}
    return {"ok": True, "objects": objects, "state": state}


def test_task1_tries_next_candidate_after_bounded_failure():
    cfg = AppConfig()
    outside = {"green": (250.0, -230.0), "blue": (350.0, 0.0)}
    inside = {"red": (220.0, 10.0), "yellow": (230.0, 10.0),
              "wood": (240.0, 10.0)}
    calls = []

    def call(name, args):
        calls.append((name, args))
        if name == "observe_scene":
            return scene(outside, inside)
        if name == "recover_and_home":
            return {"ok": True, "reason": "ok", "arm_at_home": True}
        if args["color"] == "green" and len([x for x in calls if x[0] == name]) == 1:
            return {"ok": False, "reason": "neighbour_clearance",
                    "retry_advice": "try_other_target", "state": {"holding": None}}
        color = args["color"]
        inside[color] = outside.pop(color)
        return {"ok": True, "reason": "released", "state": {"holding": None}}

    result = PrimitiveMission(cfg, calibration(), call).run(1)
    transfers = [args["color"] for name, args in calls if name == "move_block_to_slot"]
    assert transfers == ["green", "blue", "green"]
    assert result["status"] == "complete"
    assert result["failures"][0]["reason"] == "neighbour_clearance"


def test_task2_uses_floors_in_order_without_model_calls():
    cfg = AppConfig()
    outside = {
        "red": (210.0, 0.0), "yellow": (220.0, 0.0),
        "green": (230.0, 0.0), "blue": (240.0, 0.0),
        "wood": (250.0, 0.0),
    }
    inside = {}
    floors = []
    clock = [0.0]

    def call(name, args):
        if name == "observe_scene":
            return scene(outside, inside)
        floors.append(args["floor"])
        color = args["color"]
        inside[color] = outside.pop(color)
        return {"ok": True, "reason": "released", "state": {"holding": None}}

    def sleep(seconds):
        clock[0] += seconds

    result = PrimitiveMission(cfg, calibration(), call,
                              clock=lambda: clock[0], sleep=sleep).run(2)
    assert floors == [0, 1, 2, 3, 4]
    assert result["status"] == "placed_unverified"
    assert clock[0] >= 4.99


def test_held_block_failure_opens_homes_and_tries_another_target():
    cfg = AppConfig()
    calls = []
    outside = {"blue": (300.0, 0.0), "green": (200.0, 0.0)}
    inside = {}

    def call(name, args):
        calls.append(name)
        if name == "observe_scene":
            return scene(outside, inside)
        if name == "recover_and_home":
            return {"ok": True, "reason": "ok", "arm_at_home": True}
        if args["color"] == "blue":
            return {"ok": False, "reason": "ik_gate", "state": {"holding": "blue"}}
        inside["green"] = outside.pop("green")
        return {"ok": True, "reason": "released", "state": {"holding": None}}

    result = PrimitiveMission(cfg, calibration(), call).run(1)
    assert result["status"] == "incomplete"
    assert calls[:5] == ["observe_scene", "move_block_to_slot",
                         "recover_and_home", "observe_scene", "move_block_to_slot"]
    assert result["colors"] == ["green"]


def test_release_then_failed_check_is_reobserved_before_retry():
    cfg = AppConfig()
    outside = {"green": (300.0, 0.0)}
    inside = {color: (200.0, 0.0) for color in ("red", "blue", "yellow", "wood")}
    calls = []

    def call(name, args):
        calls.append(name)
        if name == "observe_scene":
            return scene(outside, inside)
        if name == "recover_and_home":
            return {"ok": True, "reason": "ok", "arm_at_home": True}
        inside["green"] = outside.pop("green")
        return {"ok": False, "reason": "task_incomplete", "failed_stage": "verify",
                "state": {"holding": None}}

    result = PrimitiveMission(cfg, calibration(), call).run(1)
    assert result["status"] == "complete"
    assert calls == ["observe_scene", "move_block_to_slot", "recover_and_home", "observe_scene"]


def test_operator_stop_does_not_restart_mission():
    cfg = AppConfig()
    stopped = [False]
    calls = []

    def call(name, args):
        calls.append(name)
        if name == "observe_scene":
            return scene({"green": (300.0, 0.0)}, {})
        stopped[0] = True
        return {"ok": False, "reason": "cancelled"}

    result = PrimitiveMission(cfg, calibration(), call, stopped=lambda: stopped[0]).run(1)
    assert result["status"] == "stopped"
    assert calls == ["observe_scene", "move_block_to_slot"]


def test_service_mission_never_calls_the_provider(monkeypatch):
    from agent.primitive_mission import PrimitiveMission
    from agent.control import ControlState
    from test_agent_service import _service

    called = []
    monkeypatch.setattr(PlaneCalibration, "load", lambda _path: calibration())

    def run(self, task):
        called.append(task)
        self.emit({"type": "mission_result", "task": task, "status": "complete",
                   "detail": "fake"})
        return {"status": "complete"}

    monkeypatch.setattr(PrimitiveMission, "run", run)
    service, _skills, events = _service([])
    token = service.acquire_lease(None)
    assert service.mission(token, 1) == (202, {"accepted": True, "task": 1, "mode": "no_llm"})
    service.wait_idle()
    assert called == [1]
    assert service.runner.history == []
    assert service.gate.state is ControlState.IDLE
    assert any(event.get("type") == "mission_result" for event in events)
    service.shutdown()


def test_service_mission_recovery_uses_worker_without_provider(monkeypatch):
    from agent.primitive_mission import PrimitiveMission
    from agent.control import ControlState
    from test_agent_service import _service

    monkeypatch.setattr(PlaneCalibration, "load", lambda _path: calibration())

    def run(self, task):
        recovered = self.call("recover_and_home", {})
        self.emit({"type": "mission_result", "task": task, "status": "incomplete"})
        return {"status": "incomplete" if recovered["ok"] else "needs_recovery"}

    monkeypatch.setattr(PrimitiveMission, "run", run)
    service, skills, events = _service([])
    token = service.acquire_lease(None)
    assert service.mission(token, 1)[0] == 202
    service.wait_idle()
    assert skills.homed == 1
    assert service.gate.state is ControlState.IDLE
    assert any(event.get("name") == "recover_and_home"
               for event in events if event.get("type") == "tool_result")
    service.shutdown()

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


def test_held_block_failure_stops_before_selecting_another():
    cfg = AppConfig()
    calls = []

    def call(name, args):
        calls.append(name)
        if name == "observe_scene":
            return scene({"blue": (300.0, 0.0)}, {})
        return {"ok": False, "reason": "ik_gate", "state": {"holding": "blue"}}

    result = PrimitiveMission(cfg, calibration(), call).run(1)
    assert result["status"] == "needs_recovery"
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

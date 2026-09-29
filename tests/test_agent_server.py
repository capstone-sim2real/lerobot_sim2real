"""HTTP surface of so101-agent (skipped when the agent extra is not installed)."""


import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from agent.provider.fake import ScriptedProvider  # noqa: E402
from agent.provider.types import TextDelta, TurnEnd  # noqa: E402
from agent.server import CONTROL_UI_VERSION, EventHub, create_app  # noqa: E402
from agent.service import AgentService  # noqa: E402
from config import AppConfig  # noqa: E402
from session.cancel import CancelToken  # noqa: E402

from test_agent_service import Skills  # noqa: E402


class FakePerceptionBackend:
    def __init__(self):
        self.backend = "cv"

    def status(self):
        return {"cv": {"available": True}, "yoloe": {"available": True, "control_capable": True},
                "selected": self.backend, "control_backend": self.backend}

    def set_backend(self, backend):
        if backend not in {"cv", "yoloe"}:
            raise ValueError("bad backend")
        self.backend = backend
        return backend

    def close(self):
        pass


def _client():
    cfg = AppConfig()
    hub = EventHub()
    perception = FakePerceptionBackend()

    def builder(publish):
        cancel = CancelToken()
        service = AgentService(cfg, provider=ScriptedProvider([[TextDelta("hi"), TurnEnd("end_turn")]]),
                               skills_factory=lambda: Skills(cancel), cancel=cancel, publish=publish,
                               system_prompt="sys", transcript_dir="")
        service.start = lambda timeout_s=None: (service._worker.start(), setattr(service, "started", True))
        return service

    return TestClient(create_app(cfg, builder, hub, perception_backend=perception))


def test_lease_is_shared_stop_is_open_and_commands_need_the_shared_token():
    with _client() as client:
        token = client.post("/api/lease").json()["token"]
        assert client.post("/api/lease").json()["token"] == token
        assert client.post("/api/lease", headers={"X-Operator-Token": "stale"}).json()["token"] == token
        assert client.post("/api/lease", headers={"X-Operator-Token": token}).json()["token"] == token
        assert client.post("/api/chat", json={"text": "hi"}).status_code == 403
        assert client.post("/api/home", headers={"X-Operator-Token": token}).status_code == 409
        assert client.post("/api/chat", json={"text": "hi"}, headers={"X-Operator-Token": token}).status_code == 202
        assert client.post("/api/jog", json={"left_mm": "x"}, headers={"X-Operator-Token": token}).status_code == 400
        assert client.post("/api/jog", json={"left_mm": 10}, headers={"X-Operator-Token": token}).status_code == 428
        config = client.get("/api/config").json()
        assert config["mjpeg_path"] == "/video/shoulder.mjpg" and config["provider"] == "fake"
        assert config["perception_backends"]["cv"]["available"] is True
        assert config["perception_backends"]["control_backend"] == "cv"
        assert config["perception_backends"]["yoloe"]["control_capable"] is True
        assert client.post("/api/perception/backend", json={"backend": "yoloe"}).status_code == 403
        switched = client.post(
            "/api/perception/backend", json={"backend": "yoloe"},
            headers={"X-Operator-Token": token},
        )
        assert switched.status_code == 200 and switched.json()["control_backend"] == "yoloe"
        assert client.post("/api/lease/force-release").status_code in (200, 403)
        stop = client.post("/api/stop").json()
        assert stop["stopped"] is True and stop["state"] == "stopped"
        assert client.post("/api/chat", json={"text": "hi"}, headers={"X-Operator-Token": token}).status_code == 409


def test_llm_free_mission_requires_current_ui_and_operator():
    with _client() as client:
        token = client.post("/api/lease").json()["token"]
        path = "/api/mission"
        assert client.post(path, json={"task": 1}).status_code == 428
        headers = {"X-Operator-Token": token,
                   "X-SO101-Control-Version": CONTROL_UI_VERSION}
        assert client.post(path, json={"task": 3}, headers=headers).status_code == 400
        assert client.post(path, json={"task": 1},
                           headers={"X-SO101-Control-Version": CONTROL_UI_VERSION}).status_code == 403

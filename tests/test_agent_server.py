"""HTTP surface of so101-agent (skipped when the agent extra is not installed)."""

import re

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


def _client():
    cfg = AppConfig()
    hub = EventHub()

    def builder(publish):
        cancel = CancelToken()
        service = AgentService(cfg, provider=ScriptedProvider([[TextDelta("hi"), TurnEnd("end_turn")]]),
                               skills_factory=lambda: Skills(cancel), cancel=cancel, publish=publish,
                               system_prompt="sys", transcript_dir="")
        service.start = lambda timeout_s=None: (service._worker.start(), setattr(service, "started", True))
        return service

    return TestClient(create_app(cfg, builder, hub))


def test_lease_is_shared_stop_is_open_and_commands_need_the_shared_token():
    with _client() as client:
        token = client.post("/api/lease").json()["token"]
        assert client.post("/api/lease").json()["token"] == token
        assert client.post("/api/lease", headers={"X-Operator-Token": "stale"}).json()["token"] == token
        assert client.post("/api/lease", headers={"X-Operator-Token": token}).json()["token"] == token
        assert client.post("/api/chat", json={"text": "hi"}).status_code == 403
        assert client.post("/api/home", headers={"X-Operator-Token": token}).status_code == 409
        stop = client.post("/api/stop").json()
        assert stop["stopped"] is False and stop["state"] == "idle"
        assert client.post("/api/chat", json={"text": "hi"}, headers={"X-Operator-Token": token}).status_code == 202
        assert client.post("/api/jog", json={"left_mm": "x"}, headers={"X-Operator-Token": token}).status_code == 400
        assert client.post("/api/jog", json={"left_mm": 10}, headers={"X-Operator-Token": token}).status_code == 428
        config = client.get("/api/config").json()
        assert config["mjpeg_path"] == "/video/shoulder.mjpg" and config["provider"] == "fake"
        assert client.post("/api/lease/force-release").status_code in (200, 403)

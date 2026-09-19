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
        index = client.get("/")
        assert index.status_code == 200
        assert "SO-101" in index.text
        assert index.headers["cache-control"].startswith("no-store")
        assert re.search(r'/app\.js\?v=\d+', index.text)
        assert re.search(r'/app\.css\?v=\d+', index.text)
        script_response = client.get("/app.js")
        assert script_response.headers["cache-control"].startswith("no-store")
        assert 'addEventListener("mousedown"' not in script_response.text
        assert 'addEventListener("touchstart"' not in script_response.text
        assert 'button.addEventListener("click"' in script_response.text
        assert "window.SpeechRecognition || window.webkitSpeechRecognition" in script_response.text
        assert '$("mic").hidden = true' not in script_response.text
        assert "if (!window.isSecureContext)" not in script_response.text
        from html.parser import HTMLParser
        class Attributes(HTMLParser):
            def handle_starttag(self, tag, attrs):
                attributes = dict(attrs)
                if attributes.get("id") == "mic":
                    self.mic = attributes
        parsed = Attributes()
        parsed.feed(index.text)
        assert "hidden" in parsed.mic
        assert re.search(r'class="voice-option"[^>]*hidden', index.text)
        assert 'id="voice-status"' in index.text
        assert "음성 인식 끝나면 자동 전송" in index.text
        assert 'recognition.continuous = true' in script_response.text
        assert 'finalText || latestText' in script_response.text
        assert "heardAudio" in script_response.text and "heardSpeech" in script_response.text
        assert "function renderMarkdown" in script_response.text
        assert "appendInlineMarkdown" in script_response.text
        assert "element.replaceChildren()" in script_response.text
        assert ".innerHTML" not in script_response.text
        assert f'CONTROL_UI_VERSION = "{CONTROL_UI_VERSION}"' in script_response.text
        assert client.get("/app.css").headers["cache-control"].startswith("no-store")
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


def test_manual_endpoint_validates_and_allowlists_tools():
    with _client() as client:
        token = client.post("/api/lease").json()["token"]
        headers = {
            "X-Operator-Token": token,
            "X-SO101-Control-Version": CONTROL_UI_VERSION,
        }
        assert client.post("/api/manual", json={}, headers=headers).status_code == 400
        assert client.post("/api/manual", json={"tool": 5}, headers=headers).status_code == 400
        assert client.post("/api/manual", json={"tool": "run_task1"}, headers=headers).status_code == 400
        assert client.post("/api/manual", json={"tool": "rotate_gripper", "arguments": "x"},
                           headers=headers).status_code == 400
        assert client.post(
            "/api/manual",
            json={"tool": "rotate_gripper", "arguments": {"delta_deg": 30}},
            headers={"X-Operator-Token": token},
        ).status_code == 428
        response = client.post("/api/manual", json={"tool": "rotate_gripper", "arguments": {"delta_deg": 30}},
                               headers=headers)
        assert response.status_code == 202

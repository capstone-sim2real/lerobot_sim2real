"""Camera bridging uses read-only allowlisted paths and releases upstream streams."""
import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.camera_proxy import camera_router


class Stream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


def client_for(handler):
    app = FastAPI()
    app.include_router(camera_router("http://camera.test:8090", "shoulder",
                                    transport=httpx.MockTransport(handler)))
    return TestClient(app)


@pytest.mark.parametrize("resource,path,mime", [
    ("video", "/video/shoulder.mjpg", "multipart/x-mixed-replace; boundary=frame"),
    ("events", "/events/shoulder", "text/event-stream"),
    ("config", "/overlay-config/shoulder.json", "application/json"),
    ("reference", "/gripper-reference.json", "application/json"),
])
def test_proxy_preserves_stream_and_closes_upstream(resource, path, mime):
    stream = Stream([b"first\n", b"second\n"])

    def upstream(request):
        assert str(request.url) == "http://camera.test:8090" + path
        assert request.method == "GET"
        return httpx.Response(200, headers={"content-type": mime}, stream=stream)

    with client_for(upstream) as client:
        response = client.get("/api/camera/" + resource)
    assert response.status_code == 200
    assert response.content == b"first\nsecond\n"
    assert response.headers["content-type"] == mime
    assert response.headers["x-accel-buffering"] == "no"
    assert stream.closed


def test_proxy_cannot_forward_writes_or_arbitrary_paths():
    calls = []

    def upstream(request):
        calls.append(request)
        return httpx.Response(200)

    with client_for(upstream) as client:
        assert client.post("/api/camera/reference").status_code == 405
        assert client.get("/api/camera/stop").status_code == 404
        assert client.get("/api/camera/http:%2F%2Fevil.test").status_code == 404
    assert calls == []


@pytest.mark.parametrize("code,expected", [(404, 404), (503, 503), (500, 502), (302, 502)])
def test_proxy_error_and_redirect_are_closed_without_following(code, expected):
    stream = Stream([b"upstream details"])
    with client_for(lambda request: httpx.Response(code, stream=stream,
                      headers={"location": "http://other.test"})) as client:
        response = client.get("/api/camera/events")
    assert response.status_code == expected
    assert "upstream details" not in response.text
    assert stream.closed


def test_unreachable_camera_is_a_readable_502():
    def upstream(request):
        raise httpx.ConnectError("unreachable", request=request)
    with client_for(upstream) as client:
        response = client.get("/api/camera/config")
    assert response.status_code == 502
    assert "카메라" in response.json()["error"]


def test_broken_stream_releases_upstream():
    class Broken(Stream):
        async def __aiter__(self):
            yield b"data: {}\n\n"
            raise httpx.ReadError("camera disconnected")
    stream = Broken([])
    with client_for(lambda request: httpx.Response(200, stream=stream)) as client:
        assert client.get("/api/camera/events").content == b"data: {}\n\n"
    assert stream.closed


def test_camera_selection_is_allowlisted_and_wrist_reference_is_not_shared():
    paths=[]
    def upstream(request):
        paths.append(request.url.path)
        return httpx.Response(200,stream=Stream([b'frame']))
    with client_for(upstream) as client:
        assert client.get('/api/camera/video?camera=wrist').status_code==200
        assert client.get('/api/camera/video?camera=shoulder').status_code==200
        assert client.get('/api/camera/video?camera=../../other').status_code==404
        assert client.get('/api/camera/reference?camera=wrist').status_code==404
    assert paths==['/video/wrist.mjpg','/video/shoulder.mjpg']

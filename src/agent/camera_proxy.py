"""Read-only camera bridge. Camera ownership remains with camera.server."""
import math
from urllib.parse import quote

from config import AgentCameraViewConfig


def camera_router(base_url: str, camera_name: str, *, settings=None, transport=None):
    import httpx
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    settings = settings or AgentCameraViewConfig()
    for key, value in vars(settings).items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"agent.camera_view.{key} must be finite and positive")
    router = APIRouter()
    @router.get("/api/camera/{resource}")
    async def camera_resource(resource: str, camera: str | None = None):
        selected = camera if camera is not None else camera_name
        if selected not in {camera_name, "shoulder", "wrist"}:
            return JSONResponse({"error": "지원하지 않는 카메라입니다."}, status_code=404)
        selected = quote(selected, safe="")
        paths = {
            "video": f"/video/{selected}.mjpg",
            "config": f"/overlay-config/{selected}.json",
            "events": f"/events/{selected}",
            "reference": "/gripper-reference.json",
        }
        if resource not in paths or (selected == "wrist" and resource == "reference"):
            return JSONResponse({"error": "지원하지 않는 카메라 경로입니다."}, status_code=404)
        # The configured host and exact path allowlist are never client supplied.
        client = httpx.AsyncClient(
            transport=transport, timeout=httpx.Timeout(settings.read_timeout_s, connect=settings.connect_timeout_s),
            follow_redirects=False, trust_env=False,
        )
        upstream = None
        try:
            upstream = await client.send(
                client.build_request("GET", base_url.rstrip("/") + paths[resource]),
                stream=True,
            )
            if upstream.status_code != 200:
                code = upstream.status_code if upstream.status_code in (404, 503) else 502
                await upstream.aclose()
                await client.aclose()
                return JSONResponse({"error": "카메라 데이터를 사용할 수 없습니다."}, status_code=code)
            headers = {
                "Content-Type": upstream.headers.get("content-type", "application/octet-stream"),
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            }
            if resource in ("config", "reference"):
                data = await upstream.aread()
                await upstream.aclose()
                await client.aclose()
                return Response(content=data, headers=headers)
        except (httpx.HTTPError, OSError):
            if upstream is not None:
                await upstream.aclose()
            await client.aclose()
            return JSONResponse({"error": "카메라 서버에 연결할 수 없습니다."}, status_code=502)
        except BaseException:
            if upstream is not None:
                await upstream.aclose()
            await client.aclose()
            raise

        async def body():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            except httpx.HTTPError:
                # End the stream so the browser can reconnect.
                return
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(body(), headers=headers)

    return router

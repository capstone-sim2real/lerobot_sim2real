"""Authenticated manual-control HTTP routes for the agent web UI."""
import math


def register_manual_api(app, service, *, control_ui_version):
    """Attach state-changing routes while keeping robot policy in AgentService."""
    from fastapi import Request
    from fastapi.responses import JSONResponse

    def token_of(request):
        return request.headers.get("x-operator-token") or request.query_params.get("token")

    def reply(result):
        code, body = result
        return JSONResponse(body, status_code=code)

    def require_current_ui(request):
        if request.headers.get("x-so101-control-version") == control_ui_version:
            return None
        return JSONResponse(
            {"error": "조작 화면이 이전 버전입니다. Ctrl+Shift+R로 새로고침해 주세요."},
            status_code=428,
        )

    @app.post("/api/lease")
    async def lease(request: Request):
        return {"token": service().acquire_lease(token_of(request))}

    @app.post("/api/lease/release")
    async def lease_release(request: Request):
        return {"released": service().gate.release_lease(token_of(request) or "")}

    @app.post("/api/lease/force-release")
    async def lease_force_release(request: Request):
        host = request.client.host if request.client else ""
        if host not in ("127.0.0.1", "::1", "localhost"):
            return JSONResponse({"error": "local only"}, status_code=403)
        service().gate.force_release()
        return {"released": True}

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        return reply(service().chat(token_of(request), str(body.get("text", ""))))

    @app.post("/api/jog")
    async def jog(request: Request):
        body = await request.json()
        try:
            vector = [float(body.get(k, 0.0)) for k in ("forward_mm", "left_mm", "up_mm")]
        except (TypeError, ValueError):
            return JSONResponse({"error": "numbers required"}, status_code=400)
        if not all(math.isfinite(value) for value in vector):
            return JSONResponse({"error": "numbers required"}, status_code=400)
        stale = require_current_ui(request)
        return stale or reply(service().jog(token_of(request), *vector))

    @app.post("/api/keyboard/start")
    async def keyboard_start(request: Request):
        stale = require_current_ui(request)
        return stale or reply(service().keyboard_start(token_of(request)))

    async def keyboard_update(request, *, release=False, stop=False):
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse({"error": "keyboard session required"}, status_code=400)
        if not release and not stop and not isinstance(body.get("vector"), list):
            return JSONResponse({"error": "keyboard direction required"}, status_code=400)
        if not release and not stop:
            stale = require_current_ui(request)
            if stale is not None:
                return stale
        return reply(service().keyboard_update(
            token_of(request), body.get("session_id"), body.get("seq", 0),
            body.get("vector", []), release=release, stop=stop,
        ))

    @app.post("/api/keyboard/update")
    async def update_keyboard(request: Request):
        return await keyboard_update(request)

    @app.post("/api/keyboard/release")
    async def release_keyboard(request: Request):
        return await keyboard_update(request, release=True)

    @app.post("/api/keyboard/stop")
    async def stop_keyboard(request: Request):
        return await keyboard_update(request, stop=True)

    @app.post("/api/manual")
    async def manual(request: Request):
        body = await request.json()
        name = body.get("tool")
        if not isinstance(name, str):
            return JSONResponse({"error": "'tool' is required"}, status_code=400)
        arguments = body.get("arguments") or {}
        if not isinstance(arguments, dict):
            return JSONResponse({"error": "'arguments' must be an object"}, status_code=400)
        stale = require_current_ui(request)
        return stale or reply(service().direct(token_of(request), name, arguments))

    @app.post("/api/stop")
    async def stop():
        return service().stop()

    @app.post("/api/home")
    async def home(request: Request):
        return reply(service().home(token_of(request)))

    @app.post("/api/reset")
    async def reset(request: Request):
        return reply(service().reset_chat(token_of(request)))

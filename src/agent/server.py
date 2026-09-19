"""so101-agent: FastAPI chat server for the LLM tool-calling arm agent.

    # preflight: slots, table regions, IK, jog window, provider, camera -- no robot
    so101-agent --dry-run

    # offline rehearsal: simulated arm + simulated camera, rule-based fake LLM
    so101-agent --sim --provider fake

    # real arm (so101-camera is started automatically if not already running)
    export ANTHROPIC_API_KEY=...
    so101-agent
    so101-agent --provider openai      # OPENAI_API_KEY
    so101-agent --provider gemini      # GEMINI_API_KEY

Then open http://<jetson>:8099/ . The browser's built-in Web Speech API is
used directly; Tailscale HTTPS is the most reliable remote origin, while HTTP
support depends on the browser's own permission policy.

Never run together with so101-run / so101-collect: all three own the robot
serial bus (enforced by the bus lock at agent.lock_path).
"""

# No ``from __future__ import annotations`` here: FastAPI resolves the
# ``Request`` annotations of the routes defined inside create_app at runtime.
import argparse
import asyncio
import collections
import json
import logging
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from camera.autostart import ensure_camera_server
from config import AppConfig, load_config

logger = logging.getLogger("agent")

WEB_DIR = Path(__file__).resolve().parent / "web"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_UI_VERSION = "cell-grid-v1"
API_KEY_ENV = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "fake": [],
}

def load_env_file(explicit: str | None = None) -> Path | None:
    """Read ``KEY=value`` lines from a .env file into os.environ.

    No dependency: this is the entire feature. Checked, in order: an
    explicit ``--env-file``, ``./.env`` (wherever so101-agent was launched
    from), then the repo root's ``.env`` (so it works launched from anywhere).
    Existing environment variables always win -- a real ``export`` overrides
    the file, so the file is a convenience default rather than a hidden
    override of something the operator set on purpose.
    """
    candidates = [Path(explicit)] if explicit else [Path.cwd() / ".env", REPO_ROOT / ".env"]
    for path in candidates:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if not key:
                continue
            if key not in os.environ:
                os.environ[key] = value
        return path
    return None


# ── event fan-out to SSE subscribers ─────────────────────────────────


_CLOSE = object()  # sentinel: wakes every /api/events generator so it can return


class EventHub:
    def __init__(self, backlog: int = 400):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: set[asyncio.Queue] = set()
        self._backlog: collections.deque = collections.deque(maxlen=backlog)
        self._lock = threading.Lock()
        self._seq = 0

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def close(self) -> None:
        """Unblock every open SSE stream so uvicorn's shutdown does not hang.

        A StreamingResponse generator counts as an open connection until it
        returns; without this, ``queue.get()`` sits waiting for the next
        event/heartbeat for up to ``sse_heartbeat_s`` and Ctrl-C needs a
        second, forced press. Called from the lifespan shutdown, already on
        the loop, so a direct put is fine (no cross-thread call needed).
        """
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(_CLOSE)
            except asyncio.QueueFull:
                pass

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, **event}
            if event.get("type") == "reset":
                self._backlog.clear()
            if event.get("type") != "text_delta":
                self._backlog.append(event)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._fanout, event)

    def _fanout(self, event: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def subscribe(self) -> tuple[asyncio.Queue, list[dict[str, Any]]]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        with self._lock:
            backlog = list(self._backlog)
        self._subscribers.add(queue)
        return queue, backlog

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)


# ── wiring ───────────────────────────────────────────────────────────


def make_skills_factory(cfg: AppConfig, cancel, *, sim: bool):
    """Runs on the robot thread: open the session (and IK) there."""

    def factory():
        from control.ik import TopDownIK
        from control.poses import PoseRegistry
        from session.arm_session import ArmSession
        from session.skills import Skills

        ik = TopDownIK(cfg.ik, project_root=".")
        kwargs: dict[str, Any] = {"cancel": cancel, "ik": ik}
        if sim:
            from perception.homography import PlaneCalibration
            from session.factories import calibration_grasp_z_mm

            from .sim import SimRobotIO, SimWorld

            calib = PlaneCalibration.load(cfg.perception.calibration_path)
            world = SimWorld.default(cfg, calib)
            home = PoseRegistry.load(cfg.motion.poses_path).get(cfg.motion.home_pose)
            robot = SimRobotIO(world, ik.forward_position_mm,
                               grasp_z_mm=calibration_grasp_z_mm(calib), initial_joints=home)
            holder: dict[str, Any] = {}

            def scene():
                session = holder["session"]
                return world.scene(session.calib, session.slot_centres, cfg.agent.slot_snap_radius_mm)

            session = ArmSession.open(cfg, robot=robot, acquire_bus_lock=False, prebuild_ik=True,
                                      scene_fn=scene, **kwargs)
            holder["session"] = session
        else:
            session = ArmSession.open(cfg, prebuild_ik=True, **kwargs)
        check_jog_window(session.cfg, session.grasp_z_mm)
        return Skills(session)

    return factory


def check_jog_window(cfg: AppConfig, grasp_z_mm: float) -> None:
    floor = grasp_z_mm + cfg.task2.block_height_mm
    if cfg.agent.relative.jog_min_z_mm <= floor:
        raise ValueError(
            f"agent.relative.jog_min_z_mm ({cfg.agent.relative.jog_min_z_mm:g}) must be above the "
            f"top of a block standing on the table ({floor:.1f}mm)"
        )


def create_app(cfg: AppConfig, service_builder, hub: EventHub):
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

    from agent.camera_proxy import camera_router

    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_app):
        hub.attach(asyncio.get_running_loop())
        service = service_builder(hub.publish)
        await asyncio.to_thread(service.start)
        state["service"] = service

        async def watchdog():
            while True:
                await asyncio.sleep(1.0)
                service.gate.tick()

        task = asyncio.create_task(watchdog())
        try:
            yield
        finally:
            task.cancel()
            hub.close()
            await asyncio.to_thread(service.shutdown)

    app = FastAPI(title="SO-101 Agent", lifespan=lifespan)

    app.include_router(camera_router(cfg.agent.camera_base_url, cfg.agent.camera_name, settings=cfg.agent.camera_view))

    def svc():
        return state["service"]

    def token_of(request: Request) -> str | None:
        return request.headers.get("x-operator-token") or request.query_params.get("token")

    def reply(result: tuple[int, dict]) -> JSONResponse:
        code, body = result
        return JSONResponse(body, status_code=code)

    def current_control_ui(request: Request) -> bool:
        return request.headers.get("x-so101-control-version") == CONTROL_UI_VERSION

    def stale_control_ui() -> JSONResponse:
        return JSONResponse(
            {"error": "조작 화면이 이전 버전입니다. Ctrl+Shift+R로 새로고침해 주세요."},
            status_code=428,
        )

    @app.get("/")
    async def index():
        # A cached control script is unsafe: newly added buttons can look
        # live while their event handlers do not exist, so no request ever
        # reaches the robot API. Give each asset a content-changing URL and
        # forbid caching so an asset-only edit appears on the next reload.
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        html = html.replace("__APP_JS_VERSION__", str((WEB_DIR / "app.js").stat().st_mtime_ns))
        html = html.replace("__APP_CSS_VERSION__", str((WEB_DIR / "app.css").stat().st_mtime_ns))
        html = html.replace("__OVERLAY_JS_VERSION__", str((WEB_DIR / "camera-overlay.js").stat().st_mtime_ns))
        html = html.replace("__RENDERER_VERSION__", str((WEB_DIR.parent.parent / "camera" / "overlay_renderer.js").stat().st_mtime_ns))
        return HTMLResponse(html, headers={"Cache-Control": "no-store, max-age=0"})

    @app.get("/app.js")
    async def app_js():
        return FileResponse(
            WEB_DIR / "app.js",
            media_type="application/javascript",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/app.css")
    async def app_css():
        return FileResponse(
            WEB_DIR / "app.css",
            media_type="text/css",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/camera-overlay.js")
    async def camera_overlay_js():
        return FileResponse(WEB_DIR / "camera-overlay.js", media_type="application/javascript",
                            headers={"Cache-Control": "no-store"})

    @app.get("/overlay-renderer.js")
    async def overlay_renderer_js():
        return FileResponse(WEB_DIR.parent.parent / "camera" / "overlay_renderer.js",
                            media_type="application/javascript", headers={"Cache-Control": "no-store"})

    @app.get("/api/config")
    async def ui_config():
        agent = cfg.agent
        return {
            "camera_base_url": agent.camera_base_url,
            "camera_view": asdict(agent.camera_view),
            "mjpeg_path": f"/video/{agent.camera_name}.mjpg",
            "max_jog_mm": agent.relative.max_jog_mm,
            "default_step_mm": agent.relative.default_step_mm,
            "small_step_mm": agent.relative.small_step_mm,
            "places": svc().places,
            "provider": svc().provider.name,
            "model": svc().provider.model,
        }

    @app.get("/api/telemetry")
    async def telemetry():
        return svc().telemetry()

    @app.get("/api/health")
    async def health():
        body = svc().health()
        body["camera_ok"] = await asyncio.to_thread(camera_ok, cfg)
        return body

    @app.post("/api/lease")
    async def lease(request: Request):
        token = svc().acquire_lease(token_of(request))
        return {"token": token}

    @app.post("/api/lease/release")
    async def lease_release(request: Request):
        return {"released": svc().gate.release_lease(token_of(request) or "")}

    @app.post("/api/lease/force-release")
    async def lease_force_release(request: Request):
        host = request.client.host if request.client else ""
        if host not in ("127.0.0.1", "::1", "localhost"):
            return JSONResponse({"error": "local only"}, status_code=403)
        svc().gate.force_release()
        return {"released": True}

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        return reply(svc().chat(token_of(request), str(body.get("text", ""))))

    @app.post("/api/jog")
    async def jog(request: Request):
        body = await request.json()
        try:
            vector = [float(body.get(k, 0.0)) for k in ("forward_mm", "left_mm", "up_mm")]
        except (TypeError, ValueError):
            return JSONResponse({"error": "numbers required"}, status_code=400)
        if not all(math.isfinite(v) for v in vector):
            return JSONResponse({"error": "numbers required"}, status_code=400)
        if not current_control_ui(request):
            return stale_control_ui()
        return reply(svc().jog(token_of(request), *vector))

    @app.post("/api/manual")
    async def manual(request: Request):
        body = await request.json()
        name = body.get("tool")
        if not isinstance(name, str):
            return JSONResponse({"error": "'tool' is required"}, status_code=400)
        arguments = body.get("arguments") or {}
        if not isinstance(arguments, dict):
            return JSONResponse({"error": "'arguments' must be an object"}, status_code=400)
        if not current_control_ui(request):
            return stale_control_ui()
        return reply(svc().direct(token_of(request), name, arguments))

    @app.post("/api/stop")
    async def stop():
        return svc().stop()

    @app.post("/api/home")
    async def home(request: Request):
        return reply(svc().home(token_of(request)))

    @app.post("/api/reset")
    async def reset(request: Request):
        return reply(svc().reset_chat(token_of(request)))

    @app.get("/api/events")
    async def events(request: Request):
        token = token_of(request)
        service = svc()
        queue, backlog = hub.subscribe()
        service.gate.operator_connected(token)

        async def stream():
            # uvicorn's own shutdown waits for open connections to close
            # BEFORE it fires the lifespan-shutdown event that runs
            # hub.close() -- so that alone deadlocks a live server (the
            # reason Ctrl-C needed a second, forced press). Polling the
            # Server object's own should_exit here is independent of that
            # event and closes this connection within one poll interval of
            # the real SIGINT. Absent under TestClient (no live Server), so
            # hub.close() from the lifespan handler covers that case instead.
            server = getattr(request.app.state, "uvicorn_server", None)
            poll_s = min(1.0, cfg.agent.sse_heartbeat_s)
            last_heartbeat = time.monotonic()
            try:
                yield f"data: {json.dumps({'type': 'control_state', **service.gate.snapshot()}, ensure_ascii=False)}\n\n"
                for event in backlog:
                    yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                while True:
                    if (server is not None and server.should_exit) or await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=poll_s)
                    except asyncio.TimeoutError:
                        now = time.monotonic()
                        if now - last_heartbeat >= cfg.agent.sse_heartbeat_s:
                            yield ": keepalive\n\n"
                            last_heartbeat = now
                        continue
                    if event is _CLOSE:
                        break
                    yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            finally:
                hub.unsubscribe(queue)
                service.gate.operator_disconnected(token)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def camera_ok(cfg: AppConfig) -> bool:
    from urllib.request import urlopen

    try:
        with urlopen(f"{cfg.agent.camera_base_url}/health", timeout=1.0) as response:  # nosec B310
            return response.status == 200
    except Exception:  # noqa: BLE001
        return False


# ── dry run ──────────────────────────────────────────────────────────


def _grid_edge_cells(cells: list) -> list:
    """The outermost cell at each end of every row and column.

    Reach runs out at the band's rim, so these are where the IK gate fails
    first -- checking them keeps the dry-run honest without solving IK for
    every one of the ~150 cells.
    """
    extremes: dict[tuple[str, int], tuple] = {}
    for cell in cells:
        for axis, key, value in (("row", cell.y, cell.x), ("col", cell.x, cell.y)):
            for name, pick in (("min", min), ("max", max)):
                slot = (f"{axis}-{name}", key)
                current = extremes.get(slot)
                if current is None or pick(value, current[1]) == value:
                    extremes[slot] = (cell, value)
    return list({(c.x, c.y): c for c, _ in extremes.values()}.values())


def dry_run(cfg: AppConfig, provider_name: str, fallback_name: str | None = None) -> int:
    """Everything the agent will rely on, checked without the robot bus."""
    import copy

    import numpy as np

    from camera.client import fetch_snapshot
    from control.ik import TopDownIK
    from control.poses import PoseRegistry
    from control.task1_transport import Task1TransportPlanner, over_ik_gate
    from perception.homography import PlaneCalibration
    from perception.scene import detect_scene
    from session.factories import calibration_grasp_z_mm
    from session.relative import table_region_xy
    from session.report import print_slot_table

    from .prompt import build_system_prompt
    from .tools import build_tools

    cfg = copy.deepcopy(cfg)
    agent = cfg.agent
    problems: list[str] = []
    print("so101-agent dry-run (no robot connection, no motion)\n")
    calib = PlaneCalibration.load(cfg.perception.calibration_path)
    print(f"calibration : {cfg.perception.calibration_path}")
    if not calib.zone_polygon_mm:
        print("  zone_polygon_mm missing -- run so101-zone-calibrate --write")
        return 1
    grasp_z = calibration_grasp_z_mm(calib)
    print(f"  grasp_z_mm={grasp_z:.1f}  rms_mm={calib.meta.get('rms_mm')}  loo_max_mm={calib.meta.get('loo_max_mm')}")
    print(f"  zone polygon mm: {[tuple(round(v, 1) for v in p) for p in calib.zone_polygon_mm]}")

    ik = TopDownIK(cfg.ik, project_root=".")
    planner = Task1TransportPlanner(calib, cfg, ik)
    labels = [f"{e}/{k}" for e, k in zip(agent.zone_slots.labels, agent.zone_slots.korean_labels)]
    print(f"\nzone cells ({agent.zone_slots.frame}: top = far row, left = +y = image left)")
    print_slot_table(planner.slots, labels=labels)
    px = calib.board_to_pixel(np.asarray([s.xy_mm for s in planner.slots]))
    for slot, (u, v) in zip(planner.slots, px):
        print(f"    {labels[slot.index]:>20s} -> image px ({u:6.1f}, {v:6.1f})")

    print("\ntable regions (column/row -> xy -> checks)")
    base = calib.base_xy_mm or (0.0, 0.0)
    from perception.detector import point_in_workspace
    from perception.zone import point_in_zone
    from control.task1_transport import solve_place_point

    for column in agent.table_regions.columns_deg:
        for row in agent.table_regions.rows_fraction:
            xy = table_region_xy(column, row, cfg.perception, agent.table_regions, base)
            flags = []
            if not point_in_workspace(xy, cfg.perception, base):
                flags.append("OUT-OF-WORKSPACE")
            try:
                solve_place_point(ik, cfg, xy, grasp_z + cfg.task1.release_clearance_mm,
                                  base_xy_mm=base, label="region")
            except ValueError as exc:
                flags.append(f"IK-GATE ({exc})")
            if flags:
                problems.append(f"table region {column}/{row}: {', '.join(flags)}")
            if point_in_zone(xy, calib, agent.table_zone_margin_mm):
                # directly in front of the zone: never chosen, not a fault
                flags.append("overlaps zone (always skipped)")
            status = "ok" if not flags else ", ".join(flags)
            print(f"  {column:9s} {row:6s} x={xy[0]:7.1f} y={xy[1]:7.1f}  {status}")

    grid_cfg = agent.board_grid
    if grid_cfg.enabled:
        from session.grid import bounds as grid_bounds
        from session.grid import build_grid, cells_in_view, cells_in_workspace, default_anchor_mm

        anchor = default_anchor_mm(cfg.perception, agent.table_regions, base)
        grid = build_grid(calib.board_grid, grid_cfg, anchor)
        cells = cells_in_workspace(grid, cfg.perception, agent.table_regions, base, grid_cfg)
        cells = cells_in_view(cells, lambda pts: calib.board_to_pixel(np.asarray(pts)), calib.image_size)
        span = grid_bounds((c.x, c.y) for c in cells)
        source = "measured board" if calib.board_grid else "axis-aligned fallback (no board_grid)"
        print(f"\nboard cells ({source}, cell {grid.cell_mm:.1f}mm, image x+ = right, y+ = away)")
        print(f"  {len(cells)} cells, x {span['x'][0]}..{span['x'][1]}, y {span['y'][0]}..{span['y'][1]}, "
              f"(0,0) at x={grid.origin_mm[0]:.1f} y={grid.origin_mm[1]:.1f} mm")
        if not calib.board_grid:
            problems.append(
                "no measured board grid: cells are axis-aligned guesses "
                "(run tools.calibrate_board_grid --write)"
            )
        if not cells:
            problems.append("board grid has no cell inside the workspace")
        # IK on every cell would take minutes; the band's edges are where it
        # fails first, so checking those bounds the risk without the cost.
        edge = _grid_edge_cells(cells)
        failed = []
        for cell in edge:
            try:
                solve_place_point(ik, cfg, cell.xy_mm, grasp_z + cfg.task1.release_clearance_mm,
                                  base_xy_mm=base, label="cell")
            except ValueError as exc:
                failed.append((cell, str(exc)))
        print(f"  edge cells checked: {len(edge)}, IK-GATE failures: {len(failed)}")
        for cell, reason in failed[:5]:
            print(f"    ({cell.x:+3d}, {cell.y:+3d}) x={cell.xy_mm[0]:7.1f} y={cell.xy_mm[1]:7.1f}  {reason}")
        if failed:
            problems.append(f"{len(failed)}/{len(edge)} board edge cells fail the IK gate")

    home = PoseRegistry.load(cfg.motion.poses_path).get(cfg.motion.home_pose)
    hx, hy, hz = ik.forward_position_mm(home)
    rel = agent.relative
    print(f"\nhome tool-frame position: x={hx:.1f} y={hy:.1f} z={hz:.1f}mm")
    print(f"jog window z: {rel.jog_min_z_mm:g}..{rel.jog_max_z_mm:g}mm, max {rel.max_jog_mm:g}mm per call, frame={rel.frame}")
    try:
        check_jog_window(cfg, grasp_z)
    except ValueError as exc:
        problems.append(str(exc))
        print(f"  PROBLEM: {exc}")
    probe = ik.solve(hx, hy, rel.jog_min_z_mm)
    print(f"  jog entry from home at z={rel.jog_min_z_mm:g}: IK error {probe.position_error_mm:.1f}mm"
          f" ({'ok' if not over_ik_gate(probe, cfg) else 'OVER GATE'})")

    print(f"\nprovider: {provider_name}  model: {agent.models.get(provider_name)}")
    for env in API_KEY_ENV.get(provider_name, []):
        print(f"  {env}: {'set' if os.environ.get(env) else 'NOT SET'}")
    if fallback_name is not None:
        print(f"fallback: {fallback_name}  model: {agent.models.get(fallback_name)}")
        for env in API_KEY_ENV.get(fallback_name, []):
            print(f"  {env}: {'set' if os.environ.get(env) else 'NOT SET'}")
    tools = build_tools(cfg)
    print(f"tools ({len(tools)}): {', '.join(t.spec.name for t in tools)}")
    try:
        prompt = build_system_prompt(cfg)
        print(f"system prompt: {len(prompt)} chars from {agent.system_prompt_path}")
    except OSError as exc:
        problems.append(f"system prompt: {exc}")

    print(f"\ncamera: {cfg.perception.snapshot_url}")
    camera_proc = None
    try:
        if cfg.camera.auto_start:
            camera_proc = ensure_camera_server(agent.camera_base_url, extra_args=cfg.camera.extra_args)
        frame = fetch_snapshot(cfg.perception.snapshot_url)
        scene = detect_scene(frame, calib, cfg.perception, [s.xy_mm for s in planner.slots],
                             zone_max_per_color=agent.zone_scan_max_per_color,
                             snap_radius_mm=agent.slot_snap_radius_mm)
        for block in scene.outside.values():
            print(f"  outside {block.color:6s} x={block.center_mm[0]:7.1f} y={block.center_mm[1]:7.1f}")
        for block in scene.inside.values():
            slot = agent.zone_slots.labels[block.slot_index] if block.slot_index is not None else "-"
            print(f"  inside  {block.color:6s} x={block.center_mm[0]:7.1f} y={block.center_mm[1]:7.1f} slot={slot}")
        if not scene.all():
            print("  no blocks detected")
    except Exception as exc:  # noqa: BLE001
        print(f"  unavailable ({exc}) -- start so101-camera before the demo")
    finally:
        if camera_proc is not None:
            camera_proc.stop()

    print("\n" + ("all checks passed" if not problems else "problems:\n  - " + "\n  - ".join(problems)))
    return 0 if not problems else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="key.path=value")
    parser.add_argument("--provider", choices=["anthropic", "openai", "gemini", "fake"])
    parser.add_argument("--model", help="override agent.models[<provider>]")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--sim", action="store_true", help="simulated arm and camera (no hardware)")
    parser.add_argument("--dry-run", action="store_true", help="check geometry/IK/provider/camera; no robot")
    parser.add_argument("--env-file", help="path to a .env file (default: ./.env, then the repo root's)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    env_file = load_env_file(args.env_file)
    if env_file is not None:
        logger.info("loaded API key(s) from %s", env_file)
    cfg = load_config(args.config, overrides=args.overrides)
    provider_name = args.provider or cfg.agent.provider
    fallback_name = None if args.provider else cfg.agent.fallback_provider
    if args.model:
        cfg.agent.models[provider_name] = args.model

    if args.dry_run:
        try:
            return dry_run(cfg, provider_name, fallback_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Dry-run aborted: %s", exc)
            return 1

    try:
        import uvicorn
    except ImportError:
        logger.error('so101-agent needs the agent extra: uv pip install -e ".[agent]"')
        return 1

    from session.cancel import CancelToken

    from .provider import build_provider
    from .service import AgentService

    for required_provider in (provider_name, fallback_name):
        if required_provider is None:
            continue
        present = [env for env in API_KEY_ENV.get(required_provider, []) if os.environ.get(env)]
        if API_KEY_ENV.get(required_provider) and not present:
            logger.error(
                "%s is not set for provider %s; add it to .env or use --provider fake",
                " / ".join(API_KEY_ENV[required_provider]),
                required_provider,
            )
            return 1
    try:
        provider = build_provider(cfg.agent, provider=provider_name)
        if fallback_name is not None:
            from .provider.fallback import FallbackProvider

            fallback = build_provider(cfg.agent, provider=fallback_name)
            provider = FallbackProvider(provider, fallback)
    except ImportError as exc:
        logger.error("provider %s needs its SDK (%s); see docs/guide/SO101_LLM_에이전트.md", provider_name, exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - e.g. the SDK rejecting its credentials
        logger.error("could not create the %s client: %s", provider_name, exc)
        return 1
    camera_proc = None
    if not args.sim and cfg.camera.auto_start:
        try:
            camera_proc = ensure_camera_server(cfg.agent.camera_base_url, extra_args=cfg.camera.extra_args)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1

    hub = EventHub()

    def service_builder(publish):
        cancel = CancelToken()
        return AgentService(
            cfg,
            provider=provider,
            skills_factory=make_skills_factory(cfg, cancel, sim=args.sim),
            cancel=cancel,
            publish=publish,
        )

    app = create_app(cfg, service_builder, hub)
    host = args.host or cfg.agent.host
    port = args.port or cfg.agent.port
    fallback_label = (
        f" fallback={fallback_name}/{cfg.agent.models[fallback_name]}" if fallback_name else ""
    )
    logger.info("so101-agent on http://%s:%d (provider=%s model=%s%s sim=%s)",
                host, port, provider.name, provider.model, fallback_label, args.sim)
    # Built directly (not uvicorn.run) so /api/events can poll
    # server.should_exit -- see the long comment in create_app's SSE stream()
    # for why that, not the lifespan-shutdown event, is what actually breaks
    # the "Waiting for connections to close" deadlock on Ctrl-C.
    config = uvicorn.Config(app, host=host, port=port, log_level="info", timeout_graceful_shutdown=5)
    server = uvicorn.Server(config)
    app.state.uvicorn_server = server
    try:
        server.run()
    except KeyboardInterrupt:
        # uvicorn.run() swallows this same way: after a graceful shutdown,
        # Server.capture_signals() re-raises the original SIGINT once its
        # handler is restored, so this is expected on every clean Ctrl-C,
        # not a sign shutdown failed.
        pass
    finally:
        if camera_proc is not None:
            camera_proc.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

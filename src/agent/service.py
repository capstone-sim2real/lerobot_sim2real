"""Everything the web server does, without the web framework.

Threads:

- the web event loop (FastAPI) only calls these methods and never touches
  the robot; ``stop()`` returns immediately;
- ``RobotWorker``'s thread owns the ArmSession and runs every skill;
- one short-lived thread per chat turn / jog / home request drives the LLM
  loop and blocks on the robot worker.

Events for the browser go through ``publish`` (thread-safe).
"""

from __future__ import annotations

import concurrent.futures
import itertools
import logging
import threading
import time
from typing import Any, Callable

from config import AppConfig

from .control import ControlGate, ControlState
from .prompt import build_system_prompt
from .provider.types import Provider, ToolCall
from .runner import AgentRunner, TurnOutcome
from .tools import ToolRegistry
from .worker import RobotWorker

logger = logging.getLogger(__name__)

Publish = Callable[[dict[str, Any]], None]


_SCENE_TERMS = (
    "블록", "블럭", "적재", "구역", "안에", "밖", "꺼내", "옮겨", "옮기", "모아",
    "쌓아", "쌓기", "탑", "미션", "현재상황", "현재 상황", "카메라",
    "빨강", "빨간", "노랑", "노란", "초록", "녹색", "파랑", "파란", "나무", "우드",
    "block", "zone", "stack", "mission", "task", "camera", "red", "yellow", "green",
    "blue", "wood",
)


def needs_fresh_scene(text: str) -> bool:
    """Whether a natural-language turn depends on current block positions.

    Deliberately excludes generic direction/gripper words: automatically
    homing before "right 20 mm" would destroy the pose that manual jogging is
    relative to.  Composite block skills already expect a home-view camera
    observation, so a preflight observation is safe for these scene terms.
    """
    folded = " ".join(str(text).casefold().split())
    return any(term in folded for term in _SCENE_TERMS)


class AgentService:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        provider: Provider,
        skills_factory: Callable[[], Any],
        cancel,
        publish: Publish = lambda _event: None,
        system_prompt: str | None = None,
        transcript_dir: str | None = None,
    ):
        self.cfg = cfg
        self.provider = provider
        self.cancel = cancel
        self._publish = publish
        self._worker = RobotWorker(skills_factory)
        self._ids = itertools.count(1)
        self._threads: list[threading.Thread] = []
        self.gate = ControlGate(
            on_stop=cancel.set,
            lease_grace_s=cfg.agent.lease_grace_s,
            lease_idle_timeout_s=cfg.agent.lease_idle_timeout_s,
            on_change=lambda snap: self._publish({"type": "control_state", **snap}),
        )
        self.registry = ToolRegistry(cfg, self._execute)
        self.runner = AgentRunner(
            provider,
            self.registry,
            system_prompt if system_prompt is not None else build_system_prompt(cfg),
            cfg.agent,
            emit=self._publish,
            should_stop=cancel.is_set,
            transcript_dir=transcript_dir if transcript_dir is not None else cfg.agent.transcript_dir,
        )
        self.places: dict[str, Any] = {}
        self.started = False
        self._keyboard_jog = None
        self._telemetry_future = None
        self._telemetry_cache = {}

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self, timeout_s: float | None = None) -> None:
        self._worker.start(timeout_s)
        self.places = self._worker.run(lambda skills: places_payload(skills), timeout_s=60.0)
        self.started = True

    def shutdown(self) -> None:
        if self.gate.state in (ControlState.BUSY, ControlState.HOMING):
            self.cancel.set()
        for thread in list(self._threads):
            thread.join(timeout=30.0)
        self._worker.stop()

    def _execute(self, job):
        future = self._worker.submit(job)
        try:
            return future.result(timeout=self.cfg.agent.tool_timeout_s)
        except concurrent.futures.TimeoutError:
            # the skill is still running on the robot thread: stop it there
            self.cancel.set()
            raise

    def _spawn(self, target, name: str) -> None:
        self._threads = [t for t in self._threads if t.is_alive()]
        thread = threading.Thread(target=target, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()

    def wait_idle(self, timeout_s: float = 30.0) -> None:
        """Test helper: join every request thread, including ones a finishing
        thread spawns next (e.g. STOP auto-triggering the home job) -- loops
        until none are left rather than joining one fixed snapshot."""
        deadline = time.monotonic() + timeout_s
        while True:
            threads = [t for t in self._threads if t.is_alive()]
            if not threads or time.monotonic() > deadline:
                return
            for thread in threads:
                thread.join(max(0.0, deadline - time.monotonic()))

    # ── lease ────────────────────────────────────────────────────────

    def acquire_lease(self, token: str | None) -> str | None:
        session_exists = self.gate.lease_since() is not None
        new = self.gate.acquire_lease(token)
        if new is not None and not session_exists:
            # Reset only when a shared session is created. A second browser
            # joins the existing conversation instead of erasing it.
            self.runner.reset()
            self._publish({"type": "reset"})
        return new

    # ── commands ─────────────────────────────────────────────────────

    def chat(self, token: str | None, text: str) -> tuple[int, dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return 400, {"error": "empty message"}
        if not self.gate.check(token):
            return 403, {"error": "not the operator"}
        if not self.gate.try_begin(token, "chat"):
            return 409, {"error": "busy", **self.gate.snapshot()}
        self.cancel.clear()
        self._publish({"type": "user", "text": text})

        def turn() -> None:
            try:
                current_cv = None
                if needs_fresh_scene(text):
                    call = ToolCall(
                        f"observe_scene_auto_{next(self._ids)}", "observe_scene", {"include_zone": True}
                    )
                    self._publish({
                        "type": "tool_call", "id": call.id, "name": call.name,
                        "arguments": call.arguments, "automatic": True,
                    })
                    result = self.registry.execute(call)
                    self._publish({
                        "type": "tool_result", "id": call.id, "name": call.name,
                        "result": result.content, "automatic": True,
                    })
                    current_cv = result.content
                    from session.results import ROBOT_FAULT_REASONS

                    if result.content.get("reason") in ROBOT_FAULT_REASONS:
                        self._publish({
                            "type": "assistant_text",
                            "text": "최신 장면을 확인하는 중 로봇 동작이 중단되었습니다.",
                        })
                        self._publish({"type": "turn_end", "robot_fault": True})
                        outcome = TurnOutcome(robot_fault=True, error=result.content.get("detail"))
                    else:
                        outcome = self.runner.run_turn(text, current_cv=current_cv)
                else:
                    outcome = self.runner.run_turn(text)
            except Exception as exc:  # noqa: BLE001 - a crashed turn is a fault
                logger.exception("agent turn crashed")
                self._publish({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                outcome = TurnOutcome(robot_fault=True, error=str(exc))
            self._after_command(outcome.robot_fault)

        self._spawn(turn, "so101-agent-turn")
        return 202, {"accepted": True}

    # Tools the manual control panel may call directly, bypassing the LLM.
    # Everything here is also an ordinary LLM tool (agent.tools.build_tools);
    # this only decides what a physical button on the page is allowed to fire.
    MANUAL_TOOLS = frozenset({
        "move_arm", "move_to_cell", "move_to_pixel", "place_at_pixel", "rotate_gripper", "pick_here", "place_here",
        "open_gripper", "return_to_home",
    })

    def direct(self, token: str | None, name: str, arguments: dict[str, Any]) -> tuple[int, dict]:
        """Run one tool immediately, as if a manual control-panel button fired it.

        Goes through the exact same ToolRegistry (and therefore the same
        argument validation, cancellation, and result envelope) an LLM tool
        call does -- a manual button is just a one-tool-call "turn" with no
        model in the loop.
        """
        if name not in self.MANUAL_TOOLS:
            return 400, {"error": f"'{name}' is not a manual control action"}
        if not self.gate.check(token):
            return 403, {"error": "not the operator"}
        if not self.gate.try_begin(token, name):
            return 409, {"error": "busy", **self.gate.snapshot()}
        self.cancel.clear()
        call = ToolCall(f"{name}_{next(self._ids)}", name, dict(arguments))
        logger.info("direct command accepted id=%s tool=%s arguments=%s", call.id, call.name, call.arguments)

        def run() -> None:
            self._publish({"type": "tool_call", "id": call.id, "name": call.name, "arguments": call.arguments, "direct": True})
            result = self.registry.execute(call)
            logger.info(
                "direct command finished id=%s tool=%s ok=%s reason=%s data=%s",
                call.id,
                call.name,
                result.content.get("ok"),
                result.content.get("reason"),
                result.content.get("data", {}),
            )
            self._publish({"type": "tool_result", "id": call.id, "name": call.name, "result": result.content, "direct": True})
            from session.results import ROBOT_FAULT_REASONS

            self._after_command(result.content.get("reason") in ROBOT_FAULT_REASONS)

        self._spawn(run, f"so101-agent-{name}")
        return 202, {"accepted": True}

    def jog(self, token: str | None, forward_mm: float, left_mm: float, up_mm: float) -> tuple[int, dict]:
        return self.direct(token, "move_arm",
                           {"forward_mm": float(forward_mm), "left_mm": float(left_mm), "up_mm": float(up_mm)})

    def keyboard_start(self, token):
        from .keyboard_jog import KeyboardJog
        if "move_arm" not in self.MANUAL_TOOLS:
            return 400, {"error": "keyboard jog unavailable"}
        if not self.gate.check(token):
            return 403, {"error": "not the operator"}
        try:
            import ruckig  # Optional agent dependency, checked before reserving the arm.
        except ImportError:
            return 503, {"error": "연속 가감속 기능에 필요한 ruckig 패키지가 설치되지 않았습니다."}
        if not self.gate.try_begin(token, "keyboard_jog"):
            return 409, {"error": "busy", **self.gate.snapshot()}
        stream = KeyboardJog(self.cfg.agent.relative)
        self._keyboard_jog = stream
        self.cancel.clear()

        def run():
            fault = False
            try:
                result = self._execute(lambda skills: stream.run(skills, self.cancel))
                if result is not None and not result.ok:
                    from session.results import ROBOT_FAULT_REASONS
                    fault = result.reason in ROBOT_FAULT_REASONS
                    self._publish({"type": "keyboard_jog_end", "message": result.detail})
            except Exception as exc:
                fault = True
                logger.exception("keyboard jog ended with a fault")
                self._publish({"type": "keyboard_jog_end", "message": str(exc)})
            finally:
                stream.close()
                self._keyboard_jog = None
                self._after_command(fault)
        self._spawn(run, "so101-keyboard-jog")
        return 202, {"session_id": stream.id}

    def keyboard_update(self, token, session_id, seq, vector, *, stop=False, release=False):
        import math
        if not self.gate.check(token):
            return 403, {"error": "not the operator"}
        stream = self._keyboard_jog
        if stream is None or stream.id != session_id:
            return 409, {"error": "keyboard session ended"}
        if release:
            stream.release()
            return 200, {"releasing": True}
        if stop:
            stream.close()
            return 200, {"stopped": True}
        if (type(seq) is not int or seq < 0 or len(vector) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1 for v in vector)):
            return 400, {"error": "invalid keyboard direction"}
        if not stream.update(seq, vector):
            return 409, {"error": "expired or outdated keyboard input"}
        return 200, {"accepted": True}

    def _after_command(self, robot_fault: bool) -> None:
        fault = robot_fault or self.cancel.is_set()
        self.gate.finish(robot_fault=fault, message="동작이 중단되었습니다. home 복귀가 필요합니다." if fault else None)
        if self.gate.state is ControlState.STOPPED and self.cfg.agent.stop_auto_home:
            if self.gate.begin_auto_home():
                self._spawn(self._home_job, "so101-agent-home")

    def stop(self) -> dict[str, Any]:
        stopped = self.gate.request_stop()
        self._publish({"type": "stop_pressed", "effective": stopped})
        return {"stopped": stopped, **self.gate.snapshot()}

    def home(self, token: str | None) -> tuple[int, dict[str, Any]]:
        if not self.gate.check(token):
            return 403, {"error": "not the operator"}
        if not self.gate.try_begin_home(token):
            return 409, {"error": "home is only accepted after a stop", **self.gate.snapshot()}
        self._spawn(self._home_job, "so101-agent-home")
        return 202, {"accepted": True}

    def _home_job(self) -> None:
        # Cleared here, before queuing, rather than inside the skill: a STOP
        # pressed after this point must still stop the homing motion.
        if self.gate.state is not ControlState.HOMING:
            self.gate.finish_home(False, message="home 복귀 전에 다시 정지되었습니다.")
            return
        self.cancel.clear()
        call_id = f"home_{next(self._ids)}"
        self._publish({"type": "tool_call", "id": call_id, "name": "recover_and_home", "arguments": {}, "direct": True})
        result = self.registry.run_skill("recover_and_home", lambda skills: skills.recover_and_home())
        envelope = result.to_envelope()
        self._publish({"type": "tool_result", "id": call_id, "name": "recover_and_home", "result": envelope, "direct": True})
        at_home = bool(result.ok and result.data.get("arm_at_home"))
        self.gate.finish_home(at_home, message=None if at_home else result.detail)

    def reset_chat(self, token: str | None) -> tuple[int, dict]:
        if not self.gate.check(token):
            return 403, {"error": "not the operator"}
        if self.gate.state is not ControlState.IDLE:
            return 409, {"error": "busy"}
        self.runner.reset()
        self._publish({"type": "reset"})
        return 200, {"reset": True}

    def telemetry(self):
        from .telemetry import collect
        future = self._telemetry_future
        if future is not None and future.done():
            try:
                self._telemetry_cache = future.result()
            except Exception as exc:
                self._telemetry_cache = {**self._telemetry_cache, "error": type(exc).__name__}
            self._telemetry_future = None
        age = time.time() - self._telemetry_cache.get("sampled_at", 0)
        if self._telemetry_future is None and age >= self.cfg.agent.camera_view.poll_s:
            self._telemetry_future = self._worker.submit(collect)
        return {**self._telemetry_cache, "pending": self._telemetry_future is not None,
                "control": self.gate.snapshot()}

    def health(self) -> dict[str, Any]:
        return {
            "session_connected": self.started,
            "provider": self.provider.name,
            "model": self.provider.model,
            **self.gate.snapshot(),
        }


# Arc sampling step for the reach outline. Matches the camera page's
# camera.overlay.workspace_boundary.sample_step_deg default: the two pages
# draw the same sector, so a visibly different smoothness would read as a
# different boundary.
SECTOR_STEP_DEG = 2.0


def places_payload(skills) -> dict[str, Any]:
    """Zone cells and table regions in mm and calibration pixels, for the UI."""
    import numpy as np

    from perception.detector import workspace_sector_points_mm
    from perception.zone import point_in_zone
    from session.grid import bounds as grid_bounds
    from session.relative import table_region_xy

    s = skills.s
    cfg = s.cfg
    names = cfg.agent.zone_slots
    regions = cfg.agent.table_regions

    def px(points):
        if not points:
            return []
        return s.calib.board_to_pixel(np.asarray(points, dtype=np.float64)).tolist()

    slot_xy = s.slot_centres
    region_items = [
        (c, r, table_region_xy(c, r, cfg.perception, regions, s.base_xy))
        for c in regions.columns_deg
        for r in regions.rows_fraction
    ]
    sector_mm = workspace_sector_points_mm(cfg.perception, s.base_xy, SECTOR_STEP_DEG)
    grid_payload = None
    if cfg.agent.board_grid.enabled:
        grid = skills.grid
        cells = sorted(skills.cells.items())
        # one batch transform for every corner: 4 per cell, ~600 points
        corners_mm = [c for (x, y), _xy in cells for c in grid.cell_corners_mm(x, y)]
        corners_px = px(corners_mm)
        grid_payload = {
            "cell_mm": round(grid.cell_mm, 2),
            "measured": bool(s.calib.board_grid),
            "bounds": grid_bounds(skills.cells),
            "default_layer": cfg.agent.board_grid.default_layer,
            "cells": [
                {
                    "x": x, "y": y, "xy_mm": list(xy),
                    "corners_px": corners_px[i * 4 : i * 4 + 4],
                    "in_zone": point_in_zone(xy, s.calib, cfg.agent.table_zone_margin_mm),
                }
                for i, ((x, y), xy) in enumerate(cells)
            ],
        }
    return {
        "image_size": list(s.calib.image_size),
        "zone_polygon_px": px([tuple(p) for p in s.calib.zone_polygon_mm or []]),
        # the same arc the camera page outlines, so both screens agree on
        # where the arm can reach
        "sector_px": {"arc": px(sector_mm.tolist()), "base": px([s.base_xy])[0]},
        "grid": grid_payload,
        "slots": [
            {"index": i, "label": names.labels[i], "korean": names.korean_labels[i],
             "xy_mm": list(xy), "px": p}
            for i, (xy, p) in enumerate(zip(slot_xy, px(slot_xy)))
        ],
        "regions": [
            {"column": c, "row": r, "korean": f"{regions.column_korean[c]} {regions.row_korean[r]}",
             "xy_mm": list(xy), "px": p}
            for (c, r, xy), p in zip(region_items, px([xy for _, _, xy in region_items]))
        ],
    }

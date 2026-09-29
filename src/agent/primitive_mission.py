"""Deterministic Task 1/2 orchestration of the existing guarded agent tools.

No model calls and no duplicate pick, IK, clearance, or recovery logic.
Each composite tool owns its own fresh camera observations; this runner only
selects the next block and destination from returned state.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from config import AppConfig
from fsm.task1 import block_priority
from perception.homography import PlaneCalibration

Call = Callable[[str, dict[str, Any]], dict[str, Any]]
Emit = Callable[[dict[str, Any]], None]


class PrimitiveMission:
    def __init__(
        self, cfg: AppConfig, calib: PlaneCalibration, call: Call,
        *, stopped: Callable[[], bool] = lambda: False,
        emit: Emit = lambda _event: None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cfg, self.calib, self.call = cfg, calib, call
        self.stopped, self.emit, self.clock, self.sleep = stopped, emit, clock, sleep

    def run(self, task: int) -> dict[str, Any]:
        if task not in (1, 2):
            raise ValueError("task must be 1 or 2")
        started = self.clock()
        deadline = started + (180 if task == 1 else 300)
        completed: list[str] = []
        attempted: dict[str, int] = {}
        failures: list[dict[str, str]] = []
        start_pose_recovered = False

        def recovered_home(result: dict[str, Any]) -> bool:
            return (bool(result.get("ok"))
                    and result.get("arm_at_home") is not False
                    and (result.get("state") or {}).get("arm_at_home") is not False)

        def finish(status: str, detail: str) -> dict[str, Any]:
            result = dict(task=task, status=status, detail=detail,
                          colors=completed, failures=failures,
                          elapsed_s=round(self.clock() - started, 2))
            self.emit({"type": "mission_result", **result})
            return result

        while not self.stopped() and self.clock() < deadline:
            seen = self.call("observe_scene", {})
            if not seen.get("ok"):
                return finish("incomplete", f"카메라 관찰 실패: {seen.get('reason')}")
            state = seen.get("state") or {}
            if state.get("holding") or not state.get("arm_at_home"):
                if start_pose_recovered:
                    return finish("needs_recovery", "home 복귀 후에도 안전한 시작 자세가 확인되지 않았습니다.")
                recovered = self.call("recover_and_home", {})
                if self.stopped():
                    break
                if not recovered_home(recovered):
                    return finish("needs_recovery", "시작 자세의 home 복귀에 실패했습니다.")
                start_pose_recovered = True
                continue
            start_pose_recovered = False
            objects = seen.get("objects") or []
            outside = {item["color"] for item in objects if "slot" not in item}
            inside = {item["color"] for item in objects if "slot" in item}
            if task == 1:
                completed = [color for color in completed if color not in outside]
                if len(inside) == 5 and not outside:
                    return finish("complete", "새 관찰에서 블록 5개가 적재구역 안에 확인됐습니다.")
                candidates = [item for item in objects if "slot" not in item]
                slots = state.get("zone_slots") or {}
                free_slots = [name for name in self.cfg.agent.zone_slots.labels
                              if slots.get(name) is None]
                if not candidates or not free_slots:
                    return finish("incomplete", "이동 가능한 외부 블록 또는 빈 슬롯이 확인되지 않았습니다.")
            else:
                if len(completed) >= 5:
                    # One more camera observation after the required dwell is
                    # evidence of continued planar visibility, not tower height.
                    remaining = 5.0
                    while remaining > 0 and not self.stopped():
                        step = min(0.2, remaining)
                        self.sleep(step)
                        remaining -= step
                    if self.stopped():
                        break
                    checked = self.call("observe_scene", {})
                    if not checked.get("ok"):
                        return finish("placed_unverified", "5개 해제 후 재관찰 실패. 실제 적층 높이는 미확인입니다.")
                    return finish("placed_unverified", "5개를 적층 지점에 해제했습니다. 평면 카메라로 층수와 5초 안정성은 검증할 수 없습니다.")
                candidates = [item for item in objects if item["color"] not in completed]
                free_slots = []
                if not candidates:
                    return finish("incomplete", "다음 층에 쓸 블록이 관찰되지 않았습니다.")

            if self.stopped():
                break
            choices = [item for item in candidates
                       if attempted.get(item["color"], 0) < self.cfg.fsm.max_retries_per_block]
            if not choices:
                return finish("incomplete", "관찰된 모든 후보가 이번 실행에서 실패했습니다.")
            target = max(
                choices,
                key=lambda item: block_priority(
                    (item["x_mm"], item["y_mm"]), self.calib, self.cfg),
            )
            color = target["color"]
            if task == 1:
                name, arguments = "move_block_to_slot", dict(color=color, slot=free_slots[0])
            else:
                name, arguments = "stack_block_to_floor", dict(color=color, floor=len(completed))
            self.emit({"type": "mission_step", "task": task, "color": color,
                       "action": name, "arguments": arguments})
            result = self.call(name, arguments)
            if result.get("ok"):
                if color not in completed:
                    completed.append(color)
                attempted.clear()
                continue
            failures.append(dict(color=color, reason=str(result.get("reason")),
                                 stage=str(result.get("failed_stage", ""))))
            if self.stopped():
                break
            # A composite may have released the block before its final check
            # failed. Open, return home, and observe again before deciding
            # whether the block still needs a slot or floor.
            recovered = self.call("recover_and_home", {})
            if self.stopped():
                break
            if not recovered_home(recovered):
                return finish("needs_recovery", f"{color} 실패 후 home 복귀 실패: {recovered.get('reason')}")
            attempted[color] = (self.cfg.fsm.max_retries_per_block
                                if result.get("retry_advice") == "try_other_target"
                                else attempted.get(color, 0) + 1)
        return finish("stopped" if self.stopped() else "timeout",
                      "중단되었습니다." if self.stopped() else "미션 시간 한도에 도달했습니다.")

"""Run the unmodified Task 1/2 FSM flows on this session's robot."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

from fsm.states import RunContext
from session.cancel import guard
from session.results import SkillResult

logger = logging.getLogger(__name__)


class MissionMixin:
    # ── missions ─────────────────────────────────────────────────────

    def run_task(self, task: int) -> SkillResult:
        action, t0, s, cfg = f"run_task{task}", time.monotonic(), self.s, self.cfg
        if task not in (1, 2):
            return self._result(False, action, "invalid_arguments", "미션은 1, 2만 있습니다.", t0=t0)
        if s.held is not None:
            return self._result(False, action, "already_holding",
                                f"{self._label(s.held.color)} 블록을 들고 있어 미션을 시작할 수 없습니다. 먼저 내려놓으세요.",
                                retry_advice="do_not_retry", t0=t0)
        scene = self._observe_or_fail(action, t0)
        if isinstance(scene, SkillResult):
            return scene
        from fsm.flows import build_task1_states, build_task2_stack_states
        from fsm.machine import StateMachine, TransitionLogger

        perceive = guard(s.cancel, s.task1_perceive())
        ctx = RunContext(fsm=cfg.fsm)
        warnings: list[str] = []
        if task == 1:
            # Pre-reserve cells that already hold a block so Task 1 does not
            # drop a new one on top of an earlier agent placement.
            taken = {color: index for index, color in scene.slot_occupancy.items() if color}
            if taken:
                ctx.extras["task1_slot_by_color"] = dict(taken)
            loose = [b.color for b in scene.inside.values() if b.slot_index is None]
            if loose:
                warnings.append(f"적재 구역 안에 칸에 맞지 않게 놓인 블록이 있습니다: {', '.join(loose)}")
            if len(taken) + len(scene.outside) > len(cfg.task1.slot_uv):
                return self._result(False, action, "precondition",
                                    "빈 칸보다 옮길 블록이 많습니다.", retry_advice="ask_operator", t0=t0)
            states = build_task1_states(robot=s.robot, motion=s.motion, perceive=perceive,
                                        pick_state=s.pick_state, cfg=cfg, calib=s.calib,
                                        planner=s.transport)
        else:
            stack_xy = s.stack.stack_xy_mm
            for block in scene.inside.values():
                if math.dist(block.center_mm, stack_xy) < cfg.agent.place_clear_radius_mm:
                    return self._result(
                        False, action, "precondition",
                        f"적재 지점에 이미 {block.color} 블록이 있어 쌓기를 시작할 수 없습니다.",
                        retry_advice="ask_operator", t0=t0,
                    )
            states = build_task2_stack_states(robot=s.robot, motion=s.motion, perceive=perceive,
                                              pick_state=s.pick_state, cfg=cfg, calib=s.calib,
                                              planner=s.stack)
        run_id = time.strftime(f"agent_task{task}_%Y%m%d_%H%M%S")
        log_dir = Path(cfg.logging.log_dir)
        csv_path = log_dir / f"{run_id}_transitions.csv" if cfg.logging.save_transitions else None
        machine = StateMachine(states, ctx, transition_logger=TransitionLogger(csv_path),
                               enforce_time_budget=task != 1)
        logger.info("agent: running Task %d as %s", task, run_id)
        machine.run()
        s.last_scene = None
        verification = self._observe_or_fail(action, t0)
        verification_failed = isinstance(verification, SkillResult)
        remaining = [] if verification_failed else sorted(verification.outside)
        stop_reason = ctx.extras.get("task2_stop_reason")
        if (
            task == 2
            and stop_reason is None
            and ctx.budget_exhausted()
            and (verification_failed or remaining)
        ):
            stop_reason = "time_budget_exhausted"
        data: dict[str, Any] = {
            "run_id": run_id,
            "remaining_outside": remaining,
            "place_actions": ctx.extras.get("task1_place_actions") if task == 1 else ctx.placed_count,
            "attempts": ctx.extras.get("task1_attempts_total") or ctx.attempts,
            "warnings": warnings or None,
            "stop_reason": stop_reason,
            "failed_blocks": ctx.extras.get("task2_failed_blocks"),
        }
        if task == 1:
            complete = bool(ctx.extras.get("task1_complete"))
            detail = "미션 1 완료: 적재 구역 밖에 블록이 없습니다." if complete else "미션 1이 끝났지만 완료 조건을 확인하지 못했습니다."
        else:
            complete = not verification_failed and not remaining
            if complete:
                detail = f"미션 2 완료: {ctx.placed_count}층을 쌓았습니다."
            elif verification_failed:
                detail = (
                    f"미션 2가 {ctx.placed_count}층에서 끝났지만 카메라로 완료 여부를 "
                    "확인하지 못했습니다."
                )
            else:
                detail = (
                    f"미션 2 미완료: {ctx.placed_count}층을 쌓았고 적재 구역 밖에 "
                    f"{', '.join(remaining)} 블록이 남았습니다."
                )
            if stop_reason:
                detail += f" (종료 사유: {stop_reason})"
        if warnings:
            detail += " " + " ".join(warnings)
        return self._result(complete, action, "ok" if complete else "task_incomplete", detail, t0=t0, **data)

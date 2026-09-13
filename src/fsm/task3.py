"""Task-3-only states: episode boundaries and the rearrangement prompt.

Task 3 runs Task 1's gather loop unchanged — ``VerifyState`` and
``Task1TransportState`` are injected as-is, and PICK is the same
``CvIkPickState`` with its rotated retry pinned to one attempt. Only two states
are specialised, and both only because the *episode* has boundaries the
mission does not:

``Task3SelectState``
    Closes the previous episode after the arm is back at home, opens the next
    one once a target is chosen, and turns Task 1's "region is empty"
    termination into a prompt for a new arrangement.

``Task3PlaceState``
    Marks the episode successful. PLACE completing is the only proof that the
    block was both grasped and delivered.
"""

from __future__ import annotations

import logging
from typing import Callable

from config import AppConfig
from control.motion import MotionController
from data.episode_recorder import EpisodeRecorder, StopRecording
from fsm.states import RunContext, StateName
from fsm.task1 import Task1PlaceState, Task1PerceiveFn, Task1SelectState
from perception.homography import PlaneCalibration

logger = logging.getLogger(__name__)

#: ctx.extras key set by PLACE and consumed by the next SELECT.
EPISODE_OK_KEY = "task3_episode_ok"


class Task3SelectState(Task1SelectState):
    """HOME, close the last episode, open the next one.

    The episode boundary falls out of Task 1's own structure: the parent's
    ``enter`` is where the arm returns home, so closing the episode *after*
    that call means every episode ends with the return-home motion and the
    next one starts from a settled home pose. Both ends of the cycle are the
    same pose, which is what makes the demonstration loopable.

    The camera-polling wait in ``step`` sits between episodes and is
    therefore never recorded.
    """

    def __init__(
        self,
        motion: MotionController,
        perceive: Task1PerceiveFn,
        calib: PlaneCalibration,
        cfg: AppConfig,
        recorder: EpisodeRecorder,
        *,
        prompt=input,
        stop_requested: Callable[[], bool] = lambda: False,
    ):
        super().__init__(motion, perceive, calib, cfg)
        self._recorder = recorder
        # Injected so tests can drive the round boundary without a terminal.
        self._prompt = prompt
        self._stop_requested = stop_requested
        self._saved_at_round_start = 0

    def enter(self, ctx: RunContext) -> None:
        # go_home first: its motion belongs to the episode being closed.
        super().enter(ctx)
        success = bool(ctx.extras.pop(EPISODE_OK_KEY, False))
        self._recorder.finish_episode(success=success)

    def step(self, ctx: RunContext) -> StateName | None:
        # SELECT can spend an arbitrary time polling the camera without
        # issuing a robot command, so RecordingRobotIO's command-boundary
        # stop check alone is insufficient here.
        if self._stop_requested():
            raise StopRecording("stop requested while selecting a block")
        next_state = super().step(ctx)
        if self._stop_requested():
            raise StopRecording("stop requested while selecting a block")
        if next_state is StateName.DONE:
            # Task 1 would stop here. Task 3 collects rounds indefinitely, so
            # the empty-region proof means "the arrangement is used up".
            return self._round_complete(ctx)
        if next_state is StateName.PICK:
            target_color = ctx.target_id
            assert target_color is not None
            self._recorder.begin_episode(target_color)
        return next_state

    # ── round boundary ───────────────────────────────────────────────

    def _round_complete(self, ctx: RunContext) -> StateName | None:
        """Ask for a new arrangement instead of finishing the run."""
        ctx.extras.pop("task1_complete", None)
        rounds = int(ctx.extras.get("task3_rounds", 0)) + 1
        ctx.extras["task3_rounds"] = rounds

        saved_this_round = self._recorder.saved_total - self._saved_at_round_start
        if saved_this_round == 0:
            stalled = int(ctx.extras.get("task3_stalled_rounds", 0)) + 1
            ctx.extras["task3_stalled_rounds"] = stalled
        else:
            ctx.extras["task3_stalled_rounds"] = 0
            stalled = 0

        self._announce(rounds, saved_this_round, stalled)
        if self._cfg.task3.prompt_on_round_complete:
            self._prompt("블록을 다시 배치한 뒤 Enter를 누르세요 (종료: Ctrl-C) ")

        self._reset_round(ctx)
        self._saved_at_round_start = self._recorder.saved_total
        return None

    def _announce(self, rounds: int, saved_this_round: int, stalled: int) -> None:
        totals = ", ".join(
            f"{color} {count}" for color, count in sorted(self._recorder.saved_by_color.items())
        ) or "none"
        lines = [
            "",
            "=" * 68,
            f"  라운드 {rounds} 완료 — 지정구역 밖에 블록이 없습니다.",
            f"  이번 라운드 저장: {saved_this_round}개 / 누적: {self._recorder.saved_total}개 ({totals})",
        ]
        if stalled >= self._cfg.task3.max_rounds_without_progress:
            lines.append(
                f"  경고: {stalled}개 라운드 연속으로 저장된 에피소드가 없습니다. "
                "블록이 팔의 작업 반경 안에 있는지 확인하세요."
            )
        lines += ["  블록 5개를 다시 랜덤하게 배치해 주세요.", "=" * 68, ""]
        # print, not logger: this is an instruction to the operator, and it
        # has to stand out in a log stream that is otherwise per-tick chatter.
        print("\a" + "\n".join(lines), flush=True)

    def _reset_round(self, ctx: RunContext) -> None:
        """Forget everything that was scoped to the previous arrangement."""
        self._archive_attempts(ctx, set(ctx.attempts) | set(ctx.skipped))
        # Slots are per-arrangement: the next five blocks must fill 0..4 again.
        ctx.extras.pop("task1_slot_by_color", None)
        ctx.extras.pop("task1_empty_for_s", None)
        ctx.target_id = None
        self._empty_since = None
        # The blocks moved while we were blocked on the prompt, so any frame
        # captured before now is evidence about the old arrangement.
        self._last_frame_seq = -1


class Task3PlaceState(Task1PlaceState):
    """Task 1's PLACE, plus the flag that says this episode is worth keeping.

    The flag is set only after the release and the lift back to hover have
    both completed, so "successful" means grasped *and* delivered — the rule
    an episode is saved under.
    """

    def step(self, ctx: RunContext) -> StateName | None:
        next_state = super().step(ctx)
        ctx.extras[EPISODE_OK_KEY] = True
        return next_state

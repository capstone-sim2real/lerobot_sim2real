"""State machine loop: transitions, time budget enforcement, transition log."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

from fsm.states import RunContext, State, StateName

logger = logging.getLogger(__name__)


class TransitionLogger:
    """Appends one CSV row per transition; used for post-run analysis."""

    COLUMNS = ("wall_time", "elapsed_s", "from_state", "to_state", "placed_count", "target_id", "note")

    def __init__(self, csv_path: Path | str | None):
        self._path = Path(csv_path) if csv_path is not None else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", newline="") as f:
                csv.writer(f).writerow(self.COLUMNS)

    def log(self, ctx: RunContext, from_state: StateName, to_state: StateName) -> None:
        note, ctx.last_note = ctx.last_note, ""
        logger.info(
            "%s -> %s (elapsed %.1fs, placed %d%s)",
            from_state.value,
            to_state.value,
            ctx.elapsed_s(),
            ctx.placed_count,
            f", note: {note}" if note else "",
        )
        if self._path is None:
            return
        with open(self._path, "a", newline="") as f:
            csv.writer(f).writerow(
                (
                    f"{time.time():.3f}",
                    f"{ctx.elapsed_s():.3f}",
                    from_state.value,
                    to_state.value,
                    ctx.placed_count,
                    ctx.target_id or "",
                    note,
                )
            )


class StateMachine:
    """Runs states until DONE is reached or the time budget is exhausted.

    DONE is terminal — there is no DONE handler here. Hardware cleanup
    (return home, disconnect) belongs in the runner's finally block so it
    also happens on exceptions.
    """

    def __init__(
        self,
        states: dict[StateName, State],
        ctx: RunContext,
        initial: StateName = StateName.SELECT,
        transition_logger: TransitionLogger | None = None,
    ):
        if initial not in states:
            raise ValueError(f"Initial state {initial.value!r} has no handler")
        self._states = states
        self._ctx = ctx
        self._initial = initial
        self._logger = transition_logger or TransitionLogger(None)

    def run(self) -> RunContext:
        current_name = self._initial
        while current_name is not StateName.DONE:
            state = self._states[current_name]
            state.enter(self._ctx)
            next_name: StateName | None = None
            while next_name is None:
                if self._ctx.budget_exhausted():
                    self._ctx.last_note = self._ctx.last_note or "time_budget_exhausted"
                    next_name = StateName.DONE
                    break
                next_name = state.step(self._ctx)
            state.exit(self._ctx)
            self._logger.log(self._ctx, current_name, next_name)
            current_name = next_name
        return self._ctx

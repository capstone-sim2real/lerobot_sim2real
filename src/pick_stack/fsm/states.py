"""FSM vocabulary: state names, run context, and the state handler base class.

Task flow (AGENTS.md §3):

    SELECT → PICK → VERIFY → TRANSPORT → PLACE → (blocks left && time left) → SELECT
               ↑______fail (retry / skip)____|

Task 1 and Task 2 differ only in which PLACE handler is injected; every other
state is shared. Concrete handlers arrive in later PRs — this module defines
the contract they implement so the machine and tests are stable from day one.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pick_stack.config import FsmConfig


class StateName(str, Enum):
    SELECT = "select"
    PICK = "pick"
    VERIFY = "verify"
    TRANSPORT = "transport"
    PLACE = "place"
    DONE = "done"


@dataclass
class RunContext:
    """Mutable run state shared by all handlers.

    ``target_id`` identifies the block currently being worked on (set by
    SELECT — e.g. a detection index or colour label). Attempt counting is per
    target so one stubborn block cannot eat the whole time budget.
    """

    fsm: FsmConfig
    start_time: float = field(default_factory=time.monotonic)
    placed_count: int = 0
    target_id: str | None = None
    attempts: dict[str, int] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)
    # free-form scratch space for handlers (e.g. last detection result)
    extras: dict[str, Any] = field(default_factory=dict)
    # short note attached to the next logged transition (reset after logging)
    last_note: str = ""

    def elapsed_s(self) -> float:
        return time.monotonic() - self.start_time

    def time_left_s(self) -> float:
        return self.fsm.time_budget_s - self.elapsed_s()

    def budget_exhausted(self) -> bool:
        return self.time_left_s() <= self.fsm.reserve_time_s

    def record_attempt(self, target_id: str) -> int:
        """Count one attempt on a target; returns the new attempt count."""
        self.attempts[target_id] = self.attempts.get(target_id, 0) + 1
        return self.attempts[target_id]

    def should_skip(self, target_id: str) -> bool:
        return (
            target_id in self.skipped
            or self.attempts.get(target_id, 0) >= self.fsm.max_retries_per_block
        )

    def skip(self, target_id: str) -> None:
        self.skipped.add(target_id)

    def all_blocks_done(self) -> bool:
        return self.placed_count >= self.fsm.num_blocks


class State(abc.ABC):
    """One FSM state. The machine calls enter() once, then step() repeatedly
    until it returns the next StateName, then exit().

    step() must return quickly (one control tick / one bounded motion chunk):
    the machine checks the time budget between steps, so a step that blocks
    for minutes defeats the 5-minute cutoff.
    """

    name: StateName

    def enter(self, ctx: RunContext) -> None:  # noqa: B027 (optional hook)
        pass

    @abc.abstractmethod
    def step(self, ctx: RunContext) -> StateName | None:
        """Advance one tick. Return the next state, or None to keep stepping."""

    def exit(self, ctx: RunContext) -> None:  # noqa: B027 (optional hook)
        pass

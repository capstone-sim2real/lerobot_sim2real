import csv

import pytest

from pick_stack.config import FsmConfig
from pick_stack.fsm import RunContext, State, StateMachine, StateName, TransitionLogger


class SelectMock(State):
    name = StateName.SELECT

    def step(self, ctx):
        if ctx.all_blocks_done():
            ctx.last_note = "all_blocks_placed"
            return StateName.DONE
        candidates = [f"block{i}" for i in range(ctx.fsm.num_blocks)]
        remaining = [c for c in candidates if not ctx.should_skip(c) and c not in ctx.extras.get("placed", set())]
        if not remaining:
            ctx.last_note = "no_targets_left"
            return StateName.DONE
        ctx.target_id = remaining[0]
        return StateName.PICK


class PickMock(State):
    name = StateName.PICK

    def step(self, ctx):
        ctx.record_attempt(ctx.target_id)
        return StateName.VERIFY


class VerifyMock(State):
    """Fails the grasp check the first `fail_first` times per target."""

    name = StateName.VERIFY

    def __init__(self, fail_first=0):
        self.fail_first = fail_first

    def step(self, ctx):
        if ctx.attempts[ctx.target_id] <= self.fail_first:
            if ctx.should_skip(ctx.target_id):
                ctx.skip(ctx.target_id)
                ctx.last_note = "grasp_failed_skip"
                return StateName.SELECT
            ctx.last_note = "grasp_failed_retry"
            return StateName.PICK
        return StateName.TRANSPORT


class TransportMock(State):
    name = StateName.TRANSPORT

    def step(self, ctx):
        return StateName.PLACE


class PlaceMock(State):
    name = StateName.PLACE

    def step(self, ctx):
        ctx.placed_count += 1
        ctx.extras.setdefault("placed", set()).add(ctx.target_id)
        return StateName.SELECT


class StuckState(State):
    name = StateName.PICK

    def step(self, ctx):
        return None  # never transitions on its own


def make_states(verify=None):
    return {
        StateName.SELECT: SelectMock(),
        StateName.PICK: PickMock(),
        StateName.VERIFY: verify or VerifyMock(),
        StateName.TRANSPORT: TransportMock(),
        StateName.PLACE: PlaceMock(),
    }


def make_ctx(**kwargs):
    fsm = FsmConfig(**{"num_blocks": 3, "time_budget_s": 60.0, "reserve_time_s": 0.0, **kwargs})
    return RunContext(fsm=fsm)


def test_happy_path_places_all_blocks():
    ctx = StateMachine(make_states(), make_ctx()).run()
    assert ctx.placed_count == 3
    assert ctx.extras["placed"] == {"block0", "block1", "block2"}


def test_verify_failure_retries_then_succeeds():
    ctx = StateMachine(make_states(VerifyMock(fail_first=1)), make_ctx()).run()
    assert ctx.placed_count == 3
    assert all(count == 2 for count in ctx.attempts.values())


def test_target_skipped_after_max_retries():
    # every grasp fails -> each target gets max_retries_per_block attempts, then skipped
    ctx = StateMachine(make_states(VerifyMock(fail_first=100)), make_ctx(max_retries_per_block=2)).run()
    assert ctx.placed_count == 0
    assert ctx.skipped == {"block0", "block1", "block2"}
    assert all(count == 2 for count in ctx.attempts.values())


def test_time_budget_forces_done():
    ctx = make_ctx(time_budget_s=-1.0)
    states = make_states()
    states[StateName.PICK] = StuckState()
    result = StateMachine(states, ctx).run()
    assert result.placed_count == 0


def test_missing_handler_rejected():
    states = make_states()
    del states[StateName.PLACE]
    with pytest.raises(ValueError, match="place"):
        StateMachine(states, make_ctx())


def test_transition_log_written(tmp_path):
    log_path = tmp_path / "transitions.csv"
    StateMachine(
        make_states(), make_ctx(num_blocks=1), transition_logger=TransitionLogger(log_path)
    ).run()
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    assert [r["to_state"] for r in rows] == ["pick", "verify", "transport", "place", "select", "done"]
    assert rows[-1]["note"] == "all_blocks_placed"
    assert rows[-1]["placed_count"] == "1"

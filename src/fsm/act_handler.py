"""ACT-specific implementation of the FSM PICK state.

The rest of the state machine is policy-agnostic.  Keeping this adapter in a
separate module lets the runner inject a CV+IK PICK state without making the
shared FSM depend on ACT or its gRPC transport.
"""

from __future__ import annotations

from typing import Protocol

from control.motion import MotionController
from control.poses import Pose
from fsm.states import RunContext, State, StateName
from policy.act_client import PickResult


class PickClient(Protocol):
    """Minimal interface the ACT PICK state needs from a policy client."""

    def ping(self) -> bool: ...

    def run_pick(self, retreat_pose: Pose) -> PickResult: ...


class ActPickState(State):
    """Run the legacy ACT policy until it reaches the recorded retreat pose."""

    name = StateName.PICK

    def __init__(self, client: PickClient, motion: MotionController, retreat_pose: Pose):
        self._client = client
        self._motion = motion
        self._retreat_pose = retreat_pose

    def step(self, ctx: RunContext) -> StateName | None:
        if not self._client.ping():
            ctx.last_note = "policy_server_unreachable"
            return StateName.DONE

        ctx.record_attempt(ctx.target_id)
        result = self._client.run_pick(self._retreat_pose)
        ctx.extras["pick_result"] = result
        if result.reached_retreat:
            return StateName.VERIFY

        self._motion.open_gripper()
        if ctx.should_skip(ctx.target_id):
            ctx.skip(ctx.target_id)
            ctx.last_note = f"pick_{result.outcome}_skip"
        else:
            ctx.last_note = f"pick_{result.outcome}_retry"
        return StateName.SELECT

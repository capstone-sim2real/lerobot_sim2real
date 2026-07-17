"""End-to-end FSM runs with stubbed motion/policy/perception on MockRobotIO."""

import pytest

from pick_stack.config import FsmConfig, SensingConfig
from pick_stack.fsm.handlers import SlotPlaceStrategy, StackPlaceStrategy, build_states
from pick_stack.fsm.machine import StateMachine
from pick_stack.fsm.states import RunContext
from pick_stack.control import MockRobotIO
from pick_stack.perception.detector import BlockDetection
from pick_stack.perception.select import SelectionResult, target_id_for
from pick_stack.policy.act_client import PickResult

RETREAT = {j: 0.0 for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")}


def det(color, x, y):
    return BlockDetection(color=color, center_mm=(x, y), area_mm2=1600.0, aspect=1.0,
                          solidity=1.0, fill=1.0, box_mm=[])


class StubMotion:
    """Records primitive calls; no kinematics."""

    def __init__(self):
        self.calls = []

    def go_home(self):
        self.calls.append("go_home")

    def open_gripper(self):
        self.calls.append("open_gripper")

    def transport_to_zone(self):
        self.calls.append("transport")

    def place_in_slot(self, slot_index):
        self.calls.append(f"slot_{slot_index}")

    def stack_place(self):
        self.calls.append("stack")
        return True


class StubClient:
    def __init__(self, robot, outcomes=None, reachable=True):
        # outcomes: list of "ok"/"timeout" consumed per run_pick call
        self.robot = robot
        self.outcomes = list(outcomes or [])
        self.reachable = reachable
        self.picks = 0

    def ping(self):
        return self.reachable

    def run_pick(self, retreat_pose):
        self.picks += 1
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if outcome == "ok":
            # simulate a successful grasp: gripper stopped on the block, load high
            self.robot.joints["gripper"] = 25.0
            self.robot.loads["gripper"] = 300
            return PickResult("retreat_reached", 5.0, 100, 90)
        self.robot.joints["gripper"] = 3.0
        self.robot.loads["gripper"] = 10
        return PickResult("timeout", 25.0, 700, 0)


class StubPerception:
    """Blocks disappear once placed (the stub 'places' the selected one)."""

    def __init__(self, blocks):
        self.blocks = dict(blocks)  # target_id -> detection
        self.placed = set()

    def __call__(self, skipped):
        eligible = {tid: d for tid, d in self.blocks.items() if tid not in skipped and tid not in self.placed}
        if not eligible:
            return SelectionResult(None, None, 0, list(self.blocks.values()))
        tid = sorted(eligible)[0]
        return SelectionResult(eligible[tid], tid, len(eligible), list(self.blocks.values()))

    def mark_placed(self, tid):
        self.placed.add(tid)


def make_blocks(n):
    detections = [det("red", 100 + 50 * i, 200) for i in range(n)]
    return {target_id_for(d, 40.0): d for d in detections}


def build(robot, motion, client, perceive, task=1, **fsm_kwargs):
    sensing = SensingConfig(grasp_settle_s=0.0, sample_interval_s=0.0, grasp_samples=1)
    strategy = SlotPlaceStrategy(motion) if task == 1 else StackPlaceStrategy(motion)
    states = build_states(
        robot=robot, motion=motion, perceive=perceive, client=client,
        retreat_pose=RETREAT, sensing_cfg=sensing, place_strategy=strategy,
    )
    ctx = RunContext(fsm=FsmConfig(**{"num_blocks": 3, "time_budget_s": 60.0, "reserve_time_s": 0.0, **fsm_kwargs}))
    return StateMachine(states, ctx), ctx


class PlacementAwarePerception(StubPerception):
    """Marks the selected block placed when the FSM's placed_count grows."""

    def __init__(self, blocks, ctx_ref):
        super().__init__(blocks)
        self.ctx_ref = ctx_ref
        self._last_placed = 0
        self._last_tid = None

    def __call__(self, skipped):
        ctx = self.ctx_ref()
        if ctx is not None and ctx.placed_count > self._last_placed and self._last_tid:
            self.placed.add(self._last_tid)
            self._last_placed = ctx.placed_count
        result = super().__call__(skipped)
        self._last_tid = result.target_id
        return result


def run_scenario(task=1, outcomes=None, reachable=True, num_blocks=3, **fsm_kwargs):
    robot = MockRobotIO()
    robot.connect()
    motion = StubMotion()
    client = StubClient(robot, outcomes=outcomes, reachable=reachable)
    holder = {}
    perceive = PlacementAwarePerception(make_blocks(num_blocks), lambda: holder.get("ctx"))
    machine, ctx = build(robot, motion, client, perceive, task=task,
                         num_blocks=num_blocks, **fsm_kwargs)
    holder["ctx"] = ctx
    machine.run()
    return ctx, motion, client


def test_task1_places_all_blocks_in_slot_order():
    ctx, motion, client = run_scenario(task=1)
    assert ctx.placed_count == 3
    assert client.picks == 3
    assert [c for c in motion.calls if c.startswith("slot")] == ["slot_0", "slot_1", "slot_2"]
    assert motion.calls.count("transport") == 3


def test_task2_uses_stack_strategy():
    ctx, motion, _ = run_scenario(task=2)
    assert ctx.placed_count == 3
    assert motion.calls.count("stack") == 3
    assert ctx.extras["stack_contacts"] == [True, True, True]


def test_pick_timeout_retries_then_succeeds():
    ctx, motion, client = run_scenario(outcomes=["timeout", "ok", "ok", "ok"])
    assert ctx.placed_count == 3
    assert client.picks == 4
    assert motion.calls.count("open_gripper") >= 1  # reset after the timeout


def test_block_skipped_after_repeated_failures():
    # first block: every pick times out -> 2 attempts -> skipped; others fine
    ctx, _, client = run_scenario(outcomes=["timeout", "timeout", "ok", "ok"], max_retries_per_block=2)
    assert ctx.placed_count == 2
    assert len(ctx.skipped) == 1
    assert client.picks == 4


def test_server_unreachable_stops_run():
    ctx, _, client = run_scenario(reachable=False)
    assert ctx.placed_count == 0
    assert client.picks == 0


def test_verify_failure_returns_to_select():
    # pick "succeeds" (reaches retreat) but with an empty gripper
    class EmptyHandClient(StubClient):
        def run_pick(self, retreat_pose):
            self.picks += 1
            self.robot.joints["gripper"] = 3.0  # fully closed = nothing held
            self.robot.loads["gripper"] = 10
            return PickResult("retreat_reached", 5.0, 100, 90)

    robot = MockRobotIO()
    robot.connect()
    motion = StubMotion()
    client = EmptyHandClient(robot)
    perceive = StubPerception(make_blocks(1))
    machine, ctx = build(robot, motion, client, perceive, num_blocks=1, max_retries_per_block=2)
    machine.run()
    assert ctx.placed_count == 0
    assert client.picks == 2  # retried once, then skipped
    assert len(ctx.skipped) == 1
    assert "transport" not in motion.calls  # HARD RULE: no transport without grasp

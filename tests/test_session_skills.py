"""Agent skills on the production pick/place code, against a simulated arm."""

import math

import pytest

from agent.sim import SimRobotIO
from agent.tools import ToolRegistry
from agent.provider.types import ToolCall
from fsm import ik_handler
from session.cancel import Cancelled

from agent_helpers import FakeIk, fast_cfg, make_skills


def _slot_xy(skills, index):
    return skills.s.slot_centres[index]


def test_pick_not_detected_never_commands_the_bus():
    skills, _world, robot = make_skills({"yellow": (180.0, 120.0)})
    result = skills.pick_block("blue")
    assert (result.ok, result.reason, result.retry_advice) == (False, "not_detected", "ask_operator")
    assert robot.sent_actions == []


def test_move_block_to_slot_places_and_verifies_the_cell():
    skills, world, _robot = make_skills({"yellow": (180.0, 120.0)})
    result = skills.move_block_to_slot("yellow", 0)
    assert result.ok and result.reason == "released", result.detail
    assert result.data["verified"] is True
    assert result.data["measured"]["slot"] == "top-left"
    assert math.dist(world.blocks["yellow"], _slot_xy(skills, 0)) < 1.0
    assert skills.s.held is None and skills.s.last_block_color == "yellow"


def test_recover_and_home_drops_a_held_block_and_closes_the_gripper():
    skills, world, robot = make_skills({"yellow": (200.0, 100.0)})
    assert skills.pick_block("yellow").ok
    picked_at = world.blocks["yellow"]  # SimWorld only tracks a block's xy while it is not held
    assert skills.s.held is not None

    result = skills.recover_and_home()
    assert result.ok and result.data["arm_at_home"] is True
    assert result.data["released"] == "yellow"
    assert skills.s.held is None
    # dropped close to where it was picked from (small grasp-bias tolerance),
    # not carried all the way home (home is ~150mm away)
    assert world.blocks["yellow"] == pytest.approx(picked_at, abs=15.0)
    assert robot.joints["gripper"] == pytest.approx(skills.cfg.sensing.gripper_close_pos)


# ── board cell addressing ──────────────────────────────────────────────


def _far_cell(skills):
    """An addressable cell far from home, to prove the jog cap does not apply."""
    return max(skills.cells, key=lambda key: math.dist(skills.cells[key], skills.s.arm_position_mm()[:2]))

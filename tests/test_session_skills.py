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


def test_recover_and_home_drops_a_held_block_and_opens_the_gripper():
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
    assert robot.joints["gripper"] == pytest.approx(skills.cfg.sensing.gripper_open_pos)


# ── board cell addressing ──────────────────────────────────────────────


def _far_cell(skills):
    """An addressable cell far from home, to prove the jog cap does not apply."""
    return max(skills.cells, key=lambda key: math.dist(skills.cells[key], skills.s.arm_position_mm()[:2]))


def test_selected_perception_backend_supplies_agent_observation():
    import numpy as np
    from camera.client import CameraSnapshot

    skills, world, _robot = make_skills({"yellow": (180.0, 120.0)})
    expected = world.scene(
        skills.s.calib, skills.s.slot_centres, skills.cfg.agent.slot_snap_radius_mm
    )

    class SelectedBackend:
        backend = "yoloe"
        calls = 0

        def observe_scene(self, calib, slot_xy, *, after, cancel, clock):
            self.calls += 1
            return expected, CameraSnapshot(
                np.zeros((500, 700, 3), np.uint8), expected.frame_seq, clock()
            )

    backend = SelectedBackend()
    skills.s._scene_fn = None
    skills.s._perception_backend = backend
    result = skills.observe_scene()

    assert result.ok
    assert backend.calls == 1
    assert [item["color"] for item in result.data["blocks_outside"]] == ["yellow"]
    assert skills.s.last_snapshot is not None

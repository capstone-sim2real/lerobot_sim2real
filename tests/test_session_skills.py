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


def test_occupied_cell_is_refused_before_moving():
    skills, world, robot = make_skills({"yellow": (180.0, 120.0), "green": (180.0, -120.0)})
    assert skills.move_block_to_slot("green", 0).ok
    sent = len(robot.sent_actions)
    result = skills.move_block_to_slot("yellow", 0)
    assert (result.ok, result.reason, result.retry_advice) == (False, "slot_occupied", "retry_ok")
    assert "top-center" in result.data["free_slots"]
    # at most the homing before the photo ran: the jaws were never commanded
    assert all("gripper" not in action for action in robot.sent_actions[sent:])
    assert world.blocks["yellow"] == (180.0, 120.0)


def test_unreachable_block_is_reported_without_a_grasp(monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("run_grasp_attempts must not run for an unreachable block")

    monkeypatch.setattr(ik_handler, "run_grasp_attempts", forbidden)
    skills, _world, _robot = make_skills({"red": (320.0, 0.0)}, ik=FakeIk(reach_mm=200.0))
    result = skills.pick_block("red")
    assert (result.reason, result.retry_advice) == ("unreachable", "do_not_retry")
    assert result.data["internal_retries_exhausted"] is True


def test_empty_grasp_uses_every_home_and_reapproach_round():
    cfg = fast_cfg()
    skills, _world, _robot = make_skills({"blue": (200.0, 150.0)}, cfg=cfg, grab_radius_mm=0.01)
    result = skills.pick_block("blue")
    assert (result.ok, result.reason, result.retry_advice) == (False, "grasp_empty", "ask_operator")
    assert result.data["attempts_used"] == cfg.fsm.max_retries_per_block
    assert skills.s.arm_at_home()


def test_relative_grasp_offset_accumulates_clamps_and_puts_the_block_back():
    cfg = fast_cfg()
    cfg.agent.relative.max_pick_offset_mm = 12.0
    skills, world, _robot = make_skills({"wood": (200.0, 0.0)}, cfg=cfg)
    assert skills.pick_block("wood").ok
    again = skills.pick_block("wood", forward_mm=5.0, relative_to_last=True)
    assert again.ok and again.data["pick_offset_mm"] == {"forward": 5.0, "left": 0.0}
    third = skills.pick_block("wood", forward_mm=10.0, relative_to_last=True)
    assert third.ok and third.data["pick_offset_mm"]["forward"] == pytest.approx(12.0)
    assert third.data["offset_clamped"] is True
    wrong = skills.pick_block("red", forward_mm=5.0, relative_to_last=True)
    assert wrong.reason == "already_holding"


def test_relative_grasp_without_history_is_refused():
    skills, _world, robot = make_skills({"wood": (200.0, 0.0)})
    result = skills.pick_block("wood", forward_mm=5.0, relative_to_last=True)
    assert result.reason == "precondition" and robot.sent_actions == []


def test_block_is_taken_back_out_of_the_zone_to_a_named_table_region():
    skills, world, _robot = make_skills({"green": (180.0, -120.0)})
    assert skills.move_block_to_slot("green", 1).ok
    result = skills.move_block_to_table("green", "left", "middle")
    assert result.ok, result.detail
    assert result.data["in_zone"] is False
    assert not skills.s.in_zone(world.blocks["green"])
    assert world.blocks["green"][1] > 0  # left of the camera view is +y


def test_table_region_that_overlaps_the_zone_is_never_chosen():
    skills, world, _robot = make_skills({"green": (180.0, -120.0)})
    result = skills.move_block_to_table("green", "center", "far")
    # center/far lies inside the zone: a nearby free spot or a refusal, never the zone
    if result.ok:
        assert not skills.s.in_zone(world.blocks["green"], skills.cfg.agent.table_zone_margin_mm)
    else:
        assert result.reason in ("no_free_region", "destination_in_zone")


def test_shift_block_reports_measured_displacement_in_the_arm_frame():
    skills, world, _robot = make_skills({"blue": (180.0, 0.0)})
    result = skills.shift_block("blue", 0.0, 20.0)
    assert result.ok and result.reason == "moved", result.detail
    # straight ahead, arm-frame left is +y
    assert result.data["measured_mm"]["left"] == pytest.approx(20.0, abs=6.0)
    assert world.blocks["blue"][1] > 10.0


def test_shift_into_another_block_is_refused_before_grasping():
    skills, world, robot = make_skills({"blue": (180.0, 0.0), "red": (180.0, 40.0)})
    result = skills.shift_block("blue", 0.0, 30.0)
    assert result.reason == "destination_blocked"
    assert all("gripper" not in action for action in robot.sent_actions)
    assert world.blocks["blue"] == (180.0, 0.0)


def test_jog_limits_are_refused_without_motion():
    skills, _world, robot = make_skills({})
    too_far = skills.move_arm(left_mm=60.0)
    assert too_far.reason == "limit_exceeded" and robot.sent_actions == []


def test_jog_from_home_enters_the_jog_height_then_place_here_releases():
    skills, world, _robot = make_skills({"yellow": (200.0, 100.0)})
    assert skills.pick_block("yellow").ok
    moved = skills.move_arm(left_mm=20.0)
    assert moved.ok, moved.detail
    x, y, z = skills.s.arm_position_mm()
    assert y == pytest.approx(moved.data["target"]["y"], abs=0.5)
    placed = skills.place_here()
    assert placed.ok, placed.detail
    assert math.dist(world.blocks["yellow"], (x, y)) < 1.0

    fresh, _w, robot = make_skills({})
    entered = fresh.move_arm(left_mm=10.0)
    assert entered.ok and entered.data["entered_jog_height"] is True
    assert robot.joints["elbow_flex"] == pytest.approx(fresh.cfg.agent.relative.jog_min_z_mm)
    below = fresh.move_arm(up_mm=-30.0)
    assert below.reason == "height_limit"


def test_jog_tolerates_settling_just_past_the_window_edge():
    """A lateral move (up_mm=0) must not get permanently trapped once the arm
    settles a few mm outside the strict jog window -- the same IK/arrival
    tolerance every move here already accepts is what put it there."""
    skills, _world, robot = make_skills({})
    entered = skills.move_arm(left_mm=10.0)
    assert entered.ok and entered.data["entered_jog_height"] is True
    jog_min = skills.cfg.agent.relative.jog_min_z_mm
    assert robot.joints["elbow_flex"] == pytest.approx(jog_min)

    # simulate the arm settling a couple mm below the window's exact edge --
    # well within jog_max_ik_error_mm, exactly like real hardware's
    # transit_arrival_tol does not land on the commanded height exactly
    robot.joints["elbow_flex"] = jog_min - (skills.cfg.agent.relative.jog_max_ik_error_mm - 1.0)

    again = skills.move_arm(left_mm=5.0)
    assert again.ok, again.detail
    assert not again.data.get("entered_jog_height")  # already "in" the window, not re-clamped


def test_rotate_gripper_only_moves_wrist_roll():
    skills, _world, robot = make_skills({})
    before = dict(robot.joints)
    result = skills.rotate_gripper(30.0)
    assert result.ok, result.detail
    assert robot.joints["wrist_roll"] == pytest.approx(before["wrist_roll"] + 30.0)
    for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"):
        assert robot.joints[joint] == pytest.approx(before[joint])
    assert result.data == {"from_deg": pytest.approx(before["wrist_roll"]),
                           "target_deg": pytest.approx(before["wrist_roll"] + 30.0),
                           "reached_deg": pytest.approx(before["wrist_roll"] + 30.0)}


def test_rotate_gripper_rejects_zero_and_oversized_requests():
    skills, _world, robot = make_skills({})
    assert skills.rotate_gripper(0.0).reason == "invalid_arguments"
    limit = skills.cfg.agent.relative.max_gripper_roll_deg
    too_far = skills.rotate_gripper(limit + 1.0)
    assert too_far.reason == "limit_exceeded" and robot.sent_actions == []


def test_pick_here_grabs_whatever_is_under_the_gripper_with_unknown_colour():
    skills, world, robot = make_skills({"yellow": (150.0, 30.0)})
    # position the arm over the block first, exactly like a claw machine
    moved = skills.move_arm(left_mm=30.0)
    assert moved.ok, moved.detail

    result = skills.pick_here()
    assert result.ok and result.reason == "held", result.detail
    assert skills.s.held is not None and skills.s.held.color is None
    assert skills.s.last_block_color is None
    state = skills.state_dict()
    assert state["holding"] == "unidentified"


def test_pick_here_refuses_when_already_holding_something():
    skills, _world, robot = make_skills({"yellow": (200.0, 100.0)})
    assert skills.pick_block("yellow").ok
    sent = len(robot.sent_actions)
    result = skills.pick_here()
    assert result.reason == "already_holding"
    assert robot.sent_actions[sent:] == []


def test_place_here_after_pick_here_reports_the_block_as_unidentified():
    skills, world, _robot = make_skills({"yellow": (150.0, 30.0)})
    skills.move_arm(left_mm=30.0)
    assert skills.pick_here().ok
    result = skills.place_here()
    assert result.ok, result.detail
    assert "color" not in result.data  # None fields are dropped from the envelope
    assert "정체 불명" in result.detail
    assert result.data["verified"] is False  # can't camera-verify an unknown colour


def test_stop_mid_skill_becomes_a_cancelled_fault_envelope():
    skills, _world, robot = make_skills({"yellow": (200.0, 100.0)})
    original = SimRobotIO.send_joints
    calls = {"n": 0}

    def send(self, positions):
        calls["n"] += 1
        if calls["n"] == 3:
            skills.s.cancel.set()
        return original(self, positions)

    robot.send_joints = send.__get__(robot)
    registry = ToolRegistry(skills.cfg, lambda job: job(skills))
    result = registry.execute(ToolCall("c1", "pick_block", {"color": "yellow"}))
    assert result.content["reason"] == "cancelled" and result.is_error
    assert "state" in result.content

    with pytest.raises(Cancelled):
        skills.s.robot.send_joints({"gripper": 50.0})
    recovered = skills.recover_and_home()
    assert recovered.ok and recovered.data["arm_at_home"] is True
    assert recovered.data.get("released") is None  # nothing was held when it stopped
    assert not skills.s.cancel.is_set()


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


def test_task1_keeps_cells_that_already_hold_a_block():
    skills, world, _robot = make_skills({"yellow": (180.0, 120.0), "green": (180.0, -120.0)})
    assert skills.move_block_to_slot("yellow", 0).ok
    result = skills.run_task(1)
    assert result.ok, result.detail
    assert math.dist(world.blocks["yellow"], skills.s.slot_centres[0]) < 1.0
    green_slot = min(range(5), key=lambda i: math.dist(world.blocks["green"], skills.s.slot_centres[i]))
    assert green_slot == 1
    assert result.data["remaining_outside"] == []


def test_task3_is_disabled_by_default():
    skills, _world, robot = make_skills({})
    result = skills.run_task(3)
    assert result.reason == "disabled" and robot.sent_actions == []


# ── board cell addressing ──────────────────────────────────────────────


def _far_cell(skills):
    """An addressable cell far from home, to prove the jog cap does not apply."""
    return max(skills.cells, key=lambda key: math.dist(skills.cells[key], skills.s.arm_position_mm()[:2]))


def test_move_to_cell_crosses_the_board_in_one_move():
    skills, _world, _robot = make_skills({})
    x, y = _far_cell(skills)
    target = skills.cells[(x, y)]
    start = skills.s.arm_position_mm()
    assert math.dist(start[:2], target) > skills.cfg.agent.relative.max_jog_mm
    result = skills.move_to_cell(x, y)
    assert result.ok, result.detail
    assert result.data["cell"] == {"x": x, "y": y}
    reached = skills.s.arm_position_mm()
    assert math.dist(reached[:2], target) < 1.0
    # ... while the same relative request is still refused as too far
    assert skills.move_arm(forward_mm=60.0).reason == "limit_exceeded"


def test_cells_outside_the_fan_are_refused_before_any_motion():
    skills, _world, robot = make_skills({})
    for bad in ((0, 99), (99, 0)):
        result = skills.move_to_cell(*bad)
        assert result.reason == "invalid_arguments"
        assert "부채꼴" in result.detail
    assert robot.sent_actions == []


def test_place_at_cell_releases_on_the_addressed_square():
    skills, world, _robot = make_skills({"yellow": (180.0, 120.0)})
    assert skills.pick_block("yellow").ok
    cell = min(skills.cells, key=lambda key: math.hypot(*skills.cells[key]))
    result = skills.place_at_cell(*cell)
    assert result.ok, result.detail
    assert math.dist(world.blocks["yellow"], skills.cells[cell]) < 25.0


def test_zone_cells_are_refused_and_keep_holding_the_block():
    skills, _world, _robot = make_skills({"yellow": (180.0, 120.0)})
    assert skills.pick_block("yellow").ok
    in_zone = [key for key, xy in skills.cells.items()
               if skills.s.in_zone(xy, skills.cfg.agent.table_zone_margin_mm)]
    assert in_zone, "the zone must overlap the addressable band for this test to mean anything"
    result = skills.place_at_cell(*in_zone[0])
    assert result.reason == "invalid_arguments" and "적재 구역" in result.detail
    assert skills.s.held is not None


def test_a_blocked_cell_puts_the_block_down_beside_it():
    # the blocker sits outside the zone: a zone cell is refused outright
    skills, world, _robot = make_skills({"yellow": (180.0, 120.0), "blue": (180.0, -160.0)})
    occupied = skills.grid.xy_to_cell((180.0, -160.0))
    assert occupied in skills.cells
    assert skills.pick_block("yellow").ok
    result = skills.place_at_cell(*occupied)
    assert result.ok, result.detail
    assert result.data["adjusted_by_mm"] > 0
    assert math.dist(world.blocks["yellow"], world.blocks["blue"]) >= skills.cfg.agent.place_clear_radius_mm


def test_move_block_to_cell_refuses_a_bad_address_without_grasping():
    skills, world, robot = make_skills({"green": (180.0, -120.0)})
    result = skills.move_block_to_cell("green", 0, 99)
    assert result.reason == "invalid_arguments"
    assert world.blocks["green"] == (180.0, -120.0)
    assert all("gripper" not in action for action in robot.sent_actions)

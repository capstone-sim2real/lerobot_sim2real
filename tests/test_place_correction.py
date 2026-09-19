"""The placement loop: measure where a block landed, then command the difference."""

import math

import pytest

from agent_helpers import FakeIk, make_skills
from session.place_correction import PlaceCorrection
from session.relative import offset_xy

ARM = {"frame": "arm", "base_xy_mm": (0.0, 0.0)}


class DroopIk(FakeIk):
    """An arm that reaches a fixed offset away from whatever it is commanded.

    Stands in for what the real rig does: ``release_at`` opens the jaws once
    every joint is within ``motion.arrival_tol`` (3 degrees, measured as
    ~29mm of travel at 283mm reach) and a carried block droops on top of
    that. The offset is in the arm frame, where that error actually lives.
    """

    def __init__(self, forward_mm: float = 0.0, left_mm: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.forward_mm = forward_mm
        self.left_mm = left_mm

    def solve(self, x_mm, y_mm, z_mm, yaw_deg=None, radial_tilt_deg=0.0):
        landed = offset_xy((x_mm, y_mm), self.forward_mm, self.left_mm, **ARM)
        return super().solve(landed[0], landed[1], z_mm, yaw_deg=yaw_deg,
                             radial_tilt_deg=radial_tilt_deg)


def test_correction_is_the_running_mean_of_what_it_still_misses():
    correction = PlaceCorrection()
    target = (200.0, 0.0)
    # the block lands 30mm further out than asked
    assert correction.observe(target, (230.0, 0.0), **ARM)
    assert correction.forward_mm == pytest.approx(-30.0, abs=0.1)
    assert correction.left_mm == pytest.approx(0.0, abs=0.1)
    # commanding the corrected point is what makes it land on target
    assert correction.command_xy(target, **ARM)[0] == pytest.approx(170.0, abs=0.1)

    # with that correction in force the next one lands on target, which
    # confirms the estimate rather than doubling it
    assert correction.observe(target, target, **ARM)
    assert correction.forward_mm == pytest.approx(-30.0, abs=0.1)

    # a residual that survives the correction means the offset was too small
    assert correction.observe(target, (215.0, 0.0), **ARM)
    assert correction.forward_mm == pytest.approx(-35.0, abs=0.1)


def test_a_wild_miss_is_not_treated_as_bias():
    correction = PlaceCorrection(max_sample_mm=80.0)
    assert not correction.observe((200.0, 0.0), (200.0, 150.0), **ARM)
    assert correction.samples == 0 and correction.magnitude_mm == 0.0


def test_the_learned_offset_cannot_run_away():
    correction = PlaceCorrection(max_mm=20.0)
    correction.observe((200.0, 0.0), (270.0, 0.0), **ARM)
    assert correction.magnitude_mm == pytest.approx(20.0, abs=0.01)


def test_a_drooping_arm_corrects_itself_from_the_camera_check():
    droop = 30.0
    skills, world, _robot = make_skills(
        {"yellow": (180.0, 120.0)}, ik=DroopIk(forward_mm=droop), grab_radius_mm=70.0
    )
    cell = (0, -3)  # clear of the fixture zone and its 30mm margin
    target = skills.cells[cell]

    assert skills.pick_block("yellow").ok
    first = skills.place_at_cell(*cell)
    assert first.ok, first.detail
    # the miss is reported, in millimetres and as the cell it really landed in
    assert first.data["miss_mm"] == pytest.approx(droop, abs=3.0)
    assert first.data["measured_cell"] != {"x": cell[0], "y": cell[1]}
    assert "벗어났습니다" in first.detail
    assert skills.place_correction.samples == 1
    assert skills.place_correction.forward_mm == pytest.approx(-droop, abs=3.0)

    # second attempt: the arm is sent short by what it overshot last time
    assert skills.pick_block("yellow").ok
    second = skills.place_at_cell(*cell)
    assert second.ok, second.detail
    assert second.data["miss_mm"] < first.data["miss_mm"] / 3
    assert math.dist(world.blocks["yellow"], target) < droop / 3
    assert second.data["measured_cell"] == {"x": cell[0], "y": cell[1]}


def test_learning_off_keeps_the_arm_exactly_as_it_was():
    cfg_skills, _world, _robot = make_skills({"yellow": (180.0, 120.0)},
                                             ik=DroopIk(forward_mm=30.0), grab_radius_mm=70.0)
    cfg_skills.cfg.agent.place_correction.learn = False
    assert cfg_skills.pick_block("yellow").ok
    result = cfg_skills.place_at_cell(0, -3)
    assert result.ok, result.detail
    # still measured and reported -- only the learning is off
    assert result.data["miss_mm"] > 15.0
    assert "place_correction" not in result.data
    assert cfg_skills.place_correction.samples == 0

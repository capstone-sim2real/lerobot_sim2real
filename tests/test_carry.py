"""Carrying a block across the workspace without dragging it.

The top-down envelope is strongly radius-dependent — wrist_flex sits on its
+-95 deg URDF limit throughout, so lift is bought only by folding the arm
in (measured: ~90mm at 195mm reach, ~50mm at 285mm). A block picked far out
must therefore be retracted before the swing, not carried at the reach it
was picked from.

No placo here: the hover search and the retraction are exercised against a
stub solver.
"""

from __future__ import annotations

import math

import pytest

from pick_stack.config import AppConfig
from pick_stack.control.ik import IkResult
from pick_stack.tools.demo_pick_and_place import highest_reachable_hover, pull_in


class EnvelopeIk:
    """Solves only below a ceiling that shrinks with reach, like the real arm."""

    def __init__(self, ceiling_at_195: float = 90.0):
        self._ceiling_at_195 = ceiling_at_195

    def _ceiling(self, x, y):
        r = math.hypot(x, y)
        return self._ceiling_at_195 - max(0.0, r - 195.0) * 0.45

    def solve(self, x_mm, y_mm, z_mm, yaw_deg=None):
        ok = z_mm <= self._ceiling(x_mm, y_mm)
        return IkResult({"shoulder_pan": 0.0}, 0.5 if ok else 80.0, 0.1)


# --- pull_in -------------------------------------------------------------


def test_pull_in_shortens_reach_but_keeps_the_azimuth():
    x, y = 284.9, -17.3
    nx, ny = pull_in(x, y, 195.0)
    assert math.hypot(nx, ny) == pytest.approx(195.0)
    assert math.atan2(ny, nx) == pytest.approx(math.atan2(y, x))


def test_pull_in_leaves_a_target_already_inside_alone():
    assert pull_in(150.0, 20.0, 195.0) == (150.0, 20.0)


def test_pull_in_disabled_by_zero_radius():
    assert pull_in(284.9, -17.3, 0.0) == (284.9, -17.3)


# --- highest_reachable_hover --------------------------------------------


def test_retracting_buys_lift():
    """The whole point: same block, folded in, reaches materially higher."""
    cfg = AppConfig()
    ik = EnvelopeIk()
    far = highest_reachable_hover(ik, 284.9, -17.3, 10.2, cfg)
    near_x, near_y = pull_in(284.9, -17.3, cfg.motion.transit_apex_radius_mm)
    near = highest_reachable_hover(ik, near_x, near_y, 10.2, cfg)
    assert near > far + 20.0


def test_finer_search_step_finds_more_of_the_envelope():
    cfg_coarse, cfg_fine = AppConfig(), AppConfig()
    cfg_coarse.motion.hover_search_step_mm = 10.0
    cfg_fine.motion.hover_search_step_mm = 1.0
    ik = EnvelopeIk()
    coarse = highest_reachable_hover(ik, 240.0, 0.0, 9.3, cfg_coarse)
    fine = highest_reachable_hover(ik, 240.0, 0.0, 9.3, cfg_fine)
    assert fine >= coarse


def test_floor_is_tested_before_being_returned():
    """The floor used to be returned unverified when nothing above it solved."""
    cfg = AppConfig()
    floor_reachable = EnvelopeIk(ceiling_at_195=1000.0)  # everything solves
    z = highest_reachable_hover(floor_reachable, 200.0, 0.0, 9.3, cfg)
    assert z == pytest.approx(9.3 + cfg.motion.hover_clearance_mm)

    nothing = EnvelopeIk(ceiling_at_195=-1000.0)  # nothing solves, floor included
    z = highest_reachable_hover(nothing, 200.0, 0.0, 9.3, cfg)
    assert z == pytest.approx(9.3 + cfg.motion.hover_min_clearance_mm)


def test_search_lands_on_the_floor_exactly_when_only_the_floor_solves():
    cfg = AppConfig()
    cfg.motion.hover_search_step_mm = 5.0
    floor = 9.3 + cfg.motion.hover_min_clearance_mm

    class OnlyFloor:
        def solve(self, x, y, z, yaw_deg=None):
            return IkResult({"shoulder_pan": 0.0}, 0.5 if z <= floor else 80.0, 0.1)

    assert highest_reachable_hover(OnlyFloor(), 200.0, 0.0, 9.3, cfg) == pytest.approx(floor)

"""TopDownIK tests against the real URDF. Needs placo (extra: lerobot[kinematics]),
so this module is skipped entirely in the lightweight/no-hardware test env
(docs/design.md §9) and only runs under ~/lerobot/.venv.
"""

from __future__ import annotations


import pytest

pytest.importorskip("placo")

from config import IkConfig  # noqa: E402
from control.ik import TopDownIK
from control.grasp import approach_hover_xy  # noqa: E402


@pytest.fixture(scope="module")
def ik() -> TopDownIK:
    return TopDownIK(IkConfig(), project_root=".")


def test_low_right_side_grasp_tries_past_the_three_bad_nearest_seeds(ik: TopDownIK):
    """Regression from the red-block failure on 2026-09-19.

    At this reachable point the three nearest radius/height seeds all drove
    shoulder_pan to its limit and missed by over 170mm. A valid branch lies
    within the first ten candidates.
    """
    result = ik.solve(
        x_mm=156.063,
        y_mm=-111.948,
        z_mm=4.058,
        yaw_deg=10.620,
        radial_tilt_deg=-3.0,
    )
    assert result.position_error_mm < 5.0
    assert result.tilt_error_deg < 6.0


def test_solve_holding_wrist_roll_does_not_drift_across_repeated_jogs(ik: TopDownIK):
    """``yaw_for_wrist_roll_deg``'s single-probe estimate is only ~1.5-2 deg
    accurate per call; a jog skill that re-reads the (already drifted)
    current wrist_roll every step compounds that -- 5 chained 10mm jogs
    spun the jaws ~8 degrees with no rotate command ever issued. The
    corrected solve must hold wrist_roll steady across a chain like that.
    """
    x, y, z = 220.0, 60.0, 60.0
    wrist_roll = ik.solve(x, y, z, yaw_deg=None).joints["wrist_roll"]
    for _ in range(5):
        x += 10.0
        result = ik.solve_holding_wrist_roll(x, y, z, wrist_roll)
        assert abs(result.joints["wrist_roll"] - wrist_roll) < 0.5, result.joints["wrist_roll"]
        wrist_roll = result.joints["wrist_roll"]


def test_side_tilt_follows_shoulder_axis_at_grasp_and_hover(ik: TopDownIK):
    """At the side, base-origin tilt misses while shoulder-axis tilt solves."""
    grasp_xy = (0.0, 300.0)
    hover_xy = approach_hover_xy(grasp_xy, 20.0, 55.0, -60.0, ik.pan_origin_xy_mm)
    grasp = ik.solve(*grasp_xy, 20.0, radial_tilt_deg=-60.0)
    hover = ik.solve(*hover_xy, 55.0, radial_tilt_deg=-60.0)
    assert grasp.position_error_mm < 3.0
    assert hover.position_error_mm < 3.0
    assert abs(grasp.tilt_error_deg - 60.0) < 1.0

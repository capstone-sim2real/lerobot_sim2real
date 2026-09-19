"""TopDownIK tests against the real URDF. Needs placo (extra: lerobot[kinematics]),
so this module is skipped entirely in the lightweight/no-hardware test env
(AGENTS.md §13) and only runs under ~/lerobot/.venv.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("placo")

from config import IkConfig  # noqa: E402
from control.ik import ARM_JOINTS, TopDownIK  # noqa: E402


@pytest.fixture(scope="module")
def ik() -> TopDownIK:
    return TopDownIK(IkConfig(), project_root=".")


def test_solve_within_reach_converges(ik: TopDownIK):
    result = ik.solve(x_mm=220.0, y_mm=60.0, z_mm=10.0, yaw_deg=0.0)
    assert result.position_error_mm < 2.0
    assert result.tilt_error_deg < 2.0
    assert set(result.joints) == set(ARM_JOINTS)


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


def test_solve_beyond_topdown_reach_reports_large_error(ik: TopDownIK):
    # r ~= 336mm, past the ~320mm top-down limit (AGENTS.md §7).
    result = ik.solve(x_mm=300.0, y_mm=150.0, z_mm=10.0, yaw_deg=0.0)
    assert result.position_error_mm > 15.0


def test_small_outward_tilt_releases_far_reach_wrist_saturation(ik: TopDownIK):
    vertical = ik.solve(x_mm=320.0, y_mm=0.0, z_mm=10.0)
    opened = ik.solve(
        x_mm=320.0,
        y_mm=0.0,
        z_mm=10.0,
        radial_tilt_deg=-5.0,
    )

    assert opened.position_error_mm < vertical.position_error_mm
    assert opened.joints["wrist_flex"] < vertical.joints["wrist_flex"]
    assert opened.tilt_error_deg <= 6.0


def test_default_yaw_keeps_wrist_roll_near_neutral(ik: TopDownIK):
    # A fixed base-frame yaw forces wrist_roll to swing ~80 deg across the
    # workspace to hold one absolute direction; the neutral default must not
    # (AGENTS.md §7 — this is what overheated the wrist_roll servo).
    for x, y in ((150.0, -150.0), (220.0, 0.0), (220.0, 150.0), (290.0, -80.0)):
        result = ik.solve(x_mm=x, y_mm=y, z_mm=10.0)
        assert abs(result.joints["wrist_roll"]) < 15.0, (x, y, result.joints["wrist_roll"])


def test_grasp_yaw_matches_the_block_and_prefers_the_workspace_tangent(ik: TopDownIK):
    """Turning the jaws to a block angle must prefer the local tangent.

    Holding a *fixed base-frame* yaw is what swung wrist_roll ~80 deg across
    the workspace and overheated the servo on 2026-08-31 (AGENTS.md §7).
    Square symmetry lets the runtime choose the face axis nearest that
    tangent, avoiding an unnecessary perpendicular wrist orientation.
    """
    for x, y in ((220.0, -60.0), (200.0, 120.0), (260.0, -140.0), (170.0, 170.0)):
        neutral = ik.neutral_yaw_deg(x, y, 20.0)
        tangent = math.degrees(math.atan2(y, x)) + 90.0
        for block_angle in (0.0, 20.0, 40.0, 60.0, 80.0):
            yaw = ik.grasp_yaw_deg(x, y, 20.0, block_angle)
            # a square grasps identically every 90 deg, so the commanded yaw
            # must be congruent to the block angle
            assert abs(((yaw - block_angle) + 45.0) % 90.0 - 45.0) < 1e-6
            assert abs(((yaw - tangent) + 180.0) % 360.0 - 180.0) <= 45.0 + 1e-6
            result = ik.solve(x, y, 20.0, yaw_deg=yaw)
            assert result.position_error_mm < 5.0, (x, y, block_angle)
            assert abs(result.joints["wrist_roll"]) < 60.0, (x, y, block_angle, neutral)


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

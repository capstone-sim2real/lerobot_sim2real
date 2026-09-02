"""TopDownIK tests against the real URDF. Needs placo (extra: lerobot[kinematics]),
so this module is skipped entirely in the lightweight/no-hardware test env
(AGENTS.md §13) and only runs under ~/lerobot/.venv.
"""

from __future__ import annotations

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


def test_grasp_yaw_matches_the_block_without_winding_up_the_wrist(ik: TopDownIK):
    """Turning the jaws to a block angle must stay near the neutral wrist.

    Holding a *fixed base-frame* yaw is what swung wrist_roll ~80 deg across
    the workspace and overheated the servo on 2026-08-31 (AGENTS.md §7).
    grasp_yaw_deg folds the block angle mod 90, which bounds the excursion.
    """
    for x, y in ((220.0, -60.0), (200.0, 120.0), (260.0, -140.0), (170.0, 170.0)):
        neutral = ik.neutral_yaw_deg(x, y, 20.0)
        for block_angle in (0.0, 20.0, 40.0, 60.0, 80.0):
            yaw = ik.grasp_yaw_deg(x, y, 20.0, block_angle)
            # a square grasps identically every 90 deg, so the commanded yaw
            # must be congruent to the block angle
            assert abs(((yaw - block_angle) + 45.0) % 90.0 - 45.0) < 1e-6
            assert abs(((yaw - neutral) + 180.0) % 360.0 - 180.0) <= 45.0 + 1e-6
            result = ik.solve(x, y, 20.0, yaw_deg=yaw)
            assert result.position_error_mm < 5.0, (x, y, block_angle)
            assert abs(result.joints["wrist_roll"]) < 50.0, (x, y, block_angle)

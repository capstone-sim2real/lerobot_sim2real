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


def test_default_yaw_keeps_wrist_roll_near_neutral(ik: TopDownIK):
    # A fixed base-frame yaw forces wrist_roll to swing ~80 deg across the
    # workspace to hold one absolute direction; the neutral default must not
    # (AGENTS.md §7 — this is what overheated the wrist_roll servo).
    for x, y in ((150.0, -150.0), (220.0, 0.0), (220.0, 150.0), (290.0, -80.0)):
        result = ik.solve(x_mm=x, y_mm=y, z_mm=10.0)
        assert abs(result.joints["wrist_roll"]) < 15.0, (x, y, result.joints["wrist_roll"])

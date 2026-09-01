"""TopDownIK tests against the real URDF. Needs placo (extra: lerobot[kinematics]),
so this module is skipped entirely in the lightweight/no-hardware test env
(AGENTS.md §13) and only runs under ~/lerobot/.venv.
"""

from __future__ import annotations

import pytest

pytest.importorskip("placo")

from pick_stack.config import IkConfig  # noqa: E402
from pick_stack.control.ik import ARM_JOINTS, TopDownIK  # noqa: E402


@pytest.fixture(scope="module")
def ik() -> TopDownIK:
    return TopDownIK(IkConfig(), project_root=".")


def test_solve_within_reach_converges(ik: TopDownIK):
    result = ik.solve(x_mm=220.0, y_mm=60.0, z_mm=10.0, yaw_deg=0.0)
    assert result.position_error_mm < 2.0
    assert result.tilt_error_deg < 2.0
    assert set(result.joints) == set(ARM_JOINTS)


def test_solve_matches_pan_sign_convention(ik: TopDownIK):
    # AGENTS.md §7: shoulder_pan sign is opposite atan2(y, x).
    result = ik.solve(x_mm=200.0, y_mm=100.0, z_mm=10.0, yaw_deg=0.0)
    assert result.joints["shoulder_pan"] < 0.0


def test_solve_beyond_topdown_reach_reports_large_error(ik: TopDownIK):
    # r ~= 336mm, past the ~320mm top-down limit (AGENTS.md §7).
    result = ik.solve(x_mm=300.0, y_mm=150.0, z_mm=10.0, yaw_deg=0.0)
    assert result.position_error_mm > 15.0


def test_seed_cache_is_reused(ik: TopDownIK, tmp_path):
    import time

    cfg = IkConfig(seed_cache_path=str(tmp_path / "seed.npz"))
    fresh = TopDownIK(cfg, project_root=".")
    t0 = time.perf_counter()
    fresh.solve(x_mm=200.0, y_mm=0.0, z_mm=10.0)
    first = time.perf_counter() - t0

    reloaded = TopDownIK(cfg, project_root=".")
    t0 = time.perf_counter()
    reloaded.solve(x_mm=200.0, y_mm=0.0, z_mm=10.0)
    second = time.perf_counter() - t0
    assert second < first

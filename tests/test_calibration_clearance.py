from config import CalibrationClearanceConfig
from session.calibration_clearance import directional_clearance, top_view_clearance


def square(cx, cy, half=20):
    return [(cx-half, cy-half), (cx+half, cy-half),
            (cx+half, cy+half), (cx-half, cy+half)]


def test_neighbour_blocks_only_its_matching_grasp_direction():
    cfg = CalibrationClearanceConfig(uncertainty_mm=15)
    obstacles = {"right": square(50, 0)}
    assert not directional_clearance(square(0, 0), obstacles, cfg, 0)["clear"]
    assert directional_clearance(square(0, 0), obstacles, cfg, 90)["clear"]
    assert top_view_clearance(square(0, 0), obstacles, cfg)["clear"]


def test_both_directions_must_be_blocked_before_target_is_rejected():
    cfg = CalibrationClearanceConfig(uncertainty_mm=15)
    obstacles = {"right": square(50, 0), "above": square(0, 50)}
    result = top_view_clearance(square(0, 0), obstacles, cfg)
    assert not result["clear"]
    assert not result["orientations"]["x"]["clear"]
    assert not result["orientations"]["y"]["clear"]


def test_shifted_jaw_footprint_rejects_target_top_contact():
    cfg = CalibrationClearanceConfig(
        uncertainty_mm=15,
        jaw_inner_clearance_mm=2,
    )
    result = directional_clearance(
        square(0, 0), {}, cfg, 0, jaw_center_mm=(10, 0)
    )
    assert not result["clear"]
    assert result["target_overlap"]
    assert result["target_jaw_hits"] == [0]


def test_centred_jaw_footprint_clears_target_top():
    cfg = CalibrationClearanceConfig(
        uncertainty_mm=15,
        jaw_inner_clearance_mm=2,
    )
    result = directional_clearance(
        square(0, 0), {}, cfg, 0, jaw_center_mm=(0, 0)
    )
    assert result["clear"]
    assert not result["target_overlap"]

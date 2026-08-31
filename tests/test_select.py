import numpy as np
import pytest

from pick_stack.config import SelectConfig
from pick_stack.perception import BlockDetection, PlaneCalibration, select_target, target_id_for


def det(color, x, y):
    return BlockDetection(
        color=color, center_mm=(x, y), area_mm2=1600.0, aspect=1.0, solidity=1.0, fill=1.0,
        box_mm=[(x - 20, y - 20), (x + 20, y - 20), (x + 20, y + 20), (x - 20, y + 20)],
    )


def make_calib(base=(300.0, 380.0), zone=((20, 20), (220, 20), (220, 120), (20, 120))):
    return PlaneCalibration(
        H=np.eye(3), image_size=(600, 400), square_mm=25.0,
        base_xy_mm=base, zone_polygon_mm=[tuple(map(float, p)) for p in zone],
    )


def test_nearest_first():
    detections = [det("blue", 100, 300), det("red", 290, 360), det("green", 500, 200)]
    result = select_target(detections, make_calib(), SelectConfig())
    assert result.target.color == "red"
    assert result.remaining == 3
    assert result.target_id == target_id_for(result.target, 40.0)


def test_zone_blocks_excluded():
    inside = det("blue", 100, 70)  # inside zone polygon
    near = det("wood", 230, 70)  # 10mm outside zone edge, within 20mm margin
    outside = det("green", 500, 300)
    result = select_target([inside, near, outside], make_calib(), SelectConfig())
    assert result.target.color == "green"
    assert result.remaining == 1


def test_skipped_targets_excluded():
    d1, d2 = det("red", 290, 360), det("blue", 100, 300)
    cfg = SelectConfig()
    skipped = {target_id_for(d1, cfg.target_cell_mm)}
    result = select_target([d1, d2], make_calib(), cfg, skipped=skipped)
    assert result.target.color == "blue"
    assert result.remaining == 1


def test_no_targets_left():
    result = select_target([det("red", 100, 70)], make_calib(), SelectConfig())
    assert result.target is None
    assert result.target_id is None
    assert result.remaining == 0


def test_base_required():
    with pytest.raises(ValueError, match="base"):
        select_target([det("red", 100, 300)], make_calib(base=None), SelectConfig())


def test_stable_target_ids():
    a = det("red", 118, 279)  # both quantize to the same 40mm cell
    b = det("red", 121, 282)
    assert target_id_for(a, 40.0) == target_id_for(b, 40.0)

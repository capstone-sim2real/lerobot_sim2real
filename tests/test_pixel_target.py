import math

import numpy as np
import pytest
from agent_helpers import make_skills
from control.grasp import near_vertical_pick_radius_mm
from session.pixel_target import calibration_id, pixel_preview_config, resolve_pixel


def setup(tmp_path):
    skills, world, robot = make_skills({})
    path = tmp_path / "calib.json"
    skills.s.calib.save(path)
    skills.cfg.perception.calibration_path = str(path)
    point = skills.s.calib.board_to_pixel([(180.0, 80.0)])[0]
    return skills, robot, int(round(point[0])), int(round(point[1])), calibration_id(skills.s.calib)


def test_pixel_plane_and_stale_bounds(tmp_path):
    sk, robot, u, v, key = setup(tmp_path)
    target = resolve_pixel(sk.cfg, sk.s.calib, u, v, key)
    assert target["z_mm"] == sk.s.grasp_z_mm
    assert target["x_mm"] == pytest.approx(180, abs=1)
    for args in [(-1, v, key), (u, v, "stale"), (True, v, key), (u, sk.s.calib.image_size[1], key)]:
        with pytest.raises(ValueError):
            resolve_pixel(sk.cfg, sk.s.calib, *args)
    assert not robot.sent_actions


def test_ik_failure_is_preplanned_before_first_motion(tmp_path, monkeypatch):
    sk, robot, u, v, key = setup(tmp_path)
    original = sk.s.ik.solve_holding_wrist_roll
    n = 0

    def solve(*args, **kwargs):
        nonlocal n
        n += 1
        result = original(*args, **kwargs)
        if n == 3:
            result.position_error_mm = 1000
        return result

    monkeypatch.setattr(sk.s.ik, "solve_holding_wrist_roll", solve)
    result = sk.move_to_pixel(u, v, key)
    assert not result.ok and result.reason == "ik_gate"
    assert not robot.sent_actions


def test_near_vertical_line_matches_lift_tilt_boundary(tmp_path):
    sk, _, _, _, _ = setup(tmp_path)
    preview = pixel_preview_config(sk.cfg, sk.s.calib)
    green = preview["near_vertical_arc_px"]
    outer = preview["reach_arc_px"]
    middle = len(green) // 2
    green_mm = sk.s.calib.pixel_to_board(np.asarray([green[middle]]))[0]
    outer_mm = sk.s.calib.pixel_to_board(np.asarray([outer[middle]]))[0]
    base = sk.s.base_xy
    expected = min(
        near_vertical_pick_radius_mm(sk.cfg.task1),
        math.dist(outer_mm, base),
    )
    assert math.dist(green_mm, base) == pytest.approx(expected, abs=0.1)


def test_selected_pixel_without_cv_is_usable_and_expires(tmp_path):
    from session.primitives import PrimitiveSkills

    base, robot, u, v, key = setup(tmp_path)
    sk = PrimitiveSkills(base.s)
    before = len(robot.sent_actions)
    result = sk.select_pixel_target(u, v, key)
    assert result.ok and result.data["geometry_assumed"]
    assert result.data.get("matched_color") is None
    assert len(robot.sent_actions) == before
    oid, obs = result.data["object_id"], result.data["observation_id"]
    block = sk._object(oid, obs)
    assert block.center_mm == pytest.approx((180, 80), abs=1)
    assert sk.open_gripper().ok
    assert sk.move_relative(up_mm=50).ok
    assert sk.move_to_target("object", "pregrasp", oid, obs).ok
    assert sk.align_gripper(oid, obs).ok
    sk.observe_scene()
    with pytest.raises(ValueError):
        sk._object(oid, obs)


def test_invalid_selected_pixel_cannot_create_target(tmp_path):
    from session.primitives import PrimitiveSkills

    base, robot, u, v, key = setup(tmp_path)
    sk = PrimitiveSkills(base.s)
    assert not sk.select_pixel_target(u, v, "stale").ok
    assert "selected_1" not in sk._objects
    assert not robot.sent_actions


def test_composite_source_and_pixel_tower_use_exact_target(tmp_path, monkeypatch):
    from session.primitives import PrimitiveSkills
    from session.results import SkillResult

    base, robot, u, v, key = setup(tmp_path)
    sk = PrimitiveSkills(base.s)
    source = dict(u=u, v=v, calibration_id=key)

    def transfer(color, slot):
        assert color == "selected"
        assert sk._observe_target(color).ok
        assert sk._observed_scene.find(color) is not None
        return SkillResult(True, "move_block_to_slot", "released")

    monkeypatch.setattr(sk, "_move_block_to_slot_once", transfer)
    assert sk.move_block_to_slot(slot="top-left", source=source).ok
    assert sk._transfer_source is None
    previous = sk.s._stack

    def stack(color, floor):
        assert sk.s.stack.raw_xy_mm == pytest.approx((180, 80), abs=1)
        assert sk.s.stack.stack_xy_mm == sk.s.stack.raw_xy_mm
        assert floor == 1 and color == "selected"
        return SkillResult(True, "stack_block_to_floor", "released")

    monkeypatch.setattr(sk, "_stack_block_to_floor_once", stack)
    assert sk.stack_block_to_floor(source=source, floor=1, destination=source).ok
    assert sk.s._stack is previous
    assert not robot.sent_actions

"""Shared calibrated primitives with mock motors; not physical validation."""
from dataclasses import replace
from unittest.mock import Mock

from agent_helpers import make_skills
from session.primitives import PrimitiveSkills
from session.results import SkillResult


def setup(tmp_path):
    base, world, robot = make_skills({"green": (180., 0.)})
    base.cfg.agent.primitives.calibrated_pick = True
    base.cfg.agent.collection.root = str(tmp_path)
    base.cfg.agent.calibration_clearance.expected_colors = ["green"]
    sk = PrimitiveSkills(base.s)
    cal = sk._calibration()
    # Fake IK has no URDF meshes. Only the geometry gate is stubbed here;
    # planning, correction, descent and motor IO execute the shared code.
    def clear(*args):
        cal._approach_joints = None
        return {"clear": True, "reason": "ok"}
    cal._clearance_gate = clear
    assert sk.observe_scene().ok
    assert sk.open_gripper().ok
    return sk, cal, robot


def approach(sk):
    return sk.move_to_target("object", "pregrasp", "green_1", sk.observation_id)


def test_shared_calibration_without_second_io_or_hidden_close(tmp_path):
    sk, cal, robot = setup(tmp_path)
    assert sk.s is cal.s and sk.s.robot is cal.s.robot
    sk.s.motion.open_gripper = Mock(side_effect=AssertionError("pregrasp must not open jaws"))
    assert approach(sk).ok
    assert sk._target[-1] == "pregrasp"
    assert not sk.close_gripper().ok
    before = len(robot.sent_actions)
    aligned = sk.align_gripper("green_1", sk.observation_id)
    assert aligned.ok and aligned.data["alignment_already_applied"]
    assert len(robot.sent_actions) == before
    result = sk.move_to_target("object", "grasp", "green_1", sk.observation_id)
    assert result.ok and cal.descent_ready
    assert sk.close_gripper().ok


def test_hover_shortfall_runs_existing_bounded_correction(tmp_path):
    sk, cal, robot = setup(tmp_path)
    original = cal.calibration_prepare
    def shortfall(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result.ok
        cal.attempt = None
        return SkillResult(False, "calibration_prepare", "grasp_blocked",
                           data={"stop_reason": "hover_not_settled"})
    cal.calibration_prepare = shortfall
    original_correction = cal.calibration_correct_hover
    cal.calibration_correct_hover = Mock(wraps=original_correction)
    assert approach(sk).ok
    cal.calibration_correct_hover.assert_called_once_with(dry_run=False)


def test_geometry_rejection_prevents_motion_and_close(tmp_path):
    sk, cal, robot = setup(tmp_path)
    cal._clearance_gate = lambda *a: {"clear": False, "reason": "neighbour_clearance"}
    before = len(robot.sent_actions)
    assert not approach(sk).ok
    assert not sk.close_gripper().ok
    assert len(robot.sent_actions) == before


def test_load_stop_during_calibrated_descent_cannot_close(tmp_path):
    sk, cal, robot = setup(tmp_path)
    assert approach(sk).ok
    count = [0]
    def loads():
        count[0] += 1
        return {j: (0 if count[0] <= 2 else 1000) for j in sk.cfg.sensing.contact_joints}
    robot.read_loads = loads
    result = sk.move_to_target("object", "grasp", "green_1", sk.observation_id)
    assert not result.ok and result.data["stop_reason"] == "load_increase"
    assert not sk.close_gripper().ok


def test_observe_invalidates_prepared_descent(tmp_path):
    sk, cal, robot = setup(tmp_path)
    assert approach(sk).ok
    assert sk.observe_scene().ok
    assert cal.attempt is None and not sk._pick_ready

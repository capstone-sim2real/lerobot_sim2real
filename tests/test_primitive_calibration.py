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


def test_partial_descent_does_not_authorize_early_close(tmp_path):
    sk, cal, robot = setup(tmp_path)
    assert approach(sk).ok
    result = sk.descend_step(2)
    assert result.ok and not result.data['depth_ready']
    assert not sk.close_gripper().ok
    assert cal.attempt is not None


def test_transit_rejects_measured_path_deviation(tmp_path):
    import pytest
    sk, cal, robot = setup(tmp_path)
    sk.s.arm_position_mm = Mock(side_effect=[(150.,0.,60.), (150.,0.,60.), (200.,0.,65.)])
    with pytest.raises(TimeoutError, match="path corridor"):
        sk.move_relative(up_mm=30)


def test_held_motion_keeps_calibrated_pick_tilt(tmp_path):
    sk, cal, robot = setup(tmp_path)
    assert approach(sk).ok
    assert sk.move_to_target("object", "grasp", "green_1", sk.observation_id).ok
    cal.plan.radial_tilt_deg = -3.0
    assert sk.close_gripper().ok
    sk.s.ik.solve_holding_wrist_roll = Mock(wraps=sk.s.ik.solve_holding_wrist_roll)
    assert sk.move_relative(up_mm=30).ok
    assert all(call.kwargs["radial_tilt_deg"] == -3.0
               for call in sk.s.ik.solve_holding_wrist_roll.call_args_list)
    sk.s.held = None
    sk._solve((180.,0.,60.))
    assert sk.s.ik.solve_holding_wrist_roll.call_args.kwargs["radial_tilt_deg"] == 0.0


def test_loaded_transit_applies_only_one_bounded_tracking_correction(tmp_path):
    sk, cal, robot = setup(tmp_path)
    assert approach(sk).ok
    assert sk.move_to_target("object", "grasp", "green_1", sk.observation_id).ok
    assert sk.close_gripper().ok
    sk.cfg.motion.grasp_hover_arrival_tol = 1.0
    original = sk.s.player.move_through
    calls = []
    def lagged(points, **kwargs):
        calls.append(points)
        result = original(points, **kwargs)
        if len(calls) == 1:
            robot.joints["elbow_flex"] -= 2.0
        return result
    sk.s.player.move_through = lagged
    sk.s.player.settle = lambda *a, **k: (2.0, False)
    result = sk.move_relative(up_mm=30)
    assert result.ok
    assert len(calls) == 2
    assert result.data["tracking_correction_deg"]["elbow_flex"] == 2.0
    assert max(map(abs,result.data["tracking_correction_deg"].values())) <= 3.0


def test_magnitude_contact_ignores_load_sign_reversal():
    from config import SensingConfig
    from control.sensing import ContactMonitor
    robot = Mock()
    cfg = SensingConfig(contact_joints=["elbow_flex"], contact_baseline_samples=1)
    robot.read_loads.side_effect = [{"elbow_flex":132.}, {"elbow_flex":-60.}, {"elbow_flex":-220.}]
    monitor = ContactMonitor(robot,cfg,magnitude_increase=True)
    monitor.start()
    assert not monitor.check().contact
    assert monitor.check().contact


def test_high_table_contact_never_allows_release(tmp_path, monkeypatch):
    from control.sensing import ContactReading
    sk, cal, robot = setup(tmp_path)
    assert approach(sk).ok
    assert sk.move_to_target("object", "grasp", "green_1", sk.observation_id).ok
    assert sk.close_gripper().ok
    assert sk.move_relative(up_mm=50).ok
    cell = min(sk.cells,key=lambda k: sum((a-b)**2 for a,b in zip(sk.cells[k],(180.,0.))))
    assert sk.move_to_target("cell","preplace",x=cell[0],y=cell[1]).ok
    monkeypatch.setattr("session.primitives.ContactMonitor.check",lambda self:ContactReading(True))
    assert not sk.descend_until_contact(20).ok
    assert not sk.open_gripper().ok

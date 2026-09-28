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
    result = approach(sk)
    assert not result.ok and result.reason == "neighbour_clearance"
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


def test_loaded_transit_skips_empty_arm_tracking_correction(tmp_path):
    sk, cal, robot = setup(tmp_path)
    assert approach(sk).ok
    assert sk.move_to_target("object", "grasp", "green_1", sk.observation_id).ok
    assert sk.close_gripper().ok
    original = sk.s.player.move_through
    calls = []
    def lagged(points, **kwargs):
        calls.append(points)
        result = original(points, **kwargs)
        robot.joints["elbow_flex"] -= 2.0
        return result
    sk.s.player.move_through = lagged
    result = sk.move_relative(up_mm=30)
    assert result.ok
    assert len(calls) == 1
    assert result.data["tracking_correction_deg"] == {}


def test_magnitude_contact_uses_local_free_motion_baseline():
    from config import SensingConfig
    from control.sensing import ContactMonitor
    robot = Mock()
    cfg = SensingConfig(contact_joints=["elbow_flex"], contact_baseline_samples=1)
    robot.read_loads.side_effect = [
        {"elbow_flex": 20.0},
        {"elbow_flex": 60.0},
        {"elbow_flex": 100.0},
        {"elbow_flex": 220.0},
    ]
    monitor = ContactMonitor(robot, cfg, magnitude_increase=True)
    monitor.start()
    first = monitor.check()
    assert not first.contact
    monitor.rebase(first.loads)
    second = monitor.check()
    assert not second.contact
    monitor.rebase(second.loads)
    assert monitor.check().contact



def test_preplace_reuses_task1_far_slot_tilt_and_gate(tmp_path):
    from control.task1_transport import place_tilt_deg

    sk, cal, robot = setup(tmp_path)
    assert approach(sk).ok
    assert sk.move_to_target("object", "grasp", "green_1", sk.observation_id).ok
    assert sk.close_gripper().ok
    sk._move = Mock(return_value=SkillResult(True, "move_to_target", "moved"))

    result = sk.move_to_target("slot", "preplace", slot="top-left")

    assert result.ok
    expected = place_tilt_deg(sk.s.slot_centres[0], sk.s.base_xy, sk.cfg)
    assert sk._move.call_args.kwargs["radial_tilt_deg"] == expected
    assert sk._move.call_args.kwargs["max_ik_error_mm"] == sk.cfg.ik.max_position_error_mm

def test_preplace_backs_off_correction_only_after_preflight_ik_failure(tmp_path):
    sk, cal, robot = setup(tmp_path)
    assert approach(sk).ok
    assert sk.move_to_target("object", "grasp", "green_1", sk.observation_id).ok
    assert sk.close_gripper().ok
    sk._move = Mock(side_effect=[
        SkillResult(False, "move_to_target", "ik_gate"),
        SkillResult(True, "move_to_target", "moved"),
    ])

    result = sk.move_to_target("slot", "preplace", slot="top-left")

    assert result.ok
    assert result.data["place_correction_scale"] == 0.75
    assert [row["scale"] for row in result.data["place_correction_attempts"]] == [1.0, 0.75]
    assert sk._move.call_count == 2


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

def test_tilted_held_pick_reverses_approach_before_lift(tmp_path):
    sk, cal, robot = setup(tmp_path)
    assert approach(sk).ok
    assert sk.move_to_target("object", "grasp", "green_1", sk.observation_id).ok
    grasp = cal.attempt
    hover = sk.s.ik.solve(160.0, 0.0, 45.0, radial_tilt_deg=-30.0)
    cal.attempt = replace(grasp, hover=hover, hover_xy_mm=(160.0, 0.0), hover_z_mm=45.0)
    cal.plan.radial_tilt_deg = -30.0
    assert sk.close_gripper().ok
    original = sk.s.ik.solve_holding_wrist_roll
    def strict_vertical(x, y, z, wrist_roll_deg, **kwargs):
        result = original(x, y, z, wrist_roll_deg, **kwargs)
        if x > 175.0 and 10.0 < z < 45.0:
            return replace(result, position_error_mm=5.6)
        return result
    sk.s.ik.solve_holding_wrist_roll = strict_vertical
    # Exercise planning and the clearance branch without simulating motor time.
    def arrive(goal, **kwargs):
        robot.joints.update(goal)
        return robot.read_joints()
    def traverse(goals, **kwargs):
        return arrive(goals[-1])
    sk.s.player.move_to = arrive
    sk.s.player.move_through = traverse
    result = sk.move_relative(up_mm=50.0)
    assert result.ok
    assert result.data["reverse_pick_retreat"]
    assert result.data["lateral_clearance_ready"]
    assert sk.s.arm_position_mm() == (160.0, 0.0, 60.0)
    assert sk.s.held is not None

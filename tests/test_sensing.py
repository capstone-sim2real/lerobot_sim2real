import pytest

from pick_stack.config import SensingConfig
from pick_stack.control import ContactMonitor, MockRobotIO, check_grasp


def make_cfg(**kwargs):
    defaults = dict(grasp_settle_s=0.0, sample_interval_s=0.0, grasp_samples=3, contact_baseline_samples=3)
    return SensingConfig(**{**defaults, **kwargs})


def make_robot(gripper_pos: float, gripper_load: int):
    robot = MockRobotIO(initial_joints={"gripper": gripper_pos})
    robot.loads["gripper"] = gripper_load
    return robot


def test_grasp_held_block():
    # 20mm block: gripper stops early, load stays high
    result = check_grasp(make_robot(25.0, -300), make_cfg())
    assert result.grasped
    assert result.pos_says_held and result.load_says_held
    assert result.gripper_load_abs == 300  # sign stripped


def test_grasp_empty_hand():
    result = check_grasp(make_robot(3.0, 15), make_cfg())
    assert not result.grasped
    assert not result.pos_says_held and not result.load_says_held


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("position_only", True),
        ("load_only", False),
        ("position_and_load", False),
        ("position_or_load", True),
    ],
)
def test_grasp_modes_disagreeing_signals(mode, expected):
    # position says held, load says empty (e.g. very light touch)
    result = check_grasp(make_robot(25.0, 15), make_cfg(grasp_check_mode=mode))
    assert result.grasped is expected


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="grasp_check_mode"):
        check_grasp(make_robot(25.0, 300), make_cfg(grasp_check_mode="vibes"))


def test_contact_monitor_detects_spike():
    robot = MockRobotIO()
    robot.loads.update({"shoulder_lift": 100, "elbow_flex": -50})
    monitor = ContactMonitor(robot, make_cfg(contact_load_delta=80.0))
    baseline = monitor.start()
    assert baseline == {"shoulder_lift": 100.0, "elbow_flex": -50.0}

    assert not monitor.check().contact  # unchanged

    robot.loads["shoulder_lift"] = 150  # +50: below delta
    assert not monitor.check().contact

    robot.loads["elbow_flex"] = -140  # |delta|=90: contact (load may drop on contact)
    reading = monitor.check()
    assert reading.contact
    assert reading.deltas["elbow_flex"] == pytest.approx(90.0)


def test_contact_monitor_requires_start():
    monitor = ContactMonitor(MockRobotIO(), make_cfg())
    with pytest.raises(RuntimeError, match="start"):
        monitor.check()

import numpy as np
import pytest

from pick_stack.control import MockRobotIO


def test_mock_robot_roundtrip():
    robot = MockRobotIO(initial_joints={"gripper": 50.0})
    robot.connect()
    assert robot.is_connected
    assert robot.read_joints()["gripper"] == 50.0

    sent = robot.send_joints({"shoulder_pan": 10.0, "gripper": 0.0})
    assert sent == {"shoulder_pan": 10.0, "gripper": 0.0}
    assert robot.read_joints()["shoulder_pan"] == 10.0
    assert robot.sent_actions == [{"shoulder_pan": 10.0, "gripper": 0.0}]

    robot.disconnect()
    assert not robot.is_connected


def test_mock_observation_contains_joints_and_frames():
    robot = MockRobotIO()
    robot.frames["top"] = np.zeros((480, 640, 3), dtype=np.uint8)
    obs = robot.read_observation()
    assert obs["shoulder_pan.pos"] == 0.0
    assert obs["top"].shape == (480, 640, 3)


def test_real_robot_requires_connect():
    from pick_stack.config import RobotIOConfig
    from pick_stack.control import So101RobotIO

    io = So101RobotIO(RobotIOConfig())
    assert not io.is_connected
    with pytest.raises(RuntimeError, match="connect"):
        io.read_joints()

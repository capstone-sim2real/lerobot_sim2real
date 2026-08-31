import pytest

from pick_stack.config import MotionConfig, SensingConfig
from pick_stack.control import MockRobotIO, MotionController, PoseRegistry
from pick_stack.control.robot_io import JOINT_NAMES


def full_pose(**kwargs):
    pose = {j: 0.0 for j in JOINT_NAMES}
    pose.update(kwargs)
    return pose


def make_registry():
    return PoseRegistry(
        {
            "home": full_pose(),
            "retreat": full_pose(shoulder_lift=-20.0),
            "zone_approach": full_pose(shoulder_pan=30.0, shoulder_lift=-10.0),
            "slot_0": full_pose(shoulder_pan=30.0, shoulder_lift=-40.0),
            "slot_1": full_pose(shoulder_pan=35.0, shoulder_lift=-40.0),
            "slot_2": full_pose(shoulder_pan=40.0, shoulder_lift=-40.0),
            "slot_3": full_pose(shoulder_pan=30.0, shoulder_lift=-45.0),
            "slot_4": full_pose(shoulder_pan=35.0, shoulder_lift=-45.0),
            "tower_approach": full_pose(shoulder_pan=-30.0),
            "tower_descent_0": full_pose(shoulder_pan=-30.0, shoulder_lift=-10.0),
            "tower_descent_1": full_pose(shoulder_pan=-30.0, shoulder_lift=-30.0),
        }
    )


def make_controller(robot, sensing_kwargs=None):
    motion_cfg = MotionConfig(fps=0.0, gripper_action_wait_s=0.0, place_settle_s=0.0)
    sensing_cfg = SensingConfig(
        sample_interval_s=0.0, contact_baseline_samples=1, gripper_open_pos=50.0,
        **(sensing_kwargs or {}),
    )
    return MotionController(robot, make_registry(), motion_cfg, sensing_cfg)


def test_validate_poses():
    robot = MockRobotIO()
    controller = make_controller(robot)
    controller.validate_poses(task=1)
    controller.validate_poses(task=2)

    empty = MotionController(
        robot, PoseRegistry(), MotionConfig(), SensingConfig()
    )
    with pytest.raises(KeyError, match="home"):
        empty.validate_poses(task=1)


def test_place_in_slot_sequence():
    robot = MockRobotIO()
    robot.connect()
    controller = make_controller(robot)
    controller.place_in_slot(0)

    # descended to the slot, released, lifted back out
    gripper_cmds = [a["gripper"] for a in robot.sent_actions if set(a) == {"gripper"}]
    assert gripper_cmds == [50.0]
    assert robot.read_joints()["shoulder_lift"] == pytest.approx(-10.0)  # back at zone_approach

    with pytest.raises(IndexError):
        controller.place_in_slot(5)


class ContactAfterNSteps(MockRobotIO):
    """Load spike on elbow_flex after the arm has descended n steps."""

    def __init__(self, n_steps):
        super().__init__()
        self.n_steps = n_steps
        self._descent_steps = 0

    def send_joints(self, positions):
        if set(positions) != {"gripper"}:
            self._descent_steps += 1
            if self._descent_steps >= self.n_steps:
                self.loads["elbow_flex"] = 500
        return super().send_joints(positions)


def test_stack_place_stops_on_contact():
    robot = ContactAfterNSteps(n_steps=25)
    robot.connect()
    controller = make_controller(robot, {"contact_load_delta": 80.0})
    assert controller.stack_place() is True
    # released after contact, and returned to the approach pose
    assert any(set(a) == {"gripper"} for a in robot.sent_actions)
    assert robot.read_joints()["shoulder_pan"] == pytest.approx(-30.0)
    # contact fired mid-ladder: it never commanded the ladder bottom (-30)
    lifts = [a["shoulder_lift"] for a in robot.sent_actions if "shoulder_lift" in a]
    assert min(lifts) > -30.0


def test_stack_place_without_contact_releases_at_bottom():
    robot = MockRobotIO()  # loads never change
    robot.connect()
    controller = make_controller(robot, {"contact_load_delta": 80.0})
    assert controller.stack_place() is False
    lifts = [a["shoulder_lift"] for a in robot.sent_actions if "shoulder_lift" in a]
    assert min(lifts) == pytest.approx(-30.0)  # reached ladder bottom
    assert any(set(a) == {"gripper"} for a in robot.sent_actions)

"""Continuous transit preserves tick bounds and aborts promptly."""
import pytest
from config import MotionConfig
from control.trajectory import TrajectoryPlayer

class Robot:
    def __init__(self):
        self.pose = {"joint": 7.0}
        self.sent = []
    def read_joints(self):
        return dict(self.pose)
    def send_joints(self, pose):
        self.sent.append(dict(pose))
        self.pose.update(pose)

def test_measured_start_tick_bounds_and_no_knot_settling():
    robot = Robot()
    player = TrajectoryPlayer(robot, MotionConfig(fps=0, max_step_per_tick=2))
    player.move_to = lambda *a, **k: pytest.fail("must not stop per knot")
    player.settle = lambda *a, **k: pytest.fail("must not settle per knot")
    result = player.move_through([{"joint": 11.}, {"joint": 15.}])
    assert [p["joint"] for p in robot.sent] == [9., 11., 13., 15.]
    assert result == {"joint": 15.}

def test_progress_guard_aborts_before_next_command():
    robot = Robot()
    player = TrajectoryPlayer(robot, MotionConfig(fps=0, max_step_per_tick=2))
    def stop():
        raise TimeoutError("path deviation")
    with pytest.raises(TimeoutError, match="path deviation"):
        player.move_through([{"joint": 15.}], check_progress=stop)
    assert len(robot.sent) == 1

def test_deadline_prevents_further_commands(monkeypatch):
    import control.trajectory as module
    robot = Robot()
    player = TrajectoryPlayer(robot, MotionConfig(fps=0, move_timeout_s=1))
    ticks = iter([0., 2.])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="deadline"):
        player.move_through([{"joint": 15.}])
    assert not robot.sent

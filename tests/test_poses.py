import pytest

from pick_stack.control import PoseRegistry
from pick_stack.control.robot_io import JOINT_NAMES


def full_pose(value=0.0):
    return {j: value for j in JOINT_NAMES}


def test_set_get_save_load(tmp_path):
    path = tmp_path / "poses.yaml"
    registry = PoseRegistry(path=path)
    registry.set("home", full_pose(1.5))
    registry.set("retreat", full_pose(-3.0))
    registry.save()

    loaded = PoseRegistry.load(path)
    assert loaded.names() == ["home", "retreat"]
    assert loaded.get("home")["shoulder_pan"] == 1.5
    assert "home" in loaded and "nope" not in loaded


def test_missing_pose_message_names_the_tool():
    with pytest.raises(KeyError, match="record_pose"):
        PoseRegistry().get("home")


def test_incomplete_pose_rejected(tmp_path):
    path = tmp_path / "poses.yaml"
    path.write_text("poses:\n  home:\n    shoulder_pan: 0.0\n")
    with pytest.raises(ValueError, match="missing joint"):
        PoseRegistry.load(path)
    with pytest.raises(ValueError, match="missing joint"):
        PoseRegistry().set("home", {"shoulder_pan": 0.0})


def test_require():
    registry = PoseRegistry({"home": full_pose()})
    registry.require(["home"])
    with pytest.raises(KeyError, match="retreat"):
        registry.require(["home", "retreat"])


def test_ladder_sorted_numerically():
    registry = PoseRegistry(
        {
            "tower_descent_10": full_pose(10),
            "tower_descent_2": full_pose(2),
            "tower_descent_0": full_pose(0),
            "tower_other": full_pose(99),
        }
    )
    names = [name for name, _ in registry.ladder("tower_descent")]
    assert names == ["tower_descent_0", "tower_descent_2", "tower_descent_10"]
    assert registry.ladder("nothing") == []

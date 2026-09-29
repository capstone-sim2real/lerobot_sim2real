"""Measured loaded release height is recovered before opening the gripper."""

from agent_helpers import fast_cfg, make_skills
from session.primitives import PrimitiveSkills


def test_zone_release_corrects_loaded_shoulder_sag(monkeypatch):
    cfg = fast_cfg()
    cfg.agent.relative.frame = "base"
    base, _world, robot = make_skills({"yellow": (160.0, 40.0)}, cfg=cfg)
    skills = PrimitiveSkills(base.s)
    assert skills.observe_scene().ok
    assert skills.open_gripper().ok
    assert skills.move_relative(up_mm=50).ok
    assert skills.move_to_target("object", "pregrasp", "yellow_1", skills.observation_id).ok
    assert skills.align_gripper("yellow_1", skills.observation_id).ok
    assert skills.move_to_target("object", "grasp", "yellow_1", skills.observation_id).ok
    assert skills.close_gripper().ok
    assert skills.move_relative(up_mm=50).ok
    assert skills.move_to_target("slot", "preplace", slot="top-left").ok

    move_to = skills.s.player.move_to
    calls = []

    def loaded_move(goal, **kwargs):
        result = move_to(goal, **kwargs)
        # Fake FK maps elbow_flex to Z. Reproduce the 12.7 mm shortfall
        # observed with the yellow block while preserving a held grasp.
        robot.joints["elbow_flex"] -= 12.7
        calls.append(skills.s.arm_position_mm()[2])
        return result

    monkeypatch.setattr(skills.s.player, "move_to", loaded_move)
    result = skills.drop_at_zone_target()
    assert result.ok, result
    assert len(calls) == 2
    assert result.data["first_release_fk_mm"][2] < skills.s.grasp_z_mm + cfg.task1.release_clearance_mm / 2
    assert result.data["release_fk_mm"][2] >= skills.s.grasp_z_mm + cfg.task1.release_clearance_mm / 2

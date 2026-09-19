import pytest
from agent_helpers import make_skills
from session.pixel_target import calibration_id, resolve_pixel


def setup(tmp_path):
    skills, world, robot=make_skills({})
    path=tmp_path/'calib.json'
    skills.s.calib.save(path)
    skills.cfg.perception.calibration_path=str(path)
    point=skills.s.calib.board_to_pixel([(180.,80.)])[0]
    return skills,robot,int(round(point[0])),int(round(point[1])),calibration_id(skills.s.calib)


def test_pixel_plane_and_stale_bounds(tmp_path):
    sk,robot,u,v,key=setup(tmp_path)
    target=resolve_pixel(sk.cfg,sk.s.calib,u,v,key)
    assert target['z_mm']==sk.s.grasp_z_mm
    assert target['x_mm']==pytest.approx(180,abs=1)
    for args in [(-1,v,key),(u,v,'stale'),(True,v,key),(u,sk.s.calib.image_size[1],key)]:
        with pytest.raises(ValueError):resolve_pixel(sk.cfg,sk.s.calib,*args)
    assert not robot.sent_actions


def test_ik_failure_is_preplanned_before_first_motion(tmp_path,monkeypatch):
    sk,robot,u,v,key=setup(tmp_path)
    original=sk.s.ik.solve_holding_wrist_roll
    n=0
    def solve(*args,**kwargs):
        nonlocal n
        n+=1
        result=original(*args,**kwargs)
        if n==3:result.position_error_mm=1000
        return result
    monkeypatch.setattr(sk.s.ik,'solve_holding_wrist_roll',solve)
    result=sk.move_to_pixel(u,v,key)
    assert not result.ok and result.reason=='ik_gate'
    assert not robot.sent_actions


def test_fixed_height_and_no_gripper_command(tmp_path):
    sk,robot,u,v,key=setup(tmp_path)
    result=sk.move_to_pixel(u,v,key)
    assert result.ok,result.detail
    assert result.data['target']['z_mm']==sk.s.grasp_z_mm
    assert sk.s.arm_position_mm()[2]==pytest.approx(sk.s.grasp_z_mm)
    assert all('gripper' not in action for action in robot.sent_actions)


def test_changed_calibration_and_missing_held_block_do_not_move(tmp_path):
    sk,robot,u,v,key=setup(tmp_path)
    assert sk.place_at_pixel(u,v,key).reason=='no_block_held'
    import json
    path=sk.cfg.perception.calibration_path
    data=json.load(open(path));data['meta']['grasp_z_mm_mean']+=1
    with open(path,'w') as f:json.dump(data,f)
    assert sk.move_to_pixel(u,v,key).reason=='invalid_arguments'
    assert not robot.sent_actions

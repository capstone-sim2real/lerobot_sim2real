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

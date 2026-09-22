from config import CalibrationClearanceConfig
from session.calibration_clearance import clearance_check


def test_sideways_sweep_checked_not_only_endpoint():
    cfg = CalibrationClearanceConfig()
    assert clearance_check([(300,0,4)], {"wood":(150,0)}, cfg)["clear"]
    assert not clearance_check([(150,0,60),(300,0,4)], {"wood":(150,0)}, cfg)["clear"]


def test_missing_scene_rejects_before_opening_jaws(tmp_path):
    from agent_helpers import make_skills
    from session.grasp_calibration import CalibrationSkills
    sk,world,robot = make_skills({"green":(180,0)})
    cal = CalibrationSkills(sk.s,tmp_path)
    def forbidden():
        raise AssertionError("Must not open jaws for incomplete scene")
    cal.s.motion.open_gripper = forbidden
    result = cal.calibration_prepare("green")
    assert result.reason == "scene_incomplete"
    assert cal.attempt is None


def test_hover_correction_dry_run_and_load_abort(tmp_path):
    from agent_helpers import make_skills
    from session.grasp_calibration import CalibrationSkills
    sk,_,robot=make_skills({'green':(180,0)})
    cal=CalibrationSkills(sk.s,tmp_path)
    cal.cfg.agent.calibration_clearance.expected_colors=['green']
    def clear(*args):
        cal._approach_joints=None
        return {'clear':True,'reason':'ok'}
    cal._clearance_gate=clear
    assert cal.calibration_prepare('green').ok
    a=cal.attempt;cal.attempt=None
    current={**robot.read_joints(),**a.hover.joints}
    current['elbow_flex']-=2.2
    sent=[]
    robot.read_joints=lambda:dict(current)
    robot.send_joints=lambda q:(sent.append(q) or dict(q))
    assert cal.calibration_correct_hover(dry_run=True).ok
    assert sent==[]
    count=[0]
    def loads():
        count[0]+=1
        return {j:(100 if count[0]>1 else 0) for j in cal.cfg.sensing.contact_joints}
    robot.read_loads=loads
    result=cal.calibration_correct_hover(dry_run=False)
    assert not result.ok and result.data['stop_reason']=='load_increase'
    assert cal.attempt is None and not cal.descent_ready
    assert sent[-1]=={j:current[j] for j in a.hover.joints}

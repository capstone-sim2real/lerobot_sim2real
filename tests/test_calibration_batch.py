from tools.astra_calibration_batch import run_batch,PLAN
from session.results import SkillResult

def test_batch_places_only_verified_holding_and_tries_all():
    calls=[]
    def call(name,args):
        calls.append(name)
        if name=='calibration_pick_guarded':
            if args['color']=='blue':return {'ok':False,'reason':'neighbour_clearance'}
            return {'ok':True,'state':{'holding':args['color']}}
        if name=='place_at_slot':assert args['slot'] in ['top-left','top-center','top-right','bottom-left','bottom-right']
        return {'ok':True,'verified':True,'in_zone':True,'state':{'holding':None}}
    r=run_batch(call,lambda r:None)
    assert r['attempted']==5 and r['completed']==4
    assert calls.count('place_at_slot')==4

def test_guard_stop_never_closes_places_or_homes_again():
    calls=[]
    def call(name,args):
        calls.append(name);return {'ok':False,'stop_reason':'load_increase'}
    r=run_batch(call,lambda r:None)
    assert calls==['calibration_pick_guarded']
    assert r['stop_reason']=='physical_guard_or_holding'

def test_auto_hover_failure_cannot_authorize_descent(tmp_path):
    from agent_helpers import make_skills
    from session.grasp_calibration import CalibrationSkills
    sk,_,_=make_skills({'green':(180,0)})
    cal=CalibrationSkills(sk.s,tmp_path)
    cal.calibration_prepare=lambda color:SkillResult(False,'calibration_prepare','grasp_blocked',data={'stop_reason':'hover_not_settled'})
    cal.calibration_correct_hover=lambda **kw:SkillResult(False,'calibration_correct_hover','grasp_blocked',data={'stop_reason':'load_increase'})
    def forbidden():raise AssertionError('Unsafe continuation')
    cal.calibration_descend_guarded=forbidden
    cal.calibration_close_lift=forbidden
    assert not cal.calibration_pick_guarded('green').ok

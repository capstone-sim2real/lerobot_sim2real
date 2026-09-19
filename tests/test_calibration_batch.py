from tools.astra_calibration_batch import run_batch,PLAN
from session.results import SkillResult


def test_guard_stop_never_closes_places_or_homes_again():
    calls=[]
    def call(name,args):
        calls.append(name);return {'ok':False,'stop_reason':'load_increase'}
    r=run_batch(call,lambda r:None)
    assert calls==['calibration_pick_guarded']
    assert r['stop_reason']=='physical_guard_or_holding'

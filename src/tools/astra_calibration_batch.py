"""Frozen five-block trial over the existing server; no model/image inspection."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

PLAN=[('blue',0),('yellow',2),('green',1),('wood',3),('red',4)]

def run_batch(call,save):
    report={'plan':PLAN,'llm_image_observation':False,'policy':'existing CV + bounded measured hover correction; no online parameter edits', 'attempts':[],'stop_reason':None}
    for color,slot in PLAN:
        item={'color':color,'slot':slot};report['attempts'].append(item)
        picked=call('calibration_pick_guarded',{'color':color});item['pick']=picked;save(report)
        if not picked.get('ok'):
            # A physical guard stop ends the unattended experiment in place.
            if picked.get('stop_reason') in ('load_increase','tracking_lag','timeout','shortfall') or picked.get('state',{}).get('holding'):
                report['stop_reason']='physical_guard_or_holding';save(report);break
            item['home']=call('return_to_home',{});save(report)
            if not item['home'].get('ok'):
                report['stop_reason']='home_failed';save(report);break
            continue
        if picked.get('state',{}).get('holding')!=color:
            report['stop_reason']='holding_mismatch';save(report);break
        item['place']=call('place_at_slot',{'slot':['top-left','top-center','top-right','bottom-left','bottom-right'][slot]});save(report)
        if not item['place'].get('ok') or item['place'].get('state',{}).get('holding'):
            report['stop_reason']='place_failed';save(report);break
    report['completed']=sum(bool(a.get('place',{}).get('verified')) and a.get('place',{}).get('in_zone') is True for a in report['attempts'])
    report['attempted']=len(report['attempts']);save(report)
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--dry-run',action='store_true');ap.add_argument('--resume-held-first',action='store_true');args=ap.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    (out/'plan.json').write_text(json.dumps({'plan':PLAN,'no_llm_observation':True,'dry_run':args.dry_run},indent=2))
    if args.dry_run:
        print(json.dumps({'dry_run':True,'plan':PLAN}));return
    previous=None
    if args.resume_held_first:
        previous=json.loads((out/'report.json').read_text())
        assert len(previous['attempts'])==1 and previous['attempts'][0]['place']['reason']=='invalid_arguments'
        assert previous['attempts'][0]['pick']['state']['holding']==PLAN[0][0]
        (out/'report.initial.json').write_text(json.dumps(previous,indent=2))
    def save(report):
        tmp=out/'report.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(out/'report.json')
    def call(tool,arguments):
        if previous is not None and tool=='calibration_pick_guarded' and arguments['color']==PLAN[0][0]:
            return dict(previous['attempts'][0]['pick'],resumed_existing_hold=True)
        command=[sys.executable,'-m','tools.astra_calibration_call',tool,'--arguments',json.dumps(arguments),'--output',str(out/'calls')]
        p=subprocess.run(command,capture_output=True,text=True)
        with (out/'client.log').open('a') as f:f.write(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError('Client failed; see client.log (client requests STOP on dispatch failure)')
        event=json.loads(p.stdout.strip().splitlines()[-1]);result=event['result'];result['evidence']=event['evidence']
        return result
    try:
        result=run_batch(call,save)
        print(json.dumps({'completed':result['completed'],'attempted':result['attempted'],'stop_reason':result['stop_reason']}))
    except Exception as exc:
        (out/'error.txt').write_text(str(exc));raise
if __name__=='__main__':main()

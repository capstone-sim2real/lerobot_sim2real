"""Reusable primitive-only experiment sequence; never writes motor commands."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--steps',required=True,help='JSON tool/arguments list; observation_id may be latest')
    ap.add_argument('--output',required=True)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    steps=json.loads(args.steps)
    if args.dry_run:
        print(json.dumps({'dry_run':True,'steps':steps}));return
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    report={'steps':[],'complete':False};observation=None
    for step in steps:
        values=dict(step.get('arguments',{}))
        if values.get('observation_id')=='latest':
            if observation is None:raise ValueError('observe_scene required before latest target')
            values['observation_id']=observation
        p=subprocess.run([sys.executable,'-m','tools.agent_tool_call',step['tool'],
                          '--arguments',json.dumps(values),'--output',str(out/'calls')],capture_output=True,text=True)
        with (out/'client.log').open('a') as f:f.write(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError('Tool client failed; see client.log')
        event=json.loads(p.stdout.strip().splitlines()[-1]);result=event['result']
        report['steps'].append(event)
        (out/'report.json').write_text(json.dumps(report,indent=2))
        if step['tool']=='observe_scene' and result.get('ok'):
            observation=result['observation_id']
        state=result.get('state',{})
        print(json.dumps({'tool':step['tool'],'ok':result['ok'],'reason':result['reason'],
                          'detail':result.get('detail'),'holding':state.get('holding'),
                          'arm':state.get('arm_position_mm'),'evidence':event['evidence']}),flush=True)
        if not result['ok'] or ('expect_holding' in step and state.get('holding')!=step['expect_holding']):
            return
    report['complete']=True
    (out/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()

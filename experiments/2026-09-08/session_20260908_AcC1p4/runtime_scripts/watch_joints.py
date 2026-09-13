"""Display teleop telemetry only. Never opens motor ports."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--interval',type=float,default=.5);p.add_argument('--stale-after',type=float,default=3);p.add_argument('--once',action='store_true');a=p.parse_args()
path=Path('/tmp/so101-relative-teleop/observations.jsonl')
try:
 while True:
  try:
   with path.open('rb') as f:
    f.seek(0,2);size=f.tell();f.seek(max(0,size-16384));lines=f.read().splitlines()
   row=None
   for line in reversed(lines):
    try:row=json.loads(line);break
    except ValueError:continue
   if row is None:raise ValueError('No complete telemetry row')
   age=max(0,time.time()-row['time']);j=row['follower']
   if not a.once:print('\033[2J\033[H',end='')
   print('FOLLOWER JOINTS | '+('STALE: tracking stopped or delayed' if age>a.stale_after else 'LIVE')+f' | age {age:.1f}s')
   print(f"wrist_roll: {j['wrist_roll']:+.2f} deg   (calibration: near 0 deg)")
   for key,val in j.items():
    if key!='wrist_roll':print(f'{key:16s} {val:+8.2f} '+('%' if key=='gripper' else 'deg'))
   print('Ctrl+C: close this display only; teleop continues.',flush=True)
  except (OSError,ValueError,KeyError) as e:print('Telemetry unavailable:',e,flush=True)
  if a.once:break
  time.sleep(a.interval)
except KeyboardInterrupt:pass

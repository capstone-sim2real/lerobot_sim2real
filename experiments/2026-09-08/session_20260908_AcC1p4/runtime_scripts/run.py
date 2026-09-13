import argparse,json,time,signal,logging
from pathlib import Path
from live_capture import LiveCapture
from config import RobotIOConfig
from control.robot_io import So101RobotIO
from control.trajectory import interpolate
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader
p=argparse.ArgumentParser()
p.add_argument('--dry-run',action='store_true')
p.add_argument('--capture-request',type=Path,required=True)
p.add_argument('--snapshot-url',required=True)
p.add_argument('--capture-max-delta',type=float,required=True)
p.add_argument('--offsets-json',type=Path)
p.add_argument('--fps',type=float,required=True)
p.add_argument('--startup-step',type=float)
p.add_argument('--startup-ramp-seconds',type=float,default=0)
p.add_argument('--step',type=float,required=True)
p.add_argument('--max-relative-target',type=float,required=True)
p.add_argument('--seconds',type=float,required=True)
p.add_argument('--temperature-limit',type=float,default=65)
p.add_argument('--startup-limit',type=float,required=True)
p.add_argument('--stable-seconds',type=float,required=True)
p.add_argument('--stable-delta',type=float,required=True)
a=p.parse_args()
resume_offsets=json.loads(a.offsets_json.read_text()) if a.offsets_json else None
assert a.fps>0 and a.step>0 and a.seconds>=0
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
f=SOFollower(SOFollowerRobotConfig(id='my_follower',port='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00',use_degrees=True,max_relative_target=a.max_relative_target,disable_torque_on_disconnect=False))
l=SOLeader(SOLeaderTeleopConfig(id='my_leader',port='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6085435-if00',use_degrees=True))
io=So101RobotIO(RobotIOConfig());io._robot=f
stop=False;enabled=False;capture=None

def stopping(*_):
 global stop
 stop=True
signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
try:
 for r in (f,l):r.bus.connect()
 assert f.bus.is_calibrated,'Follower calibration mismatch'
 # Use current leader calibration in memory; never rewrite either device/file.
 l.bus.calibration=l.bus.read_calibration()
 for r in (f,l):
  torque=r.bus.sync_read('Torque_Enable')
  assert len(set(torque.values()))==1,'Mixed torque state'
  if r is l:assert all(v==0 for v in torque.values()),'Leader must be torque off'
  assert all(v==0 for v in r.bus.sync_read('Operating_Mode').values()),'Expected position mode'
  assert all(r.bus.ping(m.id)==777 for m in r.bus.motors.values()),'Unexpected model'
 fp=io.read_joints();lp=l.bus.sync_read('Present_Position')
 arms=[j for j in fp if j!='gripper']
 print('STARTUP_DELTA',json.dumps({j:lp[j]-fp[j] for j in arms}),flush=True)
 if not a.dry_run:
  capture=LiveCapture(a.capture_request,a.snapshot_url,a.capture_max_delta)
  print('WAITING: align leader with follower, then hold still',flush=True)
  startup_log=0
  stable_start=None;anchor=None;deadline=time.monotonic()+a.seconds if a.seconds else float('inf')
  while not stop:
   if time.monotonic()>deadline:raise RuntimeError('Startup alignment timed out; no motion sent')
   fp=io.read_joints();lp=l.bus.sync_read('Present_Position')
   capture.poll(fp)
   aligned=max(abs(lp[j]+(resume_offsets[j] if resume_offsets else 0)-fp[j]) for j in arms)<=a.startup_limit
   if time.monotonic()-startup_log>=1:
    Path('/tmp/so101-relative-teleop/startup_status.json').write_text(json.dumps({'time':time.time(),'follower':fp,'leader':lp,'delta':{j:lp[j]+(resume_offsets[j] if resume_offsets else 0)-fp[j] for j in arms}}))
    startup_log=time.monotonic()
   if not aligned:stable_start=None;anchor=None
   elif anchor is None or max(abs(lp[j]-anchor[j]) for j in lp)>a.stable_delta:
    anchor=lp.copy();stable_start=time.monotonic()
   elif time.monotonic()-stable_start>=a.stable_seconds:break
   time.sleep(0.1)
  if stop:raise RuntimeError('Cancelled before tracking')
 offsets=resume_offsets if resume_offsets else {j:fp[j]-lp[j] for j in fp}
 print(json.dumps({'dry_run':a.dry_run,'follower_start':fp,'leader_start':lp,'offsets':offsets}),flush=True)
 if not a.dry_run:
  raw=f.bus.sync_read('Present_Position',normalize=False)
  f.bus.sync_write('Goal_Position',raw,normalize=False)
  assert f.bus.sync_read('Goal_Position',normalize=False)==raw,'Goal readback mismatch'
  io.set_torque(True);enabled=True
  assert all(v==1 for v in f.bus.sync_read('Torque_Enable').values())
  print('ACTIVE: relative leader tracking; follower torque ON; leader torque OFF',flush=True)
  started=time.monotonic();lastlog=0;command=io.read_joints()
  with open('/tmp/so101-relative-teleop/observations.jsonl','a',buffering=1) as log:
   while not stop and (a.seconds==0 or time.monotonic()-started<a.seconds):
    tick=time.monotonic();cur=io.read_joints();leader=l.bus.sync_read('Present_Position')
    capture.poll(cur)
    goal={j:leader[j]+offsets[j] for j in cur}
    goal['gripper']=min(100.,max(0.,leader['gripper']))
    step=a.startup_step if a.startup_step is not None and tick-started<a.startup_ramp_seconds else a.step
    path=interpolate(command,goal,step)
    if path:command=io.send_joints(path[0])
    if tick-lastlog>=1:
     temps=f.bus.sync_read('Present_Temperature')
     if max(temps.values())>=a.temperature_limit:raise RuntimeError('Temperature limit reached: '+json.dumps(temps))
     if not all(v==1 for v in f.bus.sync_read('Torque_Enable').values()):raise RuntimeError('Follower torque lost')
     log.write(json.dumps({'time':time.time(),'follower':cur,'leader':leader,'goal':goal,'command':command,'temperature':temps})+'\n');lastlog=tick
    time.sleep(max(0,1/a.fps-(time.monotonic()-tick)))
except Exception as e:
 print('ERROR',repr(e),flush=True)
 raise
finally:
 if enabled and f.bus.is_connected:
  # Stop tracking at measured pose; preserve holding torque on disconnect.
  try:f.bus.sync_write('Goal_Position',f.bus.sync_read('Present_Position',normalize=False),normalize=False);print('STOPPED: holding current pose, torque retained',flush=True)
  except Exception as e:print('HOLD FAILED',repr(e),flush=True)
 if capture is not None:capture.close()
 for r in (l,f):
  if r.bus.is_connected:r.bus.disconnect(disable_torque=False)

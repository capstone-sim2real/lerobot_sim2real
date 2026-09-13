"""Continuously publish measured teleop FK; never accesses serial devices."""
import json,time,os
from pathlib import Path
from tools.record_calibration_point import load_kinematics,DEFAULT_URDF,ARM_MOTORS
from config import load_config
from perception import PlaneCalibration
source=Path('/tmp/so101-relative-teleop/observations.jsonl')
out=Path('/tmp/so101-gripper-reference.json')
k=load_kinematics(DEFAULT_URDF,'gripper_frame_link')
cal=PlaneCalibration.load(load_config('src/configs/default.yaml').perception.calibration_path)
last=None
while True:
    try:
        with source.open('rb') as f:
            f.seek(0,2); size=f.tell(); f.seek(max(0,size-16384)); lines=f.read().splitlines()
        r=None
        for line in reversed(lines):
            try:
                candidate=json.loads(line)
                if 'follower' in candidate and 'time' in candidate:
                    r=candidate;break
            except (ValueError,UnicodeError): pass
        if r is None: raise ValueError('No complete telemetry record')
        fresh=time.time()-r['time']<3
        key=(r['time'],fresh)
        if key!=last:
            joints=[r['follower'][j] for j in ARM_MOTORS]
            mm=k.forward_kinematics(joints)[:3,3]*1000
            px=cal.board_to_pixel(mm[:2].reshape(1,2))[0]
            data=dict(available=fresh,frame='gripper_frame_link',source='Live measured teleop FK (1 Hz telemetry)',projection='XY on calibrated block plane; not image detection',xyz_mm=mm.tolist(),pixel=px.tolist(),measured_at=r['time'],joint_degrees=joints)
            tmp=out.with_name(out.name+'.live.tmp');tmp.write_text(json.dumps(data));tmp.replace(out)
            print(json.dumps(data),flush=True);last=key
    except Exception as e:
        print(repr(e),flush=True)
        tmp=out.with_name(out.name+'.live.tmp');tmp.write_text(json.dumps(dict(available=False,error=str(e))));tmp.replace(out)
    time.sleep(.1)

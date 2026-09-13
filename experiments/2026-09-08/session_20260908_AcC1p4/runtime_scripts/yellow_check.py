import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from config import load_config
from camera.client import fetch_snapshot
from perception import PlaneCalibration,detect_blocks
from control.ik import TopDownIK
from control.grasp import plan_grasp_attempts
from tools.record_calibration_point import load_kinematics,DEFAULT_URDF,ARM_MOTORS
p=argparse.ArgumentParser();p.add_argument('--attempt-index',type=int,default=0);p.add_argument('--dry-run',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
cfg=load_config('src/configs/default.yaml');cal=PlaneCalibration.load(cfg.perception.calibration_path)
url='http://127.0.0.1:8090/snapshot/shoulder.jpg'
frame=fetch_snapshot(url);cv2.imwrite(str(out/'before.jpg'),frame)
ds=[d for d in detect_blocks(frame,cal,cfg.perception,is_rgb=False) if d.color=='yellow']
assert len(ds)==1,f'Expected one yellow block, got {len(ds)}'
d=ds[0];ik=TopDownIK(cfg.ik,project_root='.')
gp=plan_grasp_attempts(ik,cfg,*d.center_mm,float(cal.meta['grasp_z_mm_mean']),block_angle_deg=d.angle_deg,log=print)
at=gp.attempts[a.attempt_index];assert at.reachable,'Centre attempt unreachable'
plan=dict(block_xy_mm=list(d.center_mm),block_angle_deg=d.angle_deg,aim_xy_mm=list(at.xy_mm),grasp_z_mm=at.grasp_z_mm,hover_z_mm=at.hover_z_mm,hover=at.hover.joints,grasp=at.grasp.joints,ik_error_mm=at.grasp.position_error_mm)
(out/'plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps(plan),flush=True)
if a.dry_run: print('DRY RUN PASS: no hardware access',flush=True);raise SystemExit
from control.robot_io import So101RobotIO
from control.trajectory import TrajectoryPlayer
from control.sensing import check_grasp
from control.poses import PoseRegistry
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
k=load_kinematics(DEFAULT_URDF,'gripper_frame_link')
class ObservedIO(So101RobotIO):
    last=0
    temp_at=0
    def read_joints(self):
        j=super().read_joints()
        if time.time()-self.temp_at>1:
            self.temp_at=time.time()
            temps=self.robot.bus.sync_read("Present_Temperature",normalize=False)
            if max(temps.values())>=70: raise RuntimeError(f"Temperature stop: {temps}")
        if time.time()-self.last>.2:
            self.last=time.time()
            with open('/tmp/so101-relative-teleop/observations.jsonl','a') as f:f.write(json.dumps(dict(time=self.last,follower=j,source='yellow_check'))+'\n')
        return j
robot=ObservedIO(cfg.robot)
robot._robot=SOFollower(SOFollowerRobotConfig(port=cfg.robot.port,id=cfg.robot.id,use_degrees=True,max_relative_target=cfg.robot.max_relative_target,disable_torque_on_disconnect=False))
robot.robot.bus.connect()
def record(stage,**extra):
    j=robot.read_joints();mm=k.forward_kinematics([j[n] for n in ARM_MOTORS])[:3,3]*1000
    data=dict(stage=stage,time=time.time(),joints=j,fk_xyz_mm=mm.tolist(),block_xy_mm=plan['block_xy_mm'],aim_xy_mm=plan['aim_xy_mm'],fk_minus_block_mm=(mm[:2]-d.center_mm).tolist(),**extra)
    (out/(stage+'.json')).write_text(json.dumps(data,indent=2));cv2.imwrite(str(out/(stage+'.jpg')),fetch_snapshot(url));print(json.dumps(data),flush=True)
    for _ in range(20):robot.read_joints();time.sleep(.2)
try:
    assert all(v==1 for v in robot.robot.bus.sync_read('Torque_Enable',normalize=False).values()),'Expected torque on'
    assert robot.robot.bus.read_calibration()==robot.robot.calibration,'Calibration mismatch'
    player=TrajectoryPlayer(robot,cfg.motion)
    print('MOVING HOME',flush=True);player.move_to(PoseRegistry.load(cfg.motion.poses_path).get(cfg.motion.home_pose),max_step=1.,tol=cfg.motion.transit_arrival_tol)
    player.set_gripper(cfg.sensing.gripper_open_pos)
    print('MOVING HOVER',flush=True);player.move_to(at.hover.joints,max_step=1.,tol=cfg.motion.transit_arrival_tol)
    player.settle(at.hover.joints,tol=cfg.motion.grasp_hover_arrival_tol,timeout_s=cfg.motion.grasp_hover_settle_s)
    record('hover')
    print('DESCENDING',flush=True);_,blocked=player.descend(at.grasp.joints);record('before_close',blocked=blocked)
    player.set_gripper(cfg.sensing.gripper_close_pos);check=check_grasp(robot,cfg.sensing)
    record('closed',grasped=check.grasped,gripper_pos=check.gripper_pos,load=check.gripper_load_abs)
    if check.grasped:
        print('VERIFIED: LIFTING TO HOVER',flush=True);player.move_to(at.hover.joints,max_step=1.,tol=cfg.motion.transit_arrival_tol);record('lifted')
    else:
        print('EMPTY: OPENING AND RETREATING',flush=True);player.set_gripper(cfg.sensing.gripper_open_pos);player.move_to(at.hover.joints,max_step=1.,tol=cfg.motion.transit_arrival_tol);record('failed_retreat')
finally:
    raw=robot.robot.bus.sync_read('Present_Position',normalize=False);robot.robot.bus.sync_write('Goal_Position',{n:v for n,v in raw.items() if n!='gripper'},normalize=False)
    robot.robot.bus.disconnect(disable_torque=False)
    print('FINISHED: torque retained; teleop remains paused',flush=True)

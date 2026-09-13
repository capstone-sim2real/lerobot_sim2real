import argparse,json,time
from pathlib import Path
import cv2
from config import load_config
from perception import PlaneCalibration
from control.ik import TopDownIK
from control.grasp import highest_reachable_hover
from control.poses import PoseRegistry
from camera.client import fetch_snapshot
p=argparse.ArgumentParser();p.add_argument('--x',type=float,required=True);p.add_argument('--y',type=float,required=True);p.add_argument('--source-plan',required=True);p.add_argument('--output',required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
cfg=load_config('src/configs/default.yaml');cal=PlaneCalibration.load(cfg.perception.calibration_path);ik=TopDownIK(cfg.ik,project_root='.')
z=float(cal.meta['grasp_z_mm_mean'])+cfg.task1.release_clearance_mm
source=json.loads(Path(a.source_plan).read_text());sx,sy=source['aim_xy_mm']
r=(sx*sx+sy*sy)**.5;scale=min(1.,cfg.motion.transit_apex_radius_mm/r);ax,ay=sx*scale,sy*scale
wps=[]
for name,x,y,zz in [('source_lift',sx,sy,highest_reachable_hover(ik,sx,sy,z,cfg)),('source_apex',ax,ay,highest_reachable_hover(ik,ax,ay,z,cfg)),('dest_hover',a.x,a.y,highest_reachable_hover(ik,a.x,a.y,z,cfg)),('dest_drop',a.x,a.y,z)]:
    sol=ik.solve(x,y,zz)
    assert sol.position_error_mm<=cfg.ik.max_position_error_mm and sol.tilt_error_deg<=cfg.ik.max_tilt_error_deg,(name,sol)
    wps.append((name,sol.joints));print(name,x,y,zz,sol.position_error_mm,flush=True)
(out/'transfer_plan.json').write_text(json.dumps(wps,indent=2))
if a.dry_run:print('DRY RUN PASS',flush=True);raise SystemExit
from control.robot_io import So101RobotIO
from control.trajectory import TrajectoryPlayer
from control.sensing import check_grasp
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
class IO(So101RobotIO):
    last=0;temp_at=0
    def read_joints(self):
        j=super().read_joints();now=time.time()
        if now-self.temp_at>1:
            self.temp_at=now;t=self.robot.bus.sync_read('Present_Temperature',normalize=False)
            if max(t.values())>=70:raise RuntimeError(f'Temperature stop {t}')
        if now-self.last>.2:
            self.last=now
            with open('/tmp/so101-relative-teleop/observations.jsonl','a') as f:f.write(json.dumps(dict(time=now,follower=j,source='transfer_yellow'))+'\n')
        return j
robot=IO(cfg.robot);robot._robot=SOFollower(SOFollowerRobotConfig(port=cfg.robot.port,id=cfg.robot.id,use_degrees=True,max_relative_target=cfg.robot.max_relative_target,disable_torque_on_disconnect=False));robot.robot.bus.connect()
try:
    assert robot.robot.bus.read_calibration()==robot.robot.calibration
    assert all(v==1 for v in robot.robot.bus.sync_read('Torque_Enable',normalize=False).values())
    player=TrajectoryPlayer(robot,cfg.motion)
    player.set_gripper(cfg.sensing.gripper_close_pos)
    check=check_grasp(robot,cfg.sensing);print('INITIAL_GRASP',check,flush=True);assert check.grasped,'Not holding block: abort transfer'
    player=TrajectoryPlayer(robot,cfg.motion)
    for name,j in wps:
        print('MOVE',name,flush=True)
        if name=='dest_drop':player.descend(j)
        else:
            player.move_to(j,max_step=1.,tol=cfg.motion.transit_arrival_tol)
            player.settle(j,tol=cfg.motion.grasp_hover_arrival_tol,timeout_s=cfg.motion.grasp_hover_settle_s)
        cv2.imwrite(str(out/(name+'.jpg')),fetch_snapshot('http://127.0.0.1:8090/snapshot/shoulder.jpg'))
    print('RELEASE',flush=True);player.set_gripper(cfg.sensing.gripper_open_pos);time.sleep(cfg.motion.place_settle_s)
    player.move_to(wps[-2][1],max_step=1.,tol=cfg.motion.transit_arrival_tol)
    player.move_to(PoseRegistry.load(cfg.motion.poses_path).get(cfg.motion.home_pose),max_step=1.,tol=cfg.motion.transit_arrival_tol)
    cv2.imwrite(str(out/'released.jpg'),fetch_snapshot('http://127.0.0.1:8090/snapshot/shoulder.jpg'));print('TRANSFER COMPLETE',flush=True)
finally:
    raw=robot.robot.bus.sync_read('Present_Position',normalize=False);robot.robot.bus.sync_write('Goal_Position',{n:v for n,v in raw.items() if n!='gripper'},normalize=False);robot.robot.bus.disconnect(disable_torque=False)

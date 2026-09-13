import argparse,json,time
from config import load_config
from control.poses import PoseRegistry
p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');a=p.parse_args()
cfg=load_config('src/configs/default.yaml');goal=PoseRegistry.load(cfg.motion.poses_path).get(cfg.motion.home_pose)
print('HOME goal',goal,flush=True)
if a.dry_run: print('DRY RUN: existing named home trajectory, measured start, step 1, clamp retained');raise SystemExit
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
    TrajectoryPlayer(robot,cfg.motion).move_to(goal,max_step=1.,tol=cfg.motion.transit_arrival_tol)
    print('HOME COMPLETE',flush=True)
finally:
    raw=robot.robot.bus.sync_read('Present_Position',normalize=False)
    robot.robot.bus.sync_write('Goal_Position',{n:v for n,v in raw.items() if n!='gripper'},normalize=False)
    robot.robot.bus.disconnect(disable_torque=False)

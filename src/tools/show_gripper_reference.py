"""Publish a measured FK reference snapshot for the camera UI. No motor writes."""
from pathlib import Path
import argparse,json,time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='src/configs/default.yaml')
    p.add_argument('--output',type=Path,default=Path('/tmp/so101-gripper-reference.json'))
    a=p.parse_args()
    from config import load_config
    from perception import PlaneCalibration
    from tools.record_calibration_point import load_kinematics,DEFAULT_URDF,read_follower_xyz
    cfg=load_config(a.config)
    k=load_kinematics(DEFAULT_URDF,'gripper_frame_link')
    joints,xyz=read_follower_xyz(cfg.robot.port,cfg.robot.id,k)
    measured=time.time();calib=PlaneCalibration.load(cfg.perception.calibration_path)
    mm=xyz*1000;pixel=calib.board_to_pixel(mm[:2].reshape(1,2))[0]
    data={'available':True,'frame':'gripper_frame_link','source':'FK snapshot',
          'projection':'XY on calibrated block plane; not an image detection',
          'xyz_mm':mm.tolist(),'pixel':pixel.tolist(),'measured_at':measured,
          'joint_degrees':joints.tolist(),'plane_z_mm':calib.meta.get('grasp_z_mm_mean')}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(data));tmp.replace(a.output)
    print(json.dumps(data))
if __name__=='__main__':main()

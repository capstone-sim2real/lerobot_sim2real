"""Convert captured joint degrees to FK CSV offline; never connects to an arm."""
import argparse,json
from pathlib import Path
from tools.record_calibration_point import load_kinematics,DEFAULT_URDF,ARM_MOTORS,update_csv
p=argparse.ArgumentParser();p.add_argument('record',type=Path);a=p.parse_args()
r=json.loads(a.record.read_text())
k=load_kinematics(DEFAULT_URDF,'gripper_frame_link')
pose=k.forward_kinematics([r['joints'][j] for j in ARM_MOTORS]);xyz=pose[:3,3]
row={'name':r['name'],'image':r['image'],'u_px':'','v_px':'','x_m':f'{xyz[0]:.6f}','y_m':f'{xyz[1]:.6f}','z_m':f'{xyz[2]:.6f}',**{j:f"{r['joints'][j]:.3f}" for j in ARM_MOTORS},'notes':'Torque-on teleop capture; pixel pending; block motion confirmation pending'}
update_csv(a.record.parent/'points.csv',row,False)
print(json.dumps(row))

"""Experimental measured wrist lower bound at the shared bus write seam."""
import math
from control.robot_io import BaseRobotIO

class CalibrationJointLimitIO(BaseRobotIO):
    def __init__(self,inner,cfg):
        self.inner=inner;self.cfg=cfg;self.joint_names=inner.joint_names
    def connect(self):return self.inner.connect()
    def disconnect(self):return self.inner.disconnect()
    @property
    def is_connected(self):return self.inner.is_connected
    def read_joints(self):return self.inner.read_joints()
    def read_observation(self):return self.inner.read_observation()
    def read_loads(self):return self.inner.read_loads()
    def set_torque(self,enabled):return self.inner.set_torque(enabled)
    def send_joints(self,positions):
        if "wrist_roll" in positions:
            target=positions["wrist_roll"]
            if not math.isfinite(target):raise ValueError("Non-finite wrist target")
            limit=self.cfg.wrist_roll_min_deg
            if target<limit:
                actual=self.inner.read_joints()["wrist_roll"]
                # An already-outside arm may only hold or retreat toward limit.
                if not (actual<limit and target>=actual):
                    raise ValueError(f"wrist_roll target {target} below measured limit {limit}")
        return self.inner.send_joints(positions)
    def __getattr__(self,name):return getattr(self.inner,name)

"""Scalar path-speed S curve; Ruckig preserves velocity and acceleration on retarget."""


class JogRamp:
    def __init__(self, cfg):
        from ruckig import Ruckig, InputParameter, OutputParameter, ControlInterface
        self.cfg = cfg
        self.otg = Ruckig(1, 1 / cfg.keyboard_tick_hz)
        self.input = InputParameter(1)
        self.output = OutputParameter(1)
        self.input.control_interface = ControlInterface.Velocity
        self.input.current_position = [0.]
        self.input.current_velocity = [0.]
        self.input.current_acceleration = [0.]
        self.input.target_velocity = [0.]
        self.input.target_acceleration = [0.]
        self.input.max_velocity = [cfg.keyboard_speed_mm_s]
        self.input.max_acceleration = [cfg.keyboard_acceleration_mm_s2]
        self.input.max_jerk = [cfg.keyboard_jerk_mm_s3]

    @property
    def stopped(self):
        return abs(self.input.current_velocity[0]) < 1e-8 and abs(self.input.current_acceleration[0]) < 1e-8

    def advance(self, speed):
        self.input.target_velocity = [speed]
        before = self.input.current_position[0]
        result = self.otg.update(self.input, self.output)
        if result < 0:
            raise RuntimeError(f'keyboard speed trajectory failed: {result}')
        self.output.pass_to_input(self.input)
        return max(0., self.input.current_position[0] - before)

    def braking_distance(self):
        from ruckig import InputParameter, Trajectory
        inp = InputParameter(1)
        for name in ('control_interface', 'current_position', 'current_velocity', 'current_acceleration',
                     'max_velocity', 'max_acceleration', 'max_jerk'):
            setattr(inp, name, getattr(self.input, name))
        inp.target_velocity = [0.]
        inp.target_acceleration = [0.]
        trajectory = Trajectory(1)
        result = self.otg.calculate(inp, trajectory)
        if result < 0:
            raise RuntimeError(f'keyboard braking trajectory failed: {result}')
        end, _, _ = trajectory.at_time(trajectory.duration)
        return max(0., end[0] - inp.current_position[0])

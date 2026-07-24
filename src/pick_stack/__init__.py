"""SO-101 pick & stack: hybrid FSM (ACT pick + rule-based transport/place).

Package layout (see README.md for the full architecture):
    config      dataclass tree + YAML loader, all tunables live in configs/
    control     robot IO wrapper, poses, trajectories, load sensing, motion
    perception  homography, block detection, target selection
    policy      ACT gRPC client for the PICK state
    fsm         state handlers and the state machine loop
    runners     task entrypoints
    tools       calibration / tuning CLIs
"""

__version__ = "0.1.0"

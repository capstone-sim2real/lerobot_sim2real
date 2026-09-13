"""Convert captured joint degrees to FK CSV without connecting to an arm."""

import argparse
import json
from pathlib import Path

from tools.calibration_records import update_csv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    return parser.parse_args(argv)


def finalize_record(path, kinematics, motors):
    record = json.loads(path.read_text())
    xyz = kinematics.forward_kinematics([record["joints"][name] for name in motors])[
        :3, 3
    ]
    row = {
        "name": record["name"],
        "image": record["image"],
        "u_px": "",
        "v_px": "",
        "x_m": f"{xyz[0]:.6f}",
        "y_m": f"{xyz[1]:.6f}",
        "z_m": f"{xyz[2]:.6f}",
        **{name: f"{record['joints'][name]:.3f}" for name in motors},
        "notes": "Torque-on teleop capture; pixel pending; block motion confirmation pending",
    }
    update_csv(path.parent / "points.csv", row, False)
    return row


def main(argv=None):
    args = parse_args(argv)
    from tools.record_calibration_point import load_kinematics, DEFAULT_URDF, ARM_MOTORS

    kinematics = load_kinematics(DEFAULT_URDF, "gripper_frame_link")
    print(json.dumps(finalize_record(args.record, kinematics, ARM_MOTORS)))


if __name__ == "__main__":
    main()

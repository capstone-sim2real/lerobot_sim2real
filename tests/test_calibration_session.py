import csv

from tools.capture_calibration_session import (
    PAIR_FIELDS,
    completed_points,
    export_accepted,
    next_attempt_name,
)


def row(name, complete=True):
    return {
        "name": name,
        "image": name.lower() + "_clean.jpg",
        **{key: "1.2" if complete else "" for key in PAIR_FIELDS},
    }


def test_manual_pixel_edit_preserves_recorded_joint_columns(tmp_path):
    from tools.calibration_records import read_csv, write_csv
    from tools.pick_pixels import read_rows, write_rows

    path = tmp_path / "points.csv"
    fields = ["name", "u_px", "v_px", "shoulder_pan", "wrist_roll", "notes"]
    write_csv(
        path,
        fields,
        [
            dict(
                name="P1",
                u_px="",
                v_px="",
                shoulder_pan="12.3",
                wrist_roll="0.4",
                notes="keep",
            )
        ],
    )
    rows = read_rows(path)
    rows[0]["u_px"] = "120"
    rows[0]["v_px"] = "240"
    write_rows(path, rows)
    saved_fields, saved = read_csv(path)
    assert saved_fields == fields
    assert saved[0]["wrist_roll"] == "0.4"
    assert saved[0]["shoulder_pan"] == "12.3"

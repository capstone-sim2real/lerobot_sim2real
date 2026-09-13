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


def test_resume_uses_completed_retry_and_rejects_nonfinite_or_incomplete():
    good = row("P1_retry")
    bad = row("P2")
    bad["u_px"] = "nan"
    rows = [row("P1", False), good, bad, row("P3", False)]
    assert completed_points(rows, 9) == {1: good}


def test_retry_preserves_rows_and_orphan_photos(tmp_path):
    (tmp_path / "p1_retry1_top.jpg").write_bytes(b"preserved")
    assert next_attempt_name(tmp_path, [row("P1", False)], 1) == "P1_retry2"


def test_export_excludes_failed_attempt_and_preserves_original(tmp_path):
    original = tmp_path / "points.csv"
    rows = [row("P1", False), row("P1_retry"), row("P2")]
    with original.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    before = original.read_bytes()
    assert set(export_accepted(tmp_path, 9)) == {1, 2}
    assert original.read_bytes() == before
    with (tmp_path / "accepted_points.csv").open() as stream:
        accepted = list(csv.DictReader(stream))
    assert [r["name"] for r in accepted] == ["P1_retry", "P2"]


def test_retry_name_preserves_orphan_joint_record(tmp_path):
    from tools.calibration_records import next_attempt_name

    (tmp_path / "p1_live.json").write_text("{}")
    assert next_attempt_name(tmp_path, [], 1) == "P1_retry1"


def test_shared_csv_io_retains_unknown_columns(tmp_path):
    from tools.calibration_records import read_csv, write_csv

    path = tmp_path / "points.csv"
    fields = ["name", "u_px", "operator_note"]
    rows = [{"name": "P1", "u_px": "1.25", "operator_note": "preserve, quoted text"}]
    write_csv(path, fields, rows)
    assert read_csv(path) == (fields, rows)


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

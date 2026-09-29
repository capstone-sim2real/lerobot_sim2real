"""Shared calibration rows, retry names and accepted-pair export."""

import csv
import math
import re
from pathlib import Path

CSV_FIELDS = [
    "name",
    "image",
    "u_px",
    "v_px",
    "x_m",
    "y_m",
    "z_m",
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "notes",
]


POINT_NAME = re.compile(r"P([1-9]\d*)(?:_retry\d*)?", re.IGNORECASE)
PAIR_FIELDS = (
    "u_px",
    "v_px",
    "x_m",
    "y_m",
    "z_m",
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, fields, rows) -> None:
    """Replace a complete table atomically, preserving supplied column order."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_rows(directory: Path) -> tuple[list[str], list[dict[str, str]]]:
    path = directory / "points.csv"
    return read_csv(path) if path.exists() else ([], [])


def completed_points(
    rows: list[dict[str, str]], count: int
) -> dict[int, dict[str, str]]:
    """Keep the latest complete retry for each logical point, excluding failures."""
    result = {}
    for row in rows:
        match = POINT_NAME.fullmatch(row.get("name", ""))
        if match is None or not 1 <= int(match[1]) <= count:
            continue
        try:
            complete = all(
                math.isfinite(float(row.get(key, ""))) for key in PAIR_FIELDS
            )
        except (TypeError, ValueError):
            complete = False
        if complete:
            result[int(match[1])] = row
    return result


def next_attempt_name(directory: Path, rows: list[dict[str, str]], index: int) -> str:
    names = {row["name"].lower() for row in rows}
    attempt = 0
    while True:
        name = f"P{index}" if attempt == 0 else f"P{index}_retry{attempt}"
        if (
            name.lower() not in names
            and not any(directory.glob(name.lower() + "_*.jpg"))
            and not any(directory.glob(name.lower() + "_*.json"))
        ):
            return name
        attempt += 1


def export_accepted(directory: Path, count: int) -> dict[int, dict[str, str]]:
    fields, rows = read_rows(directory)
    complete = completed_points(rows, count)
    if fields:
        target = directory / "accepted_points.csv"
        write_csv(target, fields, [complete[index] for index in sorted(complete)])
    return complete


def update_csv(csv_path: Path, row: dict[str, str], overwrite: bool) -> None:
    rows = read_csv(csv_path)[1] if csv_path.exists() else []

    exists = any(existing["name"] == row["name"] for existing in rows)
    if exists and not overwrite:
        raise RuntimeError(
            f"Point {row['name']} already exists; use --overwrite to replace it"
        )
    rows = [existing for existing in rows if existing["name"] != row["name"]]
    rows.append(row)

    write_csv(csv_path, CSV_FIELDS, rows)

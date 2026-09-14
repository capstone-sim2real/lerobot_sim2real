"""Reproduce report calibration tables/figure from immutable evidence snapshots.

Run from the repository root with:
    PYTHONPATH=src .venv/bin/python docs/report/최종보고서/analysis/summarize_calibration.py

Uses the project's existing calibration functions; never writes active profiles.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPORT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT / "src"))

from perception.homography import calibrate_from_pairs
from tools.calibrate_base_frame import (
    leave_one_out_errors,
    load_points,
    residuals_mm,
    to_pairs,
)


def main() -> None:
    evidence = REPORT / "evidence/calibration"
    manifest = json.loads((evidence / "manifest.json").read_text())
    summaries = []
    history_rows = []
    for date, sources in manifest.items():
        for item in sources.values():
            actual = hashlib.sha256((REPORT / item["snapshot"]).read_bytes()).hexdigest()
            if actual != item["sha256"]:
                raise ValueError(f"Evidence hash mismatch: {item['snapshot']}")
        profile = json.loads((REPORT / sources["profile"]["snapshot"]).read_text())
        rows = load_points(REPORT / sources["points"]["snapshot"])
        pairs = to_pairs(rows)
        h = np.asarray(profile["H"], dtype=float)
        refit = calibrate_from_pairs(pairs)
        if not np.allclose(h, refit, rtol=1e-7, atol=1e-9):
            raise ValueError(f"Saved H does not match refit: {date}")
        res = residuals_mm(h, pairs)
        loo = leave_one_out_errors(pairs)
        z = np.asarray([float(row["z_m"]) * 1000 for row in rows])
        rms = float(np.sqrt(np.mean(res**2)))
        for name, computed in [("rms_mm", rms), ("loo_max_mm", float(loo.max()))]:
            if not np.isclose(profile["meta"][name], computed, rtol=1e-7):
                raise ValueError(f"Metadata mismatch: {date} {name}")
        summary = {
            "date": date,
            "num_points": len(rows),
            "rms_mm": rms,
            "loo_max_mm": float(loo.max()),
            "z_mean_mm": float(z.mean()),
            "z_std_mm": float(z.std()),
            "per_point": [
                {"name": row["name"], "residual_mm": float(r), "loo_mm": float(l)}
                for row, r, l in zip(rows, res, loo, strict=True)
            ],
        }
        summaries.append(summary)
        label = f"9월 {int(date[-2:])}일"
        history_rows.append(f"{label} & {len(rows)} & {rms:.2f} & {loo.max():.2f} " + r"\\")
        if date != "20260913":
            continue
        latest_rows = [
            f"{row['name']} & {r:.2f} & {l:.2f} " + r"\\"
            for row, r, l in zip(rows, res, loo, strict=True)
        ]
        latest_rows.extend([r"\midrule", f"최대 & {res.max():.2f} & {loo.max():.2f} " + r"\\"])
        latest_table = [
            r"\begin{tabular}{@{}lrr@{}}",
            r"\toprule 대응점 & 적합 잔차 (mm) & LOO 오차 (mm) \\",
            r"\midrule",
            *latest_rows,
            r"\bottomrule", r"\end{tabular}",
        ]
        (REPORT / "figures/calibration_latest_table.tex").write_text("\n".join(latest_table) + "\n")
        with (REPORT / "analysis/calibration_latest_residuals.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["name", "residual_mm", "loo_mm"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(summary["per_point"])
        pixels = np.asarray([[*p, 1.0] for p, _ in pairs])
        projected = pixels @ h.T
        xy = projected[:, :2] / projected[:, 2:]
        recorded = np.asarray([p for _, p in pairs])
        figure = [
            "% Generated from 20260913 evidence; arrows at true metric scale.",
            r"\begin{tikzpicture}",
            r"\begin{axis}[width=\linewidth,height=8.1cm,axis equal image,",
            r"xmin=-245,xmax=235,ymin=35,ymax=355,xlabel={Base Y (mm)},ylabel={Base X (mm)},",
            r"grid=major,grid style={black!12},tick label style={font=\small},",
            r"legend style={font=\small,at={(0.5,1.02)},anchor=south,legend columns=2}]",
        ]
        polygon = " -- ".join(f"(axis cs:{y:.5f},{x:.5f})" for x, y in profile["zone_polygon_mm"])
        figure.append(r"\path[fill=black!7,draw=black!40] " + polygon + " -- cycle;")
        figure.append(r"\node[font=\scriptsize] at (axis cs:-10,330) {등록 영역};")
        for row, source, dest in zip(rows, recorded, xy, strict=True):
            sx, sy = source
            dx, dy = dest
            figure.extend([
                f"\\draw[-{{Latex}},thick] (axis cs:{sy:.5f},{sx:.5f}) -- (axis cs:{dy:.5f},{dx:.5f});",
                f"\\addplot[only marks,mark=o,black,forget plot] coordinates {{({sy:.5f},{sx:.5f})}};",
                f"\\addplot[only marks,mark=x,black,forget plot] coordinates {{({dy:.5f},{dx:.5f})}};",
                f"\\node[font=\\scriptsize,anchor=south] at (axis cs:{sy:.5f},{max(sx, dx)+7:.5f}) {{{row['name']}}};",
            ])
        figure.extend([
            r"\addlegendimage{only marks,mark=o}\addlegendentry{기록 FK 좌표}",
            r"\addlegendimage{only marks,mark=x}\addlegendentry{평면 변환 좌표}",
            r"\end{axis}", r"\end{tikzpicture}",
        ])
        (REPORT / "figures/calibration_latest_map.tex").write_text("\n".join(figure) + "\n")
    history_table = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule 수집 단계 & 점 수 & RMS (mm) & 최대 LOO (mm) \\",
        r"\midrule",
        r"초기 수집 (팀 집계) & 5 & 20.11 & 미기록 \\",
        r"초기 확장 (팀 집계) & 9 & 13.46 & 미기록 \\",
        *history_rows,
        r"\bottomrule", r"\end{tabular}",
    ]
    (REPORT / "figures/calibration_history_table.tex").write_text("\n".join(history_table) + "\n")
    (REPORT / "analysis/calibration_profiles_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n"
    )
    print("Verified 4 profiles; generated history, latest residual table, and latest map.")
    for summary in summaries:
        print(f"{summary['date']}: RMS {summary['rms_mm']:.4f} mm; max LOO {summary['loo_max_mm']:.4f} mm")


if __name__ == "__main__":
    main()

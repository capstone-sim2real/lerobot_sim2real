"""Reproduce report tables/figures from archived evidence; no robot access.

Run from any directory with Python 3. Output times use the task-start log as
zero, not the external timeout process start. Sensor labels are not ground truth.
"""

import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence/20260908"
OUT = ROOT / "analysis"


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_task():
    origin = None
    cycle = None
    pending = None
    cycles, attempts, transitions = [], [], []
    disconnect_s = None
    for line_number, line in enumerate((SOURCE / "task1_run.txt").read_text().splitlines(), 1):
        timestamp = re.match(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})", line)
        if not timestamp:
            continue
        now = datetime.strptime(timestamp[1], "%Y-%m-%d %H:%M:%S,%f")
        if "task starting with cv_ik PICK" in line:
            origin = now
        if origin is None:
            continue
        elapsed = round((now - origin).total_seconds(), 3)
        if "SOFollower disconnected" in line:
            disconnect_s = elapsed
        transition = re.search(r"fsm.machine INFO (\w+) -> (\w+)", line)
        if transition:
            old, new = transition.groups()
            transitions.append(dict(from_state=old, to_state=new, time_s=elapsed, source_line=line_number))
            if old == "select" and new == "pick":
                target = re.search(r"target=(\w+) slot=(\d+)", line)
                cycle = dict(cycle=len(cycles) + 1, target=target[1], slot=int(target[2]),
                             pick_start_s=elapsed, pick_end_s="", verify_end_s="",
                             transport_end_s="", release_s="", last_action_s=elapsed,
                             candidate_starts=0, held=0, empty=0, interrupted=0,
                             source_start_line=line_number, source_end_line="")
                cycles.append(cycle)
            elif cycle:
                key = {"pick": "pick_end_s", "verify": "verify_end_s", "transport": "transport_end_s"}.get(old)
                if key:
                    cycle[key] = elapsed
                if old == "place" and "released_slot=" in line:
                    cycle["release_s"] = elapsed
                cycle["source_end_line"] = line_number
        candidate = re.search(r"INFO\s+attempt '([^']+)'", line)
        if candidate:
            if pending is not None or cycle is None:
                raise ValueError(f"Unmatched candidate at line {line_number}")
            pending = dict(candidate=len(attempts) + 1, cycle=cycle["cycle"], target=cycle["target"],
                           candidate_name=candidate[1], start_s=elapsed, result_s="", outcome="",
                           sensor_position="", sensor_load="", source_start_line=line_number,
                           source_result_line="")
            attempts.append(pending)
            cycle["candidate_starts"] += 1
        result = re.search(r"-> (HELD|EMPTY|BLOCKED) pos=([\d.-]+) load=([\d.-]+)", line)
        if result:
            if pending is None:
                raise ValueError(f"Result without candidate at line {line_number}")
            pending.update(result_s=elapsed, outcome=result[1], sensor_position=float(result[2]),
                           sensor_load=float(result[3]), source_result_line=line_number)
            cycle[result[1].lower()] += 1
            pending = None
        if cycle and "fsm.ik_handler INFO" in line:
            cycle["last_action_s"] = elapsed
    if pending is not None:
        if "KeyboardInterrupt" not in (SOURCE / "task1_run.txt").read_text():
            raise ValueError("Missing result without an interruption record")
        pending["outcome"] = "INTERRUPTED_NO_SENSOR_RESULT"
        cycle["interrupted"] += 1

    counts = dict(candidate_starts=len(attempts), sensor_held=sum(c["held"] for c in cycles),
                  sensor_empty=sum(c["empty"] for c in cycles),
                  interrupted_no_result=sum(c["interrupted"] for c in cycles),
                  pick_entries=len(cycles),
                  verify_passes=sum(t["from_state"] == "verify" and t["to_state"] == "transport" for t in transitions),
                  slot_release_events=sum(c["release_s"] != "" for c in cycles),
                  done_transitions=sum(t["to_state"] == "done" for t in transitions))
    if counts != dict(candidate_starts=14, sensor_held=4, sensor_empty=9,
                      interrupted_no_result=1, pick_entries=6, verify_passes=4,
                      slot_release_events=4, done_transitions=0):
        raise ValueError(f"Report source counts changed: {counts}")
    write_csv("task1_candidates.csv", attempts)
    write_csv("task1_cycles.csv", cycles)
    write_csv("task1_transitions.csv", transitions)
    summary = dict(run_id="task1_20260908_030956", clock_origin=origin.isoformat(),
                   external_timeout_s=180, counts=counts,
                   disconnect_log_s=disconnect_s, physical_stop_s=None,
                   human_accepted_blocks=None, human_mission_success=False,
                   human_result_source="Archived session README: blue remained outside",
                   red_first_pick_s=round(cycles[2]["pick_end_s"] - cycles[2]["pick_start_s"], 3),
                   red_selection_to_release_s=round(cycles[3]["release_s"] - cycles[2]["pick_start_s"], 3),
                   timing_note="Timeout includes initialization; signal and physical stop times are not logged.")
    (OUT / "task1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    tex = [r"% Generated by analysis/summarize_evidence.py; task-start clock.",
           r"\begin{tikzpicture}",
           r"\begin{axis}[width=0.98\linewidth,height=7.4cm,xmin=0,xmax=180,ymin=0.35,ymax=6.65,",
           r"xlabel={Task elapsed time (s)},xtick={0,30,60,90,120,150,180},",
           r"ytick={1,2,3,4,5,6},yticklabels={파랑,나무,빨강 2,빨강 1,노랑,초록},",
           r"xmajorgrids=true,grid style={black!12},tick label style={font=\small},",
           r"legend style={at={(0.5,1.02)},anchor=south,legend columns=3,font=\small}]"]
    for index, c in enumerate(cycles):
        y = 6 - index
        start = c["pick_start_s"]
        end = c["pick_end_s"] if c["pick_end_s"] != "" else c["last_action_s"]
        tex.append(rf"\filldraw[fill=black!25,draw=black] (axis cs:{start},{y-.25}) rectangle (axis cs:{end},{y+.25});")
        if c["release_s"] != "":
            middle = c["transport_end_s"]
            tex.append(rf"\filldraw[fill=black!60,draw=black] (axis cs:{end},{y-.25}) rectangle (axis cs:{middle},{y+.25});")
            tex.append(rf"\filldraw[fill=black,draw=black] (axis cs:{middle},{y-.25}) rectangle (axis cs:{c['release_s']},{y+.25});")
        for a in [a for a in attempts if a["cycle"] == c["cycle"]]:
            tex.append(rf"\draw[black] (axis cs:{a['start_s']},{y-.25}) -- (axis cs:{a['start_s']},{y+.25});")
        if index == 2:
            tex.append(rf"\node[font=\scriptsize,anchor=south] at (axis cs:{(start+end)/2},{y+.28}) {{빈손 5회 / 51.5초}};")
        if index == 5:
            tex.append(rf"\draw[-{{Latex}},dashed] (axis cs:{end},{y}) -- (axis cs:177,{y});")
            tex.append(r"\node[font=\scriptsize,anchor=east] at (axis cs:148,1.5) {빈손 2회 후 후보 3 중단};")
    tex += [r"\addlegendimage{area legend,fill=black!25,draw=black}\addlegendentry{PICK}",
            r"\addlegendimage{area legend,fill=black!60,draw=black}\addlegendentry{VERIFY / TRANSPORT}",
            r"\addlegendimage{area legend,fill=black,draw=black}\addlegendentry{PLACE}",
            r"\end{axis}", r"\end{tikzpicture}"]
    (ROOT / "figures/task1_timeline.tex").write_text("\n".join(tex) + "\n")
    return summary


def calibration_map():
    venue = json.loads((SOURCE / "venue_accepted.json").read_text())
    h = venue["H"]
    raw = list(csv.DictReader((SOURCE / "accepted_points.csv").open(newline="")))
    rows = []
    for point in raw:
        u, v = float(point["u_px"]), float(point["v_px"])
        q = [r[0]*u + r[1]*v + r[2] for r in h]
        x, y = float(point["x_m"])*1000, float(point["y_m"])*1000
        hx, hy = q[0]/q[2], q[1]/q[2]
        rows.append(dict(name=point["name"], fk_x_mm=x, fk_y_mm=y, mapped_x_mm=hx, mapped_y_mm=hy,
                         dx_mm=hx-x, dy_mm=hy-y, residual_mm=math.hypot(hx-x, hy-y)))
    rms = math.sqrt(sum(p["residual_mm"]**2 for p in rows)/len(rows))
    if abs(rms-venue["meta"]["rms_mm"]) > 0.002:
        raise ValueError(f"Calibration matrix and CSV disagree: {rms}")
    write_csv("calibration_residuals.csv", rows)
    tex = [r"% Generated from saved H and FK pairs; arrows at true scale.",
           r"\begin{tikzpicture}",
           r"\begin{axis}[width=\linewidth,height=9.0cm,axis equal image,",
           r"xmin=-280,xmax=280,ymin=-25,ymax=335,xlabel={Base Y (mm)},ylabel={Base X (mm)},",
           r"xtick={-200,-100,0,100,200},ytick={0,100,200,300},grid=major,grid style={black!12},",
           r"tick label style={font=\small},legend style={font=\small,at={(0.5,1.02)},anchor=south,legend columns=2}]"]
    # Depict the nominal 200 x 100 mm zone, centered on its saved corners.
    # This schematic does not change the measured polygon or calibration fit.
    corners = venue["zone_polygon_mm"]
    center_x = sum(x for x, _ in corners) / len(corners)
    center_y = sum(y for _, y in corners) / len(corners)
    zone_width_mm, zone_depth_mm = 200.0, 100.0
    tex.append(r"\path[fill=black!7,draw=black!50,dashed] "
               + f"(axis cs:{center_y-zone_width_mm/2},{center_x-zone_depth_mm/2}) rectangle "
               + f"(axis cs:{center_y+zone_width_mm/2},{center_x+zone_depth_mm/2});")
    tex.append(rf"\node[font=\scriptsize] at (axis cs:{center_y},310) {{지정 영역 (규격)}};")
    tex.append(r"\addplot[only marks,mark=*,black,forget plot] coordinates {(0,0)};")
    tex.append(r"\node[font=\scriptsize,anchor=west] at (axis cs:8,0) {베이스 원점};")
    for index, p in enumerate(rows, 1):
        x,y,hx,hy = p["fk_x_mm"],p["fk_y_mm"],p["mapped_x_mm"],p["mapped_y_mm"]
        tex.append(rf"\draw[-{{Latex}},thick] (axis cs:{y},{x}) -- (axis cs:{hy},{hx});")
        tex.append(rf"\addplot[only marks,mark=o,black,mark size=2pt,forget plot] coordinates {{({y},{x})}};")
        tex.append(rf"\addplot[only marks,mark=x,black,mark size=2pt,forget plot] coordinates {{({hy},{hx})}};")
        tex.append(rf"\node[font=\scriptsize,anchor=south] at (axis cs:{y},{max(x,hx)+7}) {{P{index}}};")
    tex += [r"\draw[black] (axis cs:-230,30) rectangle (axis cs:-190,70);",
            r"\node[font=\scriptsize,anchor=north] at (axis cs:-210,25) {40 mm 블록};",
            r"\addlegendimage{only marks,mark=o}\addlegendentry{기록 FK 좌표}",
            r"\addlegendimage{only marks,mark=x}\addlegendentry{평면 변환 좌표}",
            r"\end{axis}",r"\end{tikzpicture}"]
    (ROOT / "figures/calibration_map.tex").write_text("\n".join(tex) + "\n")
    return rms


if __name__ == "__main__":
    summary = summarize_task()
    rms = calibration_map()
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCE.iterdir()) if p.is_file()}
    (OUT / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(dict(counts=summary["counts"], calibration_rms_mm=rms), indent=2))

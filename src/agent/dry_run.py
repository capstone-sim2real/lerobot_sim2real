"""so101-agent --dry-run: check calibration, IK, jog window, provider and camera without the robot bus."""

from __future__ import annotations

import os

from camera.autostart import ensure_camera_server
from config import AppConfig

from .server import API_KEY_ENV, check_jog_window


def _grid_edge_cells(cells: list) -> list:
    """The outermost cell at each end of every row and column.

    Reach runs out at the band's rim, so these are where the IK gate fails
    first -- checking them keeps the dry-run honest without solving IK for
    every one of the ~150 cells.
    """
    extremes: dict[tuple[str, int], tuple] = {}
    for cell in cells:
        for axis, key, value in (("row", cell.y, cell.x), ("col", cell.x, cell.y)):
            for name, pick in (("min", min), ("max", max)):
                slot = (f"{axis}-{name}", key)
                current = extremes.get(slot)
                if current is None or pick(value, current[1]) == value:
                    extremes[slot] = (cell, value)
    return list({(c.x, c.y): c for c, _ in extremes.values()}.values())


def dry_run(cfg: AppConfig, provider_name: str, fallback_name: str | None = None) -> int:
    """Everything the agent will rely on, checked without the robot bus."""
    import copy

    import numpy as np

    from camera.client import fetch_snapshot
    from control.ik import TopDownIK
    from control.poses import PoseRegistry
    from control.task1_transport import Task1TransportPlanner, over_ik_gate
    from perception.homography import PlaneCalibration
    from perception.scene import detect_scene
    from session.factories import calibration_grasp_z_mm
    from session.relative import table_region_xy
    from session.report import print_slot_table

    from .prompt import build_system_prompt
    from .tools import build_tools

    cfg = copy.deepcopy(cfg)
    agent = cfg.agent
    problems: list[str] = []
    print("so101-agent dry-run (no robot connection, no motion)\n")
    calib = PlaneCalibration.load(cfg.perception.calibration_path)
    print(f"calibration : {cfg.perception.calibration_path}")
    if not calib.zone_polygon_mm:
        print("  zone_polygon_mm missing -- run so101-zone-calibrate --write")
        return 1
    grasp_z = calibration_grasp_z_mm(calib)
    print(f"  grasp_z_mm={grasp_z:.1f}  rms_mm={calib.meta.get('rms_mm')}  loo_max_mm={calib.meta.get('loo_max_mm')}")
    print(f"  zone polygon mm: {[tuple(round(v, 1) for v in p) for p in calib.zone_polygon_mm]}")

    ik = TopDownIK(cfg.ik, project_root=".")
    planner = Task1TransportPlanner(calib, cfg, ik)
    labels = [f"{e}/{k}" for e, k in zip(agent.zone_slots.labels, agent.zone_slots.korean_labels)]
    print(f"\nzone cells ({agent.zone_slots.frame}: top = far row, left = +y = image left)")
    print_slot_table(planner.slots, labels=labels)
    px = calib.board_to_pixel(np.asarray([s.xy_mm for s in planner.slots]))
    for slot, (u, v) in zip(planner.slots, px):
        print(f"    {labels[slot.index]:>20s} -> image px ({u:6.1f}, {v:6.1f})")

    print("\ntable regions (column/row -> xy -> checks)")
    base = calib.base_xy_mm or (0.0, 0.0)
    from perception.detector import point_in_workspace
    from perception.zone import point_in_zone
    from control.task1_transport import solve_place_point

    for column in agent.table_regions.columns_deg:
        for row in agent.table_regions.rows_fraction:
            xy = table_region_xy(column, row, cfg.perception, agent.table_regions, base)
            flags = []
            if not point_in_workspace(xy, cfg.perception, base):
                flags.append("OUT-OF-WORKSPACE")
            try:
                solve_place_point(ik, cfg, xy, grasp_z + cfg.task1.release_clearance_mm,
                                  base_xy_mm=base, label="region")
            except ValueError as exc:
                flags.append(f"IK-GATE ({exc})")
            if flags:
                problems.append(f"table region {column}/{row}: {', '.join(flags)}")
            if point_in_zone(xy, calib, agent.table_zone_margin_mm):
                # directly in front of the zone: never chosen, not a fault
                flags.append("overlaps zone (always skipped)")
            status = "ok" if not flags else ", ".join(flags)
            print(f"  {column:9s} {row:6s} x={xy[0]:7.1f} y={xy[1]:7.1f}  {status}")

    grid_cfg = agent.board_grid
    if grid_cfg.enabled:
        from session.grid import bounds as grid_bounds
        from session.grid import build_grid, cells_in_view, cells_in_workspace, default_anchor_mm

        anchor = default_anchor_mm(cfg.perception, agent.table_regions, base)
        grid = build_grid(calib.board_grid, grid_cfg, anchor)
        cells = cells_in_workspace(grid, cfg.perception, agent.table_regions, base, grid_cfg)
        cells = cells_in_view(cells, lambda pts: calib.board_to_pixel(np.asarray(pts)), calib.image_size)
        span = grid_bounds((c.x, c.y) for c in cells)
        source = "measured board" if calib.board_grid else "axis-aligned fallback (no board_grid)"
        print(f"\nboard cells ({source}, cell {grid.cell_mm:.1f}mm, image x+ = right, y+ = away)")
        print(f"  {len(cells)} cells, x {span['x'][0]}..{span['x'][1]}, y {span['y'][0]}..{span['y'][1]}, "
              f"(0,0) at x={grid.origin_mm[0]:.1f} y={grid.origin_mm[1]:.1f} mm")
        if not calib.board_grid:
            problems.append(
                "no measured board grid: cells are axis-aligned guesses "
                "(run tools.calibrate_board_grid --write)"
            )
        if not cells:
            problems.append("board grid has no cell inside the workspace")
        # IK on every cell would take minutes; the band's edges are where it
        # fails first, so checking those bounds the risk without the cost.
        edge = _grid_edge_cells(cells)
        failed = []
        for cell in edge:
            try:
                solve_place_point(ik, cfg, cell.xy_mm, grasp_z + cfg.task1.release_clearance_mm,
                                  base_xy_mm=base, label="cell")
            except ValueError as exc:
                failed.append((cell, str(exc)))
        print(f"  edge cells checked: {len(edge)}, IK-GATE failures: {len(failed)}")
        for cell, reason in failed[:5]:
            print(f"    ({cell.x:+3d}, {cell.y:+3d}) x={cell.xy_mm[0]:7.1f} y={cell.xy_mm[1]:7.1f}  {reason}")
        if failed:
            problems.append(f"{len(failed)}/{len(edge)} board edge cells fail the IK gate")

    home = PoseRegistry.load(cfg.motion.poses_path).get(cfg.motion.home_pose)
    hx, hy, hz = ik.forward_position_mm(home)
    rel = agent.relative
    print(f"\nhome tool-frame position: x={hx:.1f} y={hy:.1f} z={hz:.1f}mm")
    print(f"jog window z: {rel.jog_min_z_mm:g}..{rel.jog_max_z_mm:g}mm, max {rel.max_jog_mm:g}mm per call, frame={rel.frame}")
    try:
        check_jog_window(cfg, grasp_z)
    except ValueError as exc:
        problems.append(str(exc))
        print(f"  PROBLEM: {exc}")
    probe = ik.solve(hx, hy, rel.jog_min_z_mm)
    print(f"  jog entry from home at z={rel.jog_min_z_mm:g}: IK error {probe.position_error_mm:.1f}mm"
          f" ({'ok' if not over_ik_gate(probe, cfg) else 'OVER GATE'})")

    print(f"\nprovider: {provider_name}  model: {agent.models.get(provider_name)}")
    for env in API_KEY_ENV.get(provider_name, []):
        print(f"  {env}: {'set' if os.environ.get(env) else 'NOT SET'}")
    if fallback_name is not None:
        print(f"fallback: {fallback_name}  model: {agent.models.get(fallback_name)}")
        for env in API_KEY_ENV.get(fallback_name, []):
            print(f"  {env}: {'set' if os.environ.get(env) else 'NOT SET'}")
    tools = build_tools(cfg)
    print(f"tools ({len(tools)}): {', '.join(t.spec.name for t in tools)}")
    try:
        prompt = build_system_prompt(cfg)
        print(f"system prompt: {len(prompt)} chars from {agent.system_prompt_path}")
    except OSError as exc:
        problems.append(f"system prompt: {exc}")

    print(f"\ncamera: {cfg.perception.snapshot_url}")
    camera_proc = None
    try:
        if cfg.camera.auto_start:
            camera_proc = ensure_camera_server(agent.camera_base_url, extra_args=cfg.camera.extra_args)
        frame = fetch_snapshot(cfg.perception.snapshot_url)
        scene = detect_scene(frame, calib, cfg.perception, [s.xy_mm for s in planner.slots],
                             zone_max_per_color=agent.zone_scan_max_per_color,
                             snap_radius_mm=agent.slot_snap_radius_mm)
        for block in scene.outside.values():
            print(f"  outside {block.color:6s} x={block.center_mm[0]:7.1f} y={block.center_mm[1]:7.1f}")
        for block in scene.inside.values():
            slot = agent.zone_slots.labels[block.slot_index] if block.slot_index is not None else "-"
            print(f"  inside  {block.color:6s} x={block.center_mm[0]:7.1f} y={block.center_mm[1]:7.1f} slot={slot}")
        if not scene.all():
            print("  no blocks detected")
    except Exception as exc:  # noqa: BLE001
        print(f"  unavailable ({exc}) -- start so101-camera before the demo")
    finally:
        if camera_proc is not None:
            camera_proc.stop()

    print("\n" + ("all checks passed" if not problems else "problems:\n  - " + "\n  - ".join(problems)))
    return 0 if not problems else 2

"""Pure conservative neighbour check for experimental calibration motions."""
import math


def clearance_check(points, obstacles, cfg):
    values = (cfg.tool_radius_mm, cfg.block_radius_mm,
              cfg.uncertainty_mm, cfg.obstacle_height_mm)
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Clearance envelope dimensions must be finite and positive")
    limit = cfg.tool_radius_mm + cfg.block_radius_mm + cfg.uncertainty_mm
    conflicts = set()
    nearest = math.inf
    for x, y, z in points:
        if not all(math.isfinite(v) for v in (x,y,z)):
            raise ValueError("Non-finite path")
        if z > cfg.obstacle_height_mm + cfg.tool_radius_mm + cfg.uncertainty_mm:
            continue
        for color, xy in obstacles.items():
            if not all(math.isfinite(v) for v in xy):
                raise ValueError("Non-finite obstacle")
            gap = math.hypot(x-xy[0], y-xy[1]) - limit
            nearest = min(nearest, gap)
            if gap <= 0:
                conflicts.add(color)
    return {"clear": not conflicts, "conflicts": sorted(conflicts),
            "margin_mm": None if math.isinf(nearest) else round(nearest, 1),
            "assumed_envelope": True}

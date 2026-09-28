"""Simple directional top-view clearance for block grasps."""
import math


def _polygon(value):
    if isinstance(value, dict) and "center" in value:
        cx, cy = map(float, value["center"])
        side = float(value["side_mm"])
        angle = math.radians(float(value.get("angle_deg", 0.0)))
        if not all(math.isfinite(v) for v in (cx, cy, side, angle)) or side <= 0:
            raise ValueError("Invalid fixed block outline")
        c, s = math.cos(angle), math.sin(angle)
        half = side / 2.0
        return [
            (cx + c*x - s*y, cy + s*x + c*y)
            for x, y in ((-half, -half), (half, -half), (half, half), (-half, half))
        ]
    points = value["box"] if isinstance(value, dict) else value
    result = [tuple(map(float, point)) for point in points]
    if len(result) < 3 or not all(math.isfinite(v) for point in result for v in point):
        raise ValueError("Invalid block outline")
    return result


def _interval(points, axis):
    values = [x * axis[0] + y * axis[1] for x, y in points]
    return min(values), max(values)


def _centroid(points):
    return tuple(sum(point[i] for point in points) / len(points) for i in (0, 1))


def _side_mm(points):
    area = abs(sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )) / 2.0
    return math.sqrt(area)


def _jaw_rect(center, along, across, along_min, along_max, half_span):
    return [
        (center[0] + along[0] * a + across[0] * b,
         center[1] + along[1] * a + across[1] * b)
        for a, b in ((along_min, -half_span), (along_max, -half_span),
                     (along_max, half_span), (along_min, half_span))
    ]


def _convex_intersects(first, second):
    for polygon in (first, second):
        for i, point in enumerate(polygon):
            other = polygon[(i + 1) % len(polygon)]
            axis = (-(other[1] - point[1]), other[0] - point[0])
            a = _interval(first, axis)
            b = _interval(second, axis)
            if a[1] < b[0] or b[1] < a[0]:
                return False
    return True


def directional_clearance(target_box, obstacles, cfg, axis_deg, *, jaw_center_mm=None):
    """Check two jaw pads at the predicted physical jaw centre.

    All coordinates are robot-base millimetres. This also rejects a jaw
    overlapping the target top, even when no neighbouring block exists.
    """
    depth = cfg.uncertainty_mm
    inner_clearance = cfg.jaw_inner_clearance_mm
    if (not math.isfinite(depth) or depth < 0 or not math.isfinite(axis_deg)
            or not math.isfinite(inner_clearance) or inner_clearance < 0):
        raise ValueError("Clearance values must be finite and non-negative")
    angle = math.radians(axis_deg)
    along = (math.cos(angle), math.sin(angle))
    across = (-along[1], along[0])
    target = _polygon(target_box)
    target_center = _centroid(target)
    center = target_center if jaw_center_mm is None else tuple(map(float, jaw_center_mm))
    if len(center) != 2 or not all(math.isfinite(v) for v in center):
        raise ValueError("Invalid jaw centre")

    nominal_side = cfg.block_radius_mm * math.sqrt(2.0)
    target_local = [(x-target_center[0], y-target_center[1]) for x, y in target]
    target_half_along = max(abs(value) for value in (
        x*along[0] + y*along[1] for x, y in target_local
    ))
    # Put the inner pad edges just outside the detected top projection. This
    # handles a rotated square without weakening the centre-offset check.
    half_gap = target_half_along + inner_clearance
    half_span = min(_side_mm(target), nominal_side) / 4.0
    jaws = [
        _jaw_rect(center, along, across, -half_gap-depth, -half_gap, half_span),
        _jaw_rect(center, along, across, half_gap, half_gap+depth, half_span),
    ]

    target_hits = [index for index, jaw in enumerate(jaws)
                   if _convex_intersects(jaw, target)]
    conflicts = []
    gaps = {}
    for color, value in obstacles.items():
        obstacle = _polygon(value)
        hits = [index for index, jaw in enumerate(jaws)
                if _convex_intersects(jaw, obstacle)]
        centre_gap = math.dist(target_center, _centroid(obstacle))
        gaps[color] = {"centre": round(centre_gap, 1), "jaw_hits": hits}
        if hits:
            conflicts.append(color)
    return {
        "clear": not conflicts and not target_hits,
        "conflicts": sorted(conflicts),
        "target_overlap": bool(target_hits),
        "target_jaw_hits": target_hits,
        "axis_deg": round(axis_deg % 180.0, 1),
        "margin_mm": depth,
        "gaps_mm": gaps,
        "jaw_center_mm": list(center),
        "jaw_footprints_mm": [[list(point) for point in jaw] for jaw in jaws],
        "geometry": "two-jaw top-view footprint",
    }


def top_view_clearance(target_box, obstacles, cfg):
    """Report whether either world-X or world-Y grasp direction is free."""
    x_axis = directional_clearance(target_box, obstacles, cfg, 0.0)
    y_axis = directional_clearance(target_box, obstacles, cfg, 90.0)
    return {
        "clear": x_axis["clear"] or y_axis["clear"],
        "conflicts": sorted(set(x_axis["conflicts"]) & set(y_axis["conflicts"])),
        "margin_mm": cfg.uncertainty_mm,
        "orientations": {"x": x_axis, "y": y_axis},
        "geometry": "two-direction top-view block-outline gap",
    }

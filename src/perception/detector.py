"""Block detection: colour + shape combined, never colour alone.

Runs on the homography-rectified metric view so every threshold is in mm —
independent of where the camera sits. A red block and red tape share hue but
not geometry: tape is thin/elongated/hollow at corners, a block is a filled
~40x40 mm square. The aspect/solidity/fill filters encode exactly that
(AGENTS.md §9), so red-on-red scenes resolve by form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from config import PerceptionConfig
from perception.homography import PlaneCalibration
from perception.zone import point_in_zone


@dataclass
class BlockDetection:
    color: str
    center_mm: tuple[float, float]
    area_mm2: float
    aspect: float
    solidity: float
    fill: float
    # minAreaRect corners in board mm, for debug rendering
    box_mm: list[tuple[float, float]]
    # Edge direction in board mm, folded to [0, 90) because a square grasps
    # identically every 90 deg. Default keeps positional construction working.
    angle_deg: float = 0.0
    # Median (hue, saturation) of the blob's interior. This, not the gate that
    # caught it, is what ``color`` was decided from.
    hue_sat: tuple[float, float] = (0.0, 0.0)


@dataclass
class RejectedCandidate:
    """A contour that passed the colour mask but failed one shape gate.

    Display-only: nothing in SELECT, IK or the task runner ever reads these.
    They exist because a bare ``continue`` makes "no block here" and "block
    seen, but fill was 0.48" look identical on the operator page, which is
    exactly the ambiguity that hid the dark-edge detection failures.
    """

    color: str
    center_mm: tuple[float, float]
    box_mm: list[tuple[float, float]]
    area_mm2: float
    aspect: float
    solidity: float
    fill: float
    # first gate that dropped it: "area" | "aspect" | "fill" | "solidity"
    reason: str


def _evaluate_contour(
    contour: np.ndarray,
    cfg: PerceptionConfig,
    mm2_per_px2: float,
    origin_x: float,
    origin_y: float,
) -> tuple[BlockDetection | None, RejectedCandidate | None]:
    """Apply the shape gates to one contour.

    Returns ``(detection, None)`` when every gate passes and ``(None, reject)``
    when one fails. ``(None, None)`` means the contour is degenerate and not
    worth reporting either way.
    """
    area_px = cv2.contourArea(contour)
    area_mm2 = area_px * mm2_per_px2
    (cx, cy), (rw, rh), rect_angle = cv2.minAreaRect(contour)
    if min(rw, rh) <= 0:
        return None, None
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area <= 0:
        return None, None

    aspect = max(rw, rh) / min(rw, rh)
    fill = area_px / (rw * rh)
    solidity = area_px / hull_area

    def to_mm(px: float, py: float) -> tuple[float, float]:
        return (
            origin_x + px * cfg.rectified_mm_per_px,
            origin_y + py * cfg.rectified_mm_per_px,
        )

    box_mm = [to_mm(px, py) for px, py in cv2.boxPoints(((cx, cy), (rw, rh), rect_angle))]
    centre_mm = to_mm(cx, cy)

    # Gate order is load-bearing: ``reason`` reports the *first* failure, and
    # area is checked first so oversized clutter is not mislabelled as a
    # shape problem.
    reason: str | None = None
    if not (cfg.area_mm2_min <= area_mm2 <= cfg.area_mm2_max):
        reason = "area"
    elif aspect > cfg.aspect_ratio_max:
        reason = "aspect"
    elif fill < cfg.fill_min:
        reason = "fill"
    elif solidity < cfg.solidity_min:
        reason = "solidity"

    if reason is not None:
        return None, RejectedCandidate(
            color="",  # filled in by the caller, which knows the mask colour
            center_mm=centre_mm,
            box_mm=box_mm,
            area_mm2=float(area_mm2),
            aspect=float(aspect),
            solidity=float(solidity),
            fill=float(fill),
            reason=reason,
        )

    # Taken from two adjacent corners rather than minAreaRect's own angle:
    # box_mm is already board mm (rectify() is a positive uniform scale plus a
    # translation, so it preserves angles and handedness), which sidesteps both
    # the frame conversion and OpenCV's version-dependent angle range.
    (ax, ay), (bx, by) = box_mm[0], box_mm[1]
    angle_deg = math.degrees(math.atan2(by - ay, bx - ax)) % 90.0
    return (
        BlockDetection(
            color="",  # filled in by the caller
            center_mm=centre_mm,
            area_mm2=float(area_mm2),
            aspect=float(aspect),
            solidity=float(solidity),
            fill=float(fill),
            box_mm=box_mm,
            angle_deg=float(angle_deg),
        ),
        None,
    )


def detect_blocks(
    frame: np.ndarray,
    calib: PlaneCalibration,
    cfg: PerceptionConfig,
    *,
    is_rgb: bool = True,
) -> list[BlockDetection]:
    """Detect candidate blocks in a top-camera frame.

    ``is_rgb`` is True for frames from lerobot cameras (RGB) and False for
    images loaded with cv2.imread (BGR).
    """
    detections, _ = detect_blocks_with_rejects(
        frame, calib, cfg, is_rgb=is_rgb, collect_rejects=False
    )
    return detections


def detect_blocks_with_rejects(
    frame: np.ndarray,
    calib: PlaneCalibration,
    cfg: PerceptionConfig,
    *,
    is_rgb: bool = True,
    collect_rejects: bool = True,
    min_reject_area_mm2: float = 300.0,
) -> tuple[list[BlockDetection], list[RejectedCandidate]]:
    """Same detection as :func:`detect_blocks`, plus the near-misses.

    The rejects are for the operator display only. ``min_reject_area_mm2``
    drops mask speckle so the page shows plausible blocks, not noise; set
    ``collect_rejects=False`` to skip building them entirely.
    """
    rectified, (origin_x, origin_y) = calib.rectify(frame, cfg.rectified_mm_per_px)
    hsv = cv2.cvtColor(rectified, cv2.COLOR_RGB2HSV if is_rgb else cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_kernel_px, cfg.morph_kernel_px))
    mm2_per_px2 = cfg.rectified_mm_per_px**2
    base_xy = calib.base_xy_mm or (0.0, 0.0)

    detections: list[BlockDetection] = []
    rejects: list[RejectedCandidate] = []
    for color, bands in cfg.hsv_ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo_h, lo_s, lo_v, hi_h, hi_s, hi_v in bands:
            mask |= cv2.inRange(hsv, (lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            detection, reject = _evaluate_contour(contour, cfg, mm2_per_px2, origin_x, origin_y)
            if detection is not None:
                detection.color = color
                detection.hue_sat = _median_hue_sat(hsv, contour)
                detections.append(detection)
            elif reject is not None and collect_rejects and reject.area_mm2 >= min_reject_area_mm2:
                reject.color = color
                rejects.append(reject)

    # Off the board before anything else: the arm cannot reach out there, and
    # the clutter that lives there (the wooden floor past the mat, the far
    # wall) is what produces phantom warm-coloured candidates. Dropped
    # silently rather than reported — an operator overlay covered in markers
    # for objects nothing will ever pick is noise, not diagnostics.
    if cfg.workspace_radius_mm > 0:
        detections = [d for d in detections if _in_workspace(d.center_mm, cfg, base_xy)]
        rejects = [r for r in rejects if _in_workspace(r.center_mm, cfg, base_xy)]

    # The fixed target zone is outside the detector's active region, just as
    # the area beyond the reach sector is. Do this before colour assignment:
    # an already-placed block must not consume the one allowed slot for its
    # colour and hide another block of that colour outside the zone.
    if calib.zone_polygon_mm:
        detections = [d for d in detections if not point_in_zone(d.center_mm, calib)]
        rejects = [r for r in rejects if not point_in_zone(r.center_mm, calib)]

    # One physical block can trip several gates (they overlap on purpose), so
    # merge coincident blobs into one candidate BEFORE naming it. Merging
    # after would let the same block hold two colour slots.
    if cfg.min_color_separation_mm > 0:
        detections, merged = _merge_coincident(detections, cfg.min_color_separation_mm)
        if collect_rejects:
            rejects.extend(_as_rejects(merged, "merged"))

    detections, unnamed = _assign_colors(detections, cfg)
    if collect_rejects:
        rejects.extend(_as_rejects(unnamed, "unassigned"))

    detections.sort(key=lambda d: (d.center_mm[1], d.center_mm[0]))
    rejects.sort(key=lambda r: -r.area_mm2)
    return detections, rejects


def _as_rejects(
    detections: list[BlockDetection], reason: str
) -> list[RejectedCandidate]:
    return [
        RejectedCandidate(
            color=d.color,
            center_mm=d.center_mm,
            box_mm=d.box_mm,
            area_mm2=d.area_mm2,
            aspect=d.aspect,
            solidity=d.solidity,
            fill=d.fill,
            reason=reason,
        )
        for d in detections
    ]


def _median_hue_sat(hsv: np.ndarray, contour: np.ndarray) -> tuple[float, float]:
    """Median hue and saturation inside a contour, edges eroded away.

    The border pixels of a blob are a blend of block and table, and at the
    dark edges of the board that blend is most of what a naive mean would
    see. Eroding first is what makes the measurement stable enough to name a
    colour from.
    """
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour - (x, y)], -1, 255, -1)
    eroded = cv2.erode(mask, np.ones((7, 7), np.uint8))
    if int(eroded.sum()) < 255 * 20:
        eroded = mask  # tiny blob: keep what there is rather than nothing
    patch = hsv[y : y + h, x : x + w][eroded > 0]
    if not len(patch):
        return (0.0, 0.0)
    return (float(np.median(patch[:, 0])), float(np.median(patch[:, 1])))


def _in_workspace(
    center_mm: tuple[float, float], cfg: PerceptionConfig, base_xy: tuple[float, float]
) -> bool:
    dx = center_mm[0] - base_xy[0]
    dy = center_mm[1] - base_xy[1]
    if math.hypot(dx, dy) > cfg.workspace_radius_mm:
        return False
    azimuth = math.degrees(math.atan2(dy, dx))
    return cfg.workspace_angle_min_deg <= azimuth <= cfg.workspace_angle_max_deg


def _merge_coincident(
    detections: list[BlockDetection], min_separation_mm: float
) -> tuple[list[BlockDetection], list[BlockDetection]]:
    """Collapse blobs that are the same object seen through different gates."""
    kept: list[BlockDetection] = []
    dropped: list[BlockDetection] = []
    for detection in sorted(detections, key=_blockiness, reverse=True):
        coincident = any(
            math.dist(other.center_mm, detection.center_mm) < min_separation_mm
            for other in kept
        )
        (dropped if coincident else kept).append(detection)
    return kept, dropped


def _prototype_distance(
    hue_sat: tuple[float, float], point: list[int], cfg: PerceptionConfig
) -> float:
    hue_gap = abs(hue_sat[0] - point[0])
    hue_gap = min(hue_gap, 180.0 - hue_gap)  # OpenCV hue wraps at 180
    return math.hypot(
        cfg.prototype_hue_weight * hue_gap / 128.0,
        cfg.prototype_saturation_weight * (hue_sat[1] - point[1]) / 128.0,
    )


def _nearest_prototype_distance(
    hue_sat: tuple[float, float], points: list[list[int]], cfg: PerceptionConfig
) -> float:
    """A colour's own points cover different lighting regimes; a blob only
    has to be close to whichever one applies."""
    return min(_prototype_distance(hue_sat, point, cfg) for point in points)


def _assign_colors(
    detections: list[BlockDetection], cfg: PerceptionConfig
) -> tuple[list[BlockDetection], list[BlockDetection]]:
    """Name each blob by its nearest colour prototype, one colour per block.

    Greedy over the globally best (blob, colour) pair, which is what the
    arena guarantees make correct: exactly one block of each colour exists,
    so the confident matches should claim their colours first and leave the
    ambiguous ones to sort out among what is left. That is why a blob whose
    gate said "yellow" can still come out wood — the gate only decided it was
    worth looking at.
    """
    limit = cfg.max_per_color if cfg.max_per_color > 0 else len(detections)
    pairs = sorted(
        (
            (_nearest_prototype_distance(d.hue_sat, points, cfg), index, color)
            for index, d in enumerate(detections)
            for color, points in cfg.color_prototypes.items()
        ),
        key=lambda pair: pair[0],
    )
    taken: dict[str, int] = {}
    named: dict[int, str] = {}
    for distance, index, color in pairs:
        if distance > cfg.prototype_max_distance:
            break  # sorted ascending: nothing further along can match either
        if index in named or taken.get(color, 0) >= limit:
            continue
        named[index] = color
        taken[color] = taken.get(color, 0) + 1

    kept, unnamed = [], []
    for index, detection in enumerate(detections):
        if index in named:
            detection.color = named[index]
            kept.append(detection)
        else:
            unnamed.append(detection)
    return kept, unnamed


def _blockiness(detection: BlockDetection) -> tuple[float, float]:
    """Rank surviving candidates of one colour: most block-like first.

    ``fill * solidity`` separates them well in practice — a real block fills
    its bounding rectangle and has no concavities (measured ~0.75-0.88),
    while a blob that only just cleared the gates runs ~0.55-0.70. Area
    breaks ties toward the larger candidate rather than toward the nominal
    1600 mm^2, because a block seen at high azimuth legitimately projects
    bigger (its side faces show) and must not lose to a smaller artefact.
    """
    return (detection.fill * detection.solidity, detection.area_mm2)


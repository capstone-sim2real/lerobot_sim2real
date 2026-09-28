"""YOLOE instance geometry. No motor access and no implicit CV fallback.

Coordinates project the visible silhouette onto the calibrated block-top plane.
They are estimates, NOT verified top-face centres or validated grasp targets.
The hybrid path clips the existing colour/shape detector to one YOLOE instance,
so same-colour tape cannot join the block contour.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import cv2
import numpy as np

from config import PerceptionConfig
from perception.homography import PlaneCalibration
from perception.detector import (
    _evaluate_contour,
    _median_hue_sat,
    _nearest_prototype_distance,
    point_in_workspace,
)

_MIN_HYBRID_CONTOUR_MM2 = 300.0


@dataclass(frozen=True)
class HybridFrameContext:
    """Rectified colour masks shared by every YOLOE instance in one frame."""

    hsv: np.ndarray
    origin: tuple[float, float]
    color_masks: dict[str, np.ndarray]


def prepare_hybrid_frame(
    frame: np.ndarray,
    calib: PlaneCalibration,
    cfg: PerceptionConfig,
    *,
    is_rgb: bool = False,
) -> HybridFrameContext:
    if frame.ndim != 3 or frame.shape[1::-1] != tuple(calib.image_size):
        raise ValueError("Frame resolution differs from calibration image_size")
    rectified, origin = calib.rectify(frame, cfg.rectified_mm_per_px)
    hsv = cv2.cvtColor(
        rectified, cv2.COLOR_RGB2HSV if is_rgb else cv2.COLOR_BGR2HSV
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.morph_kernel_px, cfg.morph_kernel_px)
    )
    color_masks = {}
    for color, bands in cfg.hsv_ranges.items():
        color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo_h, lo_s, lo_v, hi_h, hi_s, hi_v in bands:
            color_mask |= cv2.inRange(hsv, (lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_masks[color] = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    return HybridFrameContext(hsv=hsv, origin=origin, color_masks=color_masks)


def mask_geometry(mask: np.ndarray, calib: PlaneCalibration,
                  cfg: PerceptionConfig) -> dict:
    """Apply existing metric shape/workspace gates to a single instance mask.

    Reject resolution mismatches rather than silently misapplying calibration.
    Retain rejected objects and reasons for inspection. No one-object-per-colour
    constraint: touching same-colour instances may be distinct YOLOE masks.
    """
    if mask.ndim != 2 or mask.shape[::-1] != tuple(calib.image_size):
        raise ValueError("Mask resolution differs from calibration image_size")
    if calib.base_xy_mm is None or not np.allclose(calib.base_xy_mm, (0, 0)):
        raise ValueError("Robot-base calibration (base_xy_mm=0,0) is required")
    if not np.isfinite(calib.H).all() or abs(np.linalg.det(calib.H)) < 1e-12:
        raise ValueError("Invalid calibration homography")
    binary = (mask > 0).astype(np.uint8) * 255
    moments = cv2.moments(binary, binaryImage=True)
    if moments["m00"] == 0:
        return {"accepted": False, "reason": "empty_mask"}
    center_px = [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
    rectified, origin = calib.rectify(binary, cfg.rectified_mm_per_px)
    contours, _ = cv2.findContours((rectified >= 128).astype(np.uint8),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = {"center_px": center_px, "coordinate_frame": "robot_base_mm",
              "coordinate_method": "visible_silhouette_on_block_top_plane",
              "grasp_ready": False}
    if not contours:
        return {**result, "accepted": False, "reason": "degenerate_mask"}
    contour = max(contours, key=cv2.contourArea)
    detection, rejected = _evaluate_contour(contour, cfg,
        cfg.rectified_mm_per_px ** 2, *origin)
    geometry = detection or rejected
    if geometry is None:
        return {**result, "accepted": False, "reason": "degenerate_mask"}
    result.update(asdict(geometry))
    result["center_mm"] = [float(v) for v in geometry.center_mm]
    result["box_mm"] = [[float(v) for v in corner] for corner in geometry.box_mm]
    reason = rejected.reason if rejected else None
    if cfg.workspace_radius_mm > 0 and not point_in_workspace(geometry.center_mm, cfg, calib.base_xy_mm):
        reason = reason or "outside_workspace"
    result.update(accepted=reason is None, reason=reason)
    return result


def hybrid_color_geometry(
    frame: np.ndarray,
    mask: np.ndarray,
    calib: PlaneCalibration,
    cfg: PerceptionConfig,
    *,
    is_rgb: bool = False,
    context: HybridFrameContext | None = None,
) -> dict:
    """Measure the existing colour contour clipped to one YOLOE instance.

    This is deliberately not called a top-face segmentation. Colour boundaries
    often follow the top face, but a similarly coloured side can remain. The
    result therefore stays ``grasp_ready=False`` until physical validation.
    """
    silhouette = mask_geometry(mask, calib, cfg)
    if frame.ndim != 3 or frame.shape[:2] != mask.shape:
        raise ValueError("Frame and YOLOE mask resolutions differ")
    if not silhouette.get("center_mm"):
        return {
            "accepted": False,
            "reason": silhouette.get("reason", "degenerate_mask"),
            "grasp_ready": False,
        }

    binary = (mask > 0).astype(np.uint8) * 255
    rectified_mask, _ = calib.rectify(binary, cfg.rectified_mm_per_px)
    rectified_mask = (rectified_mask >= 128).astype(np.uint8) * 255
    context = context or prepare_hybrid_frame(
        frame, calib, cfg, is_rgb=is_rgb
    )
    candidates = []
    for color, frame_color_mask in context.color_masks.items():
        color_mask = cv2.bitwise_and(frame_color_mask, rectified_mask)
        contours, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            detection, rejected = _evaluate_contour(
                contour, cfg, cfg.rectified_mm_per_px**2, *context.origin
            )
            geometry = detection or rejected
            if geometry is None or geometry.area_mm2 < _MIN_HYBRID_CONTOUR_MM2:
                continue
            hue_sat = _median_hue_sat(context.hsv, contour)
            distance = _nearest_prototype_distance(
                hue_sat, cfg.color_prototypes[color], cfg
            )
            candidates.append(
                (
                    distance,
                    -geometry.area_mm2,
                    color,
                    detection,
                    rejected,
                    geometry,
                    hue_sat,
                )
            )

    if not candidates:
        return {
            "accepted": False,
            "reason": "no_color_contour",
            "grasp_ready": False,
        }

    distance, _, color, detection, rejected, geometry, hue_sat = min(candidates)
    reason = rejected.reason if rejected else None
    if distance > cfg.prototype_max_distance:
        reason = reason or "unassigned_color"
    if cfg.workspace_radius_mm > 0 and not point_in_workspace(
        geometry.center_mm, cfg, calib.base_xy_mm
    ):
        reason = reason or "outside_workspace"

    result = asdict(geometry)
    result.update(
        color=color,
        hue_sat=[float(value) for value in hue_sat],
        prototype_distance=float(distance),
        center_mm=[float(value) for value in geometry.center_mm],
        box_mm=[[float(value) for value in corner] for corner in geometry.box_mm],
        coordinate_frame="robot_base_mm",
        coordinate_method="yoloe_instance_intersected_with_existing_color_shape_mask",
        accepted=reason is None,
        reason=reason,
        grasp_ready=False,
    )
    return result

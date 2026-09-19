"""Dry-run console tables shared by the runners and ``so101-agent --dry-run``."""

from __future__ import annotations

from typing import Callable, Iterable

from perception.detector import BlockDetection


def print_detections(
    detections: Iterable[BlockDetection],
    *,
    title: str = "active outside-zone detections:",
    extra: Callable[[BlockDetection], str] | None = None,
) -> None:
    detections = list(detections)
    print(title)
    if not detections:
        print("  none")
        return
    for detection in detections:
        x, y = detection.center_mm
        suffix = extra(detection) if extra is not None else f"area={detection.area_mm2:.0f}mm2"
        print(f"  {detection.color:6s} x={x:7.1f} y={y:7.1f} {suffix}")


def print_slot_table(slots, *, show_tilt: bool = True, labels: list[str] | None = None) -> None:
    print("placement slots (far row first):")
    for slot in slots:
        label = f" {labels[slot.index]:>12s}" if labels is not None else ""
        tilt = f"tilt={slot.radial_tilt_deg:.1f}deg " if show_tilt else ""
        print(
            f"  {slot.index}:{label} x={slot.xy_mm[0]:7.1f} y={slot.xy_mm[1]:7.1f} "
            f"drop_z={slot.drop_z_mm:.1f} hover_z={slot.hover_z_mm:.1f} "
            f"{tilt}"
            f"errors=({slot.drop.position_error_mm:.2f}, {slot.hover.position_error_mm:.2f})mm"
        )

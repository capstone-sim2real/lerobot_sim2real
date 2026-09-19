"""Where a released block actually lands, against where it was told to.

The pixel->mm calibration is not what limits placement accuracy here. Its
own residuals are RMS ~5mm / worst ~14mm (AGENTS.md §6), while placements
miss by tens of millimetres, and the reason is written into
``motion.arrival_tol``: ``release_at`` opens the jaws once every joint is
within 3 degrees of its command, and this rig measured 3 degrees as ~29mm of
tool travel at 283mm reach (``MotionConfig.grasp_hover_arrival_tol``'s
comment). Carrying a block adds a steady-state droop on top, which
``TrajectoryPlayer.move_to`` documents as something waiting does not close.

That error is mostly *systematic*: the same target misses the same way. So
it can be measured instead of modelled -- the agent already re-observes
after every placement to verify it, which is exactly the measurement needed.
This module turns that stream of (told, landed) pairs into a running command
offset.

The offset lives in the **arm frame** (forward = radial, left = tangential,
``session.relative``), not in base x/y: droop is a function of how far the
arm is extended and how far the shoulder has swung, so a correction learned
at one azimuth only transfers to another in that frame.

Nothing here is a calibration constant: it starts at whatever config seeds
it with (0 by default), and it only moves on measurements taken on the rig
in front of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from session.relative import decompose_xy, offset_xy

XY = tuple[float, float]


@dataclass
class PlaceCorrection:
    """Running mean of the offset that makes released blocks land on target."""

    forward_mm: float = 0.0
    left_mm: float = 0.0
    samples: int = 0
    max_mm: float = 60.0
    # A miss larger than this is not bias: it is a knocked block, a bad
    # grasp, or the detector reporting a different object. Learning from it
    # would poison the mean.
    max_sample_mm: float = 80.0

    @property
    def magnitude_mm(self) -> float:
        return math.hypot(self.forward_mm, self.left_mm)

    def command_xy(self, target_xy: XY, *, base_xy_mm: XY, frame: str) -> XY:
        """The point to command so the block lands on ``target_xy``."""
        if self.forward_mm == 0.0 and self.left_mm == 0.0:
            return target_xy
        return offset_xy(
            target_xy, self.forward_mm, self.left_mm, frame=frame, base_xy_mm=base_xy_mm
        )

    def observe(self, target_xy: XY, measured_xy: XY, *, base_xy_mm: XY, frame: str) -> bool:
        """Fold one verified placement into the mean. True when it counted.

        ``measured_xy`` is where the camera found the block after the
        release, ``target_xy`` where the operator asked for it -- the
        residual between them is what the next command has to cancel.
        """
        forward, left = decompose_xy(target_xy, measured_xy, frame=frame, base_xy_mm=base_xy_mm)
        if math.hypot(forward, left) > self.max_sample_mm:
            return False
        self.samples += 1
        # running mean of (applied correction - residual): the estimate of
        # the correction that would have landed this one on target
        self.forward_mm -= forward / self.samples
        self.left_mm -= left / self.samples
        self._clamp()
        return True

    def _clamp(self) -> None:
        magnitude = self.magnitude_mm
        if magnitude > self.max_mm > 0:
            scale = self.max_mm / magnitude
            self.forward_mm *= scale
            self.left_mm *= scale

    def as_dict(self) -> dict[str, float | int]:
        return {
            "forward_mm": round(self.forward_mm, 1),
            "left_mm": round(self.left_mm, 1),
            "samples": self.samples,
        }

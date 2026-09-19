"""The one result shape every agent skill returns.

The LLM never has to learn a result format per tool: ``ok`` + a closed
``reason`` vocabulary + ``retry_advice`` decide what happens next, ``detail``
is a Korean sentence it may relay, ``data`` carries tool-specific facts and
``state`` is the world snapshot taken right after the skill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

REASONS = frozenset(
    {
        "ok", "held", "released", "moved", "not_detected", "not_in_zone", "unreachable",
        "grasp_empty", "grasp_blocked", "motion_timeout", "slot_occupied", "no_block_held",
        "already_holding", "ik_gate", "camera_stale", "camera_unreachable", "cancelled",
        "bus_lost", "precondition", "disabled", "out_of_workspace", "destination_in_zone",
        "destination_blocked", "destination_unreachable", "no_free_region", "limit_exceeded",
        "scene_incomplete", "neighbour_clearance", "scene_reposition_disabled",
        "height_limit", "invalid_arguments", "internal_error", "task_incomplete",
    }
)

# A turn that produced one of these leaves the arm somewhere unknown or
# stopped: the server refuses new commands until a verified return home.
ROBOT_FAULT_REASONS = frozenset({"cancelled", "motion_timeout", "bus_lost", "internal_error"})

RETRY_ADVICE = frozenset({"retry_ok", "do_not_retry", "ask_operator"})


@dataclass(frozen=True)
class ObservationImage:
    """JPEG bytes are separate from numeric result JSON and browser events."""
    jpeg: bytes
    camera: str
    frame_seq: int
    captured_at: float


@dataclass
class SkillResult:
    ok: bool
    action: str
    reason: str
    detail: str = ""
    retry_advice: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] | None = None
    elapsed_s: float = 0.0
    images: tuple[ObservationImage, ...] = ()

    def __post_init__(self) -> None:
        if self.reason not in REASONS:
            raise ValueError(f"unknown result reason {self.reason!r}")
        if self.retry_advice is not None and self.retry_advice not in RETRY_ADVICE:
            raise ValueError(f"unknown retry advice {self.retry_advice!r}")

    @property
    def robot_fault(self) -> bool:
        return self.reason in ROBOT_FAULT_REASONS

    def to_envelope(self) -> dict[str, Any]:
        envelope: dict[str, Any] = {
            "ok": self.ok,
            "action": self.action,
            "reason": self.reason,
            "detail": self.detail,
        }
        if self.retry_advice is not None:
            envelope["retry_advice"] = self.retry_advice
        if self.data:
            envelope.update(_round(self.data))
        envelope["elapsed_s"] = round(self.elapsed_s, 1)
        if self.state is not None:
            envelope["state"] = _round(self.state)
        return envelope


def _round(value: Any) -> Any:
    """Round floats for the model: sub-0.1mm digits are noise it would repeat back."""
    if isinstance(value, float):
        return round(value, 1)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value

"""Natural-language zone cell names -> task1.slot_uv indices."""

from __future__ import annotations

import re

from config import ZoneSlotNamesConfig, normalise_place_name

_NUMBERED = re.compile(r"(\d+)(번칸|번|칸)?")


def resolve_slot(name: str, cfg: ZoneSlotNamesConfig) -> int | None:
    """Canonical label, Korean label, alias, or a 1-based "N번"; else None."""
    key = normalise_place_name(name)
    if not key:
        return None
    n = len(cfg.labels)
    for index, label in enumerate(cfg.labels):
        if normalise_place_name(label) == key:
            return index
    for index, label in enumerate(cfg.korean_labels):
        if normalise_place_name(label) == key:
            return index
    for alias, index in cfg.aliases.items():
        if normalise_place_name(alias) == key:
            return int(index)
    match = _NUMBERED.fullmatch(key)
    if match:
        number = int(match.group(1))
        if 1 <= number <= n:
            return number - 1
    return None


def label_for(index: int, cfg: ZoneSlotNamesConfig) -> str:
    return cfg.labels[index]

"""Selection contracts."""

from config import AppConfig
from perception import (
    select_target,
    target_id_for,
)


from core_helpers import _calibration, _block


def test_selection_excludes_zone_and_skipped_blocks_with_stable_ids():
    cfg = AppConfig().select
    inside, skipped, candidate = (
        _block("red", 50, 30),
        _block("green", 220, 280),
        _block("blue", 240, 290),
    )
    assert target_id_for(_block("blue", 238, 292), cfg.target_cell_mm) == target_id_for(
        candidate, cfg.target_cell_mm
    )
    result = select_target(
        [inside, skipped, candidate],
        _calibration(),
        cfg,
        {target_id_for(skipped, cfg.target_cell_mm)},
    )
    assert result.target is candidate and result.remaining == 1

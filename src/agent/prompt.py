"""System prompt: the template file, with every number and name filled from config."""

from __future__ import annotations

from pathlib import Path
from string import Template

from config import AppConfig


def build_system_prompt(cfg: AppConfig, template_text: str | None = None) -> str:
    agent = cfg.agent
    if template_text is None:
        path = agent.system_prompt_path
        template_text = Path(path).read_text(encoding="utf-8")
    slots = agent.zone_slots
    rows = []
    for index, (label, korean) in enumerate(zip(slots.labels, slots.korean_labels)):
        row = "상단(먼 줄)" if index < 3 else "하단(가까운 줄)"
        rows.append(f"  - {korean} = slot \"{label}\" ({row})")
    regions = agent.table_regions
    columns = ", ".join(f"{k}={regions.column_korean[k]}" for k in regions.columns_deg)
    row_names = ", ".join(f"{k}={regions.row_korean[k]}" for k in regions.rows_fraction)
    grid = agent.board_grid
    grid_bounds = (
        f"한 칸은 체스판 한 칸(약 {grid.cell_mm:g}mm)이다."
        if grid.enabled
        else "이 시연장에서는 칸 좌표를 쓰지 않는다 — 이름 붙은 자리만 쓴다."
    )
    return Template(template_text).safe_substitute(
        colors=", ".join(sorted(cfg.perception.color_prototypes)),
        slot_table="\n".join(rows),
        columns=columns,
        rows=row_names,
        default_step=f"{agent.relative.default_step_mm:g}",
        small_step=f"{agent.relative.small_step_mm:g}",
        max_jog=f"{agent.relative.max_jog_mm:g}",
        grid_bounds=grid_bounds,
    )

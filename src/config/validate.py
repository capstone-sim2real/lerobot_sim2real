"""Cross-field checks run by ``load_config``."""

from __future__ import annotations

import math
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import AppConfig
    from config.robot import IkConfig, PerceptionConfig


def validate_yoloe(cfg: AppConfig) -> None:
    yoloe = cfg.yoloe
    if yoloe.imgsz != 640:
        raise ValueError("yoloe.imgsz must be 640 for the fixed TensorRT engine")
    if not 0.0 <= yoloe.trt_confidence <= 1.0:
        raise ValueError("yoloe.trt_confidence must be in [0, 1]")
    if yoloe.max_detections <= 0 or yoloe.trt_warmup_runs < 0:
        raise ValueError("invalid YOLOE TensorRT limits")
    if not math.isfinite(yoloe.web_analysis_fps) or yoloe.web_analysis_fps <= 0:
        raise ValueError("yoloe.web_analysis_fps must be finite and positive")

def validate_perception_colors(cfg: "PerceptionConfig") -> None:
    """Every gated colour must have a prototype to be identified by.

    The gates are allowed to overlap — they have to, since no fixed box
    separates wood from yellow in every arrangement. What must not happen is
    a colour that can be *gated* but never *named*: its blobs would compete
    for other colours' slots and silently evict the real blocks. So the check
    is coverage, not disjointness.
    """
    missing = sorted(set(cfg.hsv_ranges) - set(cfg.color_prototypes))
    if missing:
        raise ValueError(
            f"perception.color_prototypes is missing {missing}; every colour in "
            f"hsv_ranges needs at least one reference (hue, saturation) point"
        )
    for color, points in cfg.color_prototypes.items():
        if not points:
            raise ValueError(f"perception.color_prototypes[{color!r}] has no points")
        for point in points:
            if len(point) != 2:
                raise ValueError(
                    f"perception.color_prototypes[{color!r}] must be a list of "
                    f"[hue, saturation] points, got {point!r}"
                )
    if cfg.workspace_angle_max_deg <= cfg.workspace_angle_min_deg:
        raise ValueError(
            "perception.workspace_angle_max_deg must be greater than "
            "workspace_angle_min_deg"
        )
    profile = cfg.workspace_radius_by_angle_mm
    if profile:
        if any(len(pair) != 2 for pair in profile):
            raise ValueError(
                "perception.workspace_radius_by_angle_mm entries must be "
                "[azimuth_deg, radius_mm]"
            )
        angles = [float(pair[0]) for pair in profile]
        radii = [float(pair[1]) for pair in profile]
        if any(b <= a for a, b in zip(angles, angles[1:])):
            raise ValueError(
                "perception.workspace_radius_by_angle_mm angles must increase"
            )
        if angles[0] > cfg.workspace_angle_min_deg or angles[-1] < cfg.workspace_angle_max_deg:
            raise ValueError(
                "perception.workspace_radius_by_angle_mm must cover the workspace angle range"
            )
        if any(radius <= 0 for radius in radii):
            raise ValueError(
                "perception.workspace_radius_by_angle_mm radii must be positive"
            )


def validate_ik(cfg: "IkConfig") -> None:
    if len(cfg.shoulder_pan_origin_xy_mm) != 2 or not all(
        math.isfinite(v) for v in cfg.shoulder_pan_origin_xy_mm
    ):
        raise ValueError("ik.shoulder_pan_origin_xy_mm must be two finite coordinates")
    if cfg.seed_candidate_count <= 0:
        raise ValueError("ik.seed_candidate_count must be positive")


def validate_task1(cfg: AppConfig) -> None:
    if not 0.0 <= cfg.motion.grasp_retry_roll_deg <= 180.0:
        raise ValueError("motion.grasp_retry_roll_deg must be in [0, 180]")
    if cfg.task1.empty_timeout_s < 5.0:
        raise ValueError("task1.empty_timeout_s must be at least 5 seconds")
    if cfg.task1.scan_interval_s <= 0:
        raise ValueError("task1.scan_interval_s must be positive")
    if cfg.task1.max_frame_age_s <= 0:
        raise ValueError("task1.max_frame_age_s must be positive")
    if len(cfg.task1.slot_uv) < len(cfg.perception.color_prototypes):
        raise ValueError(
            "task1.slot_uv needs at least one slot per configured block colour"
        )
    if len(cfg.task1.slot_radial_offset_mm) != len(cfg.task1.slot_uv):
        raise ValueError("task1.slot_radial_offset_mm must have one value per slot_uv")
    if any(offset < 0 for offset in cfg.task1.slot_radial_offset_mm):
        raise ValueError("task1.slot_radial_offset_mm values must be non-negative")
    if cfg.task1.pick_near_boost_max_radius_mm < 0:
        raise ValueError("task1.pick_near_boost_max_radius_mm must be non-negative")
    if cfg.task1.pick_near_boost_mm < 0:
        raise ValueError("task1.pick_near_boost_mm must be non-negative")
    if cfg.task1.pick_tilt_start_radius_mm < 0:
        raise ValueError("task1.pick_tilt_start_radius_mm must be non-negative")
    if cfg.task1.pick_tilt_max_radius_mm <= cfg.task1.pick_tilt_start_radius_mm:
        raise ValueError(
            "task1.pick_tilt_max_radius_mm must exceed pick_tilt_start_radius_mm"
        )
    if not 0 <= cfg.task1.pick_tilt_base_deg <= cfg.task1.pick_tilt_max_deg:
        raise ValueError(
            "task1.pick_tilt_base_deg must be between zero and pick_tilt_max_deg"
        )
    if not 0 <= cfg.task1.pick_tilt_max_deg <= 60.0:
        raise ValueError("task1.pick_tilt_max_deg must be between zero and 60 degrees")
    if any(not 0 <= angle <= cfg.task1.pick_tilt_max_deg
           for angle in cfg.task1.pick_tilt_fallback_deg):
        raise ValueError("task1.pick_tilt_fallback_deg must stay within pick_tilt_max_deg")
    if not 0 <= cfg.task1.near_vertical_pick_max_deg <= cfg.task1.pick_tilt_max_deg:
        raise ValueError("task1.near_vertical_pick_max_deg must stay within pick_tilt_max_deg")
    if any(not math.isfinite(angle) or abs(angle) > cfg.agent.relative.max_gripper_roll_deg
           for angle in cfg.task1.place_yaw_fallback_offsets_deg):
        raise ValueError("task1.place_yaw_fallback_offsets_deg exceeds bounded wrist rotation")
    if cfg.task1.tilted_pick_pregrasp_clearance_mm < cfg.task1.tilted_pick_hover_clearance_mm:
        raise ValueError("task1.tilted_pick_pregrasp_clearance_mm must cover the first lift")
    if cfg.task1.tilted_pick_hover_clearance_mm < cfg.agent.calibration_clearance.obstacle_height_mm:
        raise ValueError("task1.tilted_pick_hover_clearance_mm must clear a block")
    if cfg.task1.zone_path_clearance_mm < 0:
        raise ValueError("task1.zone_path_clearance_mm must be non-negative")
    if not 0 <= cfg.task1.place_tilt_max_deg <= cfg.ik.max_tilt_error_deg:
        raise ValueError("task1.place_tilt_max_deg must be within the placement IK tilt gate")
    if not 0 < cfg.task1.place_level_tolerance_deg <= cfg.ik.max_tilt_error_deg:
        raise ValueError("task1.place_level_tolerance_deg must be within the IK tilt gate")
    if (not cfg.task1.place_tilt_candidates_deg
            or any(not math.isfinite(v) or not -30 <= v <= 0 for v in cfg.task1.place_tilt_candidates_deg)):
        raise ValueError("Task 1 placement tilts must be between -30 and 0 degrees")
    if not math.isfinite(cfg.task1.place_ik_error_mm) or not 0 < cfg.task1.place_ik_error_mm <= cfg.ik.max_position_error_mm:
        raise ValueError("Task 1 placement IK error must be positive and within the general gate")
    if not 0 < cfg.task1.place_yaw_tolerance_deg <= 45.0:
        raise ValueError("task1.place_yaw_tolerance_deg must be in (0, 45]")


def validate_task2(cfg: AppConfig) -> None:
    if len(cfg.task2.stack_uv) != 2:
        raise ValueError("task2.stack_uv must be a single [u, v] pair")
    if not all(0.0 < float(coord) < 1.0 for coord in cfg.task2.stack_uv):
        raise ValueError("task2.stack_uv must be strictly inside the zone")
    if cfg.task2.stack_radial_offset_mm < 0:
        raise ValueError("task2.stack_radial_offset_mm must be non-negative")
    if cfg.task2.block_height_mm <= 0:
        raise ValueError("task2.block_height_mm must be positive")
    if cfg.task2.max_levels < 1:
        raise ValueError("task2.max_levels must be at least one")
    if cfg.task2.upper_entry_level < 1 or cfg.task2.upper_entry_clearance_mm <= 0:
        raise ValueError("task2 upper entry level/clearance must be positive")
    if not -45 <= cfg.task2.upper_entry_radial_tilt_deg <= 0:
        raise ValueError("task2.upper_entry_radial_tilt_deg must be -45..0")
    if cfg.task2.route_standoff_mm <= 0:
        raise ValueError("task2.route_standoff_mm must be positive")
    if cfg.task2.tower_path_clearance_mm < 0:
        raise ValueError("task2.tower_path_clearance_mm must be non-negative")
    if cfg.task2.release_clearance_mm < 0:
        raise ValueError("task2.release_clearance_mm must be non-negative")
    if not 0 < cfg.task2.drop_clearance_mm <= cfg.task2.block_height_mm:
        raise ValueError("task2.drop_clearance_mm must be above zero and at most one block height")
    if not 0 < cfg.task2.hover_min_clearance_mm < cfg.task2.hover_clearance_mm:
        raise ValueError(
            "task2.hover_min_clearance_mm must be positive and below hover_clearance_mm"
        )
    if cfg.task2.hover_gate_mm <= 0:
        raise ValueError("task2.hover_gate_mm must be positive")
    if not 0 < cfg.task2.hover_squeeze_clearance_mm <= cfg.task2.hover_min_clearance_mm:
        raise ValueError(
            "task2.hover_squeeze_clearance_mm must be positive and at most "
            "hover_min_clearance_mm"
        )
    if cfg.task2.level_tilt_start_level < 1:
        raise ValueError("task2.level_tilt_start_level must be at least one")
    if cfg.task2.level_tilt_per_level_deg < 0:
        raise ValueError("task2.level_tilt_per_level_deg must be non-negative")
    if not 0 <= cfg.task2.level_tilt_max_deg <= cfg.ik.max_tilt_error_deg:
        raise ValueError(
            "task2.level_tilt_max_deg must be between zero and ik.max_tilt_error_deg"
        )


def validate_task3(cfg: AppConfig) -> None:
    """Fail at startup, never mid-collection.

    Every check here guards something that would otherwise be discovered
    after the arm has already recorded episodes into a dataset whose
    metadata is then wrong or unusable.
    """
    if cfg.task3.record_fps <= 0:
        raise ValueError("task3.record_fps must be positive")
    if cfg.task3.motion_fps_override <= cfg.task3.record_fps:
        raise ValueError(
            "task3.motion_fps_override must exceed record_fps; the recorder can only "
            "hold the dataset rate if TrajectoryPlayer's own tick sleep is shorter"
        )
    if cfg.task3.max_grasp_attempts < 1:
        raise ValueError("task3.max_grasp_attempts must be at least one")
    if not cfg.task3.repo_id:
        raise ValueError("task3.repo_id must not be empty")
    if not cfg.task3.cameras:
        raise ValueError(
            "task3.cameras must name at least one camera.server MJPEG stream; "
            "ACT requires at least one observation.images.* feature"
        )
    if cfg.task3.image_width <= 0 or cfg.task3.image_height <= 0:
        raise ValueError("task3.image_width and image_height must be positive")
    if cfg.task3.max_frame_age_s <= 0:
        raise ValueError("task3.max_frame_age_s must be positive")
    if cfg.task3.max_stale_ticks < 1:
        raise ValueError("task3.max_stale_ticks must be at least one")
    if not 0 < cfg.task3.min_episode_frames < cfg.task3.max_episode_frames:
        raise ValueError(
            "task3.min_episode_frames must be positive and below max_episode_frames"
        )
    if cfg.task3.max_rounds_without_progress < 1:
        raise ValueError("task3.max_rounds_without_progress must be at least one")
    # A colour the detector can name but the dataset cannot describe would
    # abort the run at the moment that block is first grasped.
    missing = sorted(set(cfg.perception.color_prototypes) - set(cfg.task3.task_templates))
    if missing:
        raise ValueError(
            f"task3.task_templates is missing {missing}; every detectable colour needs "
            f"a task sentence, because the colour is only known once a block is selected"
        )
    for color, sentence in cfg.task3.task_templates.items():
        if not str(sentence).strip():
            raise ValueError(f"task3.task_templates[{color!r}] is empty")


AGENT_PROVIDERS = ("anthropic", "openai", "gemini", "fake")


def normalise_place_name(name: str) -> str:
    """NFKC + casefold + drop whitespace, hyphens, underscores and punctuation."""
    text = unicodedata.normalize("NFKC", str(name)).casefold()
    return "".join(ch for ch in text if ch.isalnum())


def validate_agent(cfg: AppConfig) -> None:
    """Agent settings. Never touches the filesystem: every CLI loads this."""
    agent = cfg.agent
    if not agent.collection.root.strip():
        raise ValueError("agent.collection.root must be set")
    if agent.collection.max_steps < 1 or agent.collection.max_task_text_chars < 1:
        raise ValueError("agent.collection sequence limits must be positive")
    for name in ("max_tick_gap_s", "max_moving_gap_s", "max_total_missing_motion_s", "max_stationary_drift", "sequence_settle_window_s",
                 "sequence_settle_max_drift", "sequence_settle_timeout_s",
                 "max_mean_period_error", "idle_poll_s"):
        value = getattr(agent.collection, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"agent.collection.{name} must be finite and positive")
    if (type(agent.collection.tolerated_missing_ticks) is not int
            or not 0 <= agent.collection.tolerated_missing_ticks <= 5):
        raise ValueError("agent.collection.tolerated_missing_ticks must be in [0, 5]")
    if agent.collection.max_moving_gap_s < agent.collection.max_tick_gap_s:
        raise ValueError("max_moving_gap_s must cover max_tick_gap_s")
    if agent.collection.sequence_settle_max_drift >= agent.collection.max_stationary_drift:
        raise ValueError("agent.collection.sequence_settle_max_drift must be below max_stationary_drift")
    primitive = agent.primitives
    for name in ("target_max_age_s", "approach_clearance_mm", "lateral_clearance_mm",
                 "lateral_clearance_tolerance_mm", "zone_release_floor_margin_mm",
                 "alignment_tolerance_mm", "arrival_error_mm", "cartesian_step_mm",
                 "contact_step_mm", "contact_max_descent_mm", "contact_timeout_s",
                 "contact_backoff_mm", "wrist_roll_limit_deg"):
        value = getattr(primitive, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"agent.primitives.{name} must be finite and positive")
    if primitive.lateral_clearance_tolerance_mm >= primitive.lateral_clearance_mm:
        raise ValueError("primitive lateral clearance tolerance must be smaller than clearance")
    if primitive.zone_release_floor_margin_mm >= cfg.task1.release_clearance_mm:
        raise ValueError("primitive release floor margin must be below Task 1 drop clearance")
    if primitive.approach_clearance_mm < primitive.lateral_clearance_mm:
        raise ValueError("primitive approach clearance must cover lateral clearance")
    for name in ("observation_window_s", "observation_fps"):
        value = getattr(primitive, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"agent.primitives.{name} must be finite and positive")
    if type(primitive.max_lift_attempts) is not int or not 1 <= primitive.max_lift_attempts <= 4:
        raise ValueError("agent.primitives.max_lift_attempts must be in [1, 4]")
    if (type(primitive.image_jpeg_quality) is not int or type(primitive.image_max_width) is not int
            or not 1 <= primitive.image_jpeg_quality <= 100 or primitive.image_max_width <= 0):
        raise ValueError("invalid primitive image settings")
    if agent.provider not in AGENT_PROVIDERS:
        raise ValueError(f"agent.provider must be one of {AGENT_PROVIDERS}")
    if not str(agent.models.get(agent.provider, "")).strip():
        raise ValueError(f"agent.models has no model id for provider {agent.provider!r}")
    if agent.fallback_provider is not None:
        if agent.fallback_provider not in AGENT_PROVIDERS:
            raise ValueError(f"agent.fallback_provider must be one of {AGENT_PROVIDERS} or null")
        if agent.fallback_provider == agent.provider:
            raise ValueError("agent.fallback_provider must differ from agent.provider")
        if not str(agent.models.get(agent.fallback_provider, "")).strip():
            raise ValueError(
                f"agent.models has no model id for fallback provider {agent.fallback_provider!r}"
            )
    if agent.max_tokens <= 0:
        raise ValueError("agent.max_tokens must be positive")
    if not 0 < agent.max_tool_turns <= 50:
        raise ValueError("agent.max_tool_turns must be in (0, 50]")
    if agent.max_history_messages < 4:
        raise ValueError("agent.max_history_messages must be at least 4")
    for name in (
        "tool_timeout_s", "slot_snap_radius_mm", "place_clear_radius_mm", "camera_fresh_timeout_s",
        "lease_grace_s", "lease_idle_timeout_s", "sse_heartbeat_s",
    ):
        if getattr(agent, name) <= 0:
            raise ValueError(f"agent.{name} must be positive")
    if agent.table_zone_margin_mm < 0:
        raise ValueError("agent.table_zone_margin_mm must be non-negative")
    if agent.zone_scan_max_per_color < 0:
        raise ValueError("agent.zone_scan_max_per_color must be non-negative")

    slots = agent.zone_slots
    n_slots = len(cfg.task1.slot_uv)
    if slots.frame not in ("camera_view", "robot"):
        raise ValueError("agent.zone_slots.frame must be 'camera_view' or 'robot'")
    if len(slots.labels) != n_slots or len(slots.korean_labels) != n_slots:
        raise ValueError(
            "agent.zone_slots.labels and korean_labels need one name per task1.slot_uv cell"
        )
    seen: dict[str, int] = {}
    for index, name in enumerate([*slots.labels, *slots.korean_labels]):
        key = normalise_place_name(name)
        slot = index % n_slots
        if not key:
            raise ValueError("agent.zone_slots labels must not be empty")
        if key in seen and seen[key] != slot:
            raise ValueError(f"agent.zone_slots label {name!r} names two different cells")
        seen[key] = slot
    for alias, index in slots.aliases.items():
        key = normalise_place_name(alias)
        if not key:
            raise ValueError("agent.zone_slots.aliases keys must not be empty")
        if not isinstance(index, int) or not 0 <= index < n_slots:
            raise ValueError(f"agent.zone_slots.aliases[{alias!r}] must be a slot index")
        if key in seen and seen[key] != index:
            raise ValueError(f"agent.zone_slots alias {alias!r} contradicts a label")
        seen[key] = index

    bounds = agent.primitives
    if (not 0 < bounds.home_return_min_clearance_mm <= bounds.home_return_clearance_mm
            or bounds.home_lift_xy_limit_mm <= 0
            or not bounds.home_lift_tilt_candidates_deg
            or any(not math.isfinite(angle) for angle in bounds.home_lift_tilt_candidates_deg)):
        raise ValueError("agent.primitives home return settings are invalid")

    rel = agent.relative
    if rel.frame not in ("arm", "base"):
        raise ValueError("agent.relative.frame must be 'arm' or 'base'")
    for name in (
        "default_step_mm", "small_step_mm", "max_pick_offset_mm", "max_jog_mm", "max_shift_mm",
        "jog_max_ik_error_mm", "max_gripper_roll_deg",
    ):
        if getattr(rel, name) <= 0:
            raise ValueError(f"agent.relative.{name} must be positive")
    if not rel.jog_min_z_mm < rel.jog_max_z_mm:
        raise ValueError("agent.relative.jog_min_z_mm must be below jog_max_z_mm")
    for name in ("keyboard_speed_mm_s", "keyboard_segment_s", "keyboard_tick_hz",
                 "keyboard_heartbeat_s", "keyboard_timeout_s", "keyboard_joint_speed_deg_s",
                 "keyboard_acceleration_mm_s2", "keyboard_jerk_mm_s3", "keyboard_release_timeout_s"):
        value = getattr(cfg.agent.relative, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"agent.relative.{name} must be finite and positive")
    if cfg.agent.relative.keyboard_timeout_s <= 2 * cfg.agent.relative.keyboard_heartbeat_s:
        raise ValueError("keyboard timeout must exceed two heartbeat periods")


    regions = agent.table_regions
    if not regions.columns_deg or not regions.rows_fraction:
        raise ValueError("agent.table_regions needs at least one column and one row")
    lo, hi = cfg.perception.workspace_angle_min_deg, cfg.perception.workspace_angle_max_deg
    for column, azimuth in regions.columns_deg.items():
        if not lo <= float(azimuth) <= hi:
            raise ValueError(
                f"agent.table_regions.columns_deg[{column!r}] is outside the workspace sector"
            )
    for row, fraction in regions.rows_fraction.items():
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError(f"agent.table_regions.rows_fraction[{row!r}] must be in [0, 1]")
    if set(regions.column_korean) != set(regions.columns_deg):
        raise ValueError("agent.table_regions.column_korean must name every column")
    if set(regions.row_korean) != set(regions.rows_fraction):
        raise ValueError("agent.table_regions.row_korean must name every row")
    if regions.min_radius_mm <= 0 or regions.edge_margin_mm < 0:
        raise ValueError("agent.table_regions radii must be positive")
    if regions.search_step_mm <= 0 or regions.search_max_mm < 0:
        raise ValueError("agent.table_regions search bounds must be positive")

    grid = agent.board_grid
    if grid.cell_mm <= 0:
        raise ValueError("agent.board_grid.cell_mm must be positive")
    if grid.origin_mm is not None and len(grid.origin_mm) != 2:
        raise ValueError("agent.board_grid.origin_mm must be [x_mm, y_mm]")
    correction = agent.place_correction
    if correction.max_mm < 0 or correction.max_sample_mm <= 0:
        raise ValueError("agent.place_correction bounds must be positive")
    if math.hypot(correction.forward_mm, correction.left_mm) > correction.max_mm:
        raise ValueError("agent.place_correction seed is larger than its own max_mm")

    if grid.default_layer not in ("regions", "grid"):
        raise ValueError("agent.board_grid.default_layer must be 'regions' or 'grid'")
    for name in ("min_radius_mm", "base_keepout_depth_mm", "base_keepout_half_width_mm"):
        if getattr(grid, name) < 0:
            raise ValueError(f"agent.board_grid.{name} must be non-negative")

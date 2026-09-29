"""Camera, calibration-tool, web agent and YOLOE settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkspaceBoundaryConfig:
    """Outline of the pick workspace, drawn on the camera page.

    The geometry itself lives in ``PerceptionConfig.workspace_*`` because the
    detector gates on it — so the arc on screen is exactly the region blocks
    are reported in, rather than a decoration that can drift away from it.
    Whether a *concrete* grasp is reachable is still TopDownIK's call: this is
    the coarse "is it even on the board" test.
    """

    enabled: bool = True
    sample_step_deg: float = 2.0


@dataclass
class CameraOverlayConfig:
    """Resource limits for the non-critical operator overlay process."""

    analysis_fps: float = 5.0
    # Stabilise static block coordinates and grasp axes over recent accepted
    # observations. A colour disappears only after this many consecutive
    # misses; its history is then reset so reappearance is immediate and does
    # not blend with the old location.
    smoothing_window: int = 20
    hide_after_misses: int = 10
    worker_nice: int = 10
    opencv_threads: int = 1
    # publish near-miss contours (and which gate dropped them) to the page, so
    # "no block here" and "block seen, fill 0.48" stay distinguishable
    report_rejects: bool = True
    workspace_boundary: WorkspaceBoundaryConfig = field(
        default_factory=WorkspaceBoundaryConfig
    )


@dataclass
class CameraConfig:
    """Camera web UI settings; capture transport stays configured by its CLI."""

    overlay: CameraOverlayConfig = field(default_factory=CameraOverlayConfig)
    # so101-run / so101-collect / so101-agent start camera.server (the sole
    # /dev/video* owner) when nothing answers its health check. Set False to
    # require starting so101-camera by hand.
    auto_start: bool = True
    # Extra camera.server CLI args, e.g. ["--wrist-device", "/dev/video2"].
    extra_args: list[str] = field(default_factory=list)


@dataclass
class CalibrationCaptureConfig:
    """Raw-pixel candidate gates; provisional values, checked in the preview."""

    color: str = "yellow"
    point_count: int = 9
    area_px2_min: float = 400.0
    area_px2_max: float = 16000.0
    morph_kernel_px: int = 5


@dataclass
class SessionToolsConfig:
    """Operator-session file paths and timing; no hardware imports."""

    runtime_dir: str = "var/so101"
    snapshot_url: str = "http://127.0.0.1:8090/snapshot/shoulder.jpg"
    snapshot_timeout_s: float = 5.0
    poll_interval_s: float = 0.1
    telemetry_stale_s: float = 3.0
    capture_stale_s: float = 5.0
    capture_timeout_s: float = 15.0
    wrist_limit_deg: float = 8.0
    temperature_limit_c: float = 65.0
    telemetry_interval_s: float = 1.0
    startup_poll_s: float = 0.1
    startup_ramp_s: float = 0.0
    expected_motor_model: int = 777
    telemetry_tail_bytes: int = 16384


@dataclass
class ZoneSlotNamesConfig:
    """Human names for the ``task1.slot_uv`` cells, used by the LLM agent.

    ``frame`` states whose top/left these are. The operator looks at the
    camera page, where the far row (v -> 0) renders at the TOP of the image
    and +y mm is image-LEFT -- verified against ``meta.zone_polygon_px`` of
    the venue calibration (far corner x~312mm at py~326, +y corner at px~461).
    Re-check with ``so101-agent --dry-run`` after re-mounting the camera.
    """

    frame: str = "camera_view"
    labels: list[str] = field(
        default_factory=lambda: [
            "top-left", "top-center", "top-right", "bottom-left", "bottom-right",
        ]
    )
    korean_labels: list[str] = field(
        default_factory=lambda: ["좌상단", "상단중앙", "우상단", "좌하단", "우하단"]
    )
    # Normalised (NFKC, casefold, no spaces/hyphens) name -> slot index.
    aliases: dict[str, int] = field(
        default_factory=lambda: {
            "좌상단": 0, "왼쪽위": 0, "좌측상단": 0, "상단왼쪽": 0, "왼쪽상단": 0,
            "상단중앙": 1, "가운데위": 1, "중앙상단": 1, "상단가운데": 1, "위쪽가운데": 1,
            "우상단": 2, "오른쪽위": 2, "우측상단": 2, "상단오른쪽": 2, "오른쪽상단": 2,
            "좌하단": 3, "왼쪽아래": 3, "좌측하단": 3, "하단왼쪽": 3, "왼쪽하단": 3,
            "우하단": 4, "오른쪽아래": 4, "우측하단": 4, "하단오른쪽": 4, "오른쪽하단": 4,
            "먼쪽왼쪽": 0, "먼쪽가운데": 1, "먼쪽오른쪽": 2,
            "가까운쪽왼쪽": 3, "가까운쪽오른쪽": 4,
            "topcentre": 1, "farleft": 0, "farcenter": 1, "farcentre": 1, "farright": 2,
            "nearleft": 3, "nearright": 4,
        }
    )


@dataclass
class RelativeMotionConfig:
    """Limits for relative requests ("5mm further", "go left"). Per call."""

    # "arm": forward = radial from the base, left = tangential
    # (control.ik.gripper_frame_offset). "base": forward = +x, left = +y.
    frame: str = "arm"
    # "go left" with no distance, and "a little"
    default_step_mm: float = 20.0
    small_step_mm: float = 10.0
    # Assumed, not measured: cumulative grasp-point offset from the detected centre.
    max_pick_offset_mm: float = 25.0
    # One jog vector may not exceed this; larger requests are refused, not clamped.
    max_jog_mm: float = 50.0
    # Continuous keyboard jog defaults; nominal speed, not measured hardware speed.
    keyboard_speed_mm_s: float = 20.0
    keyboard_segment_s: float = 0.1
    keyboard_tick_hz: float = 30.0
    keyboard_heartbeat_s: float = 0.1
    keyboard_timeout_s: float = 0.4
    keyboard_joint_speed_deg_s: float = 15.0
    keyboard_acceleration_mm_s2: float = 100.0
    keyboard_jerk_mm_s3: float = 1000.0
    keyboard_release_timeout_s: float = 1.0
    # Assumed, not measured: gripper-frame z window a jog may target.
    jog_min_z_mm: float = 40.0
    jog_max_z_mm: float = 160.0
    # Assumed: a jog whose IK solve misses by more is refused rather than
    # landing somewhere else (the pick/place gate allows ik.max_position_error_mm).
    jog_max_ik_error_mm: float = 5.0
    # One shift_block vector limit.
    max_shift_mm: float = 120.0
    # One manual gripper-roll (wrist_roll) request. Refused, not clamped, like
    # every other jog here; lerobot's own max_relative_target clamp and
    # send_joints's per-tick step still bound the actual motion regardless.
    max_gripper_roll_deg: float = 90.0


@dataclass
class TableRegionsConfig:
    """Named free-placement points on the detector's workspace sector.

    A column is an azimuth (camera view: positive = left), a row is a
    fraction of [min_radius_mm, sector edge - edge_margin_mm] (camera view:
    near = bottom). Every value here is an assumption until
    ``so101-agent --dry-run`` shows each point passing the IK gate.
    """

    columns_deg: dict[str, float] = field(
        default_factory=lambda: {
            "leftmost": 75.0, "left": 40.0, "center": 0.0, "right": -40.0, "rightmost": -75.0,
        }
    )
    rows_fraction: dict[str, float] = field(
        default_factory=lambda: {"near": 0.15, "middle": 0.5, "far": 0.9}
    )
    column_korean: dict[str, str] = field(
        default_factory=lambda: {
            "leftmost": "가장 왼쪽", "left": "왼쪽", "center": "가운데",
            "right": "오른쪽", "rightmost": "가장 오른쪽",
        }
    )
    row_korean: dict[str, str] = field(
        default_factory=lambda: {"near": "아래(가까운 쪽)", "middle": "중간", "far": "위(먼 쪽)"}
    )
    min_radius_mm: float = 150.0
    edge_margin_mm: float = 20.0
    # When a named point is blocked, search rings around it at this spacing.
    search_step_mm: float = 15.0
    search_max_mm: float = 60.0


@dataclass
class PlaceCorrectionConfig:
    """Closed-loop command offset for placements (``session/place_correction``).

    Not a calibration constant. ``forward_mm``/``left_mm`` seed the offset
    (arm frame) when a venue has already measured one; left at zero the
    agent learns it from its own post-placement camera checks, which it
    performs anyway. ``max_mm`` bounds how far the learned value may go, so
    one bad measurement cannot walk placements off the table.
    """

    enabled: bool = True
    learn: bool = True
    forward_mm: float = 13.3
    left_mm: float = 5.4
    max_mm: float = 60.0
    max_sample_mm: float = 80.0


@dataclass
class BoardGridConfig:
    """Chessboard-cell addressing over the same sector (``session/grid.py``).

    The lattice itself is measured by ``tools/calibration/calibrate_board_grid.py`` and
    stored in the calibration file; only how it is *presented* lives here.
    ``origin_mm`` names the square called (0, 0) -- left unset, the operator
    page anchors it on the ``center``/``near`` table region, i.e. the bottom
    centre of the reachable band.
    """

    enabled: bool = True
    # Only used when the calibration carries no measured lattice.
    cell_mm: float = 25.0
    origin_mm: list[float] | None = None
    # Which overlay the camera page starts on: the 15 named points
    # ("regions") or the board cells ("grid"). Only one is shown at a time.
    default_layer: str = "regions"
    # Inner limits. Unlike the named points, cells go right up to the robot:
    # measured 2026-09-19 with the place IK gate, every cell that failed sat
    # in a narrow corridor straight in front of the base (|y| <= 25mm,
    # x <= 72mm) -- the gripper's 27mm lateral offset from the pan axis
    # is what makes a top-down pose impossible there, while
    # 50mm off-axis solves from 55mm out. So the keep-out is that corridor
    # plus a small circle on the base itself, not one large radius.
    min_radius_mm: float = 45.0
    base_keepout_depth_mm: float = 85.0
    base_keepout_half_width_mm: float = 40.0


@dataclass
class AgentCameraViewConfig:
    """Operator display timing defaults; not physical control limits."""

    stale_s: float = 3.0
    poll_s: float = 0.5
    retry_s: float = 5.0
    connect_timeout_s: float = 3.0
    read_timeout_s: float = 15.0


@dataclass
class CalibrationClearanceConfig:
    """Experimental conservative envelopes, assumed until physically measured.

    Shared by calibration experiments and calibrated primitives. URDF jaw
    mesh bounds are conservative; they are not a measured full-arm collision model.
    """
    # Experimental hypothesis: neutral-frame bias rotates with retry jaw yaw.
    wrist_roll_min_deg: float = -65.0
    wrist_probe_target_deg: float = 90.0
    wrist_probe_close_gripper: bool = True
    hover_correction_max_deg: float = 3.0  # bounded experimental feedforward
    # Assumed bounded trial offsets; learn from recorded outcomes, not success claims.
    trial_offsets_mm: list[list[float]] = field(default_factory=lambda: [[5.0,0.0],[10.0,0.0],[0.0,5.0],[-5.0,0.0],[-10.0,0.0],[0.0,-5.0],[0.0,10.0],[0.0,-10.0],[15.0,0.0],[15.0,-10.0]])
    trial_yaw_offsets_deg: list[float] = field(default_factory=lambda: [-15.0,15.0])
    rotate_retry_bias: bool = True
    red_separation_kernel_px: int = 31
    jaw_angle_step_deg: float = 5.0
    # Encoder span is 2490-1867 ticks at 4095 ticks/rev on this rig.
    # Closed-angle origin is a provisional URDF/side-view alignment, not a
    # measured camera calibration. Retain angle and spatial uncertainty.
    jaw_mount_yaw_deg: float = -90.0  # Live scene fixes the moving-jaw side, not only the unsigned axis.
    jaw_closed_angle_deg: float = -10.0
    jaw_span_deg: float = 54.76923076923077
    jaw_angle_uncertainty_deg: float = 5.0
    jaw_open_positions: list[float] = field(default_factory=lambda: [95.0,85.0,75.0])
    tool_radius_mm: float = 60.0
    block_radius_mm: float = 29.0
    block_side_mm: float = 40.0
    uncertainty_mm: float = 15.0
    jaw_inner_clearance_mm: float = 2.0
    obstacle_height_mm: float = 20.0  # user-confirmed flat block height; reobserve after tipping


@dataclass
class PrimitiveConfig:
    """Experimental bounds, ASSUMED until physically measured. No mission overrides."""
    calibrated_pick: bool = True  # Same calibrated primitive path as default.yaml.
    target_max_age_s: float = 120.0
    approach_clearance_mm: float = 38.0
    lateral_clearance_mm: float = 38.0
    lateral_clearance_tolerance_mm: float = 2.0  # FK/encoder settling near the lift threshold
    zone_release_floor_margin_mm: float = 2.0  # calibrated top plane to minimum release FK
    max_lift_attempts: int = 4
    observation_window_s: float = 2.0
    observation_fps: float = 5.0
    alignment_tolerance_mm: float = 25.0
    arrival_error_mm: float = 15.0
    home_fold_radius_mm: float = 40.0  # assumed low-height folding corridor around home XY
    # Assumed clearance after release: seek 55 mm, require 38 mm before folding home.
    home_return_clearance_mm: float = 55.0
    home_return_min_clearance_mm: float = 38.0
    home_lift_xy_limit_mm: float = 10.0
    home_lift_tilt_candidates_deg: list[float] = field(
        default_factory=lambda: [0.0, -5.0, -10.0, -15.0]
    )
    # Measured loaded-arm endpoint sag is 24-29mm at far slots. This applies
    # only while carrying; empty moves retain arrival_error_mm.
    loaded_arrival_error_mm: float = 30.0
    cartesian_step_mm: float = 10.0
    contact_step_mm: float = 2.0
    contact_max_descent_mm: float = 80.0
    contact_timeout_s: float = 15.0
    contact_backoff_mm: float = 2.0
    wrist_roll_limit_deg: float = 90.0
    image_max_width: int = 960
    image_jpeg_quality: int = 80


@dataclass
class AgentCollectionConfig:
    """Assumed recording quality gates; validate timing on hardware."""
    root: str = "datasets/agent"
    max_steps: int = 24
    max_task_text_chars: int = 240
    max_tick_gap_s: float = 0.1
    tolerated_missing_ticks: int = 2
    max_moving_gap_s: float = 0.5  # Provisional recording-quality bounds, not motor limits.
    max_total_missing_motion_s: float = 1.0
    max_stationary_drift: float = 1.0  # degrees, or normalized gripper percent
    sequence_settle_window_s: float = 0.3
    sequence_settle_max_drift: float = 0.25
    sequence_settle_timeout_s: float = 3.0
    max_mean_period_error: float = 0.1
    idle_poll_s: float = 0.005


@dataclass
class AgentConfig:
    """LLM tool-calling agent (so101-agent). Unused by so101-run/so101-collect."""

    # anthropic | openai | gemini | fake
    primitives: PrimitiveConfig = field(default_factory=PrimitiveConfig)
    collection: AgentCollectionConfig = field(default_factory=AgentCollectionConfig)
    provider: str = "openai"
    # Used only for the configured default provider. An explicit --provider
    # selects exactly that provider so rehearsals and diagnostics stay clear.
    fallback_provider: str | None = "gemini"
    # Model id per provider. Verify against each vendor's current model list.
    models: dict[str, str] = field(
        default_factory=lambda: {
            "anthropic": "claude-opus-5",
            "openai": "gpt-6-luna",
            "gemini": "gemini-3.8-flash",
            "fake": "fake-1",
        }
    )
    max_tokens: int = 4096
    # LLM round trips, one primitive call per response, per user message
    max_tool_turns: int = 50
    # oldest turns are dropped from the context beyond this many messages
    max_history_messages: int = 60
    # Includes local dataset/video finalization; longer calls are reported as faults
    tool_timeout_s: float = 900.0
    system_prompt_path: str = "src/configs/agent_primitives_prompt.md"
    host: str = "0.0.0.0"
    port: int = 8099
    # the browser loads the MJPEG straight from so101-camera
    camera_base_url: str = "http://127.0.0.1:8090"
    camera_name: str = "shoulder"
    camera_view: AgentCameraViewConfig = field(default_factory=AgentCameraViewConfig)
    # after the arm moves, wait up to this long for a frame captured later
    camera_fresh_timeout_s: float = 3.0
    lock_path: str = "var/so101/robot.lock"
    # zone scan runs on a copy of perception with this max_per_color
    zone_scan_max_per_color: int = 1
    # an in-zone block within this distance of a slot centre occupies it
    slot_snap_radius_mm: float = 45.0
    # Assumed: minimum centre distance between a placement and any other block
    place_clear_radius_mm: float = 55.0
    # table placements must lie at least this far outside the zone polygon
    table_zone_margin_mm: float = 30.0
    transcript_dir: str = "logs/agent"
    # operator SSE may be gone this long before the lease is dropped
    lease_grace_s: float = 15.0
    # an IDLE operator with no input for this long loses the lease
    lease_idle_timeout_s: float = 300.0
    sse_heartbeat_s: float = 15.0
    zone_slots: ZoneSlotNamesConfig = field(default_factory=ZoneSlotNamesConfig)
    relative: RelativeMotionConfig = field(default_factory=RelativeMotionConfig)
    table_regions: TableRegionsConfig = field(default_factory=TableRegionsConfig)
    board_grid: BoardGridConfig = field(default_factory=BoardGridConfig)
    place_correction: PlaceCorrectionConfig = field(default_factory=PlaceCorrectionConfig)
    calibration_clearance: CalibrationClearanceConfig = field(default_factory=CalibrationClearanceConfig)


@dataclass
class YoloeConfig:
    """Optional YOLOE backend; thresholds remain experimental candidates."""
    model: str = "models/yoloe-26s-seg.pt"
    engine: str = "models/yoloe-26s-block-vp-seg-fp16.engine"
    worker_python: str = ".venv-trt/bin/python"
    prompts: list[str] = field(default_factory=lambda: ["wooden block", "toy block", "cube"])
    device: str = "cpu"
    imgsz: int = 640
    confidence: float = 0.1
    trt_confidence: float = 0.04
    iou: float = 0.5
    cpu_threads: int = 4
    max_detections: int = 20
    trt_warmup_runs: int = 5
    web_analysis_fps: float = 2.0
    snapshot_url: str = "http://127.0.0.1:8090/snapshot/shoulder.jpg"
    http_timeout_s: float = 10.0

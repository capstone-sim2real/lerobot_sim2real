"""FSM budget and per-task (Task 1/2/3) settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FsmConfig:
    num_blocks: int = 5
    time_budget_s: float = 300.0
    # attempts per selected target before it is skipped for the run
    max_retries_per_block: int = 2
    # safety margin: force DONE when remaining budget drops below this
    reserve_time_s: float = 10.0


@dataclass
class Task1Config:
    """Transport-task completion and deterministic placement geometry."""

    # DONE is allowed only after fresh frames have continuously reported no
    # blocks in the active region for this long.
    empty_timeout_s: float = 5.0
    scan_interval_s: float = 0.2
    # A frozen/error status JPEG must never count toward the empty timeout.
    max_frame_age_s: float = 1.0
    # Score outside blocks by base distance minus this penalty at the image edge.
    # 0 means farthest-first regardless of horizontal position.
    selection_center_bias_mm: float = 50.0
    # Slot coordinates in the calibrated quadrilateral: u runs left->right
    # along its long edge, v runs far->near. Fill the far row first.
    slot_uv: list[list[float]] = field(
        default_factory=lambda: [
            [0.14, 0.20],
            [0.50, 0.20],
            [0.80, 0.20],
            [0.32, 0.76],
            [0.68, 0.76],
        ]
    )
    # Per-slot command correction away from the base for measured under-reach.
    slot_radial_offset_mm: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0]
    )
    # Picks take no distance-ramped radial correction any more -- both the
    # oblique-camera ramp and the P-row front offsets over-corrected in the
    # field. Only an ultra-near band still gets a flat outward boost, cut
    # hard at the boundary. Of the measured points P1 (157 mm) is the sole
    # one inside 175 mm; P8 (196 mm) is the next closest and takes nothing.
    pick_near_boost_max_radius_mm: float = 175.0
    pick_near_boost_mm: float = 0.0
    # At long reach a perfectly vertical gripper saturates wrist_flex near
    # +95 deg. Gradually tip the approach axis radially outward so the wrist
    # opens while remaining within ik.max_tilt_error_deg.
    pick_tilt_start_radius_mm: float = 280.0
    pick_tilt_max_radius_mm: float = 300.0
    # Outward target-axis tilt applied at every Task-1 pick. This opens
    # wrist_flex through IK while preserving the requested Cartesian point;
    # placement keeps its original far-reach-only tilt ramp.
    pick_tilt_base_deg: float = 3.0
    pick_tilt_max_deg: float = 60.0
    pick_tilt_fallback_deg: list[float] = field(default_factory=lambda: [45.0, 30.0, 15.0])
    near_vertical_pick_max_deg: float = 5.0
    place_tilt_max_deg: float = 0.0
    place_tilt_candidates_deg: list[float] = field(default_factory=lambda: [0.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0])
    place_ik_error_mm: float = 5.0
    # Model-FK tolerance for a held block's level placement approach.
    place_level_tolerance_deg: float = 3.0
    place_yaw_tolerance_deg: float = 5.0
    # Task 1 accepts any flat orientation. If zone-aligned yaw cannot reach
    # a slot, search bounded yaw alternatives before rejecting placement.
    place_yaw_fallback_offsets_deg: list[float] = field(
        default_factory=lambda: [-15.0, 15.0, -30.0, 30.0, -45.0, 45.0,
                                 -60.0, 60.0, -75.0, 75.0, -90.0, 90.0])
    tilted_pick_hover_clearance_mm: float = 30.0
    # Pre-grasp hover is higher at long reach to absorb measured empty-arm
    # sag; the first lift after closing still stops at the 30 mm value above.
    tilted_pick_pregrasp_clearance_mm: float = 50.0
    # Assumption pending hardware measurement: release 15mm above the
    # calibrated block-top plane, without seeking table contact.
    release_clearance_mm: float = 15.0
    # Assumed extra vertical gap above an observed in-zone block.
    zone_path_clearance_mm: float = 15.0


@dataclass
class Task2Config:
    """Stack every block at one point; only PLACE differs from Task 1.

    SELECT/PICK/VERIFY and the pick corrections are read from ``task1`` --
    Task 2 *is* Task 1's gather pipeline with a single destination. Tower geometry and release clearance live here.
    """

    # Tower location, same [u, v] convention as task1.slot_uv; v -> 1 is the
    # zone edge nearest the base. Fix level 1 at the centre of Task 1's
    # bottom row (v=0.76); all higher levels use the same xy.
    stack_uv: list[float] = field(default_factory=lambda: [0.50, 0.76])
    # Task 1's measured command under-reach. Needed here not for accuracy --
    # a tower only needs consistency -- but so the block physically lands
    # inside zone_polygon_mm, which is what makes the detector ignore it.
    stack_radial_offset_mm: float = 0.0
    block_height_mm: float = 20.0
    # Ceiling on the pre-solved ladder, not a promise: levels the IK cannot
    # reach are reported by the dry-run, never silently clipped.
    max_levels: int = 5

    # Assumption pending hardware measurement: smaller than task1's 5.0mm,
    # because a drop that a table absorbs will topple a tower.
    release_clearance_mm: float = 2.0
    # Assumption pending physical trials: primitive stack drops this far
    # above the nominal release plane instead of trusting load-based contact.
    drop_clearance_mm: float = 15.0
    # Assumed near-edge staging distance outside the tape, pending physical validation.
    route_standoff_mm: float = 35.0
    # Assumed extra vertical gap above the local tower footprint.
    tower_path_clearance_mm: float = 15.0
    # Fifth-floor clearance posture; model-only assumptions until hardware trial.
    upper_entry_level: int = 5
    upper_entry_clearance_mm: float = 30.0
    upper_entry_radial_tilt_deg: float = -15.0
    # Hover search bounds above the nominal release height. motion.hover_*
    # assumes a table-height target; 120mm above level four is far outside
    # the envelope and only burns failing IK solves. The lower bound is also
    # the clearance the held block has over the tower top on the way in.
    hover_clearance_mm: float = 45.0
    hover_min_clearance_mm: float = 15.0
    # Hover poses are gated harder than ik.max_position_error_mm: a hover
    # reported 12mm short would eat most of the clearance budget and make
    # the tower gap fictional (see control/grasp.py).
    hover_gate_mm: float = 3.0
    # Floor the hover search may be squeezed to when the preferred band above
    # is outside the arm's ceiling. Top-down lift collapses with height, so an
    # upper level often solves 10mm over the tower but not 15mm. Refusing the
    # level costs a whole block; approaching lower costs clearance we still
    # have. Only a release pose that itself misses stops the ladder now.
    hover_squeeze_clearance_mm: float = 8.0

    # Tilt the approach axis outward as the target rises, releasing the
    # wrist_flex saturation that caps top-down lift. Late and
    # small: a tilted release lands the block on an edge.
    level_tilt_start_level: int = 2
    level_tilt_per_level_deg: float = 1.5
    level_tilt_max_deg: float = 5.0


@dataclass
class Task3Config:
    """Automated ACT dataset collection: Task 1's gather loop, recorded.

    Task 3 *is* Task 1 -- same SELECT/PICK/VERIFY/TRANSPORT and the same five
    zone slots -- with three differences: every arm command is
    recorded into a LeRobotDataset episode, only the centre grasp point is
    tried, and the empty-region proof prompts for a new block arrangement
    instead of finishing the run. Everything else is read from ``task1``.
    """

    # Dataset fps. This is not a wish: ``RecordingRobotIO`` paces every tick
    # to it, because LeRobotDataset synthesises timestamps as frame_index/fps
    # and a policy trained on a mislabelled rate replays at the wrong speed.
    record_fps: float = 30.0
    # TrajectoryPlayer._tick_sleep sleeps 1/fps on top of the work it just
    # did, so it cannot itself hold a true rate. Task 3 raises motion.fps so
    # that sleep becomes negligible and the recorder is the only metronome.
    # motion.fps feeds nothing else -- the per-tick joint cap is
    # motion.max_step_per_tick -- so the safety envelope is unchanged.
    motion_fps_override: float = 300.0
    # Requirement: one grasp attempt per episode. A failed grasp is a
    # discarded episode, never a retry ring (which would teach the policy to
    # fumble). ``None`` would restore Task 1's five-point ring.
    max_grasp_attempts: int = 1

    repo_id: str = "local/so101_task3"
    # Empty means $HF_LEROBOT_HOME/{repo_id}. LeRobotDataset.create refuses an
    # existing directory; the runner appends a _YYYYMMDD_HHMMSS stamp.
    root: str = ""
    stamp_repo_id: bool = True
    robot_type: str = "so101_follower"
    # Orin Nano has no NVENC; h264_nvenc fails with "Operation not permitted".
    video_codec: str = "h264"

    # Dataset camera name -> camera.server MJPEG URL. camera.server is the
    # single owner of /dev/video*, so recording reads its
    # streams rather than opening either device a second time.
    cameras: dict[str, str] = field(
        default_factory=lambda: {
            "top": "http://127.0.0.1:8090/video/shoulder.mjpg",
            "wrist": "http://127.0.0.1:8090/video/wrist.mjpg",
        }
    )
    # ACT treats several observation.images.* keys as camera views and
    # requires them to share one shape, so every stream is resized to this.
    image_width: int = 640
    image_height: int = 480

    # A frame older than this is not evidence of the present; the tick is
    # skipped rather than recorded against a stale image.
    max_frame_age_s: float = 0.5
    # Consecutive skips that mean the camera is gone, not merely late. The
    # episode is discarded: a gap teaches the policy a jump that never
    # happened.
    max_stale_ticks: int = 10
    # Assumptions pending measurement of real episode lengths (AGENTS.md
    # §14.3). The floor exists because ACT's default chunk_size is 100
    # frames; the ceiling catches an arm stuck in a loop.
    min_episode_frames: int = 60
    max_episode_frames: int = 3000

    # After the outside region is proved empty, stop and ask for a new
    # arrangement instead of finishing (the arm waits at home).
    prompt_on_round_complete: bool = True
    # Sweeps that saved nothing before the operator is asked to intervene.
    # Without it an unreachable block loops forever, since Task 3 never
    # abandons a colour.
    max_rounds_without_progress: int = 3

    # One task sentence per colour, attached to every frame of that episode.
    task_templates: dict[str, str] = field(
        default_factory=lambda: {
            "red": "Pick up the red block and place it inside the red tape area.",
            "yellow": "Pick up the yellow block and place it inside the red tape area.",
            "green": "Pick up the green block and place it inside the red tape area.",
            "blue": "Pick up the blue block and place it inside the red tape area.",
            "wood": "Pick up the wooden block and place it inside the red tape area.",
        }
    )


@dataclass
class LoggingConfig:
    log_dir: str = "logs/pick_stack"
    # CSV of state transitions per run, named <run_id>_transitions.csv
    save_transitions: bool = True

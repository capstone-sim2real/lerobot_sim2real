"""Configuration tree for the SO-101 pick-and-stack project.

Every tunable lives here as a dataclass field with a default, and can be
overridden by a YAML file (``configs/default.yaml``) and/or CLI ``--set``
key=value overrides. Later PRs add their own config groups (perception,
policy, motion, ...) as new dataclasses wired into ``AppConfig``.

Usage:
    cfg = load_config(Path("src/configs/default.yaml"),
                      overrides=["fsm.time_budget_s=240"])
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class RobotIOConfig:
    """Follower arm connection. Cameras are owned by ``camera.server``."""

    port: str = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6086462-if00"
    id: str = "my_follower"
    max_relative_target: float = 10.0
    # Keep the arm holding its safe pose after a normal task shutdown.
    # Releasing torque must be an explicit manual operation.
    disable_torque_on_disconnect: bool = False
    # Kept as an escape hatch for legacy ACT experiments. The CV+IK runner
    # leaves this empty so ``camera.server`` is the sole /dev/video* owner.
    cameras: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class PerceptionConfig:
    """Top-down camera perception. Metric values are in the board frame (mm)
    defined by the venue calibration JSON (tools/calibrate_homography.py)."""

    # venue calibration produced by tools/calibrate_homography.py
    calibration_path: str = "src/configs/calib/venue_lab.json"
    # camera.server owns the USB device; runner fetches a fresh BGR JPEG here
    # at every SELECT/re-detect rather than opening /dev/video0 itself.
    snapshot_url: str = "http://127.0.0.1:8090/snapshot/shoulder.jpg"
    # chessboard square edge length on the physical board — measure it!
    square_mm: float = 25.0
    # minimal inner-corner grid to search for; CALIB_CB_LARGER extends it,
    # so partial board views (tight framing) still calibrate
    min_pattern: list[int] = field(default_factory=lambda: [5, 5])
    # rectified top-down view scale used by the detector
    rectified_mm_per_px: float = 1.0
    # colour -> list of HSV bands [h_lo, s_lo, v_lo, h_hi, s_hi, v_hi]
    # (OpenCV hue 0-179; red wraps around, hence two bands).
    #
    # These are deliberately GENEROUS gates, not classifications: they decide
    # which blobs are worth looking at, and ``color_prototypes`` below decides
    # what each blob actually is. Overlapping gates are fine and expected —
    # wood and yellow cannot be separated by any fixed box, because which axis
    # separates them depends on where the blocks are sitting (measured: in the
    # dark corners hue splits them and saturation does not; out on the bright
    # board saturation splits them and hue does not).
    # NOTE: red/yellow/wood are still the synthetic-fixture values — re-tune
    # on real frames with tools/view_detect.py before trusting them.
    # green/blue were measured on live frames (2026-09-02): near +-85 deg the
    # table edge is dark enough that a block's V median sits at ~49 while S
    # dips to ~34 at p25, so the old V>=50 / S>=90 floors cut most of the mask
    # and the survivors failed the fill/solidity gates.  The hue ceilings were
    # clipping too (green measured to 90, blue to 135).
    hsv_ranges: dict[str, list[list[int]]] = field(
        default_factory=lambda: {
            "red": [[0, 100, 55, 10, 255, 255], [170, 100, 55, 179, 255, 255]],
            "yellow": [[13, 55, 50, 40, 255, 255]],
            "green": [[35, 40, 30, 95, 255, 255]],
            "blue": [[88, 40, 30, 136, 255, 255]],
            "wood": [[3, 25, 40, 32, 120, 255]],
        }
    )
    # Reference (hue, saturation) points per colour, measured from block
    # interiors (edges eroded, ~20-45k pixels each) across dark-corner and
    # bright-board arrangements. Value is deliberately excluded: it is the
    # channel that moves most with position and carries the least identity.
    #
    # A blob is named by the nearest point across every colour's list, not by
    # which gate caught it, and each colour takes at most ``max_per_color``
    # blobs. Each colour is a LIST of points, not one, because saturation
    # alone can swing across nearly the whole axis for the same physical
    # block between a dark corner and full board light (yellow measured
    # S65-200 across sessions). A single centred point cannot cover that
    # spread without drifting into wood's territory (wood tops out around
    # S~105) — averaging the two regimes made a real yellow block closer to
    # wood's prototype than to its own. Two points, one per regime, keeps
    # each point tight enough that wood-vs-yellow still resolves correctly in
    # both: they are far apart in saturation when hue coincides, and far
    # apart in hue when saturation coincides.
    color_prototypes: dict[str, list[list[int]]] = field(
        default_factory=lambda: {
            "red": [[2, 168]],
            "yellow": [[23, 116], [26, 190]],
            "green": [[72, 148]],
            "blue": [[120, 137]],
            "wood": [[17, 81]],
        }
    )
    # Axis weights for that distance, applied to hue/128 and saturation/128
    # after folding hue onto the same scale (OpenCV hue is half-degrees, so a
    # hue unit is worth two saturation units of perceptual separation).
    prototype_hue_weight: float = 2.0
    prototype_saturation_weight: float = 1.0
    # A blob further than this from every colour's nearest point is not any
    # of the blocks. Without the ceiling the assignment would hand stray
    # blobs whichever colour slot happens to still be free, inventing a
    # block. Measured real blocks land at 0.06-0.13 from their own nearest
    # point; the nearest WRONG colour for the hardest pair (wood vs yellow)
    # is 0.22, so this sits between them with room to spare.
    prototype_max_distance: float = 0.35
    # block top face is 40x40 mm = 1600 mm^2; allow perspective/mask slack.
    # The ceiling is generous because at high azimuth the camera sees the
    # block's *side* faces too, inflating the projected blob to ~2200 mm^2.
    area_mm2_min: float = 900.0
    area_mm2_max: float = 3200.0
    # geometry filters that reject tape: elongated / hollow / sparse shapes.
    # solidity is 0.78 rather than the 0.85 a flat square would give: past
    # about +-75 deg azimuth the camera sees the block's side faces as well as
    # its top, so the silhouette is a genuinely concave hexagon. Measured
    # worst case over 30 frames with the kernel below: 0.847 (blue at +82 deg)
    # against 0.888+ for every other block. Tape and clutter still sit at
    # 0.55-0.72, and max_per_color below keeps the extra blobs harmless.
    aspect_ratio_max: float = 1.6
    solidity_min: float = 0.78
    fill_min: float = 0.65
    # 9, not 5: at the dark table edges the mask fringe flickers frame to
    # frame and eats notches into the blob, which is a *shape* failure, not a
    # colour one — loosening the HSV bands there makes it worse, because the
    # extra noise joins the block. A wider OPEN erodes that fringe away and
    # leaves a stable core. Measured on the blue block at +82 deg over 30
    # frames: solidity floor 0.691 (k=5) -> 0.802 (k=7) -> 0.847 (k=9), while
    # a 40 mm block still measures ~1900-2400 mm2, far above area_mm2_min.
    morph_kernel_px: int = 9
    # The arena holds exactly one block of each colour, so a second surviving
    # blob of the same colour is by definition not a block. Keeping only the
    # best one is what lets the HSV bands stay loose enough for the dark table
    # edges without the extra mask noise turning into phantom targets. Set to
    # 0 to keep every candidate (a venue with duplicate colours).
    max_per_color: int = 1
    # Blobs of different colours closer together than this are the same
    # physical block seen through two gates, so they are merged into one
    # candidate before the colour is decided. 0 disables the merge.
    min_color_separation_mm: float = 30.0
    # Reachable workspace, as a sector of the robot base frame. Blocks outside
    # it are not reported at all: the arm cannot pick them, and the clutter
    # out there (the wooden floor past the board, the far wall) is exactly
    # what produces phantom warm-coloured candidates. The camera page draws
    # this same sector, so what is outlined is what is detected. Radius 0
    # disables the gate.
    workspace_radius_mm: float = 320.0
    workspace_angle_min_deg: float = -90.0
    workspace_angle_max_deg: float = 90.0


@dataclass
class SelectConfig:
    # deterministic rule; must match the teleop demonstration convention
    rule: str = "nearest_first"
    # blocks within this margin of the zone polygon count as "already placed"
    zone_margin_mm: float = 0.0
    # quantization cell for stable target ids across re-detections
    target_cell_mm: float = 40.0


@dataclass
class SensingConfig:
    """Grasp verification + contact detection thresholds.

    All load values are lerobot's decoded Present_Load (signed int, sign =
    direction). Defaults are placeholders — measure real distributions with
    tools/tune_gripper_load.py before trusting them.
    """

    # gripper commands, normalized RANGE_0_100 — 100 is the end of the range
    # the servo calibration recorded, not the mechanical stop (the jaws open
    # noticeably further, but that travel is outside the recorded range and
    # is never commanded). 95 was verified against a real block. Only the
    # FSM's own open/close bookkeeping reads this — the ACT/teleop path
    # commands the gripper itself and never reads these fields.
    gripper_open_pos: float = 95.0
    gripper_close_pos: float = 2.0
    # grasp check thresholds, measured 2026-08-31 (tune_gripper_load.py,
    # --mode grasp, 6 trials): held pos=44.1..44.3 load=500(saturated);
    # empty pos=3.4 load=39..41.
    #
    # The position gate's job is to reject an EMPTY gripper, so it is anchored
    # to the empty distribution (3.4, essentially no spread) rather than to
    # the midpoint of one particular block orientation. The old 20.0 was that
    # midpoint against a block lying flat (40mm of jaw travel); a block caught
    # standing on edge gives only ~20mm of travel and landed on the threshold,
    # so a real grasp was reported EMPTY and retried. 12.0 is 3.5x the empty
    # reading and still well under any block the 70mm jaws can close on.
    gripper_empty_closed_max: float = 12.0
    # grasp check, secondary signal: sustained |Present_Load| on the gripper
    gripper_load_min: float = 200.0
    # position_only | load_only | position_and_load | position_or_load
    grasp_check_mode: str = "position_and_load"
    # let the close settle before sampling
    grasp_settle_s: float = 0.4
    grasp_samples: int = 5
    sample_interval_s: float = 0.05
    # contact detection during stack descent: |load - baseline| spike on any
    # of these joints. Contact can *reduce* load (surface takes the gravity
    # torque), hence the absolute delta.
    contact_joints: list[str] = field(default_factory=lambda: ["shoulder_lift", "elbow_flex"])
    contact_load_delta: float = 80.0
    contact_baseline_samples: int = 5


@dataclass
class MotionConfig:
    """Scripted motion (TRANSPORT / PLACE / STACK). All joint values are in
    the robot's action units (normalized; gripper 0-100) — poses recorded
    with tools/record_pose.py are stored in the same units, so they become
    invalid after recalibration and must be re-recorded."""

    poses_path: str = "src/configs/poses.yaml"
    fps: float = 30.0
    # per-tick joint delta cap for interpolation (action units); the robot's
    # own max_relative_target clamp stays on as a second net
    max_step_per_tick: float = 2.0
    # slower cap while descending onto the tower (contact must be gentle)
    descent_step_per_tick: float = 0.6
    # a move counts as arrived when every joint is within this tolerance
    arrival_tol: float = 3.0
    # looser tolerance for transit moves: holding a block leaves a
    # steady-state joint offset that no amount of extra time closes
    transit_arrival_tol: float = 8.0
    # Tolerance for the hover directly before a grasp descent. Deliberately
    # tighter than either of the above, and NOT transit_arrival_tol despite
    # being a transit: descend() interpolates from the *measured* pose, so
    # whatever error the hover move stopped at gets closed on the way down —
    # the gripper slides sideways as it descends and shoulders the block out
    # of place. Measured sweep at 283mm reach: 69mm at 8 deg, 29mm at 3 deg,
    # 21mm at 2 deg. The jaws are empty here, so the steady-state offset that
    # justifies the loose transit value does not apply.
    grasp_hover_arrival_tol: float = 2.0
    # How long to keep holding the hover trying to reach that tolerance. The
    # servos may not have the resolution for it at all, so this is a short
    # bounded wait and then the descent goes ahead from wherever it got —
    # never move_timeout_s, which would add seconds of dead time to every
    # attempt (and there are five attempts per block).
    grasp_hover_settle_s: float = 0.8
    # lift height above the grasp plane; the actual hover is the highest
    # top-down-reachable z up to this cap (the envelope shrinks with reach)
    hover_clearance_mm: float = 120.0
    hover_min_clearance_mm: float = 40.0
    # granularity of that downward search. The top-down envelope peaks near
    # 90mm and wrist_flex sits on its +-95 deg limit throughout, so every mm
    # of lift is worth finding: a coarse step throws away clearance the arm
    # actually had.
    hover_search_step_mm: float = 5.0
    # Radius to fold the arm back to while carrying a block. The top-down
    # envelope is strongly radius-dependent (measured: ~90mm of lift at
    # 195mm reach, ~50mm at 285mm), so a block picked or placed far out is
    # carried across at this reach instead of at the reach it was picked
    # from. 0 disables the retraction.
    transit_apex_radius_mm: float = 195.0
    move_timeout_s: float = 10.0
    # pause after open/close commands before moving on
    gripper_action_wait_s: float = 0.6
    # pose names (must exist in poses.yaml)
    home_pose: str = "home"
    retreat_pose: str = "retreat"
    transport_waypoints: list[str] = field(default_factory=lambda: ["zone_approach"])
    # Task 1: slot i is used for the (i+1)-th placed block
    slot_poses: list[str] = field(default_factory=lambda: ["slot_0", "slot_1", "slot_2", "slot_3", "slot_4"])
    # Task 2: approach above the tower, then descend along the ladder
    tower_approach_pose: str = "tower_approach"
    tower_ladder_prefix: str = "tower_descent"
    # ticks to reverse after contact before releasing (0 = release in place)
    contact_backoff_ticks: int = 1
    place_settle_s: float = 0.5

    # --- grasp point bias, in the GRIPPER's own frame (control/ik.py
    # gripper_frame_offset): radial = away from the base, tangential = the
    # gripper's own left. The detector reports the block centre, but the jaws
    # were measured closing on its near edge (~5-10mm into a 40mm block
    # instead of ~20mm), so the grasp point is pushed outward.
    grasp_radial_offset_mm: float = 12.0
    # Uniform +10mm toward the gripper-relative left (the tangent of the
    # base-centred reach circle), applied to every block and every retry.
    grasp_tangential_offset_mm: float = 10.0
    # Which frame the offsets above (and the retry offsets below) live in.
    #
    # False: the NEUTRAL-yaw gripper frame — radial is base -> target, and
    # tangential is perpendicular to it. This is what ik.gripper_frame_offset
    # computes and what the camera overlay draws.
    # True: the frame the JAWS actually end up in, i.e. rotated by the extra
    # yaw that grasp_yaw_deg applies to line the jaws up with the block's
    # faces (up to +-45 deg). "Left" then means the held block's left.
    #
    # Default False because it is the frame every existing measurement was
    # taken in; the two differ by rot, so a 24mm offset vector moves by
    # 2*24*sin(rot/2) — about 5mm at a 12 deg jaw turn. Flip it only with a
    # before/after grasp count, and note the overlay keeps drawing the
    # neutral-frame points either way (it never solves IK).
    grasp_offsets_follow_jaw_yaw: bool = False
    # Extra bias on the left half of the workspace (y > left_half_y_mm),
    # where the measured grasp success is lower. Adds to the global bias.
    left_half_y_mm: float = 0.0
    left_half_radial_offset_mm: float = 10.0
    left_half_tangential_offset_mm: float = 0.0
    # Optional ramp on top of that step, per 100mm of y past left_half_y_mm.
    # OFF by default: the 15 calibration points give a tangential-residual/y
    # correlation of +0.017, and a smooth positional correction is exactly
    # hypothesis 4 of docs/report/CV_IK_전환_정리.md, which was tested by
    # inverse-distance interpolation and rejected (LOO 13.99 -> 14.45mm).
    # Kept as a tunable for the hands-on observation that the left half gets
    # worse further out; turn it on only with a before/after measurement.
    left_ramp_radial_mm_per_100mm: float = 0.0
    left_ramp_tangential_mm_per_100mm: float = 0.0
    # Retry grasp points as (radial, tangential) mm from the biased centre,
    # tried in order: left, back, right, front — counter-clockwise from the
    # left in the gripper frame (+tangential is left, +radial is further out).
    # 15mm is the full half-gap the 70mm jaws leave around a 40mm block, so a
    # retry lands dead centre when the aim was off by that much in that
    # direction, and still just contains the block when the aim was right.
    # Sideways first: the left half carries the extra bias correction and is
    # where the aim is least certain, and a sideways miss is the one a
    # blocked descent also points at. Labels are derived from the signs, so
    # diagonals work here too. Empty disables retrying.
    grasp_retry_offsets_mm: list[list[float]] = field(
        default_factory=lambda: [[0.0, 15.0], [-15.0, 0.0], [0.0, -15.0], [15.0, 0.0]]
    )
    # A descent that ends this far short of its goal (action units) counts as
    # blocked rather than arrived. Only reorders the retries — the jaws are
    # closed and checked either way.
    descent_blocked_tol: float = 4.0
    # Abort the descent once the measured pose trails the pose just commanded
    # by this much (action units). Without it, a gripper that lands on a block
    # keeps receiving deeper commands and then a re-sent unreachable goal,
    # which shoves the block away and binds the arm. Raise it if normal
    # descents are misread as blocked; set it very high to disable the watch.
    descent_max_lag: float = 8.0
    # How long the grasp descent may settle against its goal. The loop exits
    # the moment it is within arrival_tol, so this only bounds the blocked
    # case: it costs a normal descent nothing, and stops the servos leaning
    # on the block for the full move_timeout_s when the jaws land on it.
    descent_settle_s: float = 5.0


@dataclass
class IkConfig:
    """Cartesian IK for the CV+IK pick path (AGENTS.md §7).

    Placo's IK is seed-sensitive: a bad seed converges to hundreds of mm of
    error, so ``TopDownIK`` pre-builds a lookup table of top-down joint
    configurations (via forward kinematics, cached to disk) and seeds every
    solve from the nearest entry.
    """

    urdf_path: str = "third_party/so101/so101.urdf"
    target_frame: str = "gripper_frame_link"
    # seed table: joint sweep step and range (degrees) per lift/elbow/wrist_flex
    seed_step_deg: float = 3.0
    seed_range_deg: float = 100.0
    # a seed config counts as "top-down" when its approach axis is within
    # this many degrees of straight down
    seed_tilt_max_deg: float = 3.0
    seed_cache_path: str = "src/configs/calib/ik_seed_table.npz"
    # pan-offset retries to absorb the gripper's lateral offset from the pan
    # axis (AGENTS.md §7 measured ~27mm)
    pan_offset_candidates_deg: list[float] = field(default_factory=lambda: [0.0, 6.0, -6.0, 12.0, -12.0])
    ik_iters: int = 8
    # reject a solve whose achieved pose misses the target by more than this
    # (signals the target is outside the top-down-reachable workspace)
    max_position_error_mm: float = 15.0
    max_tilt_error_deg: float = 6.0


@dataclass
class PolicyConfig:
    """PICK policy served remotely (Orin cannot run inference — AGENTS.md §7).

    The async chain parameters (actions_per_chunk / chunk_size_threshold /
    aggregate) mirror lerobot's validated robot_client values; tune
    chunk_size_threshold against inference latency, one variable at a time.
    """

    server_address: str = "127.0.0.1:8080"
    policy_type: str = "act"
    # path on the MACHINE RUNNING policy_server, not on the Orin
    pretrained_name_or_path: str = ""
    # must match the recording convention's single_task string
    task: str = "Pick the nearest block, lift it vertically, and move to the fixed retreat pose."
    policy_device: str = "cuda"
    actions_per_chunk: int = 50
    chunk_size_threshold: float = 0.5
    aggregate_fn_name: str = "weighted_average"  # or "latest"
    aggregate_weight: float = 0.5  # weight of the incoming action in weighted_average
    fps: float = 30.0
    connect_timeout_s: float = 5.0
    # PICK termination: episodes are trained to end at the fixed retreat pose,
    # so K consecutive ticks within tolerance = successful handoff
    retreat_tol: float = 4.0
    retreat_hold_ticks: int = 5
    # gripper excluded: its position depends on what is being held
    retreat_check_joints: list[str] = field(
        default_factory=lambda: ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    )
    pick_timeout_s: float = 25.0


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
        default_factory=lambda: [20.0, 20.0, 20.0, 20.0, 20.0]
    )
    # The oblique top camera increasingly under-estimates reach at the far
    # edge. Keep near picks untouched, then add a radial correction which
    # reaches max_offset_mm at max_radius_mm.
    pick_correction_start_radius_mm: float = 200.0
    pick_correction_max_radius_mm: float = 320.0
    pick_correction_max_offset_mm: float = 20.0
    # At long reach a perfectly vertical gripper saturates wrist_flex near
    # +95 deg. Gradually tip the approach axis radially outward so the wrist
    # opens while remaining within ik.max_tilt_error_deg.
    pick_tilt_start_radius_mm: float = 280.0
    pick_tilt_max_radius_mm: float = 320.0
    pick_tilt_max_deg: float = 5.0
    # Assumption pending hardware measurement: release just above the
    # calibrated pick plane instead of driving the held block into the table.
    release_clearance_mm: float = 5.0


@dataclass
class LoggingConfig:
    log_dir: str = "logs/pick_stack"
    # CSV of state transitions per run, named <run_id>_transitions.csv
    save_transitions: bool = True


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
    worker_nice: int = 10
    opencv_threads: int = 1
    # publish near-miss contours (and which gate dropped them) to the page, so
    # "no block here" and "block seen, fill 0.48" stay distinguishable
    report_rejects: bool = True
    workspace_boundary: WorkspaceBoundaryConfig = field(default_factory=WorkspaceBoundaryConfig)


@dataclass
class CameraConfig:
    """Camera web UI settings; capture transport stays configured by its CLI."""

    overlay: CameraOverlayConfig = field(default_factory=CameraOverlayConfig)


@dataclass
class AppConfig:
    robot: RobotIOConfig = field(default_factory=RobotIOConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    sensing: SensingConfig = field(default_factory=SensingConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    ik: IkConfig = field(default_factory=IkConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    fsm: FsmConfig = field(default_factory=FsmConfig)
    task1: Task1Config = field(default_factory=Task1Config)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build_dataclass(cls: type, data: dict[str, Any], path: str) -> Any:
    """Recursively build a dataclass from a dict, rejecting unknown keys."""
    hints = get_type_hints(cls)
    valid = {f.name: hints[f.name] for f in fields(cls)}
    unknown = set(data) - set(valid)
    if unknown:
        raise ValueError(f"Unknown config key(s) at '{path}': {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        ftype = valid[name]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _build_dataclass(ftype, value, f"{path}.{name}" if path else name)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(yaml_path: Path | str | None = None, overrides: list[str] | None = None) -> AppConfig:
    """Build AppConfig from defaults, then YAML, then ``key.path=value`` overrides."""
    data: dict[str, Any] = {}
    if yaml_path is not None:
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
    cfg = _build_dataclass(AppConfig, data, path="")
    for override in overrides or []:
        apply_override(cfg, override)
    validate_perception_colors(cfg.perception)
    validate_task1(cfg)
    return cfg


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


def validate_task1(cfg: AppConfig) -> None:
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
    if cfg.task1.pick_correction_start_radius_mm < 0:
        raise ValueError("task1.pick_correction_start_radius_mm must be non-negative")
    if cfg.task1.pick_correction_max_radius_mm <= cfg.task1.pick_correction_start_radius_mm:
        raise ValueError(
            "task1.pick_correction_max_radius_mm must exceed pick_correction_start_radius_mm"
        )
    if cfg.task1.pick_correction_max_offset_mm < 0:
        raise ValueError("task1.pick_correction_max_offset_mm must be non-negative")
    if cfg.task1.pick_tilt_start_radius_mm < 0:
        raise ValueError("task1.pick_tilt_start_radius_mm must be non-negative")
    if cfg.task1.pick_tilt_max_radius_mm <= cfg.task1.pick_tilt_start_radius_mm:
        raise ValueError("task1.pick_tilt_max_radius_mm must exceed pick_tilt_start_radius_mm")
    if not 0 <= cfg.task1.pick_tilt_max_deg <= cfg.ik.max_tilt_error_deg:
        raise ValueError(
            "task1.pick_tilt_max_deg must be between zero and ik.max_tilt_error_deg"
        )


def apply_override(cfg: AppConfig, override: str) -> None:
    """Apply one ``a.b.c=value`` override in place.

    The value is parsed with YAML semantics (so ``true``, ``3.5``, ``[1,2]``
    work), then must match the existing field's container/scalar kind.
    """
    if "=" not in override:
        raise ValueError(f"Override must look like key.path=value, got: {override!r}")
    key_path, raw_value = override.split("=", 1)
    keys = key_path.strip().split(".")
    target: Any = cfg
    for key in keys[:-1]:
        if not hasattr(target, key):
            raise ValueError(f"Unknown config group '{key}' in override {override!r}")
        target = getattr(target, key)
    leaf = keys[-1]
    if not (is_dataclass(target) and hasattr(target, leaf)):
        raise ValueError(f"Unknown config key '{key_path}' in override {override!r}")
    current = getattr(target, leaf)
    if is_dataclass(current):
        raise ValueError(f"Cannot override config group '{key_path}' directly; set its leaf keys")
    value = yaml.safe_load(raw_value)
    if current is not None and value is not None:
        if isinstance(current, bool) != isinstance(value, bool):
            raise ValueError(f"Override {override!r}: expected bool, got {type(value).__name__}")
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Override {override!r}: expected number, got {type(value).__name__}")
            value = type(current)(value)
        elif not isinstance(value, type(current)):
            raise ValueError(
                f"Override {override!r}: expected {type(current).__name__}, got {type(value).__name__}"
            )
    setattr(target, leaf, value)

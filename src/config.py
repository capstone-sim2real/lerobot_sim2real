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
import math
import unicodedata
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
    # Optional [azimuth_deg, max_raw_block_radius_mm] samples. When present,
    # they replace the circular outer edge with a polar envelope; the scalar
    # radius remains a hard cap. Venue defaults are generated from the same
    # corrected PICK + grasp-bias + TopDownIK path used by Task 1.
    workspace_radius_by_angle_mm: list[list[float]] = field(
        default_factory=lambda: [
            [-90, 275], [-80, 285], [-70, 292], [-60, 298], [-50, 301],
            [-40, 307], [-30, 311], [-20, 313], [-10, 315], [0, 315],
            [10, 315], [20, 313], [30, 309], [40, 305], [50, 300],
            [60, 296], [70, 288], [80, 283], [90, 275],
        ]
    )


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
    contact_joints: list[str] = field(
        default_factory=lambda: ["shoulder_lift", "elbow_flex"]
    )
    contact_load_delta: float = 80.0
    contact_baseline_samples: int = 5


@dataclass
class MotionConfig:
    """Scripted motion (TRANSPORT / PLACE / STACK). All joint values are in
    the robot's action units (normalized; gripper 0-100) — poses recorded
    with tools/record_pose.py are stored in the same units, so they become
    invalid after recalibration and must be re-recorded."""

    poses_path: str = "src/configs/poses.yaml"
    fps: float = 45.0
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
    # never move_timeout_s, which would add seconds of dead time to both the
    # initial attempt and its single rotated retry.
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
    slot_poses: list[str] = field(
        default_factory=lambda: ["slot_0", "slot_1", "slot_2", "slot_3", "slot_4"]
    )
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
    # instead of ~20mm), so the grasp point is pushed outward. This measured
    # bias may be reduced by the reachability fallback.
    grasp_radial_offset_mm: float = 0.0
    # Additional F/forward offset requested for every block and retry. Unlike
    # the measured bias above, this part survives reachability bias scaling.
    grasp_forward_offset_mm: float = 0.0
    # Uniform +15mm toward the gripper-relative left (the tangent of the
    # base-centred reach circle), applied to every block and every retry.
    grasp_tangential_offset_mm: float = 15.0
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
    left_half_radial_offset_mm: float = 0.0
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
    # Optional legacy position retries as (radial, tangential) mm. Production
    # leaves this empty: PICK retries once at the same XY with the jaw plane
    # rolled by ``grasp_retry_roll_deg`` instead of searching around the block.
    grasp_retry_offsets_mm: list[list[float]] = field(
        default_factory=list
    )
    # One same-position retry after an empty grasp or obstructed descent.
    # Direction follows the fan half split at left_half_y_mm: +angle
    # (counter-clockwise/left) on the left half, -angle (clockwise/right) on
    # the right half. This keeps the retry wrist turning away from the arm.
    grasp_retry_roll_deg: float = 90.0
    # A descent that ends this far short of its goal (action units) counts as
    # blocked rather than arrived. A blocked PICK descent must never close the
    # jaws; it lifts immediately and proceeds to the rotated retry.
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
    solve from the nearest configured candidates.
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
    # Number of nearest (radius, height) seed postures tried per solve. Three
    # missed a reachable low grasp at (156, -112, 4)mm by 172mm; the tenth
    # candidate converged to 2.4mm, so keep enough branches to cross that
    # discontinuity in the seed table.
    seed_candidate_count: int = 10
    # pan-offset retries to absorb the gripper's lateral offset from the pan
    # axis (AGENTS.md §7 measured ~27mm)
    pan_offset_candidates_deg: list[float] = field(
        default_factory=lambda: [0.0, 6.0, -6.0, 12.0, -12.0]
    )
    ik_iters: int = 8
    # reject a solve whose achieved pose misses the target by more than this
    # (signals the target is outside the top-down-reachable workspace)
    max_position_error_mm: float = 20.0
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
        default_factory=lambda: [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
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
    pick_tilt_max_radius_mm: float = 320.0
    # Outward target-axis tilt applied at every Task-1 pick. This opens
    # wrist_flex through IK while preserving the requested Cartesian point;
    # placement keeps its original far-reach-only tilt ramp.
    pick_tilt_base_deg: float = 3.0
    pick_tilt_max_deg: float = 5.0
    # Assumption pending hardware measurement: release just above the
    # calibrated pick plane instead of driving the held block into the table.
    release_clearance_mm: float = 5.0


@dataclass
class Task2Config:
    """Stack every block at one point; only PLACE differs from Task 1.

    SELECT/PICK/VERIFY and the pick corrections are read from ``task1`` --
    Task 2 *is* Task 1's gather pipeline with a single destination
    (AGENTS.md §3 §4). Only the tower geometry and the contact descent live
    here.
    """

    # Tower location, same [u, v] convention as task1.slot_uv; v -> 1 is the
    # zone edge nearest the base. The top-down envelope collapses with reach
    # (measured: ~90mm of lift at 195mm against ~50mm at 285mm), so every
    # millimetre pulled in buys tower height.
    stack_uv: list[float] = field(default_factory=lambda: [0.50, 0.86])
    # Task 1's measured command under-reach. Needed here not for accuracy --
    # a tower only needs consistency -- but so the block physically lands
    # inside zone_polygon_mm, which is what makes the detector ignore it.
    stack_radial_offset_mm: float = 0.0
    block_height_mm: float = 20.0  # AGENTS.md §1
    # Ceiling on the pre-solved ladder, not a promise: levels the IK cannot
    # reach are reported by the dry-run, never silently clipped.
    max_levels: int = 5

    # Assumption pending hardware measurement: smaller than task1's 5.0mm,
    # because a drop that a table absorbs will topple a tower.
    release_clearance_mm: float = 2.0
    # Task 2 does not use contact-seeking descent. Every level goes straight
    # to the solved release height and opens. Keep this compatibility field
    # fixed at zero so older CLI/config plumbing fails loudly if it tries to
    # re-enable the removed mode.
    contact_descent_levels: int = 0

    # Assumption pending hardware measurement. Three numbers have to sit in
    # one order for a landing to be detectable at all:
    #
    #   loaded trail  <  contact_shortfall  <  descent_max_lag  <  overshoot
    #
    # A block meeting the tower stops the arm ``place_overshoot_mm`` above the
    # commanded goal. If that gap is smaller than the lag we tolerate, a
    # perfect stack reads as "no contact" -- the descent stops in the right
    # place either way, but the backoff never runs and the log lies. Overshoot
    # is in mm and the other two in joint action units, so the ordering can
    # only be confirmed on the arm: watch `shortfall` in stack_contacts.
    place_overshoot_mm: float = 12.0

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
    # Below this, there is not enough travel between hover and floor for a
    # descent to prove anything.
    min_descent_travel_mm: float = 8.0

    # Tilt the approach axis outward as the target rises, releasing the
    # wrist_flex saturation that caps top-down lift (AGENTS.md §7). Late and
    # small: a tilted release lands the block on an edge.
    level_tilt_start_level: int = 2
    level_tilt_per_level_deg: float = 1.5
    level_tilt_max_deg: float = 5.0

    # Assumption pending hardware measurement: motion.descent_max_lag (8.0)
    # was tuned for an empty gripper; a carried block adds steady-state lag,
    # and too small a value reads as a jam on the first tick.
    descent_max_lag: float = 10.0
    # How far short of the goal still counts as having landed. Must clear the
    # loaded steady-state trail (motion.descent_blocked_tol = 4.0 was tuned
    # empty-handed and is too tight) but stay under descent_max_lag. Assumption
    # pending measurement; erring high is the safer mistake, because a missed
    # landing skips the backoff that relieves servo pressure before the jaws
    # open, while a false one only mislabels a release that happens anyway.
    contact_shortfall: float = 6.0
    # motion.descent_settle_s (5.0) would lean on the tower for five seconds
    # after a soft landing that never tripped the lag watch.
    descent_settle_s: float = 0.4
    # Number of descend() calls the descent is split into, with one
    # read_loads() between. Keep at 1: a per-tick read_loads once stranded
    # the arm partway down (control/trajectory.py), and chunking also
    # weakens the lag watch, which cannot accumulate across a call boundary.
    descent_probe_segments: int = 1
    # Contact in the top part of the descent is a mis-stack, not a landing.
    min_descent_fraction: float = 0.5
    max_descent_retries: int = 1


@dataclass
class Task3Config:
    """Automated ACT dataset collection: Task 1's gather loop, recorded.

    Task 3 *is* Task 1 -- same SELECT/PICK/VERIFY/TRANSPORT and the same five
    zone slots -- with three differences (AGENTS.md §3): every arm command is
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
    # single owner of /dev/video* (AGENTS.md §8), so recording reads its
    # stream rather than opening the device a second time. Add the wrist
    # entry once the camera is remounted and served by so101-camera.
    cameras: dict[str, str] = field(
        default_factory=lambda: {"top": "http://127.0.0.1:8090/video/shoulder.mjpg"}
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
    workspace_boundary: WorkspaceBoundaryConfig = field(
        default_factory=WorkspaceBoundaryConfig
    )


@dataclass
class CameraConfig:
    """Camera web UI settings; capture transport stays configured by its CLI."""

    overlay: CameraOverlayConfig = field(default_factory=CameraOverlayConfig)
    # so101-run / so101-collect / so101-agent start camera.server themselves
    # (as the sole /dev/video* owner) when nothing already answers its
    # health check, so an operator no longer has to launch so101-camera by
    # hand first. Set False to require it be started manually, as before.
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
    forward_mm: float = 0.0
    left_mm: float = 0.0
    max_mm: float = 60.0
    max_sample_mm: float = 80.0


@dataclass
class BoardGridConfig:
    """Chessboard-cell addressing over the same sector (``session/grid.py``).

    The lattice itself is measured by ``tools/calibrate_board_grid.py`` and
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
    # (AGENTS.md §7) is what makes a top-down pose impossible there, while
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
    trial_offsets_mm: list[list[float]] = field(default_factory=lambda: [[5.0,0.0],[10.0,0.0],[0.0,5.0]])
    rotate_retry_bias: bool = True
    red_separation_kernel_px: int = 31
    jaw_angle_step_deg: float = 5.0
    tool_radius_mm: float = 60.0
    block_radius_mm: float = 29.0
    uncertainty_mm: float = 15.0
    obstacle_height_mm: float = 20.0  # user-confirmed flat block height; reobserve after tipping
    expected_colors: list[str] = field(default_factory=lambda: ["red", "green", "blue", "yellow", "wood"])


@dataclass
class PrimitiveConfig:
    """Experimental bounds, ASSUMED until physically measured. No mission overrides."""
    calibrated_pick: bool = False  # Enable the measured experiment path explicitly.
    target_max_age_s: float = 120.0
    approach_clearance_mm: float = 60.0
    lateral_clearance_mm: float = 40.0
    alignment_tolerance_mm: float = 25.0
    arrival_error_mm: float = 15.0
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
    max_tick_gap_s: float = 0.1
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
            "openai": "gpt-5.6-luna",
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
    task2: Task2Config = field(default_factory=Task2Config)
    task3: Task3Config = field(default_factory=Task3Config)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    calibration_capture: CalibrationCaptureConfig = field(
        default_factory=CalibrationCaptureConfig
    )
    session_tools: SessionToolsConfig = field(default_factory=SessionToolsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

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
            kwargs[name] = _build_dataclass(
                ftype, value, f"{path}.{name}" if path else name
            )
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(
    yaml_path: Path | str | None = None, overrides: list[str] | None = None
) -> AppConfig:
    """Build AppConfig from defaults, then YAML, then ``key.path=value`` overrides."""
    data: dict[str, Any] = {}
    if yaml_path is not None:
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
    cfg = _build_dataclass(AppConfig, data, path="")
    for override in overrides or []:
        apply_override(cfg, override)
    validate_perception_colors(cfg.perception)
    validate_ik(cfg.ik)
    validate_task1(cfg)
    validate_task2(cfg)
    validate_task3(cfg)
    validate_agent(cfg)
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
    if not 0 <= cfg.task1.pick_tilt_max_deg <= cfg.ik.max_tilt_error_deg:
        raise ValueError(
            "task1.pick_tilt_max_deg must be between zero and ik.max_tilt_error_deg"
        )


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
    if cfg.task2.contact_descent_levels != 0:
        raise ValueError(
            "task2.contact_descent_levels must be 0; Task 2 contact descent is disabled"
        )
    if cfg.task2.release_clearance_mm < 0:
        raise ValueError("task2.release_clearance_mm must be non-negative")
    if cfg.task2.place_overshoot_mm <= cfg.task2.release_clearance_mm:
        raise ValueError(
            "task2.place_overshoot_mm must exceed release_clearance_mm so the "
            "descent goal sits below the nominal landing surface and contact "
            "always fires before the goal (AGENTS.md §5)"
        )
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
    if cfg.task2.min_descent_travel_mm <= 0:
        raise ValueError("task2.min_descent_travel_mm must be positive")
    if cfg.task2.level_tilt_start_level < 1:
        raise ValueError("task2.level_tilt_start_level must be at least one")
    if cfg.task2.level_tilt_per_level_deg < 0:
        raise ValueError("task2.level_tilt_per_level_deg must be non-negative")
    if not 0 <= cfg.task2.level_tilt_max_deg <= cfg.ik.max_tilt_error_deg:
        raise ValueError(
            "task2.level_tilt_max_deg must be between zero and ik.max_tilt_error_deg"
        )
    if cfg.task2.descent_max_lag <= 0:
        raise ValueError("task2.descent_max_lag must be positive")
    if not 0 < cfg.task2.contact_shortfall < cfg.task2.descent_max_lag:
        raise ValueError(
            "task2.contact_shortfall must be positive and below descent_max_lag: "
            "a landing has to register before the descent gives up on the arm"
        )
    if cfg.task2.descent_settle_s < 0:
        raise ValueError("task2.descent_settle_s must be non-negative")
    if cfg.task2.descent_probe_segments < 1:
        raise ValueError("task2.descent_probe_segments must be at least one")
    if not 0.0 <= cfg.task2.min_descent_fraction < 1.0:
        raise ValueError("task2.min_descent_fraction must be in [0, 1)")
    if cfg.task2.max_descent_retries < 0:
        raise ValueError("task2.max_descent_retries must be non-negative")


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
    for name in ("max_tick_gap_s", "max_mean_period_error", "idle_poll_s"):
        value = getattr(agent.collection, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"agent.collection.{name} must be finite and positive")
    primitive = agent.primitives
    for name in ("target_max_age_s", "approach_clearance_mm", "lateral_clearance_mm",
                 "alignment_tolerance_mm", "arrival_error_mm", "cartesian_step_mm",
                 "contact_step_mm", "contact_max_descent_mm", "contact_timeout_s",
                 "contact_backoff_mm", "wrist_roll_limit_deg"):
        value = getattr(primitive, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"agent.primitives.{name} must be finite and positive")
    if primitive.approach_clearance_mm < primitive.lateral_clearance_mm:
        raise ValueError("primitive approach clearance must cover lateral clearance")
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
        raise ValueError(
            f"Cannot override config group '{key_path}' directly; set its leaf keys"
        )
    value = yaml.safe_load(raw_value)
    if current is not None and value is not None:
        if isinstance(current, bool) != isinstance(value, bool):
            raise ValueError(
                f"Override {override!r}: expected bool, got {type(value).__name__}"
            )
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"Override {override!r}: expected number, got {type(value).__name__}"
                )
            value = type(current)(value)
        elif not isinstance(value, type(current)):
            raise ValueError(
                f"Override {override!r}: expected {type(current).__name__}, got {type(value).__name__}"
            )
    setattr(target, leaf, value)

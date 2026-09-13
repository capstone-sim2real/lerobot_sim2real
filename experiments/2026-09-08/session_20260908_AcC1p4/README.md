# 2026-09-08 hardware session checkpoint

This directory preserves experiments, including failed captures and rejected fits.
The active calibration is `teleop_ehdrms_ox0ydl6s/venue_accepted.json`, also copied
into `src/configs/calib/venue_lab.json`. Use the nine rows in
`teleop_ehdrms_ox0ydl6s/accepted_points.csv`; the parent CSV contains older trials.

## Calibration and observations

- Camera was refixed. The shortened 1-minute drift check passed at p95 0.51 px;
  this does not satisfy the normal 10-minute session gate.
- Accepted torque-on nine-point fit: RMS 8.18 mm, maximum 14.51 mm,
  maximum leave-one-out error 26.64 mm. User accepted use despite failed gates.
- V1 and V2 block/FK comparisons were 19.94 and 7.11 mm respectively. These are
  separate poses, not evidence of a correction improving accuracy. Physical
  correspondence between the URDF frame and a point on the jaws is unverified.
- FK overlay uses measured joint angles and inverse plane homography. It is
  not visual gripper detection or a height-aware 3D camera projection.

## Hardware trials

- Task1: four placement actions during a 180-second externally bounded trial;
  blue remained outside. The runner disables its internal FSM budget, so the
  external `timeout --signal=INT 180` was needed.
- That Task1 run used `--set task1.slot_radial_offset_mm=[0,0,0,0,0]` after the
  default 20 mm slot offsets failed dry-run. YAML defaults remain unchanged.
- Motion FPS was subsequently raised from 30 to 45. The later yellow trials
  used 45 FPS; Task1 as a complete mission was not repeated at that rate.
- Original yellow position (156.2, -212.1) mm: successful grasp, confirmed by
  sensor and user. Closed FK (173.1, -216.5) mm, 1.85 mm from biased target.
- Yellow was transported inward and released. At approximately (159.7, -92.6)
  mm the nominal attempt failed (empty close), despite FK within 2.13 mm of
  its target. A fresh-view left-offset retry near (159.0, -88.9) also failed,
  with FK within 1.42 mm of its target. Position and block angle both changed;
  these trials do not isolate the failure cause or establish a success rate.
- A retry planning/execution attempt was aborted before motion because the
  arm occluded the block. Home cleared the view before the executed retry.
- Transfer precheck initially failed at load 84 after the earlier shutdown
  had set gripper goal to present position. Reissuing the normal close restored
  load 500 and a passing grasp check. The archived pick/transfer helpers now
  hold arm goals on exit while preserving the gripper goal.
- Final state: arm home, follower torque retained, teleop paused, yellow on
  the inner workspace. No automatic teleop restart was performed.

Raw images, stage joint readings, plans, logs and summary are in
`pick_validation/`. Calibration snapshots and Task1 log remain in the original
session subdirectories.

## Archived runtime scripts

`runtime_scripts/` is an exact checkpoint of session-specific scripts, not a
portable supported CLI. They assume the Orin device IDs, repository CWD,
`PYTHONPATH=src`, and `/tmp/so101-relative-teleop` paths. Some also depend on saved
path files and earlier plans; review arguments and recreate paths before reuse.
Do not run a second serial reader while teleop or a motion helper owns the bus.

The selected leader profile was ehdrms `so_leader/my_leader.json`; the follower
profile was `so_follower/my_follower.json`. Copies are archived for identity
traceability, not automatic installation on another arm.

Teleop was 60 Hz requested, 5 degrees/tick, relative target clamp 10, unlimited
runtime, with the manually selected 70 C stop threshold. `start.sh` defaults to
65 C unless overridden. Actual sustained teleop rate was not measured.

`live_fk_overlay.py` loads kinematics once and consumes `observations.jsonl`
without serial access. Teleop publishes at about 1 Hz; trial helpers publish
at up to 5 Hz. Camera UI polls every 500 ms; stale data is hidden after 3 seconds.
Start the publisher from the repository CWD after restoring its script:

```bash
env PYTHONPATH=src .venv/bin/python /tmp/so101-relative-teleop/live_fk_overlay.py
```

`tools.show_gripper_reference` publishes a single measurement and uses serial;
use it only when the bus is unowned. Visual feedback before grasp is still
unimplemented. Existing camera cross-calibration UI changes are included as
part of the working tree checkpoint; this session did not validate their
physical calibration accuracy.

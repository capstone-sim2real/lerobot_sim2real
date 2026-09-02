"""Configuration, camera geometry, detection, and target-selection contracts."""

import dataclasses

import cv2
import numpy as np
import pytest

from camera.overlay import detection_metadata, reject_metadata, workspace_boundary_metadata
from camera.server import DEFAULT_OVERLAY_CONFIG, FrameRecorder, parse_args
from camera.vision_worker import VisionWorker
from camera.web_ui import render_camera_page
from config import AppConfig, PerceptionConfig, WorkspaceBoundaryConfig, load_config, validate_perception_colors
from perception import detector
from perception import (
    BlockDetection,
    PlaneCalibration,
    calibrate_from_pairs,
    detect_blocks,
    detect_blocks_with_rejects,
    select_target,
    target_id_for,
)
from runners import run_task


def test_config_loads_overrides_and_rejects_unknown_key(tmp_path):
    cfg = load_config(overrides=["fsm.time_budget_s=180", "robot.cameras={}"])
    assert cfg.fsm.time_budget_s == 180.0
    assert cfg.robot.cameras == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("fsm:\n  typo_key: 1\n")
    with pytest.raises(ValueError, match="typo_key"):
        load_config(bad)


def test_default_yaml_matches_dataclass_defaults():
    """The shipped YAML and the dataclasses duplicate every tuning value.

    Both are read at startup by different processes (camera overlay vs task
    runner), so silent drift between them makes the operator page disagree
    with what the robot actually sees.
    """
    assert dataclasses.asdict(load_config(DEFAULT_OVERLAY_CONFIG)) == dataclasses.asdict(AppConfig())


def test_camera_cli_enables_default_overlay_and_honours_explicit_options():
    defaults = parse_args([])
    assert defaults.host == "0.0.0.0"
    assert defaults.port == 8090
    assert defaults.overlay_config == str(DEFAULT_OVERLAY_CONFIG)
    assert defaults.no_overlay is False

    custom = parse_args(
        ["--host", "127.0.0.1", "--port", "9000", "--overlay-config", "custom.yaml"]
    )
    assert (custom.host, custom.port, custom.overlay_config) == (
        "127.0.0.1", 9000, "custom.yaml"
    )
    assert parse_args(["--no-overlay"]).no_overlay is True


def _block_bgr(color: str, value: int = 170) -> tuple[int, int, int]:
    """A BGR fill matching the configured reference colour for ``color``.

    Fixtures used pure saturated ink before, which no real block is anywhere
    near; the detector now names a blob by how close it sits to the measured
    reference, so a fixture has to look like the block it stands for.
    """
    hue, sat = AppConfig().perception.color_prototypes[color][0]
    patch = np.uint8([[[hue, sat, value]]])
    b, g, r = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)[0][0]
    return (int(b), int(g), int(r))


def _calibration() -> PlaneCalibration:
    pairs = [
        ((0.0, 0.0), (0.0, 0.0)), ((100.0, 0.0), (50.0, 0.0)),
        ((100.0, 100.0), (50.0, 50.0)), ((0.0, 100.0), (0.0, 50.0)),
    ]
    return PlaneCalibration(
        H=calibrate_from_pairs(pairs), image_size=(600, 400), square_mm=25.0,
        base_xy_mm=(250.0, 300.0), zone_polygon_mm=[(10.0, 10.0), (110.0, 10.0), (110.0, 60.0), (10.0, 60.0)],
    )


def test_calibration_round_trip():
    pixels = np.array([[40.0, 80.0], [500.0, 10.0]])
    calib = _calibration()
    assert calib.board_to_pixel(calib.pixel_to_board(pixels)) == pytest.approx(pixels, abs=1e-6)


def test_workspace_arc_is_projected_from_robot_base_and_matches_the_detector():
    calibration = PlaneCalibration(
        H=np.eye(3), image_size=(600, 400), square_mm=25.0, base_xy_mm=(10.0, 20.0)
    )
    perception = PerceptionConfig(
        workspace_radius_mm=100.0, workspace_angle_min_deg=-90.0, workspace_angle_max_deg=90.0
    )
    boundary = workspace_boundary_metadata(
        calibration, WorkspaceBoundaryConfig(sample_step_deg=90.0), perception
    )
    assert boundary is not None
    assert boundary["kind"] == "nominal_topdown_outer"
    np.testing.assert_allclose(
        boundary["points_px"],
        [[10.0, -80.0], [110.0, 20.0], [10.0, 120.0]],
        atol=1e-6,
    )
    # radial legs run from the robot base to the two arc ends, closing the sector
    np.testing.assert_allclose(boundary["base_px"], [10.0, 20.0], atol=1e-6)
    # the drawn arc reports the detector's own gate, not a separate number
    assert boundary["radius_mm"] == perception.workspace_radius_mm
    assert boundary["angle_min_deg"] == perception.workspace_angle_min_deg
    assert boundary["angle_max_deg"] == perception.workspace_angle_max_deg


def test_default_workspace_arc_spans_the_full_half_plane():
    cfg = AppConfig().perception
    assert (cfg.workspace_angle_min_deg, cfg.workspace_angle_max_deg) == (-90.0, 90.0)


def test_overlay_metadata_draws_existing_box_without_changing_detection_geometry():
    calibration = PlaneCalibration(H=np.eye(3), image_size=(600, 400), square_mm=25.0)
    detection = BlockDetection(
        "green", (100.0, 120.0), 1600.0, 1.0, 1.0, 1.0,
        [(80.0, 100.0), (120.0, 100.0), (120.0, 140.0), (80.0, 140.0)],
        angle_deg=0.0,
    )
    metadata = detection_metadata(detection, calibration, AppConfig())
    assert metadata["center_px"] == pytest.approx([100.0, 120.0])
    np.testing.assert_allclose(metadata["box_px"], detection.box_mm, atol=1e-6)
    assert metadata["block_angle_deg"] == 0.0
    assert metadata["display_plan"] == "nominal_full_bias"
    assert "target_label" not in metadata


def test_camera_page_keeps_mjpeg_source_while_canvas_overlay_is_enabled():
    page = render_camera_page(
        [("shoulder", "/dev/video0")], overlay_cameras={"shoulder"}
    ).decode()
    assert 'src="/video/shoulder.mjpg"' in page
    assert 'canvas class="camera-overlay"' in page
    assert 'option value="" selected' in page
    assert "/events/" in page
    assert "/overlay/shoulder.jpg" not in page


def test_vision_worker_returns_metadata_without_blocking_the_caller():
    frame = cv2.imread("docs/calibration/p1_top.jpg")
    assert frame is not None
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok

    worker = VisionWorker(DEFAULT_OVERLAY_CONFIG)
    worker.start()
    worker.subscribe("shoulder")
    try:
        worker.submit("shoulder", encoded.tobytes(), frame_seq=7, captured_at=100.0)
        payload, revision = worker.wait_for_update("shoulder", 0, timeout_s=10.0)
        assert revision == 1
        assert payload is not None
        assert payload["frame_seq"] == 7
        assert payload["display_only"] is True
        assert "detections" in payload
    finally:
        worker.unsubscribe("shoulder")
        worker.stop()


def test_detector_accepts_blocks_and_rejects_same_colour_tape_in_rgb_and_bgr():
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 80), (120, 120), _block_bgr("green"), -1)
    cv2.rectangle(image, (380, 280), (420, 320), _block_bgr("red"), -1)
    cv2.rectangle(image, (180, 191), (400, 209), _block_bgr("red"), -1)
    calib = PlaneCalibration(H=np.eye(3), image_size=(600, 400), square_mm=25.0)
    perception = AppConfig().perception
    perception.workspace_radius_mm = 0.0  # synthetic frame, not a real board
    for frame, is_rgb in ((image, False), (cv2.cvtColor(image, cv2.COLOR_BGR2RGB), True)):
        assert {d.color for d in detect_blocks(frame, calib, perception, is_rgb=is_rgb)} == {"green", "red"}


def test_reject_diagnostics_name_the_gate_without_changing_accepted_detections():
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 80), (120, 120), _block_bgr("green"), -1)
    cv2.rectangle(image, (180, 191), (400, 209), _block_bgr("red"), -1)
    calib = PlaneCalibration(H=np.eye(3), image_size=(600, 400), square_mm=25.0)
    perception = AppConfig().perception
    perception.workspace_radius_mm = 0.0  # synthetic frame, not a real board

    detections, rejects = detect_blocks_with_rejects(image, calib, perception, is_rgb=False)
    assert [d.color for d in detections] == ["green"]
    # 220x18 mm of tape is 3960 mm^2, past the area ceiling before shape matters
    assert [(r.color, r.reason) for r in rejects] == [("red", "area")]
    assert rejects[0].area_mm2 > perception.area_mm2_max

    # the plain entry point stays byte-identical for the robot control path
    plain = detect_blocks(image, calib, perception, is_rgb=False)
    assert plain == detections

    payload = reject_metadata(rejects[0], calib)
    assert payload["reason"] == "area"
    np.testing.assert_allclose(payload["center_px"], [290.0, 200.0], atol=1.0)
    assert len(payload["box_px"]) == 4


def test_every_gated_colour_must_have_a_prototype(tmp_path):
    """Gates may overlap — they must, since no fixed box separates wood from
    yellow in every arrangement. What must not exist is a colour that can be
    gated but never named: its blobs would compete for other colours' slots."""
    bad = tmp_path / "unnamed.yaml"
    bad.write_text(
        "perception:\n"
        "  hsv_ranges:\n"
        "    teal: [[80, 60, 60, 95, 255, 255]]\n"
    )
    with pytest.raises(ValueError, match="color_prototypes is missing"):
        load_config(bad)

    # the shipped palette is complete, and its gates are deliberately overlapping
    load_config(DEFAULT_OVERLAY_CONFIG)
    cfg = AppConfig().perception
    validate_perception_colors(cfg)
    wood, yellow = cfg.hsv_ranges["wood"][0], cfg.hsv_ranges["yellow"][0]
    assert wood[3] > yellow[0], "wood and yellow gates are expected to overlap in hue"


def test_one_block_is_merged_before_it_is_named():
    """A block trips several gates on purpose, so coincident blobs collapse to
    one candidate before a colour is chosen — merging afterwards would let one
    block hold two colour slots."""
    cfg = AppConfig().perception
    block = BlockDetection("yellow", (100.0, 100.0), 1700.0, 1.05, 0.95, 0.85, [], 0.0)
    same = BlockDetection("wood", (108.0, 100.0), 1700.0, 1.10, 0.90, 0.80, [], 0.0)
    other = BlockDetection("wood", (300.0, 100.0), 1500.0, 1.10, 0.90, 0.80, [], 0.0)

    kept, merged = detector._merge_coincident([same, block, other], cfg.min_color_separation_mm)
    assert kept == [block, other] and merged == [same]

    kept, merged = detector._merge_coincident([block, other], cfg.min_color_separation_mm)
    assert len(kept) == 2 and merged == []


def test_colour_comes_from_the_nearest_prototype_not_the_gate():
    """The wood/yellow pair, in both arrangements that were measured on the
    real table. In one, hue separates them and saturation does not; in the
    other it is exactly the reverse. Nearest-prototype handles both; no fixed
    band can."""
    cfg = AppConfig().perception
    for label, wood_hs, yellow_hs in (
        ("dark corners: hue splits them", (13.0, 77.0), (22.0, 105.0)),
        ("bright board: saturation splits them", (21.0, 85.0), (24.0, 127.0)),
    ):
        # both arrive mislabelled by whichever gate happened to catch them
        wood = BlockDetection("yellow", (100.0, 50.0), 1700.0, 1.0, 0.9, 0.8, [], 0.0, wood_hs)
        yellow = BlockDetection("yellow", (200.0, 50.0), 1700.0, 1.0, 0.9, 0.8, [], 0.0, yellow_hs)
        named, unnamed = detector._assign_colors([wood, yellow], cfg)
        assert unnamed == []
        assert [d.color for d in named] == ["wood", "yellow"], label


def test_yellow_keeps_a_separate_point_per_lighting_regime():
    """The regression this guards: yellow's saturation swings 65->200 between
    a dark corner and full board light. A single averaged point sits so far
    from BOTH extremes that a real dark-corner yellow block ends up closer to
    wood's prototype than to its own — this was measured to actually happen
    (own-match 0.299 vs wood-match 0.221) before yellow got a second point.
    """
    cfg = AppConfig().perception
    assert len(cfg.color_prototypes["yellow"]) >= 2, (
        "yellow needs multiple reference points to span its lighting range "
        "without drifting into wood's saturation band"
    )

    dark_corner_yellow = (22.0, 105.0)   # measured: moderate light, near wood's band
    full_board_yellow = (27.0, 198.0)    # measured: bright board, today's failure case
    wood_sample = (13.0, 70.0)

    for label, hue_sat in (
        ("dark corner", dark_corner_yellow),
        ("full board", full_board_yellow),
    ):
        to_yellow = detector._nearest_prototype_distance(
            hue_sat, cfg.color_prototypes["yellow"], cfg
        )
        to_wood = detector._nearest_prototype_distance(
            hue_sat, cfg.color_prototypes["wood"], cfg
        )
        assert to_yellow < to_wood, label
        assert to_yellow <= cfg.prototype_max_distance, label

    # and a real wood sample must still prefer wood over yellow's wide net
    assert detector._nearest_prototype_distance(
        wood_sample, cfg.color_prototypes["wood"], cfg
    ) < detector._nearest_prototype_distance(
        wood_sample, cfg.color_prototypes["yellow"], cfg
    )


def test_blocks_outside_the_reach_sector_are_not_reported():
    """Past the arc the arm cannot pick anything, and the clutter out there
    (wooden floor, far wall) is what produced phantom warm candidates."""
    cfg = AppConfig().perception
    base = (0.0, 0.0)
    inside = (200.0, 100.0)                                   # r=224, az=+27deg
    too_far = (400.0, 0.0)                                    # r=400 > 320
    behind = (-200.0, -50.0)                                  # az=-166deg
    assert detector._in_workspace(inside, cfg, base)
    assert not detector._in_workspace(too_far, cfg, base)
    assert not detector._in_workspace(behind, cfg, base)

    # the gate is measured from the robot base, not the board origin
    shifted = PerceptionConfig(workspace_radius_mm=100.0)
    assert detector._in_workspace((250.0, 0.0), shifted, (200.0, 0.0))
    assert not detector._in_workspace((250.0, 0.0), shifted, (0.0, 0.0))


def test_only_one_block_of_each_colour_survives():
    """The arena holds one block of each colour, so a second surviving blob of
    that colour cannot be a block — and the loose gates the dark table edges
    need would otherwise turn that noise into a phantom target."""
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 80), (120, 120), _block_bgr("red"), -1)
    cv2.rectangle(image, (300, 300), (340, 340), _block_bgr("red"), -1)
    calib = PlaneCalibration(H=np.eye(3), image_size=(600, 400), square_mm=25.0)
    perception = AppConfig().perception
    perception.workspace_radius_mm = 0.0  # synthetic frame, not a real board
    assert perception.max_per_color == 1

    detections, rejects = detect_blocks_with_rejects(image, calib, perception, is_rgb=False)
    assert [d.color for d in detections] == ["red"]
    assert any(r.reason == "unassigned" for r in rejects)

    perception.max_per_color = 0
    unlimited, _ = detect_blocks_with_rejects(image, calib, perception, is_rgb=False)
    assert len(unlimited) == 2


def test_reject_reason_reports_the_shape_gate_for_in_range_clutter():
    # 100x15 mm = 1500 mm^2 sits inside the area window, so the elongation
    # itself has to be what rejects it.
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (200, 193), (300, 208), _block_bgr("red"), -1)
    calib = PlaneCalibration(H=np.eye(3), image_size=(600, 400), square_mm=25.0)
    perception = AppConfig().perception
    perception.workspace_radius_mm = 0.0  # synthetic frame, not a real board

    detections, rejects = detect_blocks_with_rejects(image, calib, perception, is_rgb=False)
    assert detections == []
    assert [(r.color, r.reason) for r in rejects] == [("red", "aspect")]
    assert perception.area_mm2_min <= rejects[0].area_mm2 <= perception.area_mm2_max
    assert rejects[0].aspect > perception.aspect_ratio_max


def test_reject_collection_is_opt_out_and_drops_mask_speckle():
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (180, 191), (400, 209), _block_bgr("red"), -1)
    cv2.rectangle(image, (500, 40), (505, 45), _block_bgr("red"), -1)
    calib = PlaneCalibration(H=np.eye(3), image_size=(600, 400), square_mm=25.0)
    perception = AppConfig().perception
    perception.workspace_radius_mm = 0.0  # synthetic frame, not a real board

    _, rejects = detect_blocks_with_rejects(image, calib, perception, is_rgb=False)
    assert [r.reason for r in rejects] == ["area"]  # the 6x6 speck is below the floor

    _, none_collected = detect_blocks_with_rejects(
        image, calib, perception, is_rgb=False, collect_rejects=False
    )
    assert none_collected == []


def _block(color: str, x: float, y: float) -> BlockDetection:
    return BlockDetection(color, (x, y), 1600.0, 1.0, 1.0, 1.0, [])


def test_selection_excludes_zone_and_skipped_blocks_with_stable_ids():
    cfg = AppConfig().select
    inside, skipped, candidate = _block("red", 50, 30), _block("green", 220, 280), _block("blue", 240, 290)
    assert target_id_for(_block("blue", 238, 292), cfg.target_cell_mm) == target_id_for(candidate, cfg.target_cell_mm)
    result = select_target([inside, skipped, candidate], _calibration(), cfg, {target_id_for(skipped, cfg.target_cell_mm)})
    assert result.target is candidate and result.remaining == 1


def test_runner_perception_uses_http_snapshot_as_bgr(monkeypatch):
    cfg, calibration = AppConfig(), _calibration()
    candidate = _block("blue", 240, 290)
    calls = {}
    monkeypatch.setattr(run_task, "fetch_snapshot", lambda url: calls.setdefault("url", url) and np.zeros((1, 1, 3), dtype=np.uint8))
    monkeypatch.setattr(run_task, "detect_blocks", lambda frame, calib, perception, *, is_rgb: calls.setdefault("is_rgb", is_rgb) or [candidate])
    result = run_task.make_perceive(calibration, cfg)(set())
    assert result.target is candidate
    assert calls == {"url": cfg.perception.snapshot_url, "is_rgb": False}


def test_camera_frame_recorder_saves_atomically_at_the_requested_interval(tmp_path):
    recorder = FrameRecorder(tmp_path, interval_s=1.0)
    first = recorder.record("shoulder", b"one", now=100.123)
    assert first is not None and first.read_bytes() == b"one"
    assert recorder.record("shoulder", b"two", now=100.5) is None
    second = recorder.record("shoulder", b"three", now=101.124)
    assert second is not None and second.read_bytes() == b"three"
    assert recorder.saved == 2
    assert not list((tmp_path / "shoulder").glob("*.tmp"))


def test_camera_frame_recorder_can_save_only_significant_scene_changes(tmp_path):
    def jpeg(value: int) -> bytes:
        ok, encoded = cv2.imencode(".jpg", np.full((90, 160, 3), value, dtype=np.uint8))
        assert ok
        return encoded.tobytes()

    recorder = FrameRecorder(tmp_path, interval_s=1.0, change_threshold=8.0, max_interval_s=5.0)
    assert recorder.record("shoulder", jpeg(0), now=100.0) is not None
    assert recorder.record("shoulder", jpeg(0), now=101.0) is None
    assert recorder.record("shoulder", jpeg(40), now=102.0) is not None
    assert recorder.record("shoulder", jpeg(40), now=103.0) is None
    assert recorder.record("shoulder", jpeg(40), now=107.0) is not None
    assert recorder.saved == 3


def test_detects_block_rotation_for_jaw_alignment():
    """A diamond-oriented block must report its edge angle, not 0.

    The gripper grips two faces only if it can be turned to the block's
    edges; without an angle it always meets two corners and slips.
    """
    pairs = [((0.0, 0.0), (0.0, 0.0)), ((400.0, 0.0), (400.0, 0.0)),
             ((400.0, 300.0), (400.0, 300.0)), ((0.0, 300.0), (0.0, 300.0))]
    calib = PlaneCalibration(H=calibrate_from_pairs(pairs), image_size=(400, 300),
                             square_mm=25.0, base_xy_mm=(200.0, 400.0))
    cfg = AppConfig().perception
    cfg.rectified_mm_per_px = 1.0
    cfg.workspace_radius_mm = 0.0  # synthetic frame, not a real board
    for drawn in (0, 20, 45, 70):
        frame = np.full((300, 400, 3), 200, np.uint8)
        box = cv2.boxPoints(((200.0, 150.0), (40.0, 40.0), float(drawn)))
        cv2.fillPoly(frame, [box.astype(np.int32)], _block_bgr("green"))
        found = detect_blocks(frame, calib, cfg, is_rgb=False)
        assert found, drawn
        # folded to [0, 90): a square grasps identically every quarter turn
        off_by = ((found[0].angle_deg - drawn) + 45.0) % 90.0 - 45.0
        assert abs(off_by) < 3.0, (drawn, found[0].angle_deg)

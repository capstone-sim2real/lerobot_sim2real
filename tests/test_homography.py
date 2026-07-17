import cv2
import numpy as np
import pytest

from pick_stack.perception import PlaneCalibration, calibrate_from_chessboard, calibrate_from_pairs

SQUARE_PX = 40
SQUARE_MM = 25.0


def make_chessboard(rows=7, cols=9, square_px=SQUARE_PX, margin=60):
    h = rows * square_px + 2 * margin
    w = cols * square_px + 2 * margin
    img = np.full((h, w), 200, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y, x = margin + r * square_px, margin + c * square_px
                img[y : y + square_px, x : x + square_px] = 20
    return img


def test_chessboard_calibration_recovers_metric_scale():
    board = make_chessboard()
    # mild perspective warp so the test is not a pure-scale special case
    h, w = board.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[15, 10], [w - 25, 20], [w - 10, h - 15], [5, h - 30]])
    H_warp = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(board, H_warp, (w, h), borderValue=200)

    H, info = calibrate_from_chessboard(warped, SQUARE_MM, min_pattern=(5, 5))
    assert info["rms_mm"] < 1.0
    assert min(info["grid"]) >= 5

    # two pixels that were one square apart on the original board must map
    # to points SQUARE_MM apart in the board frame
    p1 = cv2.perspectiveTransform(np.float64([[[100, 100]]]), H_warp).reshape(2)
    p2 = cv2.perspectiveTransform(np.float64([[[100 + SQUARE_PX, 100]]]), H_warp).reshape(2)
    calib = PlaneCalibration(H=H, image_size=(w, h), square_mm=SQUARE_MM)
    m1, m2 = calib.pixel_to_board(np.array([p1, p2]))
    assert np.linalg.norm(m1 - m2) == pytest.approx(SQUARE_MM, abs=0.5)


def test_chessboard_not_found_raises():
    blank = np.full((480, 640), 128, dtype=np.uint8)
    with pytest.raises(RuntimeError, match="Chessboard not found"):
        calibrate_from_chessboard(blank, SQUARE_MM)


def make_pairs_calib(scale_mm_per_px=0.5):
    pairs = [
        ((0.0, 0.0), (0.0, 0.0)),
        ((100.0, 0.0), (100 * scale_mm_per_px, 0.0)),
        ((100.0, 100.0), (100 * scale_mm_per_px, 100 * scale_mm_per_px)),
        ((0.0, 100.0), (0.0, 100 * scale_mm_per_px)),
    ]
    H = calibrate_from_pairs(pairs)
    return PlaneCalibration(H=H, image_size=(640, 480), square_mm=SQUARE_MM)


def test_pairs_calibration_and_roundtrip():
    calib = make_pairs_calib()
    pts_px = np.array([[40.0, 80.0], [600.0, 10.0]])
    pts_mm = calib.pixel_to_board(pts_px)
    assert pts_mm == pytest.approx(pts_px * 0.5, abs=1e-6)
    assert calib.board_to_pixel(pts_mm) == pytest.approx(pts_px, abs=1e-6)


def test_pairs_require_four_points():
    with pytest.raises(ValueError, match="4 point pairs"):
        calibrate_from_pairs([((0, 0), (0, 0)), ((1, 0), (1, 0)), ((0, 1), (0, 1))])


def test_rectify_scale_and_origin():
    calib = make_pairs_calib(scale_mm_per_px=0.5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:110, 200:210] = (0, 0, 255)  # marker at px (200,100) -> mm (100,50)
    rectified, (ox, oy) = calib.rectify(frame, mm_per_px=1.0)
    assert (ox, oy) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert rectified.shape[0] == pytest.approx(240, abs=2)
    assert rectified.shape[1] == pytest.approx(320, abs=2)
    ys, xs = np.where(rectified[..., 2] > 128)
    assert xs.mean() == pytest.approx(102, abs=2)  # (200..210)px * 0.5 mm/px
    assert ys.mean() == pytest.approx(52, abs=2)


def test_save_load_roundtrip(tmp_path):
    calib = make_pairs_calib()
    calib.base_xy_mm = (160.0, 230.0)
    calib.zone_polygon_mm = [(10.0, 10.0), (110.0, 10.0), (110.0, 60.0), (10.0, 60.0)]
    calib.meta["venue"] = "test"
    path = tmp_path / "calib.json"
    calib.save(path)
    loaded = PlaneCalibration.load(path)
    assert np.allclose(loaded.H, calib.H)
    assert loaded.base_xy_mm == calib.base_xy_mm
    assert loaded.zone_polygon_mm == calib.zone_polygon_mm
    assert loaded.image_size == calib.image_size
    assert loaded.meta["venue"] == "test"

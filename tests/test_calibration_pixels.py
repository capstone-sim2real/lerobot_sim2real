import csv

import cv2
import numpy as np
import pytest

from config import AppConfig
from tools.calibration_pixels import complete_pixel_pair, detect_pixel_candidate, make_pixel_preview


def scene(count=1):
    cfg = AppConfig()
    hue, sat = cfg.perception.color_prototypes['yellow'][0]
    bgr = tuple(map(int, cv2.cvtColor(np.uint8([[[hue, sat, 170]]]), cv2.COLOR_HSV2BGR)[0, 0]))
    frame = np.zeros((240, 700, 3), dtype=np.uint8)
    for i in range(count):
        cv2.rectangle(frame, (420 + i * 100, 100), (460 + i * 100, 140), bgr, -1)
    return frame, cfg


def test_raw_pixels_do_not_use_old_workspace_or_homography():
    frame, cfg = scene()
    cfg.perception.calibration_path = '/nonexistent/stale.json'
    cfg.perception.workspace_radius_mm = 1
    assert detect_pixel_candidate(frame, cfg).center_mm == pytest.approx((440, 120), abs=1)


@pytest.mark.parametrize('count', [0, 2])
def test_missing_or_ambiguous_block_is_rejected(count):
    frame, cfg = scene(count)
    with pytest.raises(ValueError, match=f'found {count}'):
        detect_pixel_candidate(frame, cfg)


def test_preview_does_not_commit_pixels_and_acceptance_preserves_fk(tmp_path):
    frame, cfg = scene()
    photo = tmp_path / 'p1_clean.png'
    cv2.imwrite(str(photo), frame)
    csv_path = tmp_path / 'points.csv'
    original = 'name,image,u_px,v_px,x_m,y_m,z_m,wrist_roll,notes\nP1,p1_top.jpg,,,0.1,-0.2,0.01,5,original\n'
    csv_path.write_text(original)
    centre, preview = make_pixel_preview(photo, cfg)
    assert preview.exists() and csv_path.read_text() == original
    complete_pixel_pair(csv_path, 'P1', centre, photo.name)
    with csv_path.open() as stream:
        row = next(csv.DictReader(stream))
    assert float(row['u_px']) == pytest.approx(440, abs=1)
    assert row['x_m'] == '0.1' and row['wrist_roll'] == '5'
    assert row['image'] == 'p1_clean.png'
    with pytest.raises(ValueError, match='already has pixels'):
        complete_pixel_pair(csv_path, 'P1', centre, photo.name)

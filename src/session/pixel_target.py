"""Fixed head-camera pixel addresses on the calibrated one-block top plane."""
import hashlib
import json
import math

from perception.detector import point_in_workspace, workspace_radius_at_angle
from session.grid import in_base_keepout
from session.factories import calibration_grasp_z_mm


def calibration_id(calib):
    payload = [calib.H.tolist(), list(calib.image_size), calibration_grasp_z_mm(calib)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def resolve_pixel(cfg, calib, u, v, frame_id):
    if cfg.agent.camera_name != 'shoulder':
        raise ValueError('헤드캠 보정만 사용할 수 있습니다.')
    if frame_id != calibration_id(calib):
        raise ValueError('보정 정보가 변경되었습니다. 화면을 새로고침하고 다시 선택하세요.')
    w, h = calib.image_size
    if type(u) is not int or type(v) is not int or not (0 <= u < w and 0 <= v < h):
        raise ValueError('원본 헤드캠 영상 안의 정수 픽셀을 선택하세요.')
    import numpy as np
    point = tuple(float(x) for x in calib.pixel_to_board(np.array([[u, v]]))[0])
    base = calib.base_xy_mm or (0.0, 0.0)
    dx, dy = point[0]-base[0], point[1]-base[1]
    if (not all(math.isfinite(x) for x in point)
        or not point_in_workspace(point, cfg.perception, base)
        or in_base_keepout(point, base, cfg.agent.board_grid)
        or math.hypot(dx,dy) > workspace_radius_at_angle(cfg.perception, math.degrees(math.atan2(dy,dx))) - cfg.agent.table_regions.edge_margin_mm):
        raise ValueError('선택한 픽셀은 허용 범위 밖이거나 로봇 베이스 금지 영역입니다.')
    z = calibration_grasp_z_mm(calib)
    if not math.isfinite(z):
        raise ValueError('블록 윗면 보정 높이가 유효하지 않습니다.')
    return {'u':u, 'v':v, 'x_mm':point[0], 'y_mm':point[1], 'z_mm':z, 'calibration_id':frame_id}


def pixel_preview_config(cfg, calib):
    """Read-only browser preview; execution still uses resolve_pixel and IK."""
    return {
        'camera_name': cfg.agent.camera_name,
        'H': calib.H.tolist(), 'image_size': list(calib.image_size),
        'base_xy_mm': list(calib.base_xy_mm or (0.0, 0.0)),
        'z_mm': calibration_grasp_z_mm(calib), 'calibration_id': calibration_id(calib),
        'radius_mm': cfg.perception.workspace_radius_mm,
        'radius_by_angle_mm': cfg.perception.workspace_radius_by_angle_mm,
        'angle_min_deg': cfg.perception.workspace_angle_min_deg,
        'angle_max_deg': cfg.perception.workspace_angle_max_deg,
        'edge_margin_mm': cfg.agent.table_regions.edge_margin_mm,
        'min_radius_mm': cfg.agent.board_grid.min_radius_mm,
        'keepout_half_width_mm': cfg.agent.board_grid.base_keepout_half_width_mm,
        'keepout_depth_mm': cfg.agent.board_grid.base_keepout_depth_mm,
    }

"""Browser geometry must agree with the server without per-pointer requests."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from agent_helpers import make_skills
from session.pixel_target import calibration_id, pixel_preview_config, resolve_pixel


@pytest.mark.parametrize('profile', [[], [[-90, 200], [0, 320], [90, 240]]])
def test_browser_preview_matches_server(profile):
    if not shutil.which('node'):
        pytest.skip('Node required for browser geometry parity')
    skills, _, robot = make_skills({})
    cfg, calib = skills.cfg, skills.s.calib
    cfg.perception.workspace_radius_by_angle_mm = profile
    rules = pixel_preview_config(cfg, calib)
    w, h = calib.image_size
    points = [(u, v) for u in range(-1, w + 1, 13) for v in range(-1, h + 1, 13)]
    points += [(w, h), (0, 0), (w - 1, h - 1)]
    params = [dict(u=u, v=v, width=w, height=h) for u, v in points]
    expected = []
    for u, v in points:
        try:
            resolve_pixel(cfg, calib, u, v, calibration_id(calib))
            expected.append('valid')
        except ValueError:
            expected.append('invalid')
    assert 'valid' in expected and 'invalid' in expected
    source = Path('src/agent/web/app.js').read_text()
    function = source[source.index('function previewPixel('):source.index('function checkHoverPixel(')]
    script = function + '\nconst data=JSON.parse(require("fs").readFileSync(0,"utf8"));console.log(JSON.stringify(data.params.map(p=>previewPixel(p,data.rules))));'
    result = subprocess.run(['node', '-e', script], input=json.dumps(dict(rules=rules, params=params)), text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == expected
    assert not robot.sent_actions

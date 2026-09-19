"""Numerical S-curve limits, including retargeting while still accelerating."""
import pytest
pytest.importorskip('ruckig')
from config import RelativeMotionConfig
from agent.jog_ramp import JogRamp


def test_acceleration_jerk_and_speed_stay_bounded_when_retargeted():
    cfg = RelativeMotionConfig()
    ramp = JogRamp(cfg)
    dt = 1/cfg.keyboard_tick_hz
    old_a = 0.
    for target in ([20.]*4 + [0.]*2 + [20.]*30 + [0.]*30):
        distance = ramp.advance(target)
        v, a = ramp.input.current_velocity[0], ramp.input.current_acceleration[0]
        assert distance >= 0
        assert -1e-8 <= v <= cfg.keyboard_speed_mm_s+1e-8
        assert abs(a) <= cfg.keyboard_acceleration_mm_s2+1e-8
        assert abs(a-old_a) <= cfg.keyboard_jerk_mm_s3*dt+1e-8
        old_a = a
    assert ramp.stopped


def test_braking_distance_matches_generated_deceleration():
    cfg = RelativeMotionConfig()
    ramp = JogRamp(cfg)
    for _ in range(20):ramp.advance(20.)
    predicted = ramp.braking_distance()
    assert predicted > 0
    travelled = 0.
    for _ in range(100):
        travelled += ramp.advance(0.)
        if ramp.stopped:break
    assert ramp.stopped
    assert travelled == pytest.approx(predicted, abs=1e-8)
    assert travelled == pytest.approx(3., abs=1e-8)

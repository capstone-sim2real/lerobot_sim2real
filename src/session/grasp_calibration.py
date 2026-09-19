"""Compatibility server facade over shared calibrated motion."""
from session.calibration_motion import CalibrationMotion
from session.primitives import PrimitiveSkills


class CalibrationSkills(CalibrationMotion, PrimitiveSkills):
    pass

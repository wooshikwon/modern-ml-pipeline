"""Calibration Component Module"""

# Import modules to trigger auto-registration
from . import modules
from .base import BaseCalibrator
from .registry import CalibrationRegistry

__all__ = [
    "BaseCalibrator",
    "CalibrationRegistry",
    "modules",
]

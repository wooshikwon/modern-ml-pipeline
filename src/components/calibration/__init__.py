"""Calibration Component Module"""
from .registry import CalibrationRegistry

# Import modules to trigger auto-registration
from . import modules

__all__ = [
    "CalibrationRegistry",
    "modules",
]
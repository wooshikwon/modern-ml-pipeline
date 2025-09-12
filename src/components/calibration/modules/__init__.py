"""Calibration Methods Module"""

# Import modules to trigger auto-registration
from . import beta_calibration
from . import isotonic_regression

__all__ = [
    "beta_calibration", 
    "isotonic_regression",
]
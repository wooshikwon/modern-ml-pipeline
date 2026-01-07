"""
Calibration Registry - Self-registration pattern for calibrators.
"""

from __future__ import annotations

from typing import Dict, Type

from src.components.base_registry import BaseRegistry

from .base import BaseCalibrator


class CalibrationRegistry(BaseRegistry[BaseCalibrator]):
    """컴포넌트 레벨 Calibration 레지스트리"""

    _registry: Dict[str, Type[BaseCalibrator]] = {}
    _base_class = BaseCalibrator

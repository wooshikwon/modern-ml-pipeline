"""Engine Module Public API"""

from src.components.calibration.calibration_evaluator import (
    CalibrationEvaluator,
    CalibrationEvaluatorWrapper,
)
from src.utils.deps.dependencies import validate_dependencies

from .factory import Factory


def bootstrap(settings) -> None:
    """
    Global bootstrap for component registration (idempotent).
    Component registries are now self-contained and auto-register on import.
    """
    # Component registration is handled by each component's self-registration
    # Factory directly uses component registries, no central registration needed
    _trigger_component_imports()
    validate_dependencies(settings)


def _trigger_component_imports():
    """Trigger component imports to activate self-registration"""
    # Import component packages to trigger self-registration
    import src.components.adapter
    import src.components.evaluator
    import src.components.fetcher
    import src.components.preprocessor
    import src.components.trainer


__all__ = [
    "Factory",
    "CalibrationEvaluator",
    "CalibrationEvaluatorWrapper",
    "bootstrap",
]

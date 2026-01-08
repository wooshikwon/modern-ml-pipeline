"""Engine Module Public API"""

from mmp.components.calibration.calibration_evaluator import (
    CalibrationEvaluator,
    CalibrationEvaluatorWrapper,
)
from mmp.utils.deps.dependencies import validate_dependencies

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
    import mmp.components.adapter
    import mmp.components.evaluator
    import mmp.components.fetcher
    import mmp.components.preprocessor
    import mmp.components.trainer


__all__ = [
    "Factory",
    "CalibrationEvaluator",
    "CalibrationEvaluatorWrapper",
    "bootstrap",
]

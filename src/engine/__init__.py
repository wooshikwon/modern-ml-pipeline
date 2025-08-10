"""Engine Module Public API"""
from .factory import Factory
from ._registry import AdapterRegistry, EvaluatorRegistry, AugmenterRegistry, register_all_components


from src.utils.system.dependencies import validate_dependencies

def bootstrap(settings) -> None:
    """
    Global bootstrap for component registration (idempotent).
    Call this once at process start or pipeline entry.
    """
    register_all_components()
    validate_dependencies(settings)

__all__ = [
    "Factory",
    "AdapterRegistry",
    "EvaluatorRegistry",
    "AugmenterRegistry",
    "register_all_components",
    "bootstrap",
] 
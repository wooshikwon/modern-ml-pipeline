"""Engine Module Public API"""
from .factory import Factory
from ._registry import AdapterRegistry, EvaluatorRegistry, AugmenterRegistry, register_all_components


from src.utils.system.dependencies import validate_dependencies

def bootstrap(settings) -> None:
    """
    Global bootstrap for component registration (idempotent).
    Call this once at process start or pipeline entry.
    """
    _register_adapters_explicitly()
    _register_other_components()
    validate_dependencies(settings)


def _register_adapters_explicitly():
    """어댑터들을 명시적으로 등록하여 순환 import 방지"""
    from src.utils.adapters.storage_adapter import StorageAdapter
    from src.utils.adapters.sql_adapter import SqlAdapter
    
    AdapterRegistry.register("storage", StorageAdapter)
    AdapterRegistry.register("sql", SqlAdapter)
    
    # 선택적 의존성: feast adapter
    try:
        from src.utils.adapters.feast_adapter import FeastAdapter
        AdapterRegistry.register("feast", FeastAdapter)
        AdapterRegistry.register("feature_store", FeastAdapter)  # Factory에서 feature_store로 요청함
    except ImportError:
        pass


def _register_other_components():
    """기타 컴포넌트들 등록 (기존 로직 유지)"""
    # Evaluators: 명시적 등록
    from src.components._evaluator import (
        ClassificationEvaluator,
        RegressionEvaluator, 
        ClusteringEvaluator,
        CausalEvaluator,
    )
    EvaluatorRegistry.register("classification", ClassificationEvaluator)
    EvaluatorRegistry.register("regression", RegressionEvaluator)
    EvaluatorRegistry.register("clustering", ClusteringEvaluator)
    EvaluatorRegistry.register("causal", CausalEvaluator)

__all__ = [
    "Factory",
    "AdapterRegistry",
    "EvaluatorRegistry",
    "AugmenterRegistry",
    "register_all_components",
    "bootstrap",
] 
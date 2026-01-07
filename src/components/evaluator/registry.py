"""
Evaluator Registry - Self-registration pattern for evaluators.
"""

from __future__ import annotations

from typing import Dict, List, Type

from src.components.base_registry import BaseRegistry

from .base import BaseEvaluator


class EvaluatorRegistry(BaseRegistry[BaseEvaluator]):
    """컴포넌트 레벨 평가자 레지스트리"""

    _registry: Dict[str, Type[BaseEvaluator]] = {}
    _base_class = BaseEvaluator

    @classmethod
    def get_available_metrics_for_task(cls, task_type: str) -> List[str]:
        """Task별 사용 가능한 메트릭 목록 반환"""
        evaluator_class = cls._registry.get(task_type)
        if not evaluator_class:
            return []
        return getattr(evaluator_class, "METRIC_KEYS", [])

    @classmethod
    def get_default_optimization_metric(cls, task_type: str) -> str:
        """Task별 기본 최적화 메트릭 반환"""
        evaluator_class = cls._registry.get(task_type)
        if not evaluator_class:
            return "accuracy"
        return getattr(evaluator_class, "DEFAULT_OPTIMIZATION_METRIC", "accuracy")

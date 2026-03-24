"""
Optimizer Registry - 하이퍼파라미터 최적화기 전용 Registry.

TrainerRegistry에서 분리된 단일 책임 Registry.
"""

from __future__ import annotations

from typing import Dict, Type

from mmp.components.base_registry import BaseRegistry

from .base import BaseOptimizer


class OptimizerRegistry(BaseRegistry[BaseOptimizer]):
    """
    Optimizer 전용 Registry.

    BaseOptimizer를 기반 클래스로 사용하여 등록 시 타입 검증을 수행한다.
    """

    _registry: Dict[str, Type[BaseOptimizer]] = {}
    _base_class = BaseOptimizer

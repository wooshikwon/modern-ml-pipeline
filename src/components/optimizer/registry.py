"""
Optimizer Registry - 하이퍼파라미터 최적화기 전용 Registry.

TrainerRegistry에서 분리된 단일 책임 Registry.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from src.components.base_registry import BaseRegistry


class OptimizerRegistry(BaseRegistry[Any]):
    """
    Optimizer 전용 Registry.

    Optimizer는 공통 기반 클래스가 없으므로 _base_class = None으로 설정.
    추후 BaseOptimizer 도입 시 타입 검증 추가 가능.
    """

    _registry: Dict[str, Type[Any]] = {}
    _base_class = None  # Optimizer는 현재 공통 기반 클래스 없음

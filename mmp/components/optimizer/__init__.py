"""
Optimizer 컴포넌트 패키지.
하이퍼파라미터 최적화를 위한 Registry 및 모듈 제공.
"""

from .base import BaseOptimizer
from .registry import OptimizerRegistry

__all__ = ["BaseOptimizer", "OptimizerRegistry"]

# Self-registration trigger: optimizer 모듈 import시 자동 등록
from . import modules

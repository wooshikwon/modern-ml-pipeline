"""
PreprocessorStep Registry - Self-registration pattern for preprocessor steps.
"""

from __future__ import annotations

from typing import Dict, Type

from src.components.base_registry import BaseRegistry

from .base import BasePreprocessor


class PreprocessorStepRegistry(BaseRegistry[BasePreprocessor]):
    """컴포넌트 레벨 전처리 스텝 레지스트리"""

    _registry: Dict[str, Type[BasePreprocessor]] = {}
    _base_class = BasePreprocessor

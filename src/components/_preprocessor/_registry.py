from __future__ import annotations
from typing import Dict, Type
from src.interface import BasePreprocessor
from src.utils.system.logger import logger

class PreprocessorStepRegistry:
    """컴포넌트 레벨 전처리 스텝 레지스트리 (엔진 의존성 제거)."""
    _steps: Dict[str, Type[BasePreprocessor]] = {}

    @classmethod
    def register(cls, step_type: str, step_class: Type[BasePreprocessor]):
        if not issubclass(step_class, BasePreprocessor):
            raise TypeError(f"{step_class.__name__} must be a subclass of BasePreprocessor")
        cls._steps[step_type] = step_class
        logger.debug(f"[components] Preprocessor step registered: {step_type} -> {step_class.__name__}")

    @classmethod
    def create(cls, step_type: str, **kwargs) -> BasePreprocessor:
        step_class = cls._steps.get(step_type)
        if not step_class:
            available = list(cls._steps.keys())
            raise ValueError(f"Unknown preprocessor step type: '{step_type}'. Available types: {available}")
        logger.debug(f"[components] Creating preprocessor step instance: {step_type}")
        return step_class(**kwargs) 
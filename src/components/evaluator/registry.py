from __future__ import annotations
from typing import Dict, Type
from src.interface import BaseEvaluator
from src.utils.system.logger import logger

class EvaluatorRegistry:
    """컴포넌트 레벨 평가자 레지스트리 (엔진 의존성 제거)."""
    _evaluators: Dict[str, Type[BaseEvaluator]] = {}

    @classmethod
    def register(cls, task_type: str, evaluator_class: Type[BaseEvaluator]):
        if not issubclass(evaluator_class, BaseEvaluator):
            raise TypeError(f"{evaluator_class.__name__} must be a subclass of BaseEvaluator")
        cls.evaluators[task_type] = evaluator_class
        logger.debug(f"[components] Evaluator registered: {task_type} -> {evaluator_class.__name__}")

    @classmethod
    def create(cls, task_type: str, *args, **kwargs) -> BaseEvaluator:
        evaluator_class = cls.evaluators.get(task_type)
        if not evaluator_class:
            available = list(cls.evaluators.keys())
            raise ValueError(f"Unknown task type for evaluator: '{task_type}'. Available types: {available}")
        logger.debug(f"[components] Creating evaluator instance for task: {task_type}")
        return evaluator_class(*args, **kwargs)

    @classmethod
    def get_available_tasks(cls) -> list[str]:
        """등록된 모든 task type 목록 반환."""
        return list(cls.evaluators.keys())

    @classmethod 
    def get_evaluator_class(cls, task_type: str) -> Type[BaseEvaluator]:
        """Task type에 해당하는 Evaluator 클래스 반환."""
        evaluator_class = cls.evaluators.get(task_type)
        if not evaluator_class:
            available = list(cls.evaluators.keys())
            raise ValueError(f"Unknown task type: '{task_type}'. Available types: {available}")
        return evaluator_class
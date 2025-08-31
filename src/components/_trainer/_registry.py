from __future__ import annotations
from typing import Dict, Type
from src.interface import BaseTrainer
from src.utils.system.logger import logger

class TrainerRegistry:
    """컴포넌트 레벨 트레이너 레지스트리 (엔진 의존성 제거)."""
    _trainers: Dict[str, Type[BaseTrainer]] = {}

    @classmethod
    def register(cls, trainer_type: str, trainer_class: Type[BaseTrainer]):
        if not issubclass(trainer_class, BaseTrainer):
            raise TypeError(f"{trainer_class.__name__} must be a subclass of BaseTrainer")
        cls._trainers[trainer_type] = trainer_class
        logger.debug(f"[components] Trainer registered: {trainer_type} -> {trainer_class.__name__}")

    @classmethod
    def create(cls, trainer_type: str, *args, **kwargs) -> BaseTrainer:
        trainer_class = cls._trainers.get(trainer_type)
        if not trainer_class:
            available = list(cls._trainers.keys())
            raise ValueError(f"Unknown trainer type: '{trainer_type}'. Available types: {available}")
        logger.debug(f"[components] Creating trainer instance: {trainer_type}")
        return trainer_class(*args, **kwargs)

    @classmethod
    def get_available_types(cls) -> list[str]:
        """등록된 모든 trainer type 목록 반환."""
        return list(cls._trainers.keys())

    @classmethod 
    def get_trainer_class(cls, trainer_type: str = "default") -> Type[BaseTrainer]:
        """Trainer type에 해당하는 Trainer 클래스 반환."""
        trainer_class = cls._trainers.get(trainer_type)
        if not trainer_class:
            available = list(cls._trainers.keys())
            raise ValueError(f"Unknown trainer type: '{trainer_type}'. Available types: {available}")
        return trainer_class
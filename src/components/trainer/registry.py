from __future__ import annotations
from typing import Dict, Type, Any
from src.interface import BaseTrainer
from src.utils.core.logger import logger
from src.utils.core.console import get_console

class TrainerRegistry:
    """컴포넌트 레벨 트레이너 레지스트리 (엔진 의존성 제거)."""
    trainers: Dict[str, Type[BaseTrainer]] = {}
    optimizers: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, name: str, klass: Type[Any]):
        """단일 register 메서드로 Trainer 또는 Optimizer 등록.

        - BaseTrainer 서브클래스면 trainers에 등록
        - 그 외에는 optimizer로 간주하여 optimizers에 등록
        """
        console = get_console()
        try:
            if isinstance(klass, type) and issubclass(klass, BaseTrainer):
                cls.trainers[name] = klass
                logger.debug(f"[components] Trainer registered: {name} -> {klass.__name__}")
                try:
                    console.debug(f"[TrainerRegistry] Trainer가 등록되었습니다: {name} -> {klass.__name__}",
                                 rich_message=f"📝 [TrainerRegistry] Trainer registered: [cyan]{name}[/cyan] → [green]{klass.__name__}[/green]")
                except AttributeError:
                    # debug 메서드가 없으면 info 사용
                    console.info(f"[TrainerRegistry] Trainer가 등록되었습니다: {name} -> {klass.__name__}",
                                rich_message=f"📝 [TrainerRegistry] Trainer registered: [cyan]{name}[/cyan] → [green]{klass.__name__}[/green]")
            else:
                cls.optimizers[name] = klass
                logger.debug(f"[components] Optimizer registered: {name} -> {klass.__name__}")
                try:
                    console.debug(f"[TrainerRegistry] Optimizer가 등록되었습니다: {name} -> {klass.__name__}",
                                 rich_message=f"📝 [TrainerRegistry] Optimizer registered: [cyan]{name}[/cyan] → [green]{klass.__name__}[/green]")
                except AttributeError:
                    # debug 메서드가 없으면 info 사용
                    console.info(f"[TrainerRegistry] Optimizer가 등록되었습니다: {name} -> {klass.__name__}",
                                rich_message=f"📝 [TrainerRegistry] Optimizer registered: [cyan]{name}[/cyan] → [green]{klass.__name__}[/green]")
        except TypeError:
            # klass가 type이 아니거나 issubclass 불가능한 경우 optimizer로 처리
            cls.optimizers[name] = klass
            logger.debug(f"[components] Optimizer registered: {name} -> {getattr(klass, '__name__', str(klass))}")
            try:
                console.debug(f"[TrainerRegistry] Optimizer가 등록되었습니다: {name} -> {getattr(klass, '__name__', str(klass))}",
                             rich_message=f"📝 [TrainerRegistry] Optimizer registered: [cyan]{name}[/cyan] → [green]{getattr(klass, '__name__', str(klass))}[/green]")
            except AttributeError:
                # debug 메서드가 없으면 info 사용
                console.info(f"[TrainerRegistry] Optimizer가 등록되었습니다: {name} -> {getattr(klass, '__name__', str(klass))}",
                            rich_message=f"📝 [TrainerRegistry] Optimizer registered: [cyan]{name}[/cyan] → [green]{getattr(klass, '__name__', str(klass))}[/green]")

    @classmethod
    def create(cls, trainer_type: str, *args, **kwargs) -> BaseTrainer:
        trainer_class = cls.trainers.get(trainer_type)
        if not trainer_class:
            available = list(cls.trainers.keys())
            raise ValueError(f"Unknown trainer type: '{trainer_type}'. Available types: {available}")
        logger.debug(f"[components] Creating trainer instance: {trainer_type}")
        console = get_console()
        try:
            console.debug(f"[TrainerRegistry] Trainer 인스턴스를 생성합니다: {trainer_type}",
                         rich_message=f"🔧 [TrainerRegistry] Creating trainer: [cyan]{trainer_type}[/cyan]")
        except AttributeError:
            # debug 메서드가 없으면 info 사용
            console.info(f"[TrainerRegistry] Trainer 인스턴스를 생성합니다: {trainer_type}",
                        rich_message=f"🔧 [TrainerRegistry] Creating trainer: [cyan]{trainer_type}[/cyan]")
        return trainer_class(*args, **kwargs)

    @classmethod
    def get_available_types(cls) -> list[str]:
        """등록된 모든 trainer type 목록 반환."""
        return list(cls.trainers.keys())

    @classmethod 
    def get_trainer_class(cls, trainer_type: str = "default") -> Type[BaseTrainer]:
        """Trainer type에 해당하는 Trainer 클래스 반환."""
        trainer_class = cls.trainers.get(trainer_type)
        if not trainer_class:
            available = list(cls.trainers.keys())
            raise ValueError(f"Unknown trainer type: '{trainer_type}'. Available types: {available}")
        return trainer_class

    # Optimizer 생성/조회 (동일 레이어 내)
    @classmethod
    def create_optimizer(cls, name: str, *args, **kwargs):
        optimizer_class = cls.optimizers.get(name)
        if not optimizer_class:
            available = list(cls.optimizers.keys())
            raise ValueError(f"Unknown optimizer: '{name}'. Available: {available}")
        logger.debug(f"[components] Creating optimizer instance: {name}")
        console = get_console()
        try:
            console.debug(f"[TrainerRegistry] Optimizer 인스턴스를 생성합니다: {name}",
                         rich_message=f"🔧 [TrainerRegistry] Creating optimizer: [cyan]{name}[/cyan]")
        except AttributeError:
            # debug 메서드가 없으면 info 사용
            console.info(f"[TrainerRegistry] Optimizer 인스턴스를 생성합니다: {name}",
                        rich_message=f"🔧 [TrainerRegistry] Creating optimizer: [cyan]{name}[/cyan]")
        return optimizer_class(*args, **kwargs)

    @classmethod
    def get_available_optimizer_types(cls) -> list[str]:
        return list(cls.optimizers.keys())
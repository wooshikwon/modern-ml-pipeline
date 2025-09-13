"""
Registry Connector - Component Registry 통합 인터페이스

각 컴포넌트의 Registry 시스템과 연동하여 validation 로직에서
사용할 수 있는 통합된 인터페이스를 제공합니다.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Type, Any
from src.utils.core.logger import logger

class RegistryConnector:
    """
    Component Registry들과의 통합 인터페이스

    각 컴포넌트의 Registry 시스템을 lazy import하여
    순환 의존성을 방지하고 validation 로직에서 사용할 수 있도록 합니다.
    """

    _preprocessor_registry = None
    _evaluator_registry = None
    _trainer_registry = None
    _fetcher_registry = None
    _adapter_registry = None
    _calibration_registry = None
    _datahandler_registry = None

    @classmethod
    def _get_preprocessor_registry(cls):
        """PreprocessorStepRegistry를 lazy import"""
        if cls._preprocessor_registry is None:
            try:
                from src.components.preprocessor.registry import PreprocessorStepRegistry
                cls._preprocessor_registry = PreprocessorStepRegistry
                logger.debug("[validation] PreprocessorStepRegistry loaded")
            except ImportError as e:
                logger.warning(f"[validation] Failed to import PreprocessorStepRegistry: {e}")
                cls._preprocessor_registry = None
        return cls._preprocessor_registry

    @classmethod
    def _get_evaluator_registry(cls):
        """EvaluatorRegistry를 lazy import"""
        if cls._evaluator_registry is None:
            try:
                from src.components.evaluator.registry import EvaluatorRegistry
                cls._evaluator_registry = EvaluatorRegistry
                logger.debug("[validation] EvaluatorRegistry loaded")
            except ImportError as e:
                logger.warning(f"[validation] Failed to import EvaluatorRegistry: {e}")
                cls._evaluator_registry = None
        return cls._evaluator_registry

    @classmethod
    def _get_trainer_registry(cls):
        """TrainerRegistry를 lazy import"""
        if cls._trainer_registry is None:
            try:
                from src.components.trainer.registry import TrainerRegistry
                cls._trainer_registry = TrainerRegistry
                logger.debug("[validation] TrainerRegistry loaded")
            except ImportError as e:
                logger.warning(f"[validation] Failed to import TrainerRegistry: {e}")
                cls._trainer_registry = None
        return cls._trainer_registry

    @classmethod
    def _get_fetcher_registry(cls):
        """FetcherRegistry를 lazy import"""
        if cls._fetcher_registry is None:
            try:
                from src.components.fetcher.registry import FetcherRegistry
                cls._fetcher_registry = FetcherRegistry
                logger.debug("[validation] FetcherRegistry loaded")
            except ImportError as e:
                logger.warning(f"[validation] Failed to import FetcherRegistry: {e}")
                cls._fetcher_registry = None
        return cls._fetcher_registry

    @classmethod
    def _get_adapter_registry(cls):
        """AdapterRegistry를 lazy import"""
        if cls._adapter_registry is None:
            try:
                from src.components.adapter.registry import AdapterRegistry
                cls._adapter_registry = AdapterRegistry
                logger.debug("[validation] AdapterRegistry loaded")
            except ImportError as e:
                logger.warning(f"[validation] Failed to import AdapterRegistry: {e}")
                cls._adapter_registry = None
        return cls._adapter_registry

    @classmethod
    def _get_calibration_registry(cls):
        """CalibrationRegistry를 lazy import"""
        if cls._calibration_registry is None:
            try:
                from src.components.calibration.registry import CalibrationRegistry
                cls._calibration_registry = CalibrationRegistry
                logger.debug("[validation] CalibrationRegistry loaded")
            except ImportError as e:
                logger.warning(f"[validation] Failed to import CalibrationRegistry: {e}")
                cls._calibration_registry = None
        return cls._calibration_registry

    @classmethod
    def _get_datahandler_registry(cls):
        """DataHandlerRegistry를 lazy import"""
        if cls._datahandler_registry is None:
            try:
                from src.components.datahandler.registry import DataHandlerRegistry
                cls._datahandler_registry = DataHandlerRegistry
                logger.debug("[validation] DataHandlerRegistry loaded")
            except ImportError as e:
                logger.warning(f"[validation] Failed to import DataHandlerRegistry: {e}")
                cls._datahandler_registry = None
        return cls._datahandler_registry

    # Preprocessor 관련 메서드들
    @classmethod
    def get_available_preprocessor_steps(cls) -> List[str]:
        """등록된 모든 전처리 스텝 타입 목록 반환"""
        registry = cls._get_preprocessor_registry()
        if registry:
            return list(registry.preprocessor_steps.keys())
        return []

    @classmethod
    def is_preprocessor_step_available(cls, step_type: str) -> bool:
        """전처리 스텝 타입이 등록되어 있는지 확인"""
        registry = cls._get_preprocessor_registry()
        if registry:
            return step_type in registry.preprocessor_steps
        return False

    @classmethod
    def get_preprocessor_step_class(cls, step_type: str) -> Optional[Type]:
        """전처리 스텝 클래스 반환"""
        registry = cls._get_preprocessor_registry()
        if registry:
            return registry.preprocessor_steps.get(step_type)
        return None

    # Evaluator 관련 메서드들
    @classmethod
    def get_available_task_types(cls) -> List[str]:
        """등록된 모든 태스크 타입 목록 반환"""
        registry = cls._get_evaluator_registry()
        if registry:
            return registry.get_available_tasks()
        return []

    @classmethod
    def is_task_type_available(cls, task_type: str) -> bool:
        """태스크 타입이 등록되어 있는지 확인"""
        registry = cls._get_evaluator_registry()
        if registry:
            return task_type in registry.evaluators
        return False

    @classmethod
    def get_evaluator_class(cls, task_type: str) -> Optional[Type]:
        """태스크 타입에 대한 Evaluator 클래스 반환"""
        registry = cls._get_evaluator_registry()
        if registry:
            try:
                return registry.get_evaluator_class(task_type)
            except ValueError:
                return None
        return None

    # Trainer 관련 메서드들 (구체적인 구현은 TrainerRegistry 구조에 따라 달라짐)
    @classmethod
    def get_available_trainers(cls) -> List[str]:
        """등록된 모든 트레이너 목록 반환"""
        registry = cls._get_trainer_registry()
        if registry and hasattr(registry, 'trainers'):
            return list(registry.trainers.keys())
        return []

    @classmethod
    def is_trainer_available(cls, trainer_type: str) -> bool:
        """트레이너 타입이 등록되어 있는지 확인"""
        registry = cls._get_trainer_registry()
        if registry and hasattr(registry, 'trainers'):
            return trainer_type in registry.trainers
        return False

    # Fetcher 관련 메서드들
    @classmethod
    def get_available_fetchers(cls) -> List[str]:
        """등록된 모든 페처 타입 목록 반환"""
        registry = cls._get_fetcher_registry()
        if registry and hasattr(registry, 'fetchers'):
            return list(registry.fetchers.keys())
        return []

    @classmethod
    def is_fetcher_available(cls, fetcher_type: str) -> bool:
        """페처 타입이 등록되어 있는지 확인"""
        registry = cls._get_fetcher_registry()
        if registry and hasattr(registry, 'fetchers'):
            return fetcher_type in registry.fetchers
        return False

    # Adapter 관련 메서드들
    @classmethod
    def get_available_adapters(cls) -> List[str]:
        """등록된 모든 어댑터 타입 목록 반환"""
        registry = cls._get_adapter_registry()
        if registry and hasattr(registry, 'adapters'):
            return list(registry.adapters.keys())
        return []

    @classmethod
    def is_adapter_available(cls, adapter_type: str) -> bool:
        """어댑터 타입이 등록되어 있는지 확인"""
        registry = cls._get_adapter_registry()
        if registry and hasattr(registry, 'adapters'):
            return adapter_type in registry.adapters
        return False

    # Calibration 관련 메서드들
    @classmethod
    def get_available_calibration_methods(cls) -> List[str]:
        """등록된 모든 캘리브레이션 방법 목록 반환"""
        registry = cls._get_calibration_registry()
        if registry and hasattr(registry, 'calibrators'):
            return list(registry.calibrators.keys())
        return []

    @classmethod
    def is_calibration_method_available(cls, method: str) -> bool:
        """캘리브레이션 방법이 등록되어 있는지 확인"""
        registry = cls._get_calibration_registry()
        if registry and hasattr(registry, 'calibrators'):
            return method in registry.calibrators
        return False

    # DataHandler 관련 메서드들
    @classmethod
    def get_available_data_handlers(cls) -> List[str]:
        """등록된 모든 데이터 핸들러 목록 반환"""
        registry = cls._get_datahandler_registry()
        if registry and hasattr(registry, 'handlers'):
            return list(registry.handlers.keys())
        return []

    @classmethod
    def is_data_handler_available(cls, handler_type: str) -> bool:
        """데이터 핸들러 타입이 등록되어 있는지 확인"""
        registry = cls._get_datahandler_registry()
        if registry and hasattr(registry, 'handlers'):
            return handler_type in registry.handlers
        return False

    # 종합적인 검증 메서드
    @classmethod
    def validate_component_availability(cls, component_type: str, component_name: str) -> bool:
        """컴포넌트 타입과 이름에 따라 사용 가능성 검증"""
        validation_map = {
            'preprocessor': cls.is_preprocessor_step_available,
            'evaluator': cls.is_task_type_available,
            'trainer': cls.is_trainer_available,
            'fetcher': cls.is_fetcher_available,
            'adapter': cls.is_adapter_available,
            'calibration': cls.is_calibration_method_available,
            'datahandler': cls.is_data_handler_available,
        }

        validator = validation_map.get(component_type)
        if validator:
            return validator(component_name)

        logger.warning(f"[validation] Unknown component type: {component_type}")
        return False

    @classmethod
    def get_component_catalog(cls) -> Dict[str, List[str]]:
        """전체 컴포넌트 카탈로그 반환"""
        return {
            'preprocessor_steps': cls.get_available_preprocessor_steps(),
            'task_types': cls.get_available_task_types(),
            'trainers': cls.get_available_trainers(),
            'fetchers': cls.get_available_fetchers(),
            'adapters': cls.get_available_adapters(),
            'calibration_methods': cls.get_available_calibration_methods(),
            'data_handlers': cls.get_available_data_handlers(),
        }
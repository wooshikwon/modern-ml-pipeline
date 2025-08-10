"""
Factory Registry Pattern Implementation
Blueprint - Architecture Excellence

이 모듈은 Blueprint 원칙 3 "단순성과 명시성의 원칙"의 완전한 구현을 위한
확장성 있는 Registry 패턴을 제공합니다.

핵심 특징:
- 복잡성 제거: 데코레이터, 메타클래스, 자동 스캔 등 암시적 로직 배제
- 명시적 등록: 필요한 어댑터를 한 곳에서 명시적으로 등록
- 단순성: 순수한 딕셔너리 기반으로 동작하여 예측 가능성 극대화
- 확장성: 새로운 어댑터 추가 시, Factory 수정 없이 이 파일에 한 줄만 추가하면 됨
"""
from __future__ import annotations
import importlib
from typing import Dict, Type, TYPE_CHECKING
from src.utils.system.logger import logger

if TYPE_CHECKING:
    from src.interface import BaseAdapter, BaseEvaluator, BasePreprocessor, BaseAugmenter
    from src.settings import Settings


class AdapterRegistry:
    """
    단순하고 명시적인 어댑터 등록 및 생성 시스템.
    Blueprint 원칙 3: "단순성과 명시성의 원칙"을 구현합니다.
    """
    _adapters: Dict[str, Type[BaseAdapter]] = {}

    @classmethod
    def register(cls, adapter_type: str, adapter_class: Type[BaseAdapter]):
        """
        어댑터를 Registry에 명시적으로 등록합니다.
        데코레이터나 자동 스캔이 아닌, 직접 호출 방식입니다.
        """
        from src.interface import BaseAdapter
        if not issubclass(adapter_class, BaseAdapter):
            raise TypeError(f"{adapter_class.__name__} must be a subclass of BaseAdapter")
        
        cls._adapters[adapter_type] = adapter_class
        logger.debug(f"Adapter registered: {adapter_type} -> {adapter_class.__name__}")

    @classmethod
    def create(cls, adapter_type: str, settings: Settings, **kwargs) -> BaseAdapter:
        """등록된 어댑터의 인스턴스를 생성합니다."""
        adapter_class = cls._adapters.get(adapter_type)
        if not adapter_class:
            available = list(cls._adapters.keys())
            raise ValueError(f"Unknown adapter type: '{adapter_type}'. Available types: {available}")
        
        logger.debug(f"Creating adapter instance: {adapter_type}")
        return adapter_class(settings=settings, **kwargs)


class EvaluatorRegistry:
    """
    Evaluator 컴포넌트의 등록 및 생성을 관리합니다.
    새로운 task_type에 대한 플러그인 아키텍처를 지원합니다.
    """
    _evaluators: Dict[str, Type[BaseEvaluator]] = {}

    @classmethod
    def register(cls, task_type: str, evaluator_class: Type[BaseEvaluator]):
        """Evaluator를 Registry에 명시적으로 등록합니다."""
        from src.interface import BaseEvaluator
        if not issubclass(evaluator_class, BaseEvaluator):
            raise TypeError(f"{evaluator_class.__name__} must be a subclass of BaseEvaluator")
        
        cls._evaluators[task_type] = evaluator_class
        logger.debug(f"Evaluator registered: {task_type} -> {evaluator_class.__name__}")

    @classmethod
    def create(cls, task_type: str, *args, **kwargs) -> BaseEvaluator:
        """등록된 Evaluator의 인스턴스를 생성합니다."""
        evaluator_class = cls._evaluators.get(task_type)
        if not evaluator_class:
            available = list(cls._evaluators.keys())
            raise ValueError(f"Unknown task type for evaluator: '{task_type}'. Available types: {available}")
        
        logger.debug(f"Creating evaluator instance for task: {task_type}")
        return evaluator_class(*args, **kwargs)


class PreprocessorStepRegistry:
    """
    개별 전처리기 '블록(step)'의 등록 및 생성을 관리합니다.
    recipe에서 type 이름으로 전처리기를 사용할 수 있도록 지원합니다.
    """
    _steps: Dict[str, Type[BasePreprocessor]] = {}

    @classmethod
    def register(cls, step_type: str, step_class: Type[BasePreprocessor]):
        """전처리기 블록을 Registry에 명시적으로 등록합니다."""
        from src.interface import BasePreprocessor
        if not issubclass(step_class, BasePreprocessor):
            raise TypeError(f"{step_class.__name__} must be a subclass of BasePreprocessor")
        
        cls._steps[step_type] = step_class
        logger.debug(f"Preprocessor step registered: {step_type} -> {step_class.__name__}")

    @classmethod
    def create(cls, step_type: str, **kwargs) -> BasePreprocessor:
        """등록된 전처리기 블록의 인스턴스를 생성합니다."""
        step_class = cls._steps.get(step_type)
        if not step_class:
            available = list(cls._steps.keys())
            raise ValueError(f"Unknown preprocessor step type: '{step_type}'. Available types: {available}")
        
        logger.debug(f"Creating preprocessor step instance: {step_type}")
        return step_class(**kwargs)


class AugmenterRegistry:
    """Augmenter 타입 등록/생성 레지스트리."""
    _augmenters: Dict[str, Type[BaseAugmenter]] = {}

    @classmethod
    def register(cls, augmenter_type: str, augmenter_class: Type[BaseAugmenter]):
        from src.interface import BaseAugmenter
        if not issubclass(augmenter_class, BaseAugmenter):
            raise TypeError(f"{augmenter_class.__name__} must be a subclass of BaseAugmenter")
        cls._augmenters[augmenter_type] = augmenter_class
        logger.debug(f"Augmenter registered: {augmenter_type} -> {augmenter_class.__name__}")

    @classmethod
    def create(cls, augmenter_type: str, **kwargs) -> BaseAugmenter:
        augmenter_class = cls._augmenters.get(augmenter_type)
        if not augmenter_class:
            available = list(cls._augmenters.keys())
            raise ValueError(f"Unknown augmenter type: '{augmenter_type}'. Available types: {available}")
        logger.debug(f"Creating augmenter instance: {augmenter_type}")
        return augmenter_class(**kwargs)


def register_all_components():
    """
    시스템의 모든 동적 컴포넌트(어댑터, 평가자 등)를 등록합니다.
    이 함수는 애플리케이션 시작 시 한 번만 호출되어야 합니다.
    """
    # 각 컴포넌트 모듈을 임포트하여 자체 등록 로직을 트리거합니다.
    try:
        import importlib
        importlib.import_module('src.utils.adapters.sql_adapter')
        importlib.import_module('src.utils.adapters.storage_adapter')
        # feast는 선택 의존성
        try:
            importlib.import_module('src.utils.adapters.feast_adapter')
        except Exception:
            pass
        # Evaluators: 명시적 등록
        from src.components._evaluator import (
            ClassificationEvaluator,
            RegressionEvaluator,
            ClusteringEvaluator,
            CausalEvaluator,
        )
        EvaluatorRegistry.register("classification", ClassificationEvaluator)
        EvaluatorRegistry.register("regression", RegressionEvaluator)
        EvaluatorRegistry.register("clustering", ClusteringEvaluator)
        EvaluatorRegistry.register("causal", CausalEvaluator)

        # Preprocessor steps: 명시적 등록을 위한 모듈 임포트 (각 모듈에서 로컬 레지스트리에 등록됨)
        importlib.import_module('src.components._preprocessor._steps._encoder')
        importlib.import_module('src.components._preprocessor._steps._discretizer')
        importlib.import_module('src.components._preprocessor._steps._imputer')
        importlib.import_module('src.components._preprocessor._steps._missing')
        importlib.import_module('src.components._preprocessor._steps._feature_generator')
        importlib.import_module('src.components._preprocessor._steps._scaler')

        # Augmenters: 생성은 Factory에서 직접 클래스 사용
        importlib.import_module('src.components._augmenter._augmenter')
        importlib.import_module('src.components._augmenter._pass_through')
    except ImportError as e:
        logger.warning(f"Could not import components for registration: {e}")

# 모듈 로드 시 모든 컴포넌트를 등록합니다.
register_all_components() 
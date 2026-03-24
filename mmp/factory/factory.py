from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Optional

import pandas as pd

from mmp.components.preprocessor import BasePreprocessor
from mmp.factory.component_factory import ComponentCreator
from mmp.factory.model_factory import ModelCreator
from mmp.settings import Settings
from mmp.utils.core.logger import log_fact, logger

if TYPE_CHECKING:
    from mmp.components.adapter.base import BaseAdapter
    from mmp.components.fetcher.base import BaseFetcher
    from mmp.utils.integrations.pyfunc_wrapper import PyfuncWrapper


def cached(cache_key_fn: Callable[..., str]):
    """
    Factory 메서드의 결과를 캐싱하는 데코레이터.
    None 반환값은 캐싱하지 않음 (설정 미완료 상태를 캐싱하지 않기 위함).

    Args:
        cache_key_fn: 캐시 키를 생성하는 함수. 메서드 인자를 받아 문자열 반환.

    사용 예시:
        @cached(lambda adapter_type=None: f"adapter_{adapter_type or 'auto'}")
        def create_data_adapter(self, adapter_type: Optional[str] = None):
            ...
    """

    def decorator(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs) -> Any:
            key = cache_key_fn(*args, **kwargs)
            if key in self._component_cache:
                return self._component_cache[key]

            result = method(self, *args, **kwargs)
            # None은 캐싱하지 않음 (설정 미완료 상태 처리)
            if result is not None:
                self._component_cache[key] = result
            return result

        return wrapper

    return decorator


class Factory:
    """
    3-Tier Component Architecture를 통한 MLOps 컴포넌트 중앙 팩토리 클래스.
    Recipe 설정(settings.recipe)에 기반하여 계층별 패턴에 따라 컴포넌트를 생성합니다.

    Architecture Overview:
    =====================

    🔹 Tier 1: Atomic Components (원자적 컴포넌트)
       - Registry Pattern: XXXRegistry.create()
       - 단일 책임, 독립적 기능
       - 예: Evaluator, Fetcher, DataAdapter, Calibrator

    🔹 Tier 2: Composite Components (조합형 컴포넌트)
       - Factory-aware Registry Pattern: XXXRegistry.create(..., factory_provider=self)
       - 다른 컴포넌트들에 대한 의존성 필요
       - 예: Trainer, DataHandler

    🔹 Tier 3: Orchestrator Components (오케스트레이션 컴포넌트)
       - Direct Instantiation Pattern: Class(settings=self.settings)
       - 복수의 하위 컴포넌트들을 동적 조합/관리
       - 예: Preprocessor (다수 전처리 스텝 조합)

    내부적으로 ComponentCreator와 ModelCreator에 생성 로직을 위임하고,
    캐싱과 공개 API를 제공하는 파사드 역할을 합니다.
    """

    # 클래스 변수: 컴포넌트 등록 상태 추적
    _components_registered: ClassVar[bool] = False

    def __init__(self, settings: Settings):
        # 컴포넌트 자동 등록 (최초 1회만)
        self._ensure_components_registered()

        self.settings = settings

        # Recipe 구조 검증
        if not self.settings.recipe:
            raise ValueError("Recipe 구조가 필요합니다. settings.recipe가 없습니다.")

        # 자주 사용하는 경로 캐싱 (일관된 접근 패턴)
        self._recipe = settings.recipe
        self._config = settings.config
        self._data = self._recipe.data
        self._model = self._recipe.model

        # 생성된 컴포넌트 캐싱
        self._component_cache: Dict[str, Any] = {}

        # 위임 객체 생성
        self._component_creator = ComponentCreator(settings)
        self._model_creator = ModelCreator(settings)

        # Factory 초기화 정보 로깅
        log_fact(f"초기화 완료 - Recipe: {self._recipe.name}, Task: {self._recipe.task_choice}")

        # 환경 설정 요약 추가
        env_name = (
            self._config.environment.name if hasattr(self._config, "environment") else "local"
        )
        data_source_type = (
            getattr(self._config.data_source, "adapter_type", "unknown")
            if hasattr(self._config, "data_source")
            else "unknown"
        )
        feature_store_provider = (
            self._config.feature_store.provider
            if hasattr(self._config, "feature_store") and self._config.feature_store
            else "none"
        )

        logger.debug(
            f"[FACT] 환경 설정 - Environment: {env_name}, DataSource: {data_source_type}, FeatureStore: {feature_store_provider}"
        )

    @classmethod
    def _ensure_components_registered(cls) -> None:
        """
        컴포넌트들이 Registry에 등록되었는지 확인하고, 필요시 등록합니다.
        이 메서드는 Factory 인스턴스가 처음 생성될 때 한 번만 실행됩니다.
        """
        from mmp.components.adapter.registry import AdapterRegistry
        from mmp.components.calibration.registry import CalibrationRegistry
        from mmp.components.datahandler.registry import DataHandlerRegistry
        from mmp.components.evaluator.registry import EvaluatorRegistry
        from mmp.components.fetcher.registry import FetcherRegistry
        from mmp.components.optimizer.registry import OptimizerRegistry
        from mmp.components.preprocessor.registry import PreprocessorStepRegistry
        from mmp.components.trainer.registry import TrainerRegistry

        all_registries = [
            AdapterRegistry,
            CalibrationRegistry,
            DataHandlerRegistry,
            EvaluatorRegistry,
            FetcherRegistry,
            OptimizerRegistry,
            PreprocessorStepRegistry,
            TrainerRegistry,
        ]

        any_empty = any(not registry.list_keys() for registry in all_registries)

        if not cls._components_registered or any_empty:
            # 컴포넌트 모듈들을 import하여 self-registration 트리거
            try:
                import mmp.components.adapter
                import mmp.components.calibration
                import mmp.components.datahandler
                import mmp.components.evaluator
                import mmp.components.fetcher
                import mmp.components.optimizer
                import mmp.components.preprocessor
                import mmp.components.trainer
            except ImportError as e:
                logger.warning(f"[FACT] 일부 컴포넌트 로드 실패: {e}")

            # 등록 후 비어있는 Registry 경고 (optional extras로 인해 에러는 아님)
            for registry in all_registries:
                if not registry.list_keys():
                    logger.warning(
                        f"[FACT] {registry.__name__}에 등록된 컴포넌트가 없습니다."
                    )

            cls._components_registered = True

    # ===============================
    # Tier 1: Atomic Components
    # Registry Pattern - 단일 책임 컴포넌트
    # ===============================

    @cached(lambda adapter_type=None: f"adapter_{adapter_type or 'auto'}")
    def create_data_adapter(self, adapter_type: Optional[str] = None) -> "BaseAdapter":
        """
        데이터 어댑터 생성 (일관된 접근 패턴).

        Args:
            adapter_type: 명시적 어댑터 타입 (선택사항)

        Returns:
            BaseAdapter 인스턴스
        """
        return self._component_creator.create_data_adapter(adapter_type)

    @cached(lambda run_mode=None: f"fetcher_{(run_mode or 'batch').lower()}")
    def create_fetcher(self, run_mode: Optional[str] = None) -> "BaseFetcher":
        """
        Fetcher 생성 (일관된 접근 패턴).

        Args:
            run_mode: 실행 모드 (batch/serving)

        Returns:
            BaseFetcher 인스턴스
        """
        return self._component_creator.create_fetcher(run_mode=run_mode, factory_ref=self)

    @cached(lambda: "evaluator")
    def create_evaluator(self) -> Any:
        """
        Evaluator 생성 (일관된 접근 패턴).

        Returns:
            Evaluator 인스턴스
        """
        return self._component_creator.create_evaluator()

    @cached(lambda method=None: f"calibrator_{method or 'default'}")
    def create_calibrator(self, method: Optional[str] = None) -> Optional[Any]:
        """
        Calibrator 생성 (조건에 따른 생성 분기 처리).

        Args:
            method: 캘리브레이션 방법 ('beta', 'isotonic' 등)

        Returns:
            BaseCalibrator 인스턴스 또는 None
        """
        return self._component_creator.create_calibrator(method)

    @cached(lambda: "feature_store_adapter")
    def create_feature_store_adapter(self) -> "BaseAdapter":
        """
        Feature Store 어댑터 생성 (일관된 접근 패턴).

        Returns:
            Feature Store 어댑터 인스턴스
        """
        return self._component_creator.create_feature_store_adapter()

    # ===============================
    # Tier 2: Composite Components
    # Factory-aware Registry Pattern - 의존성 주입이 필요한 컴포넌트
    # ===============================

    @cached(lambda trainer_type=None: f"trainer_{trainer_type or 'default'}")
    def create_trainer(self, trainer_type: Optional[str] = None) -> Any:
        """
        Trainer 생성 (일관된 접근 패턴).

        Args:
            trainer_type: 트레이너 타입 (None이면 'default' 사용)

        Returns:
            Trainer 인스턴스
        """
        return self._component_creator.create_trainer(
            trainer_type=trainer_type, factory_provider=lambda: self
        )

    @cached(lambda: "datahandler")
    def create_datahandler(self) -> Any:
        """
        DataHandler 생성 (일관된 접근 패턴).
        task_type에 따라 적절한 DataHandler를 자동으로 선택합니다.

        Returns:
            BaseDataHandler 인스턴스
        """
        return self._component_creator.create_datahandler()

    # ===============================
    # Tier 3: Orchestrator Components
    # Direct Instantiation Pattern - 복잡한 조합 로직을 내장한 컴포넌트
    # ===============================

    @cached(lambda: "preprocessor")
    def create_preprocessor(self) -> Optional[BasePreprocessor]:
        """
        Preprocessor 생성 (일관된 접근 패턴).

        Returns:
            BasePreprocessor 인스턴스 또는 None
        """
        return self._component_creator.create_preprocessor()

    # ===============================
    # Specialized Creation Methods
    # Recipe 구조 기반 동적 생성 - 설정에 따른 조건부 생성
    # ===============================

    @cached(lambda: "model")
    def create_model(self) -> Any:
        """
        Model 생성 (일관된 접근 패턴).

        Returns:
            모델 인스턴스
        """
        return self._model_creator.create_model()

    @cached(lambda: "optuna_integration")
    def create_optuna_integration(self) -> Any:
        """
        Optuna Integration 생성 (일관된 접근 패턴).

        Returns:
            OptunaIntegration 인스턴스
        """
        return self._model_creator.create_optuna_integration()

    def create_calibration_evaluator(self, trained_model, trained_calibrator) -> Optional[Any]:
        """
        Calibration Evaluator 생성 및 실행 (모든 복잡한 로직 처리)

        Args:
            trained_model: 학습된 모델
            trained_calibrator: 학습된 calibrator

        Returns:
            Calibration metrics 또는 None
        """
        return self._model_creator.create_calibration_evaluator(trained_model, trained_calibrator)

    def create_pyfunc_wrapper(
        self,
        trained_model: Any,
        trained_datahandler: Any,
        trained_preprocessor: Optional[BasePreprocessor],
        trained_fetcher: Optional["BaseFetcher"],
        trained_calibrator: Optional[Any] = None,
        training_df: Optional[pd.DataFrame] = None,
        training_results: Optional[Dict[str, Any]] = None,
    ) -> PyfuncWrapper:
        """PyfuncWrapper 생성 (PyfuncFactory 위임)"""
        from mmp.factory.pyfunc_factory import PyfuncFactory

        pyfunc_factory = PyfuncFactory(self.settings, self._recipe)
        return pyfunc_factory.create(
            trained_model=trained_model,
            trained_datahandler=trained_datahandler,
            trained_preprocessor=trained_preprocessor,
            trained_fetcher=trained_fetcher,
            trained_calibrator=trained_calibrator,
            training_df=training_df,
            training_results=training_results,
        )

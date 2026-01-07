from __future__ import annotations

import functools
import importlib
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Optional

import pandas as pd

from src.components.adapter import AdapterRegistry
from src.components.adapter.base import BaseAdapter
from src.components.calibration.calibration_evaluator import CalibrationEvaluator
from src.components.evaluator import EvaluatorRegistry
from src.components.fetcher import FetcherRegistry
from src.components.preprocessor import BasePreprocessor, Preprocessor
from src.settings import Settings
from src.utils.core.logger import log_fact, logger

if TYPE_CHECKING:
    from src.components.fetcher.base import BaseFetcher
    from src.utils.integrations.pyfunc_wrapper import PyfuncWrapper


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
                logger.debug(f"캐시된 컴포넌트 반환: {key}")
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

    일관된 접근 패턴과 캐싱을 통해 효율적인 컴포넌트 생성을 보장합니다.
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
        # Check if calibration components are already registered
        from src.components.calibration.registry import CalibrationRegistry

        is_calibration_registered = bool(CalibrationRegistry.list_keys())

        if not cls._components_registered or not is_calibration_registered:
            # 컴포넌트 모듈들을 import하여 self-registration 트리거
            try:
                import src.components.adapter
                import src.components.calibration
                import src.components.datahandler
                import src.components.evaluator
                import src.components.fetcher
                import src.components.optimizer
                import src.components.preprocessor
                import src.components.trainer
            except ImportError as e:
                logger.warning(f"[FACT] 일부 컴포넌트 로드 실패: {e}")

            cls._components_registered = True

    def _create_from_class_path(self, class_path: str, hyperparameters: Dict[str, Any]) -> Any:
        """
        클래스 경로로부터 동적으로 객체를 생성하는 헬퍼 메서드.

        Args:
            class_path: 전체 클래스 경로 (예: 'sklearn.ensemble.RandomForestClassifier')
            hyperparameters: 클래스 초기화 파라미터

        Returns:
            생성된 객체 인스턴스
        """
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            # 하이퍼파라미터 전처리 (callable 파라미터 처리)
            processed_params = self._process_hyperparameters(hyperparameters)

            instance = model_class(**processed_params)
            logger.debug(f"[FACT] 클래스 인스턴스 생성 완료 - {class_path}")
            return instance

        except Exception as e:
            logger.error(f"[FACT] 클래스 인스턴스 생성 실패 - {class_path}: {e}")
            raise ValueError(f"Could not load class: {class_path}") from e

    def _process_hyperparameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """하이퍼파라미터 전처리 (문자열을 객체로 변환 등)."""
        processed = params.copy()

        for key, value in processed.items():
            if isinstance(value, str) and "." in value and ("_fn" in key or "_class" in key):
                try:
                    module_path, func_name = value.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    processed[key] = getattr(module, func_name)
                    logger.debug(f"Hyperparameter '{key}'를 callable로 변환했습니다: {value}")
                except (ImportError, AttributeError):
                    logger.debug(f"Hyperparameter '{key}'를 문자열로 유지합니다: {value}")

        return processed

    def _detect_adapter_type_from_uri(self, source_uri: str) -> str:
        """
        source_uri 패턴을 분석하여 필요한 어댑터 타입을 자동으로 결정합니다.

        패턴:
        - .sql 파일 또는 SQL 쿼리 → 'sql'
        - .csv, .parquet, .json → 'storage'
        - s3://, gs://, az:// → 'storage'
        - bigquery:// → 'sql'
        """
        uri_lower = source_uri.lower()

        # SQL 패턴
        if uri_lower.endswith(".sql") or "select" in uri_lower or "from" in uri_lower:
            return "sql"

        # BigQuery 패턴 → SQL adapter로 통합
        if uri_lower.startswith("bigquery://"):
            return "sql"  # 'bigquery' 대신 'sql' 반환

        # Cloud Storage 패턴
        if any(uri_lower.startswith(prefix) for prefix in ["s3://", "gs://", "az://"]):
            return "storage"

        # File 패턴
        if any(uri_lower.endswith(ext) for ext in [".csv", ".parquet", ".json", ".tsv"]):
            return "storage"

        # 기본값
        logger.warning(f"[FACT] URI 패턴 인식 실패: {source_uri} -> storage 어댑터 사용")
        return "storage"

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
        # 어댑터 타입 결정
        if adapter_type:
            target_type = adapter_type
        else:
            # 1순위: config에서 명시된 adapter_type 사용
            config_adapter_type = getattr(self.settings.config.data_source, "adapter_type", None)
            if config_adapter_type:
                target_type = config_adapter_type
                logger.debug(f"[FACT] 설정된 adapter 유형 사용: {target_type}")
            else:
                # 2순위: source_uri에서 자동 감지
                source_uri = self._data.loader.source_uri
                target_type = self._detect_adapter_type_from_uri(source_uri)
                logger.debug(f"[FACT] URI에서 adapter 유형 자동 감지: {target_type}")

        # Registry를 통한 생성
        try:
            adapter = AdapterRegistry.create(target_type, self.settings)
            return adapter
        except Exception as e:
            available = AdapterRegistry.list_keys()
            logger.error(f"'{target_type}' adapter 생성에 실패했습니다. Available: {available}")
            raise ValueError(
                f"Failed to create adapter '{target_type}'. Available: {available}"
            ) from e

    @cached(lambda run_mode=None: f"fetcher_{(run_mode or 'batch').lower()}")
    def create_fetcher(self, run_mode: Optional[str] = None) -> "BaseFetcher":
        """
        Fetcher 생성 (일관된 접근 패턴).

        Args:
            run_mode: 실행 모드 (batch/serving)

        Returns:
            BaseFetcher 인스턴스
        """
        mode = (run_mode or "batch").lower()

        # 설정 접근
        provider = (
            self.settings.config.feature_store.provider
            if self.settings.config.feature_store
            else "none"
        )
        fetch_conf = self._recipe.data.fetcher if hasattr(self._recipe.data, "fetcher") else None
        fetch_type = fetch_conf.type if fetch_conf else None

        # serving 모드 검증
        if mode == "serving":
            if fetch_type in (None, "pass_through") or provider in (None, "none"):
                raise TypeError(
                    "Serving 모드에서는 Feature Store 연결이 필요합니다. "
                    "pass_through 또는 feature_store 미구성은 허용되지 않습니다."
                )

        # Fetcher 생성: 환경이 아닌 명시적 설정(provider, fetch_type)으로 결정
        try:
            if provider == "none" or fetch_type == "pass_through" or not fetch_conf:
                fetcher = FetcherRegistry.create("pass_through")
            elif fetch_type == "feature_store" and provider in {"feast", "mock", "dynamic"}:
                fetcher = FetcherRegistry.create(fetch_type, settings=self.settings, factory=self)
            else:
                raise ValueError(
                    f"적절한 fetcher를 선택할 수 없습니다. "
                    f"provider={provider}, fetch_type={fetch_type}, mode={mode}"
                )

            return fetcher

        except Exception as e:
            logger.error(f"Fetcher 생성에 실패했습니다: {e} (mode={mode}, provider={provider})")
            raise

    @cached(lambda: "evaluator")
    def create_evaluator(self) -> Any:
        """
        Evaluator 생성 (일관된 접근 패턴).

        Returns:
            Evaluator 인스턴스
        """
        task_choice = self._recipe.task_choice

        try:
            evaluator = EvaluatorRegistry.create(task_choice, self.settings)
            return evaluator

        except Exception:
            available = EvaluatorRegistry.list_keys()
            logger.error(
                f"'{task_choice}'에 대한 Evaluator 생성에 실패했습니다. Available: {available}"
            )
            raise

    @cached(lambda method=None: f"calibrator_{method or 'default'}")
    def create_calibrator(self, method: Optional[str] = None) -> Optional[Any]:
        """
        Calibrator 생성 (조건에 따른 생성 분기 처리).

        Args:
            method: 캘리브레이션 방법 ('beta', 'isotonic' 등)

        Returns:
            BaseCalibrator 인스턴스 또는 None
        """
        # Task와 calibration 설정 확인
        task_type = self._recipe.task_choice
        calibration_config = getattr(self._recipe.model, "calibration", None)

        if not calibration_config or not getattr(calibration_config, "enabled", False):
            logger.debug("[FACT] Calibration 비활성화 - 스킵")
            return None

        if task_type != "classification":
            logger.debug(f"[FACT] {task_type} task에서는 Calibration 미지원 - 스킵")
            return None

        # Method 결정
        calibration_method = method or getattr(calibration_config, "method", None)
        if not calibration_method:
            raise ValueError(
                "Calibration method가 설정되지 않았습니다. "
                "Recipe에서 model.calibration.method를 설정하세요. "
                "사용 가능: 'beta', 'isotonic', 'temperature'"
            )

        try:
            from src.components.calibration.registry import CalibrationRegistry

            calibrator = CalibrationRegistry.create(calibration_method)
            return calibrator

        except Exception:
            from src.components.calibration.registry import CalibrationRegistry

            available = CalibrationRegistry.list_keys()
            logger.error(
                f"'{calibration_method}' Calibrator 생성에 실패했습니다. Available: {available}"
            )
            raise

    @cached(lambda: "feature_store_adapter")
    def create_feature_store_adapter(self) -> "BaseAdapter":
        """
        Feature Store 어댑터 생성 (일관된 접근 패턴).

        Returns:
            Feature Store 어댑터 인스턴스
        """
        # 검증
        if not self.settings.config.feature_store:
            raise ValueError("Feature Store settings are not configured.")

        try:
            adapter = AdapterRegistry.create("feature_store", self.settings)
            return adapter

        except Exception as e:
            logger.error(f"Feature Store adapter 생성에 실패했습니다: {e}")
            raise

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
        from src.components.trainer import TrainerRegistry

        trainer_type = trainer_type or "default"

        try:
            trainer = TrainerRegistry.create(
                trainer_type, settings=self.settings, factory_provider=lambda: self
            )
            return trainer

        except Exception:
            available = TrainerRegistry.list_keys()
            logger.error(f"'{trainer_type}' Trainer 생성에 실패했습니다. Available: {available}")
            raise

    @cached(lambda: "datahandler")
    def create_datahandler(self) -> Any:
        """
        DataHandler 생성 (일관된 접근 패턴).
        task_type에 따라 적절한 DataHandler를 자동으로 선택합니다.

        Returns:
            BaseDataHandler 인스턴스
        """
        from src.components.datahandler import DataHandlerRegistry

        task_choice = self._recipe.task_choice

        try:
            model_class_path = getattr(self._recipe.model, "class_path", None)
            datahandler = DataHandlerRegistry.get_handler_for_task(
                task_choice, self.settings, model_class_path=model_class_path
            )
            return datahandler

        except Exception:
            available = DataHandlerRegistry.list_keys()
            logger.error(
                f"'{task_choice}'에 대한 DataHandler 생성에 실패했습니다. Available: {available}"
            )
            raise

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
        preprocessor_config = getattr(self._recipe, "preprocessor", None)

        if not preprocessor_config:
            return None

        try:
            preprocessor = Preprocessor(settings=self.settings)
            return preprocessor

        except Exception as e:
            logger.error(f"Preprocessor 생성에 실패했습니다: {e}")
            raise

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
        class_path = self._model.class_path

        # 하이퍼파라미터 추출 (tuning 메타데이터 제외)
        hyperparameters = {}
        if hasattr(self._model.hyperparameters, "tuning_enabled"):
            if self._model.hyperparameters.tuning_enabled:
                # 튜닝 활성화시: fixed 파라미터만 사용
                if (
                    hasattr(self._model.hyperparameters, "fixed")
                    and self._model.hyperparameters.fixed
                ):
                    hyperparameters = self._model.hyperparameters.fixed.copy()
            else:
                # 튜닝 비활성화시: values 파라미터 사용
                if (
                    hasattr(self._model.hyperparameters, "values")
                    and self._model.hyperparameters.values
                ):
                    hyperparameters = self._model.hyperparameters.values.copy()
        else:
            # 레거시 구조: 전체 dict에서 tuning 메타데이터 제외
            hyperparameters = (
                dict(self._model.hyperparameters)
                if hasattr(self._model.hyperparameters, "__dict__")
                else {}
            )
            tuning_keys = [
                "tuning_enabled",
                "optimization_metric",
                "direction",
                "n_trials",
                "timeout",
                "fixed",
                "tunable",
                "values",
            ]
            for key in tuning_keys:
                hyperparameters.pop(key, None)

        try:
            model = self._create_from_class_path(class_path, hyperparameters)
            return model

        except Exception as e:
            logger.error(f"{class_path}에서 Model 생성에 실패했습니다: {e}")
            raise

    # ===============================
    # Specialized Creation Methods
    # Recipe 구조 기반 동적 생성 - 설정에 따른 조건부 생성
    # ===============================

    @cached(lambda: "optuna_integration")
    def create_optuna_integration(self) -> Any:
        """
        Optuna Integration 생성 (일관된 접근 패턴).

        Returns:
            OptunaIntegration 인스턴스
        """
        tuning_config = getattr(self._model, "hyperparameters", None)

        if not tuning_config:
            raise ValueError("Hyperparameter tuning settings are not configured.")

        try:
            from src.utils.integrations.optuna_integration import OptunaIntegration

            integration = OptunaIntegration(tuning_config)
            return integration

        except ImportError:
            logger.error("Optuna가 설치되지 않았습니다. pip install optuna로 설치해주세요")
            raise
        except Exception as e:
            logger.error(f"Optuna integration 생성에 실패했습니다: {e}")
            raise

    def create_calibration_evaluator(self, trained_model, trained_calibrator) -> Optional[Any]:
        """
        Calibration Evaluator 생성 및 실행 (모든 복잡한 로직 처리)

        Args:
            trained_model: 학습된 모델
            trained_calibrator: 학습된 calibrator

        Returns:
            Calibration metrics 또는 None
        """
        # Task와 calibrator 확인
        task_type = self._recipe.task_choice
        if task_type != "classification" or not trained_calibrator:
            return None

        # 모델이 predict_proba를 지원하는지 확인
        if not hasattr(trained_model, "predict_proba"):
            logger.warning("Model이 predict_proba를 지원하지 않아 calibration 평가를 건너뜁니다")
            return None

        return CalibrationEvaluator(trained_model, trained_calibrator)

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
        from src.factory.pyfunc_factory import PyfuncFactory

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

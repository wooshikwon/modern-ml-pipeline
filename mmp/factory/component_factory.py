"""Tier 1/2/3 컴포넌트 생성 전담 클래스."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from mmp.components.adapter import AdapterRegistry
from mmp.components.evaluator import EvaluatorRegistry
from mmp.components.fetcher import FetcherRegistry
from mmp.components.preprocessor import BasePreprocessor, Preprocessor
from mmp.utils.core.logger import logger

if TYPE_CHECKING:
    from mmp.components.adapter.base import BaseAdapter
    from mmp.components.fetcher.base import BaseFetcher
    from mmp.settings import Settings


class ComponentCreator:
    """
    Tier 1/2/3 컴포넌트 생성 로직을 담당하는 클래스.
    캐싱은 Factory(파사드)가 담당하므로, 이 클래스는 순수 생성 로직만 포함한다.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._recipe = settings.recipe
        self._config = settings.config
        self._data = self._recipe.data
        self._model = self._recipe.model

    # ===============================
    # Tier 1: Atomic Components
    # ===============================

    def create_data_adapter(self, adapter_type: Optional[str] = None) -> BaseAdapter:
        """데이터 어댑터 생성."""
        if adapter_type:
            target_type = adapter_type
        else:
            config_adapter_type = getattr(self._settings.config.data_source, "adapter_type", None)
            if config_adapter_type:
                target_type = config_adapter_type
                logger.debug(f"[FACT] 설정된 adapter 유형 사용: {target_type}")
            else:
                source_uri = self._data.loader.source_uri
                target_type = self._detect_adapter_type_from_uri(source_uri)
                logger.debug(f"[FACT] URI에서 adapter 유형 자동 감지: {target_type}")

        try:
            adapter = AdapterRegistry.create(target_type, self._settings)
            return adapter
        except Exception as e:
            available = AdapterRegistry.list_keys()
            logger.error(f"'{target_type}' adapter 생성에 실패했습니다. Available: {available}")
            raise ValueError(
                f"'{target_type}' 어댑터 생성에 실패했습니다. 사용 가능: {available}"
            ) from e

    def create_fetcher(
        self, run_mode: Optional[str] = None, factory_ref: Any = None
    ) -> BaseFetcher:
        """Fetcher 생성."""
        mode = (run_mode or "batch").lower()

        provider = (
            self._settings.config.feature_store.provider
            if self._settings.config.feature_store
            else "none"
        )
        fetch_conf = self._recipe.data.fetcher if hasattr(self._recipe.data, "fetcher") else None
        fetch_type = fetch_conf.type if fetch_conf else None

        if mode == "serving":
            if fetch_type in (None, "pass_through") or provider in (None, "none"):
                raise TypeError(
                    "Serving 모드에서는 Feature Store 연결이 필요합니다. "
                    "pass_through 또는 feature_store 미구성은 허용되지 않습니다."
                )

        try:
            if provider == "none" or fetch_type == "pass_through" or not fetch_conf:
                fetcher = FetcherRegistry.create("pass_through")
            elif fetch_type == "feature_store" and provider in {"feast", "mock", "dynamic"}:
                fetcher = FetcherRegistry.create(
                    fetch_type, settings=self._settings, factory=factory_ref
                )
            else:
                raise ValueError(
                    f"적절한 fetcher를 선택할 수 없습니다. "
                    f"provider={provider}, fetch_type={fetch_type}, mode={mode}"
                )

            return fetcher

        except Exception as e:
            logger.error(f"Fetcher 생성에 실패했습니다: {e} (mode={mode}, provider={provider})")
            raise

    def create_evaluator(self) -> Any:
        """Evaluator 생성."""
        task_choice = self._recipe.task_choice

        try:
            evaluator = EvaluatorRegistry.create(task_choice, self._settings)
            return evaluator

        except Exception:
            available = EvaluatorRegistry.list_keys()
            logger.error(
                f"'{task_choice}'에 대한 Evaluator 생성에 실패했습니다. Available: {available}"
            )
            raise

    def create_calibrator(self, method: Optional[str] = None) -> Optional[Any]:
        """Calibrator 생성."""
        task_type = self._recipe.task_choice
        calibration_config = getattr(self._recipe.model, "calibration", None)

        if not calibration_config or not getattr(calibration_config, "enabled", False):
            logger.debug("[FACT] Calibration 비활성화 - 스킵")
            return None

        if task_type != "classification":
            logger.debug(f"[FACT] {task_type} task에서는 Calibration 미지원 - 스킵")
            return None

        calibration_method = method or getattr(calibration_config, "method", None)
        if not calibration_method:
            raise ValueError(
                "Calibration method가 설정되지 않았습니다. "
                "Recipe에서 model.calibration.method를 설정하세요. "
                "사용 가능: 'beta', 'isotonic', 'temperature'"
            )

        try:
            from mmp.components.calibration.registry import CalibrationRegistry

            calibrator = CalibrationRegistry.create(calibration_method)
            return calibrator

        except Exception:
            from mmp.components.calibration.registry import CalibrationRegistry

            available = CalibrationRegistry.list_keys()
            logger.error(
                f"'{calibration_method}' Calibrator 생성에 실패했습니다. Available: {available}"
            )
            raise

    def create_feature_store_adapter(self) -> BaseAdapter:
        """Feature Store 어댑터 생성."""
        if not self._settings.config.feature_store:
            raise ValueError("Feature Store 설정이 구성되지 않았습니다.")

        try:
            adapter = AdapterRegistry.create("feature_store", self._settings)
            return adapter

        except Exception as e:
            logger.error(f"Feature Store adapter 생성에 실패했습니다: {e}")
            raise

    # ===============================
    # Tier 2: Composite Components
    # ===============================

    def create_trainer(
        self, trainer_type: Optional[str] = None, factory_provider: Any = None
    ) -> Any:
        """Trainer 생성."""
        from mmp.components.trainer import TrainerRegistry

        trainer_type = trainer_type or "default"

        try:
            trainer = TrainerRegistry.create(
                trainer_type, settings=self._settings, factory_provider=factory_provider
            )
            return trainer

        except Exception:
            available = TrainerRegistry.list_keys()
            logger.error(f"'{trainer_type}' Trainer 생성에 실패했습니다. Available: {available}")
            raise

    def create_datahandler(self) -> Any:
        """DataHandler 생성."""
        from mmp.components.datahandler import DataHandlerRegistry

        task_choice = self._recipe.task_choice

        try:
            model_class_path = getattr(self._recipe.model, "class_path", None)
            datahandler = DataHandlerRegistry.get_handler_for_task(
                task_choice, self._settings, model_class_path=model_class_path
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
    # ===============================

    def create_preprocessor(self) -> Optional[BasePreprocessor]:
        """Preprocessor 생성."""
        preprocessor_config = getattr(self._recipe, "preprocessor", None)

        if not preprocessor_config:
            return None

        try:
            preprocessor = Preprocessor(settings=self._settings)
            return preprocessor

        except Exception as e:
            logger.error(f"Preprocessor 생성에 실패했습니다: {e}")
            raise

    # ===============================
    # Internal Utilities
    # ===============================

    @staticmethod
    def _detect_adapter_type_from_uri(source_uri: str) -> str:
        """source_uri 패턴을 분석하여 어댑터 타입을 자동 결정."""
        uri_lower = source_uri.lower()

        if uri_lower.endswith(".sql") or "select" in uri_lower or "from" in uri_lower:
            return "sql"

        if uri_lower.startswith("bigquery://"):
            return "sql"

        if any(uri_lower.startswith(prefix) for prefix in ["s3://", "gs://", "az://"]):
            return "storage"

        if any(uri_lower.endswith(ext) for ext in [".csv", ".parquet", ".json", ".tsv"]):
            return "storage"

        logger.warning(f"[FACT] URI 패턴 인식 실패: {source_uri} -> storage 어댑터 사용")
        return "storage"

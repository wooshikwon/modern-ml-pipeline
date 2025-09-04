from __future__ import annotations
import importlib
from typing import Optional, Dict, Any, TYPE_CHECKING

import pandas as pd

from src.components._fetcher import FetcherRegistry
from src.components._preprocessor import Preprocessor, BasePreprocessor
from src.components._adapter import AdapterRegistry
from src.components._evaluator import EvaluatorRegistry
from src.interface import BaseAdapter
from src.settings import Settings
from src.utils.system.logger import logger

if TYPE_CHECKING:
    from src.engine._artifact import PyfuncWrapper
    from src.interface import BaseFetcher


class Factory:
    """
    현대화된 Recipe 설정(settings.recipe)에 기반하여 모든 핵심 컴포넌트를 생성하는 중앙 팩토리 클래스.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # 현대화된 Recipe 구조 검증
        if not self.settings.recipe:
            raise ValueError("현대화된 Recipe 구조가 필요합니다. settings.recipe가 없습니다.")
        
        logger.info(f"Factory initialized with Recipe: {self.settings.recipe.name}")

    @property 
    def model_config(self):
        """현재 모델 설정 반환 (현대화된 Recipe.model)"""
        return self.settings.recipe.model

    def _detect_adapter_type_from_uri(self, source_uri: str) -> str:
        """
        source_uri 패턴을 분석하여 필요한 어댑터 타입을 자동으로 결정합니다.
        
        패턴:
        - .sql 파일 또는 SQL 쿼리 → 'sql'
        - .csv, .parquet, .json → 'storage'
        - s3://, gs://, az:// → 'storage'
        - bigquery:// → 'bigquery'
        """
        uri_lower = source_uri.lower()
        
        # SQL 패턴
        if uri_lower.endswith('.sql') or 'select' in uri_lower or 'from' in uri_lower:
            return 'sql'
        
        # BigQuery 패턴
        if uri_lower.startswith('bigquery://'):
            return 'bigquery'
        
        # Cloud Storage 패턴
        if any(uri_lower.startswith(prefix) for prefix in ['s3://', 'gs://', 'az://']):
            return 'storage'
        
        # File 패턴
        if any(uri_lower.endswith(ext) for ext in ['.csv', '.parquet', '.json', '.tsv']):
            return 'storage'
        
        # 기본값
        logger.warning(f"source_uri 패턴을 인식할 수 없습니다: {source_uri}. 'storage' 어댑터를 사용합니다.")
        return 'storage'
    
    def create_data_adapter(self, adapter_type: Optional[str] = None) -> "BaseAdapter":
        """
        데이터 어댑터 생성. 
        1. 인자로 전달된 adapter_type 우선
        2. Recipe에 adapter 필드가 있으면 사용 (backward compatibility)
        3. 없으면 source_uri 패턴으로 자동 감지
        """
        # 우선순위 1: 명시적 인자
        if adapter_type:
            target_type = adapter_type
        else:
            # Recipe에서 loader 정보 가져오기
            loader = self.settings.recipe.data.loader
            
            # 우선순위 2: Recipe에 adapter 필드가 있으면 사용 (backward compatibility)
            if hasattr(loader, 'adapter') and loader.adapter:
                target_type = loader.adapter
                logger.debug(f"Recipe에서 adapter 타입 사용: '{target_type}'")
            else:
                # 우선순위 3: source_uri 패턴으로 자동 감지
                source_uri = loader.source_uri
                target_type = self._detect_adapter_type_from_uri(source_uri)
                logger.info(f"source_uri '{source_uri}'에서 adapter 타입 자동 감지: '{target_type}'")
        
        logger.debug(f"DataAdapter 생성: type='{target_type}'")
        try:
            return AdapterRegistry.create_adapter(target_type, self.settings)
        except Exception as e:
            available = list(AdapterRegistry.list_adapters().keys())
            raise ValueError(
                f"지원하지 않는 DataAdapter 타입입니다: '{target_type}'. 사용 가능한 어댑터: {available}"
            ) from e
    
    def create_fetcher(self, run_mode: Optional[str] = None):
        """
        설정과 실행 모드에 따라 적절한 fetcher 인스턴스를 생성합니다.
        """
        mode = (run_mode or "batch").lower()
        env = self.settings.environment.env_name if hasattr(self.settings, "environment") else "local"
        provider = (self.settings.feature_store.provider if getattr(self.settings, "feature_store", None) else "none")
        aug_conf = getattr(self.settings.recipe.model, "fetcher", None)
        aug_type = getattr(aug_conf, "type", None) if aug_conf else None

        # serving에서는 PassThrough/SqlFallback 금지
        if mode == "serving":
            if aug_type in (None, "pass_through") or provider in (None, "none"):
                raise TypeError("Serving에서는 pass_through/feature_store 미구성 사용이 금지됩니다. Feature Store 연결이 필요합니다.")

        # local 또는 provider=none 또는 명시적 pass_through → PassThrough
        if env == "local" or provider == "none" or aug_type == "pass_through" or not aug_conf:
            return FetcherRegistry.create(aug_type)

        # Feature Store 요청 + provider OK → FeatureStorefetcher
        if aug_type == "feature_store" and provider in {"feast", "mock", "dynamic"}:
            return FetcherRegistry.create(aug_type, settings=self.settings, factory=self)


        raise ValueError(
            f"적절한 fetcher를 선택할 수 없습니다. env={env}, provider={provider}, aug_type={aug_type}, mode={mode}"
        )

    def create_preprocessor(self) -> Optional[BasePreprocessor]:
        """전처리기 생성"""
        if not self.model_config.preprocessor: 
            return None
        return Preprocessor(settings=self.settings)

    def create_model(self):
        """외부 라이브러리 기반 동적 모델 생성"""
        class_path = self.model_config.class_path
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            
            hyperparams = self.model_config.hyperparameters.get_fixed_params() if hasattr(self.model_config.hyperparameters, 'get_fixed_params') else self.model_config.hyperparameters.copy()

            for key, value in hyperparams.items():
                if isinstance(value, str) and "." in value and ("_fn" in key or "_class" in key):
                    try:
                        module_path, class_name = value.rsplit('.', 1)
                        obj_module = importlib.import_module(module_path)
                        hyperparams[key] = getattr(obj_module, class_name)
                        logger.info(f"Hyperparameter '{key}' converted to object: {value}")
                    except (ImportError, AttributeError):
                        logger.warning(f"Could not convert hyperparameter '{key}' to object: {value}. Keeping as string.")

            logger.info(f"Creating model instance from: {class_path}")
            return model_class(**hyperparams)
        except Exception as e:
            logger.error(f"Failed to load model: {class_path}", exc_info=True)
            raise ValueError(f"Could not load model class: {class_path}") from e

    def create_evaluator(self):
        """task_type에 따라 동적 evaluator 생성"""
        task_type = self.model_config.data_interface.task_type
        logger.info(f"Creating evaluator for task: {task_type}")
        return EvaluatorRegistry.create(task_type, self.model_config.data_interface)

    def create_feature_store_adapter(self):
        """환경별 Feature Store 어댑터 생성"""
        if not self.settings.feature_store:
            raise ValueError("Feature Store settings are not configured.")
        logger.info("Creating Feature Store adapter.")
        return AdapterRegistry.create_adapter('feature_store', self.settings)
    
    def create_optuna_integration(self):
        """Optuna Integration 생성"""
        tuning_config = self.model_config.hyperparameter_tuning
        if not tuning_config:
            raise ValueError("Hyperparameter tuning settings are not configured.")
        
        from src.utils.integrations.optuna_integration import OptunaIntegration
        logger.info("Creating Optuna integration.")
        return OptunaIntegration(tuning_config)

    def create_pyfunc_wrapper(
        self, 
        trained_model: Any, 
        trained_preprocessor: Optional[BasePreprocessor],
        trained_fetcher: Optional['BaseFetcher'],
        training_df: Optional[pd.DataFrame] = None,
        training_results: Optional[Dict[str, Any]] = None
    ) -> PyfuncWrapper:
        """🔄 Phase 5: 완전한 스키마 정보가 캡슐화된 Enhanced Artifact 생성"""
        from src.engine._artifact import PyfuncWrapper
        logger.info("Creating PyfuncWrapper artifact...")
        
        signature, data_schema = None, None
        if training_df is not None:
            logger.info("Generating model signature and data schema from training_df...")
            from src.utils.integrations.mlflow_integration import create_enhanced_model_signature_with_schema
            
            entity_schema = self.model_config.loader.entity_schema
            data_interface = self.model_config.data_interface
            
            # 학습 시점에 timestamp 컬럼을 datetime으로 변환하여 스키마 일관성 확보
            try:
                ts_col = entity_schema.timestamp_column
                if ts_col and ts_col in training_df.columns:
                    import pandas as pd
                    if not pd.api.types.is_datetime64_any_dtype(training_df[ts_col]):
                        training_df = training_df.copy()
                        training_df[ts_col] = pd.to_datetime(training_df[ts_col], errors='coerce')
            except Exception:
                pass

            data_interface_config = {
                'entity_columns': entity_schema.entity_columns,
                'timestamp_column': entity_schema.timestamp_column,
                'task_type': data_interface.task_type,
                'target_column': data_interface.target_column,
                'treatment_column': getattr(data_interface, 'treatment_column', None),
            }
            
            signature, data_schema = create_enhanced_model_signature_with_schema(
                training_df, 
                data_interface_config
            )
            logger.info("✅ Signature and data schema created successfully.")
        
        return PyfuncWrapper(
            settings=self.settings,
            trained_model=trained_model,
            trained_preprocessor=trained_preprocessor,
            trained_fetcher=trained_fetcher,
            training_results=training_results,
            signature=signature,
            data_schema=data_schema,
        )

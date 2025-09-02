from __future__ import annotations
import importlib
from typing import Optional, Dict, Any, TYPE_CHECKING

import pandas as pd

from src.components._fetcher import FeatureStoreAugmenter, PassThroughAugmenter
from src.components._preprocessor import Preprocessor, BasePreprocessor
from src.components._adapter import AdapterRegistry
from src.components._evaluator import EvaluatorRegistry
from src.interface import BaseAdapter
from src.settings import Settings
from src.utils.system.logger import logger

if TYPE_CHECKING:
    from src.engine._artifact import PyfuncWrapper
    from src.interface import BaseAugmenter


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

    def create_data_adapter(self, adapter_type: Optional[str] = None) -> "BaseAdapter":
        """
        데이터 어댑터 생성. 인자 우선, 미지정 시 레시피의 loader.adapter 사용.
        """
        target_type = adapter_type or self.settings.recipe.model.loader.adapter
        logger.debug(f"DataAdapter 생성: type='{target_type}'")
        try:
            return AdapterRegistry.create_adapter(target_type, self.settings)
        except Exception as e:
            available = list(AdapterRegistry.list_adapters().keys())
            raise ValueError(
                f"지원하지 않는 DataAdapter 타입입니다: '{target_type}'. 사용 가능한 어댑터: {available}"
            ) from e
    
    def create_augmenter(self, run_mode: Optional[str] = None):
        """
        설정과 실행 모드에 따라 적절한 Augmenter 인스턴스를 생성합니다.
        """
        mode = (run_mode or "batch").lower()
        env = self.settings.environment.env_name if hasattr(self.settings, "environment") else "local"
        provider = (self.settings.feature_store.provider if getattr(self.settings, "feature_store", None) else "none")
        aug_conf = getattr(self.settings.recipe.model, "augmenter", None)
        aug_type = getattr(aug_conf, "type", None) if aug_conf else None

        # serving에서는 PassThrough/SqlFallback 금지
        if mode == "serving":
            if aug_type in (None, "pass_through") or provider in (None, "none"):
                raise TypeError("Serving에서는 pass_through/feature_store 미구성 사용이 금지됩니다. Feature Store 연결이 필요합니다.")

        # local 또는 provider=none 또는 명시적 pass_through → PassThrough
        if env == "local" or provider == "none" or aug_type == "pass_through" or not aug_conf:
            return PassThroughAugmenter()

        # Feature Store 요청 + provider OK → FeatureStoreAugmenter
        if aug_type == "feature_store" and provider in {"feast", "mock", "dynamic"}:
            return FeatureStoreAugmenter(settings=self.settings, factory=self)


        raise ValueError(
            f"적절한 Augmenter를 선택할 수 없습니다. env={env}, provider={provider}, aug_type={aug_type}, mode={mode}"
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
        trained_augmenter: Optional['BaseAugmenter'],
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
            trained_augmenter=trained_augmenter,
            training_results=training_results,
            signature=signature,
            data_schema=data_schema,
        )

from __future__ import annotations
import importlib
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

import mlflow
import pandas as pd

from src.components._augmenter import Augmenter, PassThroughAugmenter, BaseAugmenter
from src.components._preprocessor import Preprocessor, BasePreprocessor
from src.interface import BaseAdapter
from src.settings import Settings
from src.utils.system.logger import logger
from src.engine._registry import AdapterRegistry, EvaluatorRegistry
from src.components._augmenter._augmenter import AugmenterRegistry

if TYPE_CHECKING:
    from src.engine._artifact import PyfuncWrapper


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

    def create_data_adapter(self) -> "BaseAdapter":
        """
        레시피에 명시된 `adapter` 타입을 기반으로 데이터 어댑터를 생성합니다.
        더 이상 `default_loader`나 파일 확장자와 같은 암묵적인 규칙에 의존하지 않고,
        오직 레시피의 명시적인 선언만을 따릅니다.
        """
        # 1. 레시피에서 명시적으로 선언된 어댑터 타입을 가져옴
        adapter_type = self.settings.recipe.model.loader.adapter
        logger.debug(f"레시피에 명시된 어댑터 타입 '{adapter_type}'으로 DataAdapter 생성을 시도합니다.")

        # 2. 레지스트리를 사용하여 해당 타입의 어댑터 인스턴스를 생성
        try:
            return AdapterRegistry.create(adapter_type, settings=self.settings)
        except KeyError:
            # 3. 요청된 타입이 레지스트리에 없으면 명확한 오류 발생
            available = list(AdapterRegistry._adapters.keys())
            raise ValueError(
                f"지원하지 않는 DataAdapter 타입입니다: '{adapter_type}'. "
                f"사용 가능한 어댑터: {available}"
            )
    
    def create_augmenter(self) -> BaseAugmenter:
        """
        설정(Settings)을 기반으로 적절한 Augmenter 인스턴스를 생성합니다.
        
        환경 설정(`settings.feature_store.provider`)이 레시피 설정보다 우선합니다.
        만약 provider가 'passthrough'로 설정되어 있다면, 레시피의 augmenter 설정과
        상관없이 PassThroughAugmenter를 반환합니다.
        """
        # 1. 환경의 인프라 제약(config)을 최우선으로 확인
        feature_store_provider = self.settings.feature_store.provider
        
        if feature_store_provider == "passthrough":
            logger.debug("환경 설정에 따라 'PassThroughAugmenter'를 생성합니다.")
            return PassThroughAugmenter(settings=self.settings)

        # 2. 'passthrough'가 아닐 경우, 레시피의 논리적 요구사항을 확인
        if not self.settings.recipe.model.augmenter:
            logger.debug("레시피에 Augmenter가 정의되지 않았으므로 'PassThroughAugmenter'를 생성합니다.")
            return PassThroughAugmenter(settings=self.settings)
            
        augmenter_type = self.settings.recipe.model.augmenter.type
        
        # 3. 레지스트리에서 요청된 타입의 Augmenter 클래스를 찾아 생성
        try:
            augmenter_class = AugmenterRegistry.get_augmenter(augmenter_type)
            logger.debug(f"레시피 설정에 따라 '{augmenter_class.__name__}'를 생성합니다.")
            return augmenter_class(settings=self.settings)
        except KeyError:
            # 4. 요청된 타입이 레지스트리에 없으면 오류 발생
            raise ValueError(f"지원하지 않는 Augmenter 타입입니다: {augmenter_type}")

    def create_preprocessor(self) -> BasePreprocessor:
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
        # 하드코딩된 맵 대신 Registry를 사용
        return EvaluatorRegistry.create(evaluator_type, self.settings)

    def create_feature_store_adapter(self):
        """환경별 Feature Store 어댑터 생성"""
        if not self.settings.feature_store:
            raise ValueError("Feature Store settings are not configured.")
        logger.info("Creating Feature Store adapter.")
        return AdapterRegistry.create('feature_store', self.settings)
    
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
        trained_augmenter: Optional[BaseAugmenter],
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

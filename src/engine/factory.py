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

    def create_data_adapter(self, adapter_type: str) -> BaseAdapter:
        """데이터 어댑터 생성"""
        logger.debug(f"Requesting to create data adapter of type: {adapter_type}")
        return AdapterRegistry.create(adapter_type, self.settings)

    def create_augmenter(self) -> "BaseAugmenter":
        """Augmenter 생성 (환경별 차등 기능)"""
        # Blueprint 원칙 9: 환경별 차등적 기능 분리
        if self.settings.environment.app_env == "local":
            logger.info("LOCAL env: Creating PassThroughAugmenter.")
            return PassThroughAugmenter(settings=self.settings)
        else:
            logger.info("DEV/PROD env: Creating standard Augmenter.")
            return Augmenter(settings=self.settings, factory=self)

    def create_preprocessor(self) -> "BasePreprocessor":
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
            
            # 하이퍼파라미터 처리 (현대화된 구조)
            hyperparams = self.model_config.hyperparameters.get_fixed_params() if hasattr(self.model_config.hyperparameters, 'get_fixed_params') else self.model_config.hyperparameters
            
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
        return EvaluatorRegistry.create(task_type, self.model_config.data_interface)

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
            # Schema generation logic can be added here
            pass
        
        return PyfuncWrapper(
            settings=self.settings,
            trained_model=trained_model,
            trained_preprocessor=trained_preprocessor,
            trained_augmenter=trained_augmenter,
            training_results=training_results,
            signature=signature,
            data_schema=data_schema,
        )

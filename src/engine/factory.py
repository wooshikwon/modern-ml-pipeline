from __future__ import annotations
import importlib
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

import mlflow
import pandas as pd

from src.components.augmenter import Augmenter, LocalFileAugmenter, BaseAugmenter, PassThroughAugmenter
from src.components.preprocessor import BasePreprocessor, Preprocessor
from src.interface.base_adapter import BaseAdapter
from src.settings import Settings
from src.utils.system.logger import logger
from src.engine.registry import AdapterRegistry

if TYPE_CHECKING:
    from src.engine.artifact import PyfuncWrapper


class Factory:
    """
    설정(settings)에 기반하여 모든 핵심 컴포넌트의 인스턴스를 생성하는 중앙 팩토리 클래스.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        logger.info("Factory initialized.")

    def create_data_adapter(self, adapter_type: str) -> BaseAdapter:
        """
        단순화된 데이터 어댑터 생성 메서드.
        모든 생성 책임은 AdapterRegistry에 위임합니다.
        """
        logger.debug(f"Requesting to create data adapter of type: {adapter_type}")
        return AdapterRegistry.create(adapter_type, self.settings)

    def create_augmenter(self) -> "BaseAugmenter":
        """Augmenter 생성"""
        # Blueprint 원칙 9: 환경별 차등적 기능 분리
        if self.settings.environment.app_env == "local":
            logger.info("LOCAL env: Creating PassThroughAugmenter.")
            return PassThroughAugmenter(settings=self.settings)
        else:
            logger.info("DEV/PROD env: Creating standard Augmenter.")
            return Augmenter(settings=self.settings, factory=self)

    def create_preprocessor(self) -> "BasePreprocessor":
        preprocessor_config = self.settings.model.preprocessor
        if not preprocessor_config: 
            return None
        return Preprocessor(settings=self.settings)

    def create_model(self):
        """외부 라이브러리 기반 동적 모델 생성"""
        class_path = self.settings.model.class_path
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            hyperparams = self.settings.model.hyperparameters.root
            logger.info(f"Creating model instance from: {class_path}")
            return model_class(**hyperparams)
        except Exception as e:
            logger.error(f"Failed to load model: {class_path}", exc_info=True)
            raise ValueError(f"Could not load model class: {class_path}") from e

    def create_evaluator(self):
        """task_type에 따른 동적 evaluator 생성"""
        from src.components.evaluator import (
            ClassificationEvaluator,
            RegressionEvaluator,
            ClusteringEvaluator,
            CausalEvaluator,
        )
        
        task_type = self.settings.model.data_interface.task_type
        evaluator_map = {
            "classification": ClassificationEvaluator,
            "regression": RegressionEvaluator,
            "clustering": ClusteringEvaluator,
            "causal": CausalEvaluator,
        }
        
        evaluator_class = evaluator_map.get(task_type)
        if not evaluator_class:
            raise ValueError(f"Unsupported task_type for evaluator: '{task_type}'")
        
        logger.info(f"Creating evaluator for task: {task_type}")
        return evaluator_class(self.settings.model.data_interface)

    def create_feature_store_adapter(self):
        """환경별 Feature Store 어댑터 생성"""
        if not self.settings.feature_store:
            raise ValueError("Feature Store settings are not configured.")
        logger.info("Creating Feature Store adapter.")
        return AdapterRegistry.create('feature_store', self.settings)
    
    def create_optuna_integration(self):
        """Optuna Integration 생성"""
        if not self.settings.hyperparameter_tuning:
            raise ValueError("Hyperparameter tuning settings are not configured.")
        
        from src.utils.integrations.optuna_integration import OptunaIntegration
        logger.info("Creating Optuna integration directly.")
        return OptunaIntegration(self.settings.hyperparameter_tuning)

    def create_tuning_utils(self):
        """하이퍼파라미터 튜닝 유틸리티 생성"""
        logger.info("Creating Tuning utils.")
        from src.utils.system.tuning_utils import TuningUtils
        return TuningUtils()

    def create_pyfunc_wrapper(
        self, 
        trained_model, 
        trained_preprocessor: Optional[BasePreprocessor],
        training_results: Optional[Dict[str, Any]] = None
    ) -> PyfuncWrapper:
        """
        완전한 Wrapped Artifact 생성.
        모든 생성 로직을 헬퍼 메서드로 분리.
        """
        from src.engine.artifact import PyfuncWrapper
        logger.info("Creating PyfuncWrapper artifact.")
        
        return PyfuncWrapper(
            trained_model=trained_model,
            trained_preprocessor=trained_preprocessor,
            trained_augmenter=self.create_augmenter(),
            loader_sql_snapshot=self._create_loader_sql_snapshot(),
            augmenter_config_snapshot=self._create_augmenter_config_snapshot(),
            recipe_yaml_snapshot=self._create_recipe_yaml_snapshot(),
            model_class_path=self.settings.model.class_path,
            hyperparameter_optimization=training_results.get('hyperparameter_optimization'),
            training_methodology=training_results.get('training_methodology'),
        )
    
    def _create_loader_sql_snapshot(self) -> str:
        """
        Loader SQL 스냅샷 생성.
        loaders.py에서 이미 렌더링된 SQL 문자열이 담겨 있으므로, 그대로 반환합니다.
        """
        return self.settings.model.loader.source_uri
    
    def _create_augmenter_config_snapshot(self) -> Dict[str, Any]:
        """Augmenter 설정 스냅샷 생성"""
        if self.settings.model.augmenter:
            return self.settings.model.augmenter.dict()
        return {}
    
    def _create_recipe_yaml_snapshot(self) -> str:
        """Recipe YAML 스냅샷 생성"""
        recipe_file = self.settings.model.computed.get("recipe_file")
        if recipe_file:
            recipe_path = Path(f"recipes/{recipe_file}.yaml")
            if recipe_path.exists():
                return recipe_path.read_text(encoding="utf-8")
        return ""

from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Tuple, Optional, TYPE_CHECKING, Callable
from sklearn.model_selection import train_test_split

from src.settings import Settings
from src.utils.system.logger import logger
from src.interface import BaseTrainer, BaseModel, BaseAugmenter, BasePreprocessor, BaseEvaluator
from ._data_handler import split_data, prepare_training_data
from ._optimizer import OptunaOptimizer

if TYPE_CHECKING:
    pass

class Trainer(BaseTrainer):
    """
    모델 학습 및 평가 전체 과정을 관장하는 오케스트레이터 클래스.
    """
    def __init__(self, settings: Settings, factory_provider: Optional[Callable[[], Any]] = None):
        self.settings = settings
        self.factory_provider = factory_provider
        logger.info("Trainer가 초기화되었습니다.")
        self.training_results = {}

    def _get_factory(self):
        if self.factory_provider is None:
            raise RuntimeError("Factory provider가 주입되지 않았습니다. 엔진 의존성은 외부에서 주입되어야 합니다.")
        return self.factory_provider()

    def train(
        self,
        df: pd.DataFrame,
        model: Any,
        augmenter: BaseAugmenter,
        preprocessor: BasePreprocessor,
        evaluator: BaseEvaluator,
        context_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, BasePreprocessor, Dict[str, float], Dict[str, Any]]:
        
        # 데이터 분할 및 전처리
        train_df, test_df = split_data(df, self.settings)
        X_train, y_train, _ = prepare_training_data(train_df, self.settings)
        X_test, y_test, _ = prepare_training_data(test_df, self.settings)

        # 전처리 적용
        if preprocessor:
            preprocessor.fit(X_train)
            X_train = preprocessor.transform(X_train)
            X_test = preprocessor.transform(X_test)

        # 하이퍼파라미터 최적화 또는 직접 학습
        global_tuning_enabled = self.settings.hyperparameter_tuning.enabled
        recipe_tuning_config = self.settings.recipe.model.hyperparameter_tuning
        is_tuning_enabled_in_recipe = recipe_tuning_config and recipe_tuning_config.enabled

        use_tuning = global_tuning_enabled and is_tuning_enabled_in_recipe

        if use_tuning:
            logger.info("하이퍼파라미터 최적화를 시작합니다. (전역 및 레시피 설정 모두에서 활성화됨)")
            optimizer = OptunaOptimizer(settings=self.settings, factory_provider=self._get_factory)
            best = optimizer.optimize(train_df, self._single_training_iteration)
            self.training_results['hyperparameter_optimization'] = best
            trained_model = best['model']
        else:
            if not global_tuning_enabled:
                logger.info("하이퍼파라미터 튜닝을 건너뜁니다. 이유: 전역 설정(config)에서 비활성화되었습니다.")
            elif not is_tuning_enabled_in_recipe:
                logger.info("하이퍼파라미터 튜닝을 건너뜁니다. 이유: 레시피(recipe)에서 비활성화되었거나 설정이 없습니다.")
            logger.info("고정된 하이퍼파라미터로 모델을 학습합니다.")
            model.fit(X_train, y_train)
            trained_model = model
            self.training_results['hyperparameter_optimization'] = {'enabled': False}

        # 4. 모델 평가
        metrics = evaluator.evaluate(trained_model, X_test, y_test)
        self.training_results['evaluation_metrics'] = metrics

        # 5. 학습 방법론 메타데이터 저장
        self.training_results['training_methodology'] = self._get_training_methodology()
        
        logger.info(f"모델 평가 완료. 주요 지표: {metrics}")
        
        return trained_model, preprocessor, metrics, self.training_results

    def _single_training_iteration(self, train_df, params, seed):
        """Data Leakage 방지를 보장하는 단일 학습/검증 사이클."""
        train_data, val_data = train_test_split(
            train_df, test_size=0.2, random_state=seed, stratify=train_df.get(self._get_stratify_col())
        )
        
        X_train, y_train, additional_data = prepare_training_data(train_data, self.settings)
        X_val, y_val, _ = prepare_training_data(val_data, self.settings)
        
        factory = self._get_factory()
        preprocessor = factory.create_preprocessor()
        
        if preprocessor:
            preprocessor.fit(X_train)
            X_train_processed = preprocessor.transform(X_train)
            X_val_processed = preprocessor.transform(X_val)
        else:
            X_train_processed, X_val_processed = X_train, X_val
        
        model_instance = factory.create_model()
        model_instance.set_params(**params)
        self._fit_model(model_instance, X_train_processed, y_train, additional_data)
        
        evaluator = factory.create_evaluator()
        metrics = evaluator.evaluate(model_instance, X_val_processed, y_val, val_data)
        
        score = metrics.get(self.settings.recipe.model.hyperparameter_tuning.metric, 0.0)
        
        return {'model': model_instance, 'preprocessor': preprocessor, 'score': score}

    def _fit_model(self, model, X, y, additional_data):
        """task_type에 따라 모델을 학습시킵니다."""
        if not isinstance(model, BaseModel):
            from sklearn.base import is_classifier, is_regressor
            if not (is_classifier(model) or is_regressor(model) or hasattr(model, 'fit')):
                 raise TypeError("전달된 모델 객체는 BaseModel 인터페이스를 따르거나 scikit-learn 호환 모델이어야 합니다.")
        
        task_type = self.settings.recipe.model.data_interface.task_type
        if task_type in ["classification", "regression"]:
            model.fit(X, y)
        elif task_type == "clustering":
            model.fit(X)
        elif task_type == "causal":
            model.fit(X, additional_data['treatment'], y)
        else:
            raise ValueError(f"지원하지 않는 task_type: {task_type}")

    def _get_training_methodology(self):
        return {
            'train_test_split_method': 'stratified',
            'train_ratio': 0.8,
            'validation_strategy': 'train_validation_split',
            'random_state': 42,
            'preprocessing_fit_scope': 'train_only'
        }

    def _get_stratify_col(self):
        di = self.settings.recipe.model.data_interface
        return di.target_column if di.task_type == "classification" else di.treatment_column if di.task_type == "causal" else None
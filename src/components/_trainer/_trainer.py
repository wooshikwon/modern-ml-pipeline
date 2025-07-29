from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Tuple, Optional, TYPE_CHECKING
from sklearn.model_selection import train_test_split

from src.settings import Settings
from src.utils.system.logger import logger
from src.interface import BaseTrainer, BaseModel
from src.utils.system.schema_utils import validate_schema
from ._data_handler import split_data, prepare_training_data
from ._optimizer import OptunaOptimizer

if TYPE_CHECKING:
    from src.components._augmenter import BaseAugmenter
    from src.components._preprocessor import BasePreprocessor

class Trainer(BaseTrainer):
    """
    모델 학습 및 평가 전체 과정을 관장하는 오케스트레이터 클래스.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        logger.info("Trainer가 초기화되었습니다.")

    def train(self, df, model, augmenter, preprocessor, context_params=None):
        """
        학습 진입점. HPO 활성화 여부에 따라 경로를 분기합니다.
        """
        tuning_config = self.settings.recipe.model.hyperparameter_tuning
        if tuning_config and tuning_config.enabled:
            return self._train_with_hpo(df, augmenter, preprocessor, context_params)
        else:
            return self._train_with_fixed_params(df, model, augmenter, preprocessor, context_params)

    def _train_with_hpo(self, df, augmenter, preprocessor, context_params):
        """하이퍼파라미터 최적화(HPO)를 사용하여 모델을 학습시킵니다."""
        train_df, test_df = split_data(df, self.settings)
        
        if augmenter:
            train_df = augmenter.augment(train_df, run_mode="batch", context_params=context_params)
            test_df = augmenter.augment(test_df, run_mode="batch", context_params=context_params)

        optimizer = OptunaOptimizer(self.settings)
        hpo_results = optimizer.optimize(train_df, self._single_training_iteration)
        
        # 최적 파라미터로 최종 모델 학습
        final_result = self._single_training_iteration(train_df, hpo_results['best_params'], seed=42)
        trained_model = final_result['model']
        trained_preprocessor = final_result['preprocessor']

        # 최종 평가
        from src.engine import Factory
        factory = Factory(self.settings)
        evaluator = factory.create_evaluator()
        X_test, y_test, _ = prepare_training_data(test_df, self.settings)
        X_test_processed = trained_preprocessor.transform(X_test) if trained_preprocessor else X_test
        final_metrics = evaluator.evaluate(trained_model, X_test_processed, y_test, test_df)
        
        training_results = {
            'hyperparameter_optimization': hpo_results,
            'training_methodology': self._get_training_methodology()
        }
        return trained_model, trained_preprocessor, final_metrics, training_results

    def _train_with_fixed_params(self, df, model, augmenter, preprocessor, context_params):
        """고정된 하이퍼파라미터를 사용하여 모델을 학습시킵니다."""
        train_df, test_df = split_data(df, self.settings)
        
        if augmenter:
            train_df = augmenter.augment(train_df, run_mode="batch", context_params=context_params)
            test_df = augmenter.augment(test_df, run_mode="batch", context_params=context_params)
        
        if getattr(model, 'handles_own_preprocessing', False):
            logger.info(f"모델 '{type(model).__name__}'이(가) 내장된 전처리 로직을 사용합니다.")
            target_col = self.settings.recipe.model.data_interface.target_column
            model.fit(train_df.drop(columns=[target_col]), train_df[target_col])
            trained_preprocessor = None # 메인 preprocessor는 사용되지 않음
        else:
            logger.info("파이프라인의 선언적 Preprocessor를 사용합니다.")
            X_train, y_train, additional_data = prepare_training_data(train_df, self.settings)
            
            if preprocessor:
                preprocessor.fit(X_train)
                X_train_processed = preprocessor.transform(X_train)
            else:
                X_train_processed = X_train
            
            self._fit_model(model, X_train_processed, y_train, additional_data)
            trained_preprocessor = preprocessor

        # 평가 로직 (공통)
        from src.engine import Factory
        factory = Factory(self.settings)
        evaluator = factory.create_evaluator()
        
        X_test, y_test, _ = prepare_training_data(test_df, self.settings)
        # 모델 타입에 따라 평가에 사용할 X_test 결정
        if trained_preprocessor:
            X_test_processed = trained_preprocessor.transform(X_test)
        else: # 자체 전처리 모델의 경우, predict 메서드가 전처리를 내장하고 있음
            X_test_processed = test_df # 원본에 가까운 데이터 전달
            
        metrics = evaluator.evaluate(model, X_test_processed, y_test, test_df)
        
        training_results = {
            'hyperparameter_optimization': {'enabled': False},
            'training_methodology': self._get_training_methodology()
        }
        return model, trained_preprocessor, metrics, training_results

    def _single_training_iteration(self, train_df, params, seed):
        """Data Leakage 방지를 보장하는 단일 학습/검증 사이클."""
        train_data, val_data = train_test_split(
            train_df, test_size=0.2, random_state=seed, stratify=train_df.get(self._get_stratify_col())
        )
        
        X_train, y_train, additional_data = prepare_training_data(train_data, self.settings)
        X_val, y_val, _ = prepare_training_data(val_data, self.settings)
        
        from src.engine import Factory
        factory = Factory(self.settings)
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
            # scikit-learn과 같은 외부 라이브러리 모델들을 위한 임시 래퍼
            from sklearn.base import is_classifier, is_regressor
            if not (is_classifier(model) or is_regressor(model) or hasattr(model, 'fit')):
                 raise TypeError(f"전달된 모델 객체는 BaseModel 인터페이스를 따르거나 scikit-learn 호환 모델이어야 합니다.")
        
        task_type = self.settings.recipe.model.data_interface.task_type
        if task_type in ["classification", "regression"]: model.fit(X, y)
        elif task_type == "clustering": model.fit(X)
        elif task_type == "causal": model.fit(X, additional_data['treatment'], y)
        else: raise ValueError(f"지원하지 않는 task_type: {task_type}")

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
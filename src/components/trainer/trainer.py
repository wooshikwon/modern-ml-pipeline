from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Tuple, Optional, TYPE_CHECKING, Callable
from sklearn.model_selection import train_test_split

from src.settings import Settings
from src.utils.system.logger import logger
from src.interface import BaseTrainer, BaseModel, BaseFetcher, BasePreprocessor, BaseEvaluator, BaseDataHandler
from .modules.optimizer import OptunaOptimizer

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
        fetcher: BaseFetcher,
        datahandler: BaseDataHandler,
        preprocessor: BasePreprocessor,
        evaluator: BaseEvaluator,
        context_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, BasePreprocessor, Dict[str, float], Dict[str, Any]]:
        
        # 데이터 분할 및 전처리
        train_df, test_df = datahandler.split_data(df)
        X_train, y_train, additional_train_data = datahandler.prepare_data(train_df)
        X_test, y_test, additional_test_data = datahandler.prepare_data(test_df)

        # 전처리 적용
        if preprocessor:
            preprocessor.fit(X_train)
            X_train = preprocessor.transform(X_train)
            X_test = preprocessor.transform(X_test)

        # 하이퍼파라미터 최적화 또는 직접 학습 (Recipe 설정만 사용)
        recipe_hyperparams = self.settings.recipe.model.hyperparameters
        use_tuning = recipe_hyperparams and getattr(recipe_hyperparams, 'tuning_enabled', False)

        if use_tuning:
            logger.info("하이퍼파라미터 최적화를 시작합니다. (Recipe에서 활성화됨)")
            optimizer = OptunaOptimizer(settings=self.settings, factory_provider=self._get_factory)
            best = optimizer.optimize(train_df, lambda train_df, params, seed: self._single_training_iteration(train_df, params, seed, datahandler))
            self.training_results['hyperparameter_optimization'] = best
            trained_model = best['model']
        else:
            logger.info("하이퍼파라미터 튜닝을 건너뜁니다. 이유: Recipe에서 비활성화되었거나 설정이 없습니다.")
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

    def _single_training_iteration(self, train_df, params, seed, datahandler):
        """
        Data Leakage 방지를 보장하는 단일 학습/검증 사이클.
        
        Optuna 튜닝 시에만 사용되며, 이미 분할된 Train 데이터를
        다시 Train(80%) / Validation(20%)로 분할하여 튜닝합니다.
        """
        train_data, val_data = train_test_split(
            train_df, test_size=0.2, random_state=seed, stratify=train_df.get(self._get_stratify_col())
        )
        
        X_train, y_train, additional_data = datahandler.prepare_data(train_data)
        X_val, y_val, _ = datahandler.prepare_data(val_data)
        
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
        
        optimization_metric = self.settings.recipe.model.hyperparameters.optimization_metric or "accuracy"
        score = metrics.get(optimization_metric, 0.0)
        
        return {'model': model_instance, 'preprocessor': preprocessor, 'score': score}

    def _fit_model(self, model, X, y, additional_data):
        """task_type에 따라 모델을 학습시킵니다."""
        if not isinstance(model, BaseModel):
            from sklearn.base import is_classifier, is_regressor
            if not (is_classifier(model) or is_regressor(model) or hasattr(model, 'fit')):
                 raise TypeError("전달된 모델 객체는 BaseModel 인터페이스를 따르거나 scikit-learn 호환 모델이어야 합니다.")
        
        task_type = self.settings.recipe.data.data_interface.task_type
        if task_type in ["classification", "regression"]:
            model.fit(X, y)
        elif task_type == "clustering":
            model.fit(X)
        elif task_type == "causal":
            model.fit(X, additional_data['treatment'], y)
        elif task_type == "timeseries":
            model.fit(X, y)
        else:
            raise ValueError(f"지원하지 않는 task_type: {task_type}")

    def _get_training_methodology(self):
        """학습 방법론 메타데이터를 반환합니다."""
        validation_config = self.settings.recipe.validation
        hyperparams_config = self.settings.recipe.model.hyperparameters
        task_type = self.settings.recipe.data.data_interface.task_type
        
        # stratification 여부 결정
        stratify_col = self._get_stratify_col()
        split_method = 'stratified' if stratify_col else 'simple'
        
        # validation strategy 결정
        if hyperparams_config.enabled:
            validation_strategy = 'train_validation_split'  # Optuna 시 train에서 validation 분할
            note = f'Optuna 사용 시 Train({1-validation_config.test_size:.0%})을 다시 Train(80%)/Val(20%)로 분할'
        else:
            validation_strategy = validation_config.method
            note = f'Hyperparameter tuning 비활성화, {validation_config.method} 사용'
        
        return {
            'train_test_split_method': split_method,
            'train_ratio': 1 - validation_config.test_size,
            'test_ratio': validation_config.test_size,
            'validation_strategy': validation_strategy,
            'random_state': validation_config.random_state,
            'stratify_column': stratify_col,
            'task_type': task_type,
            'preprocessing_fit_scope': 'train_only',
            'hyperparameter_optimization': hyperparams_config.enabled,
            'n_trials': hyperparams_config.n_trials if hyperparams_config.enabled else None,
            'optimization_metric': hyperparams_config.optimization_metric if hyperparams_config.enabled else None,
            'note': note
        }

    def _get_stratify_col(self):
        di = self.settings.recipe.data.data_interface
        return di.target_column if di.task_type == "classification" else di.treatment_column if di.task_type == "causal" else None

# Self-registration
from .registry import TrainerRegistry
TrainerRegistry.register("default", Trainer)
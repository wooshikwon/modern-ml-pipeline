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

        # 전처리 산출물 저장 (선택)
        try:
            output_cfg = getattr(self.settings.config, 'output', None)
            if output_cfg and getattr(output_cfg.preprocessed, 'enabled', True):
                factory = self._get_factory()
                target = output_cfg.preprocessed
                # run_id 확보 (MLflow 활성 런 기준)
                run = mlflow.active_run() if 'mlflow' in globals() else None
                run_id = run.info.run_id if run else "no_run"
                if target.adapter_type == "storage":
                    storage_adapter = factory.create_data_adapter("storage")
                    base_path = target.config.get('base_path', './artifacts/preprocessed')
                    storage_adapter.write(X_train, f"{base_path}/preprocessed_train_{run_id}.parquet")
                    storage_adapter.write(X_test, f"{base_path}/preprocessed_test_{run_id}.parquet")
                elif target.adapter_type == "sql":
                    sql_adapter = factory.create_data_adapter("sql")
                    table = target.config.get('table')
                    if not table:
                        raise ValueError("output.preprocessed.config.table이 필요합니다.")
                    sql_adapter.write(X_train, f"{table}_train", if_exists='append', index=False)
                    sql_adapter.write(X_test, f"{table}_test", if_exists='append', index=False)
                elif target.adapter_type == "bigquery":
                    bq_adapter = factory.create_data_adapter("bigquery")
                    project_id = target.config.get('project_id')
                    dataset = target.config.get('dataset_id')
                    table = target.config.get('table')
                    location = target.config.get('location')
                    if not (project_id and dataset and table):
                        raise ValueError("BigQuery 출력에는 project_id, dataset_id, table이 필요합니다.")
                    bq_adapter.write(X_train, f"{dataset}.{table}_train", options={"project_id": project_id, "location": location, "if_exists": "append"})
                    bq_adapter.write(X_test, f"{dataset}.{table}_test", options={"project_id": project_id, "location": location, "if_exists": "append"})
                else:
                    logger.warning(f"알 수 없는 output 어댑터 타입: {target.adapter_type}. 전처리 저장을 스킵합니다.")
        except Exception as e:
            logger.error(f"전처리 산출물 저장 중 오류: {e}", exc_info=True)

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
        """task_choice에 따라 모델을 학습시킵니다."""
        if not isinstance(model, BaseModel):
            from sklearn.base import is_classifier, is_regressor
            if not (is_classifier(model) or is_regressor(model) or hasattr(model, 'fit')):
                 raise TypeError("전달된 모델 객체는 BaseModel 인터페이스를 따르거나 scikit-learn 호환 모델이어야 합니다.")
        
        task_choice = self.settings.recipe.task_choice
        if task_choice in ["classification", "regression"]:
            model.fit(X, y)
        elif task_choice == "clustering":
            model.fit(X)
        elif task_choice == "causal":
            model.fit(X, additional_data['treatment'], y)
        elif task_choice == "timeseries":
            model.fit(X, y)
        else:
            raise ValueError(f"지원하지 않는 task_choice: {task_choice}")

    def _get_training_methodology(self):
        """학습 방법론 메타데이터를 반환합니다."""
        validation_config = self.settings.recipe.evaluation.validation
        hyperparams_config = self.settings.recipe.model.hyperparameters
        task_choice = self.settings.recipe.task_choice
        
        # stratification 여부 결정
        stratify_col = self._get_stratify_col()
        split_method = 'stratified' if stratify_col else 'simple'
        
        # validation strategy 결정
        if hyperparams_config.tuning_enabled:
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
            'task_choice': task_choice,
            'preprocessing_fit_scope': 'train_only',
            'hyperparameter_optimization': hyperparams_config.tuning_enabled,
            'n_trials': hyperparams_config.n_trials if hyperparams_config.tuning_enabled else None,
            'optimization_metric': hyperparams_config.optimization_metric if hyperparams_config.tuning_enabled else None,
            'note': note
        }

    def _get_stratify_col(self):
        di = self.settings.recipe.data.data_interface
        task_choice = self.settings.recipe.task_choice
        return di.target_column if task_choice == "classification" else di.treatment_column if task_choice == "causal" else None

# Self-registration
from .registry import TrainerRegistry
TrainerRegistry.register("default", Trainer)
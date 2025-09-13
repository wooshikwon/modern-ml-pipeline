from __future__ import annotations
from typing import Dict, Any, Optional, TYPE_CHECKING, Callable

from src.settings import Settings
from src.utils.core.console_manager import get_console
from src.interface import BaseTrainer, BaseModel
from .registry import TrainerRegistry

if TYPE_CHECKING:
    pass

class Trainer(BaseTrainer):
    """
    모델 학습 및 평가 전체 과정을 관장하는 오케스트레이터 클래스.
    """
    def __init__(self, settings: Settings, factory_provider: Optional[Callable[[], Any]] = None):
        self.settings = settings
        self.factory_provider = factory_provider
        self.console = get_console(settings)
        self.console.info("[Trainer] 초기화 시작합니다")
        self.console.component_init("Trainer", "success")
        self.console.info("[Trainer] 초기화 완료되었습니다",
                         rich_message="✅ [Trainer] initialized")
        self.training_results = {}

    def _get_factory(self):
        if self.factory_provider is None:
            raise RuntimeError("Factory provider가 주입되지 않았습니다. 엔진 의존성은 외부에서 주입되어야 합니다.")
        return self.factory_provider()

    def train(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        model: Any,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> tuple[Any, dict]:
        """준비된 데이터로 순수 학습만 수행합니다. (HPO 포함)"""
        self.console.info("[Trainer] 모델 학습을 시작합니다",
                         rich_message="🎯 [Trainer] Starting model training...")

        additional_data = additional_data or {}

        recipe_hyperparams = self.settings.recipe.model.hyperparameters
        use_tuning = recipe_hyperparams and getattr(recipe_hyperparams, 'tuning_enabled', False)

        if use_tuning:
            self.console.info("[Trainer] 하이퍼파라미터 최적화를 시작합니다 (Recipe에서 활성화됨)",
                            rich_message="🎯 [Trainer] Hyperparameter optimization started")
            optimizer = TrainerRegistry.create_optimizer(
                "optuna",
                settings=self.settings,
                factory_provider=self._get_factory
            )

            def _objective_callback(_ignored_train_df, params, seed):
                # 새 모델 인스턴스 생성 후 파라미터 적용
                factory = self._get_factory()
                model_instance = factory.create_model()
                try:
                    model_instance.set_params(**(params or {}))
                except Exception:
                    pass
                # 학습
                self._fit_model(model_instance, X_train, y_train, additional_data.get('train'))
                # 검증 점수 계산
                evaluator = factory.create_evaluator()
                metrics = evaluator.evaluate(model_instance, X_val, y_val, additional_data.get('val'))
                optimization_metric = self.settings.recipe.model.hyperparameters.optimization_metric or "accuracy"
                return {
                    optimization_metric: metrics.get(optimization_metric, 0.0),
                    'score': metrics.get(optimization_metric, 0.0)
                }

            best = optimizer.optimize(train_df=None, training_callback=_objective_callback)  # train_df 미사용
            # 최적 파라미터로 최종 모델에 적용 후 재학습
            try:
                model.set_params(**(best.get('best_params') or {}))
            except Exception:
                pass
            self._fit_model(model, X_train, y_train, additional_data.get('train'))
            trained_model = model
            self.training_results['hyperparameter_optimization'] = best
        else:
            self.console.info("하이퍼파라미터 튜닝을 건너뜁니다. 이유: Recipe에서 비활성화되었거나 설정이 없습니다.", rich_message="⚙️ Using fixed hyperparameters (optimization disabled)")
            self.console.info("고정된 하이퍼파라미터로 모델을 학습합니다.", rich_message="🎯 Training with fixed hyperparameters")
            self._fit_model(model, X_train, y_train, additional_data.get('train'))
            trained_model = model
            self.training_results['hyperparameter_optimization'] = {'enabled': False}

        return trained_model, dict(self.training_results)

    # 기존 단일 학습/검증 분할 로직은 파이프라인으로 이동하여 제거됨

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
TrainerRegistry.register("default", Trainer)
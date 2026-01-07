from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from src.components.optimizer import OptimizerRegistry
from src.components.trainer.base import BaseTrainer
from src.components.trainer.registry import TrainerRegistry
from src.models.base import BaseModel
from src.settings import Settings
from src.utils.core.logger import (
    log_component,
    log_train,
    log_train_debug,
    logger,
)

if TYPE_CHECKING:
    pass


class Trainer(BaseTrainer):
    """
    모델 학습 및 평가 전체 과정을 관장하는 오케스트레이터 클래스.
    """

    def __init__(self, settings: Settings, factory_provider: Optional[Callable[[], Any]] = None):
        self.settings = settings
        self.factory_provider = factory_provider
        log_component("Trainer", "초기화 완료")
        self.training_results = {}

    def _get_factory(self):
        if self.factory_provider is None:
            raise RuntimeError(
                "Factory provider가 주입되지 않았습니다. 엔진 의존성은 외부에서 주입되어야 합니다."
            )
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
        log_train("모델 학습 시작")

        additional_data = additional_data or {}

        recipe_hyperparams = self.settings.recipe.model.hyperparameters
        use_tuning = recipe_hyperparams and getattr(recipe_hyperparams, "tuning_enabled", False)

        if use_tuning:
            log_train("하이퍼파라미터 최적화 시작")

            # 파이프라인 연결 정보 로깅
            log_train_debug(
                f"OptunaOptimizer 생성: n_trials={recipe_hyperparams.n_trials}, metric={recipe_hyperparams.optimization_metric}"
            )

            optimizer = OptimizerRegistry.create(
                "optuna", settings=self.settings, factory_provider=self._get_factory
            )

            # Factory와 Evaluator는 trial 간 재사용 (성능 최적화)
            factory = self._get_factory()
            evaluator = factory.create_evaluator()
            optimization_metric = (
                self.settings.recipe.model.hyperparameters.optimization_metric or "accuracy"
            )

            def _objective_callback(_ignored_train_df, params, seed):
                # 모델만 trial마다 새로 생성 (파라미터가 다르므로)
                model_instance = factory.create_model()
                try:
                    model_instance.set_params(**(params or {}))
                except Exception:
                    pass
                # 학습
                self._fit_model(model_instance, X_train, y_train, additional_data.get("train"))
                # 검증 점수 계산
                metrics = evaluator.evaluate(
                    model_instance, X_val, y_val, additional_data.get("val")
                )
                return {
                    optimization_metric: metrics.get(optimization_metric, 0.0),
                    "score": metrics.get(optimization_metric, 0.0),
                }

            best = optimizer.optimize(
                train_df=None, training_callback=_objective_callback
            )  # train_df 미사용

            # 최적화 완료 로깅
            log_train(
                f"HPO 완료 - best_score={best.get('best_score')}, trials={best.get('total_trials')}"
            )

            # 최적 파라미터로 최종 모델에 적용 후 재학습
            try:
                model.set_params(**(best.get("best_params") or {}))
                log_train_debug("최적 파라미터 적용 완료")
            except Exception as e:
                logger.warning(f"[TRAIN] 최적 파라미터 적용 일부 실패: {str(e)}")

            log_train("최적 파라미터로 최종 모델 학습")
            self._fit_model(model, X_train, y_train, additional_data.get("train"))
            trained_model = model
            self.training_results["hyperparameter_optimization"] = best
        else:
            log_train_debug("HPO 비활성화 - 고정 파라미터로 학습")
            self._fit_model(model, X_train, y_train, additional_data.get("train"))
            trained_model = model
            self.training_results["hyperparameter_optimization"] = {"enabled": False}

        return trained_model, dict(self.training_results)

    # 기존 단일 학습/검증 분할 로직은 파이프라인으로 이동하여 제거됨

    def _fit_model(self, model, X, y, additional_data):
        """task_choice에 따라 모델을 학습시킵니다."""
        import pandas as pd

        if not isinstance(model, BaseModel):
            from sklearn.base import is_classifier, is_regressor

            if not (is_classifier(model) or is_regressor(model) or hasattr(model, "fit")):
                logger.error("[TRAIN] 모델 객체가 BaseModel 또는 scikit-learn 호환이 아닙니다")
                raise TypeError(
                    "전달된 모델 객체는 BaseModel 인터페이스를 따르거나 scikit-learn 호환 모델이어야 합니다."
                )

        task_choice = self.settings.recipe.task_choice

        # 데이터 정보 로깅
        training_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        features = X.shape[1] if hasattr(X, "shape") and len(X.shape) > 1 else "N/A"
        log_train_debug(f"학습 데이터: {training_samples}샘플, {features}피처, task={task_choice}")

        # [필독] 최종 데이터 타입 검증 (Transparency 원칙)
        # 전처리가 끝난 후에도 문자열(object) 데이터가 남아있으면 명확한 에러 발생
        if hasattr(X, "select_dtypes"):
            obj_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
            if obj_cols:
                logger.error(f"[TRAIN] 모델 학습 실패: 처리되지 않은 문자열 컬럼이 발견되었습니다: {obj_cols}")
                suggestion = (
                    f"\n[해결 방법] 모델({model.__class__.__name__})은 수치형 데이터만 지원합니다.\n"
                    f"Recipe의 preprocessor 섹션에 아래와 같이 인코더를 추가하여 {obj_cols} 컬럼을 변환하세요:\n"
                    "preprocessor:\n"
                    "  steps:\n"
                    "    - type: onehot_encoder  # 또는 label_encoder\n"
                    f"      columns: {obj_cols}\n"
                )
                logger.info(suggestion)
                raise TypeError(f"문자열 컬럼({obj_cols})이 전처리되지 않았습니다. 인코더를 추가하세요.")

        try:
            if task_choice in ["classification", "regression"]:
                model.fit(X, y)
            elif task_choice == "clustering":
                model.fit(X)
            elif task_choice == "causal":
                model.fit(X, additional_data["treatment"], y)
            elif task_choice == "timeseries":
                model.fit(X, y)
            else:
                logger.error(f"[TRAIN] 지원하지 않는 task_choice: {task_choice}")
                raise ValueError(f"지원하지 않는 task_choice: {task_choice}")

            # 학습 완료 로깅
            log_train(f"{task_choice} 모델 학습 완료 - {model.__class__.__name__}")

        except TypeError as e:
            logger.error(f"[TRAIN] 모델 학습 타입 에러: {str(e)}")
            raise
        except ValueError as e:
            error_msg = str(e).lower()
            # NaN/결측값 관련 에러 감지 및 전처리기 추가 제안
            if (
                "nan" in error_msg
                or "missing" in error_msg
                or "null" in error_msg
                or "infinite" in error_msg
            ):
                logger.error(f"[TRAIN] 결측값/NaN 관련 에러 발생: {str(e)}")
                suggestion = (
                    "\n[해결 방법] Recipe의 preprocessor 섹션에 결측값 처리 전처리기를 추가하세요:\n"
                    "preprocessor:\n"
                    "  steps:\n"
                    "    - type: simple_imputer\n"
                    "      strategy: median  # 또는 mean, most_frequent\n"
                    "      create_missing_indicators: false\n"
                )
                logger.info(suggestion)
                raise ValueError(f"{str(e)}\n{suggestion}") from e
            raise
        except Exception as e:
            logger.error(f"[TRAIN] 모델 학습 에러: {str(e)}")
            raise

    def _get_training_methodology(self):
        """학습 방법론 메타데이터를 반환합니다."""
        validation_config = self.settings.recipe.evaluation.validation
        hyperparams_config = self.settings.recipe.model.hyperparameters
        task_choice = self.settings.recipe.task_choice

        # stratification 여부 결정
        stratify_col = self._get_stratify_col()
        split_method = "stratified" if stratify_col else "simple"

        # validation strategy 결정
        if hyperparams_config.tuning_enabled:
            validation_strategy = "train_validation_split"  # Optuna 시 train에서 validation 분할
            note = f"Optuna 사용 시 Train({1-validation_config.test_size:.0%})을 다시 Train(80%)/Val(20%)로 분할"
        else:
            validation_strategy = validation_config.method
            note = f"Hyperparameter tuning 비활성화, {validation_config.method} 사용"

        return {
            "train_test_split_method": split_method,
            "train_ratio": 1 - validation_config.test_size,
            "test_ratio": validation_config.test_size,
            "validation_strategy": validation_strategy,
            "random_state": validation_config.random_state,
            "stratify_column": stratify_col,
            "task_choice": task_choice,
            "preprocessing_fit_scope": "train_only",
            "hyperparameter_optimization": hyperparams_config.tuning_enabled,
            "n_trials": hyperparams_config.n_trials if hyperparams_config.tuning_enabled else None,
            "optimization_metric": (
                hyperparams_config.optimization_metric
                if hyperparams_config.tuning_enabled
                else None
            ),
            "note": note,
        }

    def _get_stratify_col(self):
        di = self.settings.recipe.data.data_interface
        task_choice = self.settings.recipe.task_choice
        return (
            di.target_column
            if task_choice == "classification"
            else di.treatment_column if task_choice == "causal" else None
        )


# Self-registration
TrainerRegistry.register("default", Trainer)

"""모델/HPO 관련 생성 전담 클래스."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Dict, Optional

from mmp.components.calibration.calibration_evaluator import CalibrationEvaluator
from mmp.utils.core.logger import logger

if TYPE_CHECKING:
    from mmp.settings import Settings


class ModelCreator:
    """
    모델 생성 및 HPO 관련 로직을 담당하는 클래스.
    캐싱은 Factory(파사드)가 담당하므로, 이 클래스는 순수 생성 로직만 포함한다.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._recipe = settings.recipe
        self._model = self._recipe.model

    def create_model(self) -> Any:
        """Model 생성."""
        class_path = self._model.class_path

        hyperparameters = self._extract_model_hyperparameters()

        try:
            model = self._create_from_class_path(class_path, hyperparameters)
            return model

        except Exception as e:
            logger.error(f"{class_path}에서 Model 생성에 실패했습니다: {e}")
            raise

    def create_optuna_integration(self) -> Any:
        """Optuna Integration 생성."""
        tuning_config = getattr(self._model, "hyperparameters", None)

        if not tuning_config:
            raise ValueError("하이퍼파라미터 튜닝 설정이 구성되지 않았습니다.")

        try:
            from mmp.utils.integrations.optuna_integration import OptunaIntegration

            integration = OptunaIntegration(tuning_config)
            return integration

        except ImportError:
            logger.error("Optuna가 설치되지 않았습니다. pip install optuna로 설치해주세요")
            raise
        except Exception as e:
            logger.error(f"Optuna integration 생성에 실패했습니다: {e}")
            raise

    def create_calibration_evaluator(self, trained_model, trained_calibrator) -> Optional[Any]:
        """Calibration Evaluator 생성."""
        task_type = self._recipe.task_choice
        if task_type != "classification" or not trained_calibrator:
            return None

        if not hasattr(trained_model, "predict_proba"):
            logger.warning("Model이 predict_proba를 지원하지 않아 calibration 평가를 건너뜁니다")
            return None

        return CalibrationEvaluator(trained_model, trained_calibrator)

    # ===============================
    # Internal Utilities
    # ===============================

    def _extract_model_hyperparameters(self) -> Dict[str, Any]:
        """모델 하이퍼파라미터를 추출 (tuning 메타데이터 제외)."""
        if hasattr(self._model.hyperparameters, "tuning_enabled"):
            if self._model.hyperparameters.tuning_enabled:
                # 튜닝 활성화시: fixed 파라미터만 사용
                if (
                    hasattr(self._model.hyperparameters, "fixed")
                    and self._model.hyperparameters.fixed
                ):
                    return self._model.hyperparameters.fixed.copy()
                return {}
            else:
                # 튜닝 비활성화시: values 파라미터 사용
                if (
                    hasattr(self._model.hyperparameters, "values")
                    and self._model.hyperparameters.values
                ):
                    return self._model.hyperparameters.values.copy()
                return {}
        else:
            # 레거시 구조: 전체 dict에서 tuning 메타데이터 제외
            hyperparameters = (
                dict(self._model.hyperparameters)
                if hasattr(self._model.hyperparameters, "__dict__")
                else {}
            )
            tuning_keys = [
                "tuning_enabled",
                "optimization_metric",
                "direction",
                "n_trials",
                "timeout",
                "fixed",
                "tunable",
                "values",
            ]
            for key in tuning_keys:
                hyperparameters.pop(key, None)
            return hyperparameters

    def _create_from_class_path(self, class_path: str, hyperparameters: Dict[str, Any]) -> Any:
        """클래스 경로로부터 동적으로 객체를 생성."""
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            processed_params = self._process_hyperparameters(hyperparameters)

            instance = model_class(**processed_params)
            logger.debug(f"[FACT] 클래스 인스턴스 생성 완료 - {class_path}")
            return instance

        except Exception as e:
            logger.error(f"[FACT] 클래스 인스턴스 생성 실패 - {class_path}: {e}")
            raise ValueError(f"클래스를 로드할 수 없습니다: {class_path}") from e

    @staticmethod
    def _process_hyperparameters(params: Dict[str, Any]) -> Dict[str, Any]:
        """하이퍼파라미터 전처리 (문자열을 객체로 변환 등)."""
        processed = params.copy()

        for key, value in processed.items():
            if isinstance(value, str) and "." in value and ("_fn" in key or "_class" in key):
                try:
                    module_path, func_name = value.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    processed[key] = getattr(module, func_name)
                    logger.debug(f"Hyperparameter '{key}'를 callable로 변환했습니다: {value}")
                except (ImportError, AttributeError):
                    logger.debug(f"Hyperparameter '{key}'를 문자열로 유지합니다: {value}")

        return processed

# src/components/_evaluator/_regression.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.components.evaluator.base import BaseEvaluator
from src.settings import Settings
from src.utils.core.logger import log_eval, log_eval_debug


class RegressionEvaluator(BaseEvaluator):
    METRIC_KEYS = ["r2_score", "mean_squared_error"]
    DEFAULT_OPTIMIZATION_METRIC = "r2_score"

    def __init__(self, settings: Settings):
        super().__init__(settings)

    def evaluate(self, model, X, y, source_df=None):
        log_eval(f"회귀 모델 평가 시작 - {len(X)}샘플")

        # 타겟 변수 기본 통계
        y_mean = np.mean(y)
        y_std = np.std(y)
        log_eval_debug(f"타겟 통계: 평균={y_mean:.4f}, 표준편차={y_std:.4f}")

        predictions = model.predict(X)

        # 주요 메트릭 계산
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)

        metrics = {
            "r2_score": r2,
            "mean_squared_error": mse,
            "root_mean_squared_error": rmse,
            "mean_absolute_error": mae,
        }

        # 메트릭 상세 로깅
        log_eval_debug(f"R²: {r2:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # 평가 완료 요약
        log_eval(f"평가 완료 - R²: {r2:.4f}, RMSE: {rmse:.4f}")

        return metrics


# Self-registration
from ..registry import EvaluatorRegistry

EvaluatorRegistry.register("regression", RegressionEvaluator)

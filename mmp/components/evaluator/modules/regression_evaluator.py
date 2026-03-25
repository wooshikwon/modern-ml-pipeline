# mmp/components/_evaluator/_regression.py
import re

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mmp.components.evaluator.base import BaseEvaluator
from mmp.settings import Settings
from mmp.utils.core.logger import log_eval, log_eval_debug


def _pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Compute pinball (quantile) loss."""
    residual = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.where(residual >= 0, quantile * residual, (quantile - 1) * residual)))


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

        # Detect quantile mode: DataFrame with pred_pN columns
        if isinstance(predictions, pd.DataFrame):
            quantile_cols = [c for c in predictions.columns if re.fullmatch(r"pred_p\d+", c)]
            if quantile_cols:
                return self._evaluate_quantile(y, predictions, quantile_cols)

        # --- Standard regression path (unchanged) ---
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

    def _evaluate_quantile(self, y, predictions: pd.DataFrame, quantile_cols: list[str]) -> dict:
        """Evaluate quantile regression predictions."""
        log_eval(f"분위 회귀 평가 모드 - {len(quantile_cols)}개 분위수 감지")

        metrics: dict[str, float] = {}

        # Parse quantile levels from column names and sort
        col_quantiles: list[tuple[str, int]] = []
        for col in quantile_cols:
            n = int(re.search(r"\d+", col).group())  # type: ignore[union-attr]
            col_quantiles.append((col, n))
        col_quantiles.sort(key=lambda x: x[1])

        # 1. Per-quantile pinball loss
        pinball_losses: list[float] = []
        for col, n in col_quantiles:
            quantile = n / 100.0
            loss = _pinball_loss(y, predictions[col], quantile)
            metrics[f"pinball_loss_p{n}"] = loss
            pinball_losses.append(loss)
            log_eval_debug(f"Pinball loss p{n} (q={quantile:.2f}): {loss:.4f}")

        # 2. Mean pinball loss
        mean_pl = float(np.mean(pinball_losses))
        metrics["mean_pinball_loss"] = mean_pl
        log_eval_debug(f"Mean pinball loss: {mean_pl:.4f}")

        # 3. Standard regression metrics on p50 (if available)
        if "pred_p50" in predictions.columns:
            y_arr = np.asarray(y)
            p50 = np.asarray(predictions["pred_p50"])
            r2 = r2_score(y_arr, p50)
            mse = mean_squared_error(y_arr, p50)
            rmse = float(np.sqrt(mse))
            mae = mean_absolute_error(y_arr, p50)
            metrics["r2_score"] = r2
            metrics["mean_squared_error"] = mse
            metrics["root_mean_squared_error"] = rmse
            metrics["mean_absolute_error"] = mae
            log_eval_debug(f"p50 기준 - R²: {r2:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # 4. Interval coverage (lowest to highest quantile)
        if len(col_quantiles) >= 2:
            lowest_col = col_quantiles[0][0]
            highest_col = col_quantiles[-1][0]
            y_arr = np.asarray(y)
            lower = np.asarray(predictions[lowest_col])
            upper = np.asarray(predictions[highest_col])
            coverage = float(np.mean((y_arr >= lower) & (y_arr <= upper)))
            metrics["interval_coverage"] = coverage
            log_eval_debug(
                f"구간 커버리지 (p{col_quantiles[0][1]}-p{col_quantiles[-1][1]}): {coverage:.4f}"
            )

        # 5. Per-quantile MAE
        for col, n in col_quantiles:
            q_mae = mean_absolute_error(y, predictions[col])
            metrics[f"mae_p{n}"] = q_mae
            log_eval_debug(f"MAE p{n}: {q_mae:.4f}")

        # 평가 완료 요약
        summary_parts = [f"mean_pinball={mean_pl:.4f}"]
        if "r2_score" in metrics:
            summary_parts.append(f"R²(p50)={metrics['r2_score']:.4f}")
        if "interval_coverage" in metrics:
            summary_parts.append(f"coverage={metrics['interval_coverage']:.4f}")
        log_eval(f"분위 회귀 평가 완료 - {', '.join(summary_parts)}")

        return metrics


# Self-registration
from ..registry import EvaluatorRegistry

EvaluatorRegistry.register("regression", RegressionEvaluator)

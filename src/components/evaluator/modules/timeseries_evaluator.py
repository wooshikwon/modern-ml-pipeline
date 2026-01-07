import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.components.evaluator.base import BaseEvaluator
from src.settings import Settings
from src.utils.core.logger import log_eval, log_eval_debug


class TimeSeriesEvaluator(BaseEvaluator):
    """시계열 태스크 전용 Evaluator - MAPE, SMAPE 등 시계열 특화 메트릭 제공"""

    METRIC_KEYS = ["mse", "rmse", "mae", "mape", "smape"]
    DEFAULT_OPTIMIZATION_METRIC = "rmse"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        log_eval_debug("TimeSeriesEvaluator 초기화 완료")

    def evaluate(self, model, X, y, source_df=None):
        """
        시계열 모델 평가 메트릭 계산

        Args:
            model: 학습된 시계열 모델
            X: 테스트 피처 데이터
            y: 실제 시계열 값
            source_df: 원본 테스트 데이터프레임 (선택사항)

        Returns:
            Dict[str, float]: 시계열 평가 메트릭들
        """
        log_eval(f"시계열 모델 평가 시작 - {len(X)}샘플, {X.shape[1]}피처")

        predictions = model.predict(X)

        # 1-D로 표준화 (예: (n,1) -> (n,))
        y_arr = np.ravel(np.array(y))
        pred_arr = np.ravel(np.array(predictions))

        # 시계열 예측 결과 기본 통계
        y_mean, y_std = np.mean(y_arr), np.std(y_arr)
        pred_mean, pred_std = np.mean(pred_arr), np.std(pred_arr)
        log_eval_debug(f"실제값: {y_mean:.4f}±{y_std:.4f}, 예측값: {pred_mean:.4f}±{pred_std:.4f}")

        # 기본 회귀 메트릭 계산
        mse = mean_squared_error(y_arr, pred_arr)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_arr, pred_arr)

        # 시계열 특화 메트릭 계산
        mape = self._calculate_mape(y_arr, pred_arr)
        smape = self._calculate_smape(y_arr, pred_arr)

        metrics = {
            # 기본 회귀 메트릭
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            # 시계열 특화 메트릭
            "mape": mape,
            "smape": smape,
        }

        # 최종 결과 요약
        log_eval(f"평가 완료 - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
        return metrics

    def _calculate_mape(self, y_true, y_pred):
        """
        Mean Absolute Percentage Error (MAPE) 계산

        Args:
            y_true: 실제 값 (1-D)
            y_pred: 예측 값 (1-D)

        Returns:
            float: MAPE 값 (0-100%)
        """
        y_true = np.ravel(np.array(y_true))
        y_pred = np.ravel(np.array(y_pred))

        # 0 값 처리: 0인 경우 해당 데이터 포인트 제외
        mask = y_true != 0
        if not mask.any():
            return float("inf")  # 모든 실제 값이 0인 경우

        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return float(mape)

    def _calculate_smape(self, y_true, y_pred):
        """
        Symmetric Mean Absolute Percentage Error (SMAPE) 계산

        Args:
            y_true: 실제 값 (1-D)
            y_pred: 예측 값 (1-D)

        Returns:
            float: SMAPE 값 (0-100%)
        """
        y_true = np.ravel(np.array(y_true))
        y_pred = np.ravel(np.array(y_pred))

        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0

        if not mask.any():
            return 0.0  # 실제값과 예측값이 모두 0인 경우

        smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        return float(smape)


# Self-registration
from ..registry import EvaluatorRegistry

EvaluatorRegistry.register("timeseries", TimeSeriesEvaluator)

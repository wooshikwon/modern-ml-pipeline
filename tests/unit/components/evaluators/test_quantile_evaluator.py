"""
Quantile Regression Evaluator Tests

RegressionEvaluator의 quantile 분기 테스트.
기존 standard regression 경로는 test_regression_evaluator.py에서 커버.
"""

import numpy as np
import pandas as pd

from mmp.components.evaluator.modules.regression_evaluator import (
    RegressionEvaluator,
    _pinball_loss,
)


class TestPinballLoss:
    """pinball loss 계산 정확성 테스트"""

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert _pinball_loss(y, y, 0.5) == 0.0

    def test_median_quantile_symmetric(self):
        """q=0.5일 때 pinball loss = 0.5 * MAE"""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])  # all over-predicted by 0.5
        # residual = y_true - y_pred = [-0.5, -0.5, -0.5] (all negative)
        # q=0.5: (0.5 - 1) * (-0.5) = 0.25 per sample
        assert abs(_pinball_loss(y_true, y_pred, 0.5) - 0.25) < 1e-10

    def test_high_quantile_penalizes_underprediction(self):
        """q=0.9일 때 과소예측이 더 큰 페널티"""
        y_true = np.array([10.0])
        y_pred_under = np.array([5.0])  # underprediction
        y_pred_over = np.array([15.0])  # overprediction

        loss_under = _pinball_loss(y_true, y_pred_under, 0.9)
        loss_over = _pinball_loss(y_true, y_pred_over, 0.9)
        # At q=0.9, underprediction is penalized 9x more than overprediction
        assert loss_under > loss_over

    def test_low_quantile_penalizes_overprediction(self):
        """q=0.1일 때 과대예측이 더 큰 페널티"""
        y_true = np.array([10.0])
        y_pred_under = np.array([5.0])
        y_pred_over = np.array([15.0])

        loss_under = _pinball_loss(y_true, y_pred_under, 0.1)
        loss_over = _pinball_loss(y_true, y_pred_over, 0.1)
        assert loss_over > loss_under


class TestRegressionEvaluatorQuantileMode:
    """RegressionEvaluator의 quantile 분기 테스트"""

    def _make_quantile_model(self, predictions_df: pd.DataFrame):
        """predict()가 multi-column DataFrame을 반환하는 mock model."""

        class _MockQuantileModel:
            def predict(self, X):
                return predictions_df

        return _MockQuantileModel()

    def test_detects_quantile_mode(self, settings_builder):
        """pred_pN 컬럼이 있으면 quantile 모드로 전환"""
        X = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        preds = pd.DataFrame({
            "pred_p50": [1.1, 2.1, 3.1, 4.1, 5.1],
            "pred_p90": [2.0, 3.0, 4.0, 5.0, 6.0],
        })
        model = self._make_quantile_model(preds)

        settings = settings_builder.with_task("regression").build()
        evaluator = RegressionEvaluator(settings)
        metrics = evaluator.evaluate(model, X, y)

        # quantile 모드 메트릭 확인
        assert "pinball_loss_p50" in metrics
        assert "pinball_loss_p90" in metrics
        assert "mean_pinball_loss" in metrics
        assert "interval_coverage" in metrics

    def test_standard_mode_for_non_quantile_dataframe(self, settings_builder):
        """pred_pN 패턴이 아닌 DataFrame은 standard 모드"""

        class _SingleColModel:
            def predict(self, X):
                return pd.DataFrame({"prediction": [1.0, 2.0, 3.0]})

        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([1.0, 2.0, 3.0])

        settings = settings_builder.with_task("regression").build()
        evaluator = RegressionEvaluator(settings)
        metrics = evaluator.evaluate(_SingleColModel(), X, y)

        # standard 메트릭
        assert "r2_score" in metrics
        assert "mean_squared_error" in metrics
        # quantile 메트릭 없음
        assert "mean_pinball_loss" not in metrics

    def test_pinball_loss_values(self, settings_builder):
        """pinball loss가 올바르게 계산되는지"""
        y = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        preds = pd.DataFrame({
            "pred_p50": y.values.copy(),  # perfect p50
            "pred_p90": y.values + 5.0,   # over-predicted
        })
        model = self._make_quantile_model(preds)

        settings = settings_builder.with_task("regression").build()
        evaluator = RegressionEvaluator(settings)
        metrics = evaluator.evaluate(model, pd.DataFrame({"a": range(5)}), y)

        # p50 perfect prediction → pinball loss = 0
        assert metrics["pinball_loss_p50"] == 0.0
        # p90 over-predicted → positive pinball loss
        assert metrics["pinball_loss_p90"] > 0.0

    def test_interval_coverage(self, settings_builder):
        """interval coverage 계산 검증"""
        y = pd.Series([1.0, 5.0, 10.0, 15.0, 20.0])
        preds = pd.DataFrame({
            "pred_p10": [0.0, 4.0, 9.0, 14.0, 19.0],
            "pred_p90": [2.0, 6.0, 11.0, 16.0, 21.0],
        })
        model = self._make_quantile_model(preds)

        settings = settings_builder.with_task("regression").build()
        evaluator = RegressionEvaluator(settings)
        metrics = evaluator.evaluate(model, pd.DataFrame({"a": range(5)}), y)

        # All values fall within [p10, p90]
        assert metrics["interval_coverage"] == 1.0

    def test_partial_coverage(self, settings_builder):
        """일부만 구간에 포함되는 경우"""
        y = pd.Series([1.0, 5.0, 100.0, 15.0])  # 100.0 is outlier
        preds = pd.DataFrame({
            "pred_p25": [0.0, 4.0, 9.0, 14.0],
            "pred_p75": [2.0, 6.0, 11.0, 16.0],
        })
        model = self._make_quantile_model(preds)

        settings = settings_builder.with_task("regression").build()
        evaluator = RegressionEvaluator(settings)
        metrics = evaluator.evaluate(model, pd.DataFrame({"a": range(4)}), y)

        # 3 out of 4 within interval
        assert metrics["interval_coverage"] == 0.75

    def test_p50_standard_metrics(self, settings_builder):
        """p50이 있을 때 standard regression 메트릭도 계산"""
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = pd.DataFrame({
            "pred_p50": [1.0, 2.0, 3.0, 4.0, 5.0],  # perfect
            "pred_p90": [2.0, 3.0, 4.0, 5.0, 6.0],
        })
        model = self._make_quantile_model(preds)

        settings = settings_builder.with_task("regression").build()
        evaluator = RegressionEvaluator(settings)
        metrics = evaluator.evaluate(model, pd.DataFrame({"a": range(5)}), y)

        assert metrics["r2_score"] == 1.0
        assert metrics["mean_squared_error"] == 0.0

    def test_per_quantile_mae(self, settings_builder):
        """quantile별 MAE 메트릭"""
        y = pd.Series([10.0, 20.0])
        preds = pd.DataFrame({
            "pred_p50": [11.0, 19.0],  # MAE = 1.0
            "pred_p90": [15.0, 25.0],  # MAE = 5.0
        })
        model = self._make_quantile_model(preds)

        settings = settings_builder.with_task("regression").build()
        evaluator = RegressionEvaluator(settings)
        metrics = evaluator.evaluate(model, pd.DataFrame({"a": range(2)}), y)

        assert abs(metrics["mae_p50"] - 1.0) < 1e-10
        assert abs(metrics["mae_p90"] - 5.0) < 1e-10

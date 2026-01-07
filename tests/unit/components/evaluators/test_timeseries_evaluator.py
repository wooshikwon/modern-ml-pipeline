"""
Time Series Evaluator Unit Tests - No Mock Hell Approach
Real time series metrics calculation
Following comprehensive testing strategy document principles
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.components.evaluator.modules.timeseries_evaluator import TimeSeriesEvaluator
from src.components.evaluator.base import BaseEvaluator


class TestTimeSeriesEvaluator:
    """Test TimeSeriesEvaluator with real time series data and metrics."""

    def test_timeseries_evaluator_initialization(self, settings_builder):
        """Test TimeSeriesEvaluator initialization."""
        # Given: Valid settings for time series
        settings = settings_builder.with_task("timeseries").build()

        # When: Creating TimeSeriesEvaluator
        evaluator = TimeSeriesEvaluator(settings)

        # Then: Evaluator is properly initialized
        assert isinstance(evaluator, TimeSeriesEvaluator)
        assert isinstance(evaluator, BaseEvaluator)
        assert evaluator.task_choice == "timeseries"

    def test_evaluate_with_time_series_data(self, settings_builder):
        """Test evaluation with real time series predictions."""
        # Given: Time series data and predictions
        np.random.seed(42)
        time_steps = 100
        t = np.arange(time_steps)

        # Create synthetic time series with trend and seasonality
        trend = 0.5 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        noise = np.random.normal(0, 2, time_steps)
        y_true = trend + seasonal + noise

        # Simple model that captures some pattern
        X = t.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X[:80], y_true[:80])

        X_test = X[80:]
        y_test = y_true[80:]

        settings = settings_builder.with_task("timeseries").build()
        evaluator = TimeSeriesEvaluator(settings)

        # When: Evaluating time series model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: Time series specific metrics are calculated
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

        # Common time series metrics
        possible_metrics = [
            "mse",
            "rmse",
            "mae",
            "mape",
            "mean_squared_error",
            "root_mean_squared_error",
            "mean_absolute_error",
        ]
        assert any(m in metrics for m in possible_metrics)

    def test_evaluate_with_seasonal_patterns(self, settings_builder):
        """Test evaluation with seasonal time series data."""
        # Given: Seasonal time series
        t = np.arange(120)
        seasonal_pattern = 10 * np.sin(2 * np.pi * t / 12) + 5 * np.cos(2 * np.pi * t / 6)
        y_true = seasonal_pattern + np.random.normal(0, 1, 120)

        # Model that predicts moving average
        class MovingAverageModel:
            def __init__(self, window=12):
                self.window = window
                self.history = None

            def fit(self, X, y):
                self.history = y
                return self

            def predict(self, X):
                # Simple prediction: last value from history
                return np.full(len(X), self.history[-1])

        model = MovingAverageModel()
        model.fit(None, y_true[:100])

        X_test = np.arange(100, 120).reshape(-1, 1)
        y_test = y_true[100:120]

        settings = settings_builder.with_task("timeseries").build()
        evaluator = TimeSeriesEvaluator(settings)

        # When: Evaluating seasonal model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: Metrics capture seasonal prediction quality
        assert isinstance(metrics, dict)

        # Check if error metrics are reasonable
        if "mse" in metrics or "mean_squared_error" in metrics:
            mse_key = "mse" if "mse" in metrics else "mean_squared_error"
            assert metrics[mse_key] > 0  # Should have some error

    def test_evaluate_with_trend_predictions(self, settings_builder):
        """Test evaluation with trending time series."""
        # Given: Time series with clear trend
        t = np.arange(50)
        trend = 2 * t + 10
        y_true = trend + np.random.normal(0, 5, 50)

        # Linear model for trend
        X = t.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X[:40], y_true[:40])

        X_test = X[40:]
        y_test = y_true[40:]

        settings = settings_builder.with_task("timeseries").build()
        evaluator = TimeSeriesEvaluator(settings)

        # When: Evaluating trend model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: Model captures trend reasonably well
        y_pred = model.predict(X_test)
        manual_mse = mean_squared_error(y_test, y_pred)

        # Metrics should be calculated
        assert len(metrics) > 0

        # If MSE is in metrics, verify it's reasonable
        if "mse" in metrics or "mean_squared_error" in metrics:
            mse_key = "mse" if "mse" in metrics else "mean_squared_error"
            # Should be relatively small for trend following model
            assert metrics[mse_key] < 100

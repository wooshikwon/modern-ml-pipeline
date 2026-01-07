"""
Regression Evaluator Unit Tests - No Mock Hell Approach
Real metrics calculation, real model evaluation
Following comprehensive testing strategy document principles
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from src.components.evaluator.modules.regression_evaluator import RegressionEvaluator
from src.components.evaluator.base import BaseEvaluator


class TestRegressionEvaluator:
    """Test RegressionEvaluator with real models and metrics."""

    def test_regression_evaluator_initialization(self, settings_builder):
        """Test RegressionEvaluator initialization."""
        # Given: Valid settings for regression
        settings = settings_builder.with_task("regression").build()

        # When: Creating RegressionEvaluator
        evaluator = RegressionEvaluator(settings)

        # Then: Evaluator is properly initialized
        assert isinstance(evaluator, RegressionEvaluator)
        assert isinstance(evaluator, BaseEvaluator)
        assert evaluator.task_choice == "regression"

    def test_evaluate_with_linear_regression(self, settings_builder, test_data_generator):
        """Test evaluation with real LinearRegression model."""
        # Given: Trained LinearRegression model and test data
        X, y = test_data_generator.regression_data(n_samples=100, n_features=5)
        X_train, y_train = X[:70], y[:70]
        X_test, y_test = X[70:], y[70:]

        model = LinearRegression()
        model.fit(X_train, y_train)

        settings = settings_builder.with_task("regression").build()
        evaluator = RegressionEvaluator(settings)

        # When: Evaluating model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: Metrics are calculated correctly
        assert isinstance(metrics, dict)
        assert "mean_squared_error" in metrics
        assert "r2_score" in metrics

        # Verify metrics are reasonable
        y_pred = model.predict(X_test)
        expected_mse = mean_squared_error(y_test, y_pred)
        expected_r2 = r2_score(y_test, y_pred)

        assert abs(metrics["mean_squared_error"] - expected_mse) < 0.001
        assert abs(metrics["r2_score"] - expected_r2) < 0.001

    def test_evaluate_with_random_forest_regressor(self, settings_builder, test_data_generator):
        """Test evaluation with RandomForestRegressor."""
        # Given: Trained RandomForestRegressor
        X, y = test_data_generator.regression_data(n_samples=80, n_features=4)
        X_train, y_train = X[:60], y[:60]
        X_test, y_test = X[60:], y[60:]

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        settings = settings_builder.with_task("regression").build()
        evaluator = RegressionEvaluator(settings)

        # When: Evaluating model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: All regression metrics are valid
        assert "mean_squared_error" in metrics
        assert "r2_score" in metrics

        assert metrics["mean_squared_error"] >= 0  # MSE is always non-negative
        assert metrics["r2_score"] <= 1  # R2 score is at most 1

    def test_evaluate_with_perfect_predictions(self, settings_builder):
        """Test evaluation when model makes perfect predictions."""

        # Given: Model that makes perfect predictions
        class PerfectModel:
            def __init__(self, true_values):
                self.true_values = true_values

            def predict(self, X):
                return self.true_values

        y_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model = PerfectModel(y_test)
        X_test = np.random.randn(5, 3)

        settings = settings_builder.with_task("regression").build()
        evaluator = RegressionEvaluator(settings)

        # When: Evaluating perfect model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: Metrics should be perfect
        assert "mean_squared_error" in metrics
        assert "r2_score" in metrics

        assert metrics["mean_squared_error"] == 0.0
        assert metrics["r2_score"] == 1.0

    def test_evaluate_with_poor_predictions(self, settings_builder):
        """Test evaluation with poor model predictions."""

        # Given: Model with poor predictions
        class PoorModel:
            def predict(self, X):
                # Always predict zero
                return np.zeros(len(X))

        model = PoorModel()
        X_test = np.random.randn(20, 3)
        y_test = np.random.randn(20) + 10  # True values far from zero

        settings = settings_builder.with_task("regression").build()
        evaluator = RegressionEvaluator(settings)

        # When: Evaluating poor model
        metrics = evaluator.evaluate(model, X_test, y_test)

        # Then: Metrics should reflect poor performance
        assert "mean_squared_error" in metrics
        assert "r2_score" in metrics

        assert metrics["mean_squared_error"] > 50  # Large error
        # R2 score can be negative for very poor models

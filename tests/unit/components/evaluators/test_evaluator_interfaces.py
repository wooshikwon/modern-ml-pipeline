"""
Evaluator Interface Contract Tests - No Mock Hell Approach
Testing BaseEvaluator contract compliance with real implementations
Following comprehensive testing strategy document principles
"""

import inspect
from typing import Dict

import numpy as np
import pandas as pd

from src.components.evaluator.modules.classification_evaluator import ClassificationEvaluator
from src.components.evaluator.modules.clustering_evaluator import ClusteringEvaluator
from src.components.evaluator.modules.regression_evaluator import RegressionEvaluator
from src.components.evaluator.modules.timeseries_evaluator import TimeSeriesEvaluator
from src.components.evaluator.base import BaseEvaluator


class CustomTestEvaluator(BaseEvaluator):
    """Custom evaluator implementation for testing BaseEvaluator contract."""

    def evaluate(self, model, X, y, source_df=None) -> Dict[str, float]:
        """Implement evaluate method according to BaseEvaluator contract."""
        # Simple implementation: return dummy metrics
        metrics = {"custom_metric_1": 0.85, "custom_metric_2": 0.92, "custom_accuracy": 0.88}

        # Use model to make predictions (contract validation)
        if hasattr(model, "predict"):
            predictions = model.predict(X)
            # Calculate a real metric
            if y is not None:
                metrics["data_points"] = float(len(y))

        return metrics


class TestEvaluatorInterfaceContract:
    """Test BaseEvaluator interface contract compliance."""

    def test_base_evaluator_interface_definition(self):
        """Test that BaseEvaluator defines required interface."""
        # Given: BaseEvaluator class

        # Then: Required methods and attributes
        assert hasattr(BaseEvaluator, "evaluate")
        assert hasattr(BaseEvaluator, "__init__")
        assert inspect.isabstract(BaseEvaluator)

        # Verify evaluate method signature
        evaluate_sig = inspect.signature(BaseEvaluator.evaluate)
        assert "model" in evaluate_sig.parameters
        assert "X" in evaluate_sig.parameters
        assert "y" in evaluate_sig.parameters
        assert "source_df" in evaluate_sig.parameters

    def test_custom_evaluator_implements_base_evaluator(self, settings_builder):
        """Test custom evaluator properly implements BaseEvaluator interface."""
        # Given: Custom evaluator and mock model
        settings = settings_builder.with_task("classification").build()
        evaluator = CustomTestEvaluator(settings)

        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))

        model = MockModel()
        X = np.random.randn(10, 3)
        y = np.random.randint(0, 2, 10)

        # When: Using evaluator through BaseEvaluator interface
        assert isinstance(evaluator, BaseEvaluator)
        metrics = evaluator.evaluate(model, X, y)

        # Then: Returns dict of metrics
        assert isinstance(metrics, Dict)
        assert all(isinstance(k, str) for k in metrics.keys())
        assert all(isinstance(v, (int, float)) for v in metrics.values())

    def test_all_evaluators_follow_contract(self, settings_builder):
        """Test that all evaluator implementations follow BaseEvaluator contract."""
        # Given: All evaluator implementations
        evaluator_classes = [
            ClassificationEvaluator,
            RegressionEvaluator,
            ClusteringEvaluator,
            TimeSeriesEvaluator,
        ]

        tasks = ["classification", "regression", "clustering", "timeseries"]

        for evaluator_class, task in zip(evaluator_classes, tasks):
            # Create evaluator instance with fresh builder for each task
            from tests.conftest import SettingsBuilder

            fresh_builder = SettingsBuilder()
            settings = fresh_builder.with_task(task).build()
            evaluator = evaluator_class(settings)

            # Then: Each evaluator follows the contract
            assert isinstance(evaluator, BaseEvaluator)
            assert hasattr(evaluator, "evaluate")
            assert hasattr(evaluator, "settings")
            assert hasattr(evaluator, "task_choice")
            assert evaluator.task_choice == task

            # Verify evaluate method exists and has correct signature
            evaluate_method = getattr(evaluator, "evaluate")
            sig = inspect.signature(evaluate_method)
            assert "model" in sig.parameters
            assert "X" in sig.parameters
            assert "y" in sig.parameters

    def test_evaluator_handles_none_y_for_unsupervised(self, settings_builder):
        """Test evaluators handle None y for unsupervised tasks."""
        # Given: Clustering evaluator (unsupervised)
        settings = settings_builder.with_task("clustering").build()
        evaluator = ClusteringEvaluator(settings)

        class MockClusteringModel:
            def __init__(self):
                self.labels_ = np.array([0, 0, 1, 1, 2, 2])

            def fit_predict(self, X):
                return self.labels_[: len(X)]

            def predict(self, X):
                return self.labels_[: len(X)]

        model = MockClusteringModel()
        X = np.random.randn(6, 3)

        # When: Evaluating without y (unsupervised)
        metrics = evaluator.evaluate(model, X, y=None)

        # Then: Evaluation works without y
        assert isinstance(metrics, dict)

    def test_evaluator_with_source_dataframe(self, settings_builder, test_data_generator):
        """Test evaluators can use optional source_df parameter."""
        # Given: Evaluator and data with source DataFrame
        X, y = test_data_generator.classification_data(n_samples=20)
        source_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        source_df["target"] = y
        source_df["extra_col"] = np.random.randn(20)

        settings = settings_builder.with_task("classification").build()
        evaluator = CustomTestEvaluator(settings)

        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))

        model = MockModel()

        # When: Evaluating with source_df
        metrics = evaluator.evaluate(model, X, y, source_df=source_df)

        # Then: Evaluation completes successfully
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

"""
Component Interface Contract Tests
Week 1, Days 6-7: Factory & Component Registration Tests

Tests that Factory-created components properly implement their interface contracts
following comprehensive testing strategy - No Mock Hell approach with real components.
"""

import numpy as np
import pandas as pd

from src.factory import Factory
from src.components.adapter.base import BaseAdapter
from src.components.datahandler.base import BaseDataHandler
from src.components.evaluator.base import BaseEvaluator
from src.components.fetcher.base import BaseFetcher
from src.components.trainer.base import BaseTrainer


class TestBaseAdapterInterfaceContract:
    """Test BaseAdapter interface contract implementation (1 test)."""

    def test_adapter_interface_contract_compliance(self, settings_builder, test_data_files):
        """Test that Factory-created adapters properly implement BaseAdapter interface."""
        settings = (
            settings_builder.with_data_source("storage")
            .with_data_path(str(test_data_files["classification"]))
            .build()
        )

        factory = Factory(settings)
        adapter = factory.create_data_adapter("storage")

        # Validate interface compliance
        assert isinstance(adapter, BaseAdapter)

        # Test required methods exist
        assert hasattr(adapter, "read")
        assert hasattr(adapter, "write")
        assert callable(adapter.read)
        assert callable(adapter.write)

        # Test read() method contract
        df = adapter.read(str(test_data_files["classification"]))

        # read() should return pandas DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 0

        # Test write() method contract
        test_output_path = test_data_files["classification"].parent / "test_output.csv"

        # write() should accept DataFrame and path
        adapter.write(df.head(5), str(test_output_path))

        # Verify write worked by reading back
        written_df = adapter.read(str(test_output_path))
        assert isinstance(written_df, pd.DataFrame)
        assert len(written_df) == 5  # Should have written 5 rows

        # Cleanup
        test_output_path.unlink(missing_ok=True)


class TestBaseEvaluatorInterfaceContract:
    """Test BaseEvaluator interface contract implementation (1 test)."""

    def test_evaluator_interface_contract_compliance(
        self, settings_builder, small_real_models_cache
    ):
        """Test that Factory-created evaluators properly implement BaseEvaluator interface."""
        settings = (
            settings_builder.with_task("classification")
            .with_model("sklearn.linear_model.LogisticRegression")
            .build()
        )

        factory = Factory(settings)
        evaluator = factory.create_evaluator()

        # Validate interface compliance
        assert isinstance(evaluator, BaseEvaluator)

        # Test required methods exist
        assert hasattr(evaluator, "evaluate")
        assert callable(evaluator.evaluate)

        # Test evaluate() method contract with real data
        models, data = small_real_models_cache
        X_cls = data["X_cls"]
        y_cls = data["y_cls"]
        trained_model = models["logistic"]

        # evaluate() should accept (model, X, y) and return dict
        metrics = evaluator.evaluate(trained_model, X_cls, y_cls)

        # Contract requirements
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

        # Should contain at least one numeric metric
        numeric_metrics = [k for k, v in metrics.items() if isinstance(v, (int, float, np.number))]
        assert len(numeric_metrics) > 0

        # Metrics should be reasonable values (0-1 for most classification metrics)
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                if (
                    "accuracy" in metric_name.lower()
                    or "f1" in metric_name.lower()
                    or "precision" in metric_name.lower()
                    or "recall" in metric_name.lower()
                ):
                    assert (
                        0.0 <= value <= 1.0
                    ), f"Metric {metric_name} = {value} should be in [0,1] range"


class TestModelInterfaceContract:
    """Test ML Model interface contract implementation (1 test)."""

    def test_model_interface_contract_compliance(self, settings_builder):
        """Test that Factory-created models properly implement sklearn-like interface."""
        settings = (
            settings_builder.with_task("classification")
            .with_model(
                "sklearn.ensemble.RandomForestClassifier",
                hyperparameters={"n_estimators": 5, "random_state": 42},
            )
            .build()
        )

        factory = Factory(settings)
        model = factory.create_model()

        # Test required sklearn-like methods exist
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert callable(model.fit)
        assert callable(model.predict)

        # Test fit() method contract with real data
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=30, n_features=4, random_state=42)

        # fit() should accept (X, y) and return self
        fitted_model = model.fit(X, y)
        assert fitted_model is model  # Should return self

        # Test predict() method contract
        predictions = model.predict(X)

        # predict() should return array-like with same length as input
        assert hasattr(predictions, "__len__")  # Array-like
        assert len(predictions) == len(X)
        assert len(predictions) == len(y)

        # For classification, predictions should be class labels
        unique_predictions = set(predictions)
        unique_labels = set(y)
        assert unique_predictions.issubset(
            unique_labels
        )  # Predictions should be valid class labels

        # Test that model learns (predictions should be reasonable)
        accuracy = sum(predictions == y) / len(y)
        assert accuracy >= 0.5  # Should be better than random for simple synthetic data


class TestBaseFetcherInterfaceContract:
    """Test BaseFetcher interface contract implementation (1 test)."""

    def test_fetcher_interface_contract_compliance(self, settings_builder):
        """Test that Factory-created fetchers properly implement BaseFetcher interface."""
        settings = settings_builder.with_environment("local").build()

        factory = Factory(settings)
        fetcher = factory.create_fetcher()

        # Validate interface compliance
        assert isinstance(fetcher, BaseFetcher)

        # Test required methods exist
        assert hasattr(fetcher, "fetch")
        assert callable(fetcher.fetch)

        # Test fetch() method contract
        # For pass-through fetcher, should return input data as-is
        test_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": ["a", "b", "c", "d", "e"],
                "target": [0, 1, 0, 1, 0],
            }
        )

        # fetch() should accept DataFrame and return DataFrame
        result = fetcher.fetch(test_data)

        # Contract requirements
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(test_data)
        assert list(result.columns) == list(test_data.columns)

        # For pass-through fetcher, should be identical
        pd.testing.assert_frame_equal(result, test_data)


class TestFactoryInterfaceContractEnforcement:
    """Test Factory enforces interface contracts across all components (1 test)."""

    def test_factory_enforces_all_interface_contracts(self, settings_builder, test_data_files):
        """Test that Factory ensures all created components comply with their interface contracts."""
        settings = (
            settings_builder.with_data_source("storage")
            .with_data_path(str(test_data_files["classification"]))
            .with_task("classification")
            .with_model(
                "sklearn.ensemble.RandomForestClassifier",
                hyperparameters={"n_estimators": 5, "random_state": 42},
            )
            .build()
        )

        factory = Factory(settings)

        # Create all major components
        adapter = factory.create_data_adapter()
        model = factory.create_model()
        evaluator = factory.create_evaluator()
        trainer = factory.create_trainer()
        datahandler = factory.create_datahandler()
        fetcher = factory.create_fetcher()

        # Test that all components implement their base interfaces
        assert isinstance(adapter, BaseAdapter)
        assert isinstance(evaluator, BaseEvaluator)
        assert isinstance(trainer, BaseTrainer)
        assert isinstance(datahandler, BaseDataHandler)
        assert isinstance(fetcher, BaseFetcher)

        # Test that all components have their required methods
        interface_requirements = {
            adapter: ["read", "write"],
            evaluator: ["evaluate"],
            trainer: ["train"],
            datahandler: ["prepare_data"],
            fetcher: ["fetch"],
        }

        for component, required_methods in interface_requirements.items():
            for method_name in required_methods:
                assert hasattr(
                    component, method_name
                ), f"{type(component).__name__} missing required method: {method_name}"
                assert callable(
                    getattr(component, method_name)
                ), f"{type(component).__name__}.{method_name} is not callable"

        # Test real workflow demonstrates interface compatibility
        df = adapter.read(str(test_data_files["classification"]))

        # Data preprocessing through datahandler
        prepared_data = datahandler.prepare_data(df)
        assert isinstance(prepared_data, (pd.DataFrame, tuple))

        # Model training (simplified)
        features = df.drop(["target", "entity_id"], axis=1, errors="ignore")
        target = df["target"]
        model.fit(features, target)

        # Model evaluation
        metrics = evaluator.evaluate(model, features, target)
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

        # All interfaces should work together without type errors
        # This validates that Factory creates compatible components

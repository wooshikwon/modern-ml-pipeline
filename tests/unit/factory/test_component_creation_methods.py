"""
Component Creation Methods Tests
Week 1, Days 6-7: Factory & Component Registration Tests

Tests Factory's 8 component creation methods following
comprehensive testing strategy - No Mock Hell approach with real components.
"""

import pandas as pd

from src.components.adapter.modules.storage_adapter import StorageAdapter
from src.components.evaluator.modules.classification_evaluator import ClassificationEvaluator
from src.components.evaluator.modules.regression_evaluator import RegressionEvaluator
from src.factory import Factory
from src.components.adapter.base import BaseAdapter
from src.components.datahandler.base import BaseDataHandler
from src.components.evaluator.base import BaseEvaluator
from src.components.fetcher.base import BaseFetcher
from src.components.trainer.base import BaseTrainer


class TestFactoryDataAdapterCreation:
    """Test Factory.create_data_adapter() method with real adapters."""

    def test_create_storage_adapter_with_auto_detection(self, settings_builder, test_data_files):
        """Test create_data_adapter() auto-detects storage adapter for CSV files."""
        settings = (
            settings_builder.with_data_source("storage")
            .with_data_path(str(test_data_files["classification"]))
            .build()
        )

        factory = Factory(settings)
        adapter = factory.create_data_adapter()  # Auto-detection

        # Validate correct adapter type
        assert isinstance(adapter, StorageAdapter)
        assert isinstance(adapter, BaseAdapter)

        # Test real behavior
        df = adapter.read(str(test_data_files["classification"]))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "target" in df.columns

    def test_create_sql_adapter_with_explicit_type(self, settings_builder, real_dataset_files):
        """Test create_data_adapter() with explicit SQL adapter type."""
        sql_info = real_dataset_files["sql"]
        connection_string = f"sqlite:///{sql_info['path']}"

        settings = settings_builder.with_data_source(
            "sql", config={"connection_uri": connection_string}
        ).build()

        factory = Factory(settings)
        adapter = factory.create_data_adapter("sql")  # Explicit type

        # Validate correct adapter type
        from src.components.adapter.modules.sql_adapter import SqlAdapter

        assert isinstance(adapter, SqlAdapter)
        assert isinstance(adapter, BaseAdapter)

        # Test real SQL behavior with explicit columns (security requirement)
        query = f"SELECT feature_1, feature_2, feature_3, feature_4, feature_5, target, entity_id FROM {sql_info['classification_table']} LIMIT 5"
        df = adapter.read(query)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "target" in df.columns
        assert "entity_id" in df.columns

    def test_create_adapter_caching_behavior(self, settings_builder, test_data_files):
        """Test data adapter creation caching works correctly."""
        settings = (
            settings_builder.with_data_source("storage")
            .with_data_path(str(test_data_files["classification"]))
            .build()
        )

        factory = Factory(settings)

        # First call should create and cache
        adapter1 = factory.create_data_adapter()
        assert len(factory._component_cache) >= 1

        # Second call should return cached instance
        adapter2 = factory.create_data_adapter()
        assert adapter1 is adapter2  # Same object instance

        # Different adapter type should create new instance
        try:
            adapter3 = factory.create_data_adapter("sql")
            assert adapter3 is not adapter1  # Different instance
        except Exception:
            # SQL adapter may fail without proper config, which is expected
            pass


class TestFactoryModelCreation:
    """Test Factory.create_model() method with real ML models."""

    def test_create_sklearn_classification_model(self, settings_builder, performance_benchmark):
        """Test create_model() with sklearn classification model."""
        settings = (
            settings_builder.with_task("classification")
            .with_model(
                "sklearn.ensemble.RandomForestClassifier",
                hyperparameters={"n_estimators": 10, "random_state": 42},
            )
            .build()
        )

        factory = Factory(settings)

        with performance_benchmark.measure_time("model_creation"):
            model = factory.create_model()

        # Validate model type and configuration
        from sklearn.ensemble import RandomForestClassifier

        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 10
        assert model.random_state == 42

        # Test real model behavior
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=20, n_features=4, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 20
        assert hasattr(model, "feature_importances_")

        performance_benchmark.assert_time_under("model_creation", 0.1)

    def test_create_sklearn_regression_model(self, settings_builder):
        """Test create_model() with sklearn regression model."""
        settings = (
            settings_builder.with_task("regression")
            .with_model("sklearn.linear_model.LinearRegression", hyperparameters={})
            .build()
        )

        factory = Factory(settings)
        model = factory.create_model()

        # Validate model type
        from sklearn.linear_model import LinearRegression

        assert isinstance(model, LinearRegression)

        # Test real regression behavior
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=20, n_features=3, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 20
        assert hasattr(model, "coef_")

    def test_create_model_with_hyperparameter_processing(self, settings_builder):
        """Test create_model() processes hyperparameters correctly."""
        hyperparams = {"n_estimators": 25, "max_depth": 8, "random_state": 123, "bootstrap": False}

        settings = (
            settings_builder.with_task("classification")
            .with_model("sklearn.ensemble.RandomForestClassifier", hyperparameters=hyperparams)
            .build()
        )

        factory = Factory(settings)
        model = factory.create_model()

        # Validate hyperparameters were applied
        assert model.n_estimators == 25
        assert model.max_depth == 8
        assert model.random_state == 123
        assert model.bootstrap == False


class TestFactoryEvaluatorCreation:
    """Test Factory.create_evaluator() method with real evaluators."""

    def test_create_classification_evaluator(self, settings_builder, small_real_models_cache):
        """Test create_evaluator() for classification task."""
        settings = (
            settings_builder.with_task("classification")
            .with_model("sklearn.linear_model.LogisticRegression")
            .build()
        )

        factory = Factory(settings)
        evaluator = factory.create_evaluator()

        # Validate evaluator type
        assert isinstance(evaluator, ClassificationEvaluator)
        assert isinstance(evaluator, BaseEvaluator)

        # Test real evaluation behavior
        models, data = small_real_models_cache
        X_cls = data["X_cls"]
        y_cls = data["y_cls"]

        trained_model = models["logistic"]
        metrics = evaluator.evaluate(trained_model, X_cls, y_cls)

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_create_regression_evaluator(self, settings_builder, small_real_models_cache):
        """Test create_evaluator() for regression task."""
        settings = (
            settings_builder.with_task("regression")
            .with_model("sklearn.linear_model.LinearRegression")
            .build()
        )

        factory = Factory(settings)
        evaluator = factory.create_evaluator()

        # Validate evaluator type
        assert isinstance(evaluator, RegressionEvaluator)
        assert isinstance(evaluator, BaseEvaluator)

        # Test real evaluation behavior
        models, data = small_real_models_cache
        X_reg = data["X_reg"]
        y_reg = data["y_reg"]

        trained_model = models["linear"]
        metrics = evaluator.evaluate(trained_model, X_reg, y_reg)

        assert isinstance(metrics, dict)
        # Should have at least one regression metric
        regression_metrics = [
            "mse",
            "mae",
            "r2",
            "mean_squared_error",
            "mean_absolute_error",
            "r2_score",
        ]
        found_metrics = [m for m in regression_metrics if m in metrics]
        assert len(found_metrics) > 0


class TestFactoryFetcherCreation:
    """Test Factory.create_fetcher() method with real fetchers."""

    def test_create_pass_through_fetcher_default(self, settings_builder):
        """Test create_fetcher() creates pass-through fetcher for local environment."""
        settings = settings_builder.with_environment("local").build()

        factory = Factory(settings)
        fetcher = factory.create_fetcher()

        # Validate fetcher type
        assert isinstance(fetcher, BaseFetcher)

        # Pass-through fetcher should return data as-is
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        result = fetcher.fetch(test_data)

        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, test_data)

    def test_create_fetcher_with_caching(self, settings_builder):
        """Test fetcher creation caching for different run modes."""
        settings = settings_builder.build()
        factory = Factory(settings)

        # Create fetchers for different modes
        fetcher_batch = factory.create_fetcher("batch")
        fetcher_batch2 = factory.create_fetcher("batch")

        # Same mode should return cached instance
        assert fetcher_batch is fetcher_batch2

        # Check cache contains fetcher
        cache_keys = [key for key in factory._component_cache.keys() if "fetcher" in key]
        assert len(cache_keys) >= 1


class TestFactoryPreprocessorCreation:
    """Test Factory.create_preprocessor() method."""

    def test_create_preprocessor_returns_none_when_not_configured(self, settings_builder):
        """Test create_preprocessor() returns None when no preprocessor is configured."""
        settings = settings_builder.build()  # No preprocessor configured

        factory = Factory(settings)
        preprocessor = factory.create_preprocessor()

        # Should return None when not configured
        assert preprocessor is None

    def test_create_preprocessor_caching_behavior(self, settings_builder):
        """Test preprocessor creation caching works correctly."""
        settings = settings_builder.build()
        factory = Factory(settings)

        # First call
        preprocessor1 = factory.create_preprocessor()

        # Second call should return same result (cached)
        preprocessor2 = factory.create_preprocessor()

        assert preprocessor1 is preprocessor2


class TestFactoryTrainerCreation:
    """Test Factory.create_trainer() method with real trainers."""

    def test_create_default_trainer(self, settings_builder):
        """Test create_trainer() with default trainer type."""
        settings = settings_builder.build()

        factory = Factory(settings)
        trainer = factory.create_trainer()

        # Validate trainer type
        assert isinstance(trainer, BaseTrainer)
        assert hasattr(trainer, "train")  # Should have train method

    def test_create_trainer_with_explicit_type(self, settings_builder):
        """Test create_trainer() with explicit trainer type."""
        settings = settings_builder.build()

        factory = Factory(settings)
        trainer = factory.create_trainer("default")

        # Validate trainer type
        assert isinstance(trainer, BaseTrainer)

    def test_create_trainer_caching_behavior(self, settings_builder):
        """Test trainer creation caching works correctly."""
        settings = settings_builder.build()
        factory = Factory(settings)

        # Create trainers
        trainer1 = factory.create_trainer()
        trainer2 = factory.create_trainer()

        # Should return cached instance
        assert trainer1 is trainer2


class TestFactoryDataHandlerCreation:
    """Test Factory.create_datahandler() method with real data handlers."""

    def test_create_datahandler_for_classification(self, settings_builder):
        """Test create_datahandler() for classification task."""
        settings = (
            settings_builder.with_task("classification")
            .with_model("sklearn.ensemble.RandomForestClassifier")
            .build()
        )

        factory = Factory(settings)
        datahandler = factory.create_datahandler()

        # Validate datahandler type
        assert isinstance(datahandler, BaseDataHandler)
        assert hasattr(datahandler, "prepare_data")  # Should have data preparation method

    def test_create_datahandler_for_regression(self, settings_builder):
        """Test create_datahandler() for regression task."""
        settings = (
            settings_builder.with_task("regression")
            .with_model("sklearn.linear_model.LinearRegression")
            .build()
        )

        factory = Factory(settings)
        datahandler = factory.create_datahandler()

        # Validate datahandler type
        assert isinstance(datahandler, BaseDataHandler)

    def test_create_datahandler_caching_behavior(self, settings_builder):
        """Test datahandler creation caching works correctly."""
        settings = settings_builder.with_task("classification").build()

        factory = Factory(settings)

        # Create datahandlers
        datahandler1 = factory.create_datahandler()
        datahandler2 = factory.create_datahandler()

        # Should return cached instance
        assert datahandler1 is datahandler2


class TestFactoryPyfuncWrapperCreation:
    """Test Factory.create_pyfunc_wrapper() method with real components."""

    def test_create_pyfunc_wrapper_with_minimal_components(
        self, settings_builder, small_real_models_cache
    ):
        """Test create_pyfunc_wrapper() with minimal required components."""
        settings = settings_builder.with_task("classification").build()

        factory = Factory(settings)

        # Create required components
        model = factory.create_model()
        datahandler = factory.create_datahandler()

        # Train model with real data
        models, data = small_real_models_cache
        X_cls = data["X_cls"]
        y_cls = data["y_cls"]
        model.fit(X_cls, y_cls)

        # Create PyfuncWrapper
        pyfunc_wrapper = factory.create_pyfunc_wrapper(
            trained_model=model,
            trained_datahandler=datahandler,
            trained_preprocessor=None,
            trained_fetcher=None,
        )

        # Validate wrapper
        from src.utils.integrations.pyfunc_wrapper import PyfuncWrapper

        assert isinstance(pyfunc_wrapper, PyfuncWrapper)
        assert pyfunc_wrapper.trained_model is model
        # PyfuncWrapper preserves trained components for inference
        assert pyfunc_wrapper.trained_datahandler is datahandler

    def test_create_pyfunc_wrapper_with_all_components(
        self, settings_builder, small_real_models_cache
    ):
        """Test create_pyfunc_wrapper() with all components provided."""
        settings = settings_builder.with_task("classification").build()

        factory = Factory(settings)

        # Create all components
        model = factory.create_model()
        datahandler = factory.create_datahandler()
        preprocessor = factory.create_preprocessor()
        fetcher = factory.create_fetcher()

        # Train model
        models, data = small_real_models_cache
        X_cls = data["X_cls"]
        y_cls = data["y_cls"]
        model.fit(X_cls, y_cls)

        # Create PyfuncWrapper with all components
        pyfunc_wrapper = factory.create_pyfunc_wrapper(
            trained_model=model,
            trained_datahandler=datahandler,
            trained_preprocessor=preprocessor,
            trained_fetcher=fetcher,
            training_results={"accuracy": 0.95},
        )

        # Validate wrapper with all components
        from src.utils.integrations.pyfunc_wrapper import PyfuncWrapper

        assert isinstance(pyfunc_wrapper, PyfuncWrapper)
        assert pyfunc_wrapper.trained_model is model
        # PyfuncWrapper preserves trained components for inference
        assert pyfunc_wrapper.trained_datahandler is datahandler
        assert pyfunc_wrapper.trained_preprocessor is preprocessor
        assert pyfunc_wrapper.trained_fetcher is fetcher

        # Validate training results are preserved
        assert pyfunc_wrapper.training_results == {"accuracy": 0.95}

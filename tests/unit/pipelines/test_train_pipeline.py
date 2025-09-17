"""
Unit Tests for Training Pipeline - No Mock Hell Compliant
Following test philosophy: Real components with Context patterns

Uses MLflowTestContext and ComponentTestContext for real component testing.
Only mocks external services when absolutely necessary.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.pipelines.train_pipeline import run_train_pipeline


class TestTrainPipelineWithContexts:
    """Test training pipeline with real components using Context patterns"""

    def test_train_pipeline_with_mlflow_context(
        self, mlflow_test_context, real_component_performance_tracker
    ):
        """Test full training pipeline with MLflowTestContext"""
        with mlflow_test_context.for_classification("train_pipeline") as ctx:
            with real_component_performance_tracker.measure_time("full_pipeline"):
                # Run real training pipeline with context-provided settings
                result = run_train_pipeline(ctx.settings)

            # Verify results using context helpers
            assert ctx.experiment_exists()
            assert ctx.get_experiment_run_count() > 0

            # Check metrics were logged
            metrics = ctx.get_run_metrics()
            # May have accuracy, f1, or other metrics depending on configuration
            assert len(metrics) > 0

            # Verify performance
            real_component_performance_tracker.assert_time_under("full_pipeline", 10.0)

            # Verify artifacts
            assert ctx.verify_mlflow_artifacts()

    def test_train_pipeline_with_component_context(
        self, component_test_context, mlflow_test_context
    ):
        """Test training with ComponentTestContext for component interaction"""
        with mlflow_test_context.for_classification("component_train") as mlflow_ctx:
            with component_test_context.classification_stack() as comp_ctx:
                # Use component context to verify component interactions
                # Settings from component context
                settings = comp_ctx.settings

                # Update MLflow settings from MLflow context
                settings.config.mlflow.tracking_uri = mlflow_ctx.mlflow_uri
                settings.config.mlflow.experiment_name = mlflow_ctx.experiment_name

                # Run training pipeline
                result = run_train_pipeline(settings)

                # Verify using both contexts
                assert mlflow_ctx.get_experiment_run_count() > 0
                assert comp_ctx.adapter_is_compatible_with_model()

    def test_train_with_real_data_and_preprocessor(
        self, factory_with_real_storage_adapter, mlflow_test_context
    ):
        """Test training with real data adapter and preprocessor"""
        factory, data_info = factory_with_real_storage_adapter

        with mlflow_test_context.for_classification("real_data_train") as ctx:
            # Use real factory components
            adapter = factory.create_data_adapter()
            preprocessor = factory.create_preprocessor()
            model = factory.create_model()
            evaluator = factory.create_evaluator()

            # Read real data
            df = adapter.read(data_info["path"])
            assert len(df) > 0

            # Run training with real components
            result = run_train_pipeline(factory.settings)

            # Verify results
            assert ctx.get_experiment_run_count() > 0
            metrics = ctx.get_run_metrics()
            assert len(metrics) > 0

    def test_train_pipeline_with_performance_tracking(
        self, mlflow_test_context, real_component_performance_tracker
    ):
        """Test training pipeline performance with real components"""
        with mlflow_test_context.for_classification("perf_test") as ctx:
            # Track detailed performance
            with real_component_performance_tracker.measure_time("data_loading"):
                # Data loading is part of pipeline
                pass

            with real_component_performance_tracker.measure_time("model_training"):
                result = run_train_pipeline(ctx.settings)

            # Performance assertions
            real_component_performance_tracker.assert_time_under("model_training", 5.0)

            # Verify training succeeded
            assert ctx.get_experiment_run_count() > 0


class TestTrainPipelineIntegration:
    """Integration tests for training pipeline"""

    def test_train_with_multiple_models(
        self, mlflow_test_context, settings_builder, test_data_generator
    ):
        """Test training different model types"""
        models_to_test = [
            "sklearn.ensemble.RandomForestClassifier",
            "sklearn.linear_model.LogisticRegression",
            "sklearn.tree.DecisionTreeClassifier"
        ]

        for model_class in models_to_test:
            with mlflow_test_context.for_classification(f"train_{model_class.split('.')[-1]}") as ctx:
                # Update settings with different model
                settings = settings_builder \
                    .with_task("classification") \
                    .with_model(model_class) \
                    .with_data_path(str(ctx.data_path)) \
                    .with_mlflow(ctx.mlflow_uri, ctx.experiment_name) \
                    .build()

                # Train with different model
                result = run_train_pipeline(settings)

                # Verify each model trains successfully
                assert ctx.get_experiment_run_count() > 0
                metrics = ctx.get_run_metrics()
                assert 'accuracy' in metrics or 'f1' in metrics

    def test_train_with_data_splits(
        self, component_test_context, settings_builder
    ):
        """Test training with different data split configurations"""
        with component_test_context.classification_stack() as ctx:
            # Test different split ratios
            split_configs = [
                (0.6, 0.2, 0.2, 0.0),  # 60/20/20/0
                (0.7, 0.15, 0.15, 0.0),  # 70/15/15/0
                (0.8, 0.1, 0.1, 0.0),  # 80/10/10/0
            ]

            for train, val, test, calib in split_configs:
                settings = ctx.settings_builder \
                    .with_data_split(train, val, test, calib) \
                    .build()

                # Run training with different splits
                result = run_train_pipeline(settings)

                # Basic assertion - pipeline should complete
                assert result is not None

    @patch('src.pipelines.train_pipeline.log_enhanced_model_with_schema')
    def test_train_model_logging(
        self, mock_log_model, mlflow_test_context
    ):
        """Test that models are logged correctly to MLflow"""
        # Only mock the final model logging to avoid MLflow storage
        mock_log_model.return_value = None

        with mlflow_test_context.for_classification("model_logging") as ctx:
            result = run_train_pipeline(ctx.settings)

            # Verify model logging was called
            assert mock_log_model.called

            # Verify MLflow tracking
            assert ctx.get_experiment_run_count() > 0
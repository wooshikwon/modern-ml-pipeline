"""
Unit Tests for Training Pipeline - Simplified Tests
Testing core pipeline functionality with minimal dependencies.
"""

from unittest.mock import patch

from src.pipelines.train_pipeline import run_train_pipeline


class TestTrainPipelineCore:
    """Core training pipeline tests with minimal integration dependencies"""

    def test_train_with_data_splits(self, component_test_context, settings_builder):
        """Test training with different data split configurations"""
        with component_test_context.classification_stack() as ctx:
            # Test different split ratios
            split_configs = [
                (0.6, 0.2, 0.2, 0.0),  # 60/20/20/0
                (0.7, 0.15, 0.15, 0.0),  # 70/15/15/0
                (0.8, 0.1, 0.1, 0.0),  # 80/10/10/0
            ]

            for train, val, test, calib in split_configs:
                settings = ctx.settings_builder.with_data_split(train, val, test, calib).build()

                # Run training with different splits
                result = run_train_pipeline(settings)

                # Basic assertion - pipeline should complete
                assert result is not None

    @patch("src.pipelines.train_pipeline.log_enhanced_model_with_schema")
    def test_train_model_logging(self, mock_log_model, mlflow_test_context):
        """Test that models are logged correctly to MLflow"""
        # Only mock the final model logging to avoid MLflow storage
        mock_log_model.return_value = None

        with mlflow_test_context.for_classification("model_logging") as ctx:
            result = run_train_pipeline(ctx.settings)

            # Verify model logging was called
            assert mock_log_model.called

            # Verify MLflow tracking
            assert ctx.get_experiment_run_count() > 0

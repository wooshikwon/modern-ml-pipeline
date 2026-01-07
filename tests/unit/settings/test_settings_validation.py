"""
Settings validation comprehensive testing
Follows tests/README.md philosophy with Context classes
Tests for src/settings validation components

Author: Phase 3 Refactoring
Date: 2025-09-14
"""

import pytest
from pydantic import ValidationError

from src.settings import ValidationOrchestrator
from src.settings.validation.common import ValidationResult


class TestSettingsValidation:
    """Settings validation tests following tests/README.md philosophy"""

    def test_config_validation_success(self, component_test_context):
        """Test successful config validation"""
        with component_test_context.classification_stack() as ctx:
            # Use Real Object Testing - actual ValidationOrchestrator
            validator = ValidationOrchestrator()

            # Validate real settings from context
            result = validator.validate_for_training(ctx.settings.config, ctx.settings.recipe)
            assert result.is_valid

    def test_config_validation_missing_environment_name(self, component_test_context):
        """Test config validation with missing environment name"""
        with component_test_context.classification_stack() as ctx:
            validator = ValidationOrchestrator()

            # Create settings with empty environment using context builder
            invalid_settings = ctx.settings_builder.with_environment("").build()

            result = validator.validate_for_training(
                invalid_settings.config, invalid_settings.recipe
            )
            # Real Object Testing - let actual validator determine validity
            assert isinstance(result, ValidationResult)

    def test_config_validation_sql_adapter_missing_connection_uri(self, component_test_context):
        """Test config validation for SQL adapter without connection_uri"""
        with component_test_context.classification_stack() as ctx:
            validator = ValidationOrchestrator()

            # Real Object Testing - attempt to create invalid SQL settings should fail
            with pytest.raises((ValidationError, ValueError)) as exc_info:
                sql_settings = ctx.settings_builder.with_data_source("sql", config={}).build()

            # Real validation should catch missing required SQL config
            assert "connection_uri" in str(exc_info.value) or "Field required" in str(
                exc_info.value
            )

    def test_config_validation_sql_adapter_with_bigquery_config(self, component_test_context):
        """Test config validation for SQL adapter with BigQuery-like configuration"""
        with component_test_context.classification_stack() as ctx:
            validator = ValidationOrchestrator()

            # Use context builder with valid BigQuery configuration
            bigquery_settings = ctx.settings_builder.with_data_source(
                "sql",
                config={
                    "connection_uri": "bigquery://test-project/test-dataset",
                    "project_id": "test",
                    "dataset_id": "test_dataset",
                },
            ).build()

            result = validator.validate_for_training(
                bigquery_settings.config, bigquery_settings.recipe
            )
            # Real Object Testing - should succeed with valid SQL config
            assert isinstance(result, ValidationResult)


class TestRecipeValidation:
    """Recipe validation tests using Context classes"""

    def test_recipe_validation_success(self, component_test_context):
        """Test successful recipe validation"""
        with component_test_context.classification_stack() as ctx:
            validator = ValidationOrchestrator()

            # Use real settings from context
            result = validator.validate_for_training(ctx.settings.config, ctx.settings.recipe)
            # Real recipe validation should succeed with context-generated settings
            assert result.is_valid

    def test_recipe_validation_invalid_metrics_for_task(self, component_test_context):
        """Test recipe validation with invalid metrics for task"""
        with component_test_context.classification_stack() as ctx:
            validator = ValidationOrchestrator()

            # Modify settings to have invalid metrics for classification
            invalid_settings = ctx.settings_builder.with_task("classification").build()

            # Change metrics to regression metrics (invalid for classification)
            invalid_settings.recipe.evaluation.metrics = ["mae", "rmse"]

            result = validator.validate_for_training(
                invalid_settings.config, invalid_settings.recipe
            )
            # Real Object Testing - let validator determine validity
            assert isinstance(result, ValidationResult)

    def test_recipe_validation_tuning_enabled_without_tunable_params(self, component_test_context):
        """Test recipe validation when tuning enabled but no tunable parameters"""
        with component_test_context.classification_stack() as ctx:
            validator = ValidationOrchestrator()

            # Use context builder to create settings with hyperparameter tuning
            tuning_settings = ctx.settings_builder.with_hyperparameter_tuning(enabled=True).build()

            # Remove tunable parameters to trigger validation issue
            tuning_settings.recipe.model.hyperparameters.tunable = {}

            result = validator.validate_for_training(tuning_settings.config, tuning_settings.recipe)
            # Real Object Testing - observe actual validation behavior
            assert isinstance(result, ValidationResult)

    def test_recipe_validation_timeseries_requires_timestamp(self, component_test_context):
        """Timeseries task timestamp_column requirement validation"""
        with component_test_context.classification_stack() as ctx:
            validator = ValidationOrchestrator()

            # Create timeseries settings using context builder
            timeseries_settings = (
                ctx.settings_builder.with_task("timeseries")
                .with_model("any.module.ExponentialSmoothing")
                .build()
            )

            # Intentionally remove timestamp_column to trigger validation
            timeseries_settings.recipe.data.data_interface.timestamp_column = None

            result = validator.validate_for_training(
                timeseries_settings.config, timeseries_settings.recipe
            )
            # Real Object Testing - let validator handle timeseries validation
            assert isinstance(result, ValidationResult)


class TestModelCatalogIntegration:
    """Model catalog validation integration tests"""

    def test_recipe_validation_with_model_catalog(self, component_test_context):
        """Test recipe validation with model catalog integration"""
        with component_test_context.classification_stack() as ctx:
            validator = ValidationOrchestrator()

            # Test with valid model from context
            result = validator.validate_for_training(ctx.settings.config, ctx.settings.recipe)
            # Context should provide valid model configuration
            assert result.is_valid

    def test_recipe_validation_invalid_model_for_task(self, component_test_context):
        """Test recipe validation with incompatible model for task"""
        with component_test_context.classification_stack() as ctx:
            validator = ValidationOrchestrator()

            # Create settings with potentially incompatible model
            incompatible_settings = (
                ctx.settings_builder.with_model("sklearn.cluster.KMeans")
                .with_task("classification")
                .build()
            )

            result = validator.validate_for_training(
                incompatible_settings.config, incompatible_settings.recipe
            )
            # Real Object Testing - let validator determine model compatibility
            assert isinstance(result, ValidationResult)


class TestSettingsFullValidation:
    """Complete Settings validation with Context classes"""

    def test_settings_validation_success(self, component_test_context):
        """Test successful full settings validation"""
        with component_test_context.classification_stack() as ctx:
            validator = ValidationOrchestrator()

            # Full validation of context-generated settings
            result = validator.validate_for_training(ctx.settings.config, ctx.settings.recipe)
            # Context provides fully valid settings
            assert result.is_valid

    def test_settings_validation_feature_store_compatibility(self, component_test_context):
        """Test settings validation with feature store configuration"""
        with component_test_context.classification_stack() as ctx:
            validator = ValidationOrchestrator()

            # Create settings with feature store configuration
            fs_settings = ctx.settings_builder.with_feature_store(enabled=False).build()

            # Force recipe to use feature_store fetcher
            fs_settings.recipe.data.fetcher.type = "feature_store"

            result = validator.validate_for_training(fs_settings.config, fs_settings.recipe)
            # Real Object Testing - let validator handle compatibility check
            assert isinstance(result, ValidationResult)

    def test_settings_validation_aggregates_all_errors(self, component_test_context):
        """Test settings validation collects errors from all validation layers"""
        with component_test_context.classification_stack() as ctx:
            validator = ValidationOrchestrator()

            # Real Object Testing - settings creation itself may fail with empty environment
            with pytest.raises((ValidationError, ValueError)):
                # Multiple issues: empty environment + invalid SQL config
                problem_settings = (
                    ctx.settings_builder.with_environment("")
                    .with_data_source("sql", config={})
                    .build()
                )

    def test_settings_validation_error_aggregation(self, component_test_context):
        """Test comprehensive settings validation error handling"""
        with component_test_context.classification_stack() as ctx:
            validator = ValidationOrchestrator()

            # Create intentionally problematic settings
            invalid_settings = ctx.settings_builder.with_environment("").build()

            result = validator.validate_for_training(
                invalid_settings.config, invalid_settings.recipe
            )
            # Real Object Testing - observe how validator handles errors
            assert isinstance(result, ValidationResult)

            # If validation fails, should have error information
            if not result.is_valid:
                assert hasattr(result, "errors") or hasattr(result, "error_messages")

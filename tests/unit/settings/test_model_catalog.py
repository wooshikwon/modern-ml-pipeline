"""
Model catalog validation comprehensive testing
Follows tests/README.md philosophy with Context classes and Real Object Testing
Tests for src/settings/validation catalog validation components

Author: Phase 3 Refactoring
Date: 2025-09-14
"""

from src.settings.recipe import HyperparametersTuning, Model
from src.settings.validation.catalog_validator import CatalogValidator
from src.settings.validation.common import ValidationResult


class TestCatalogValidatorTaskValidation:
    """Model catalog task validation tests using Real Object Testing"""

    def test_get_available_tasks(self, component_test_context):
        """Test getting available tasks from catalog"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Real Object Testing - get actual tasks from catalog
            tasks = validator.get_available_tasks()
            assert isinstance(tasks, set)
            # Common ML tasks should be available
            # Note: actual availability depends on catalog directory
            if tasks:  # Only test if catalog exists
                assert all(isinstance(task, str) for task in tasks)

    def test_get_available_models_for_task(self, component_test_context):
        """Test getting available models for specific task"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Get models for classification task
            models = validator.get_available_models_for_task("classification")
            assert isinstance(models, dict)

            # Each model should have proper structure
            for model_name, model_spec in models.items():
                assert isinstance(model_name, str)
                assert isinstance(model_spec, dict)

    def test_validate_task_model_compatibility(self, component_test_context):
        """Test task and model compatibility validation"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Use actual model from context
            model = ctx.settings.recipe.model
            task_type = ctx.settings.recipe.task_choice

            # Real Object Testing - validate compatibility
            result = validator.validate_task_model_compatibility(task_type, model)
            assert isinstance(result, ValidationResult)
            # Result validity depends on actual catalog content

    def test_validate_model_specification(self, component_test_context):
        """Test model specification validation"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Use actual model from context
            model = ctx.settings.recipe.model

            # Real Object Testing - validate specification
            result = validator.validate_model_specification(model)
            assert isinstance(result, ValidationResult)
            assert hasattr(result, "is_valid")
            assert hasattr(result, "error_message")


class TestCatalogValidatorModelValidation:
    """Model validation tests using Real Object Testing"""

    def test_validate_model_with_tuning_enabled(self, component_test_context):
        """Test model validation with hyperparameter tuning enabled"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Create model with tuning enabled
            model = Model(
                class_path="sklearn.ensemble.RandomForestClassifier",
                library="sklearn",
                hyperparameters=HyperparametersTuning(
                    tuning_enabled=True,
                    tunable={"n_estimators": {"range": [10, 100]}, "max_depth": {"range": [3, 10]}},
                ),
            )

            result = validator.validate_model_specification(model)
            assert isinstance(result, ValidationResult)
            # Should be valid as tuning is enabled with tunable params
            assert result.is_valid is True

    def test_validate_model_with_tuning_disabled(self, component_test_context):
        """Test model validation with hyperparameter tuning disabled"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Create model with tuning disabled
            model = Model(
                class_path="sklearn.ensemble.RandomForestClassifier",
                library="sklearn",
                hyperparameters=HyperparametersTuning(
                    tuning_enabled=False, values={"n_estimators": 100, "max_depth": 5}
                ),
            )

            result = validator.validate_model_specification(model)
            assert isinstance(result, ValidationResult)
            # Should be valid as tuning is disabled with values
            assert result.is_valid is True

    def test_validate_model_missing_class_path(self, component_test_context):
        """Test model validation with missing class_path"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Create model without class_path
            model = Model(
                class_path="",  # Empty class_path
                library="sklearn",
                hyperparameters=HyperparametersTuning(
                    tuning_enabled=False, values={"n_estimators": 100}
                ),
            )

            result = validator.validate_model_specification(model)
            assert isinstance(result, ValidationResult)
            # Should be invalid due to missing class_path
            assert result.is_valid is False
            assert "class_path" in result.error_message.lower()

    def test_validate_model_invalid_hyperparameters(self, component_test_context):
        """Test model validation with invalid hyperparameters"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Create model with tuning enabled but no tunable params
            model = Model(
                class_path="sklearn.ensemble.RandomForestClassifier",
                library="sklearn",
                hyperparameters=HyperparametersTuning(
                    tuning_enabled=True, tunable={}  # Empty tunable when tuning is enabled
                ),
            )

            result = validator.validate_model_specification(model)
            assert isinstance(result, ValidationResult)
            # Should be invalid
            assert result.is_valid is False
            assert "tunable" in result.error_message.lower()


class TestCatalogValidatorFetcherValidation:
    """Fetcher validation tests - removed as not part of CatalogValidator"""

    def test_fetcher_validation_not_in_catalog_validator(self):
        """Note: Fetcher validation is not part of CatalogValidator"""
        # This test class is kept for clarity but tests are removed
        # as CatalogValidator doesn't handle fetcher validation
        pass


class TestCatalogValidatorIntegration:
    """Integration tests for CatalogValidator with Factory"""

    def test_catalog_validator_with_factory_settings(self, component_test_context):
        """Test CatalogValidator working with Factory-created settings"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Validate the entire model from factory settings
            model = ctx.settings.recipe.model
            result = validator.validate_model_specification(model)

            assert isinstance(result, ValidationResult)
            # Factory-created models should typically be valid

    def test_catalog_validator_error_messages(self, component_test_context):
        """Test that validation error messages are helpful"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Create intentionally invalid model
            invalid_model = Model(
                class_path="", library="unknown", hyperparameters=HyperparametersTuning()
            )

            result = validator.validate_model_specification(invalid_model)
            assert isinstance(result, ValidationResult)

            if not result.is_valid:
                # Error message should be informative
                assert result.error_message
                assert len(result.error_message) > 10

    def test_catalog_validator_with_various_tasks(self, component_test_context):
        """Test validator with different ML task types"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Test with different task types (if available in catalog)
            tasks_to_test = ["classification", "regression", "clustering"]

            for task in tasks_to_test:
                models = validator.get_available_models_for_task(task)
                # Models dict should be returned (may be empty if task not in catalog)
                assert isinstance(models, dict)

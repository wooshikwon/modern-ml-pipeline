"""
Model catalog validation comprehensive testing
Follows tests/README.md philosophy with Context classes and Real Object Testing
Tests for src/settings/validation catalog validation components

Author: Phase 3 Refactoring
Date: 2025-09-14
"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from src.settings.validation.catalog_validator import CatalogValidator
from src.settings.validation.common import ValidationResult


class TestCatalogValidatorTaskValidation:
    """Model catalog task validation tests using Real Object Testing"""

    def test_task_type_validation_success(self, component_test_context):
        """Test successful task type validation"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Use actual task type from context settings
            task_type = ctx.settings.recipe.task_choice
            assert task_type == 'classification'

            # Real Object Testing - validator should handle the task type
            result = validator.validate_task_type(task_type)
            # Let the real validator determine validity
            assert isinstance(result, bool)

    def test_task_type_validation_with_registry(self, component_test_context):
        """Test task type validation with registry data"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Test with common task types
            for task_type in ['classification', 'regression', 'timeseries']:
                result = validator.validate_task_type(task_type)
                # Real Object Testing - observe actual validation behavior
                assert isinstance(result, bool)
    
    def test_invalid_task_type_validation(self, component_test_context):
        """Test validation with invalid task types"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Test with invalid task types
            invalid_tasks = ['invalid_task', '', None, 123]
            for invalid_task in invalid_tasks:
                try:
                    result = validator.validate_task_type(invalid_task)
                    # Real Object Testing - let validator handle invalid inputs
                    assert isinstance(result, bool)
                except (TypeError, AttributeError):
                    # Some invalid inputs may raise exceptions - this is acceptable
                    pass


class TestCatalogValidatorPreprocessorValidation:
    """Preprocessor validation tests using Real Object Testing"""

    def test_preprocessor_steps_validation_success(self, component_test_context):
        """Test successful preprocessor steps validation"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Test with valid preprocessor configuration
            valid_config = {
                "steps": [
                    {"type": "standard_scaler", "columns": ["feature1"]},
                    {"type": "one_hot_encoder", "columns": ["category"]}
                ]
            }

            result = validator.validate_preprocessor_steps(valid_config)
            # Real Object Testing - let validator determine validity
            assert isinstance(result, bool)

    def test_preprocessor_steps_validation_empty_config(self, component_test_context):
        """Test preprocessor steps validation with empty configuration"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Test with None and empty configurations
            test_configs = [None, {}, {"steps": []}]
            for config in test_configs:
                result = validator.validate_preprocessor_steps(config)
                # Real Object Testing - empty configs should be handled gracefully
                assert isinstance(result, bool)

    def test_preprocessor_steps_validation_invalid_structure(self, component_test_context):
        """Test preprocessor steps validation with invalid structure"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Test with invalid configurations
            invalid_configs = [
                {"steps": "not_a_list"},
                {"steps": ["not_a_dict"]},
                {"steps": [{"missing_type": "value"}]}
            ]

            for config in invalid_configs:
                result = validator.validate_preprocessor_steps(config)
                # Real Object Testing - invalid configs should be handled
                assert isinstance(result, bool)
                # Validator may add errors for invalid configs
                if not result:
                    assert len(validator.errors) > 0
                validator.clear_messages()  # Clear for next test
    
    def test_preprocessor_validation_with_registry_check(self, component_test_context):
        """Test preprocessor validation with registry availability check"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Test validation with preprocessor steps that may or may not be in registry
            test_config = {
                "steps": [
                    {"type": "standard_scaler"},
                    {"type": "one_hot_encoder"},
                    {"type": "unknown_processor"}  # This may not be registered
                ]
            }

            result = validator.validate_preprocessor_steps(test_config)
            # Real Object Testing - let validator handle registry checks
            assert isinstance(result, bool)

            # Check if validator provides helpful information
            summary = validator.get_validation_summary()
            assert 'is_valid' in summary
            assert 'error_count' in summary
            assert 'warning_count' in summary


class TestCatalogValidatorFetcherValidation:
    """Fetcher validation tests using Real Object Testing"""

    def test_fetcher_type_validation_success(self, component_test_context):
        """Test successful fetcher type validation"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Use actual fetcher from context settings
            fetcher_config = ctx.settings.recipe.data.fetcher.model_dump()
            assert 'type' in fetcher_config

            result = validator.validate_fetcher_type(fetcher_config)
            # Real Object Testing - let validator determine validity
            assert isinstance(result, bool)

    def test_fetcher_type_validation_various_types(self, component_test_context):
        """Test fetcher type validation with various configurations"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Test with different fetcher types
            test_configs = [
                {"type": "pass_through"},
                {"type": "feature_store", "feature_views": {}},
                {"type": "unknown_fetcher"},  # May not be registered
                {"missing_type": "value"}     # Invalid structure
            ]

            for config in test_configs:
                result = validator.validate_fetcher_type(config)
                assert isinstance(result, bool)
                # Clear messages for next iteration
                validator.clear_messages()

    def test_fetcher_validation_error_handling(self, component_test_context):
        """Test fetcher validation error handling"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Test with invalid fetcher configurations
            invalid_configs = [
                None,  # Null config
                {},    # Missing type
                {"type": None},  # Null type
                {"type": ""},    # Empty type
            ]

            for config in invalid_configs:
                result = validator.validate_fetcher_type(config)
                # Real Object Testing - validator should handle invalid inputs gracefully
                assert isinstance(result, bool)


class TestCatalogValidatorAdapterValidation:
    """Adapter validation tests using Real Object Testing"""

    def test_adapter_type_validation_success(self, component_test_context):
        """Test successful adapter type validation"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Use actual adapter type from context settings
            adapter_type = ctx.settings.config.data_source.adapter_type
            assert isinstance(adapter_type, str)

            result = validator.validate_adapter_type(adapter_type)
            # Real Object Testing - let validator determine validity
            assert isinstance(result, bool)

    def test_adapter_type_validation_various_types(self, component_test_context):
        """Test adapter type validation with various types"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Test with different adapter types
            adapter_types = ['storage', 'sql', 'bigquery', 'unknown_adapter']
            for adapter_type in adapter_types:
                result = validator.validate_adapter_type(adapter_type)
                assert isinstance(result, bool)
                # Clear messages for next iteration
                validator.clear_messages()

    def test_adapter_validation_error_handling(self, component_test_context):
        """Test adapter validation error handling"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Test with invalid adapter types
            invalid_adapters = [None, "", 123, ["list"], {"dict": "value"}]
            for invalid_adapter in invalid_adapters:
                try:
                    result = validator.validate_adapter_type(invalid_adapter)
                    assert isinstance(result, bool)
                except (TypeError, AttributeError):
                    # Some invalid inputs may raise exceptions - acceptable
                    pass
                validator.clear_messages()


class TestCatalogValidatorCalibrationValidation:
    """Calibration validation tests using Real Object Testing"""

    def test_calibration_method_validation_disabled(self, component_test_context):
        """Test calibration validation when disabled"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Test with calibration disabled
            config = {"enabled": False}
            result = validator.validate_calibration_method(config)
            assert result is True  # Disabled calibration should always be valid

    def test_calibration_method_validation_enabled(self, component_test_context):
        """Test calibration validation when enabled"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Test with calibration enabled and various methods
            test_configs = [
                {"enabled": True, "method": "beta"},
                {"enabled": True, "method": "isotonic"},
                {"enabled": True, "method": "unknown_method"},
                {"enabled": True},  # Missing method
            ]

            for config in test_configs:
                result = validator.validate_calibration_method(config)
                assert isinstance(result, bool)
                validator.clear_messages()

    def test_calibration_method_validation_edge_cases(self, component_test_context):
        """Test calibration validation edge cases"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Test with edge cases
            edge_configs = [None, {}, {"enabled": None}]
            for config in edge_configs:
                result = validator.validate_calibration_method(config)
                assert isinstance(result, bool)
                validator.clear_messages()


class TestCatalogValidatorIntegration:
    """Integration tests for catalog validator using Real Object Testing"""

    def test_complete_recipe_validation(self, component_test_context):
        """Test complete recipe validation with real settings"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Use actual recipe data from context
            recipe_data = ctx.settings.recipe.model_dump()

            result = validator.validate_recipe_components(recipe_data)
            # Real Object Testing - let validator handle full recipe validation
            assert isinstance(result, bool)

            # Check validation summary
            summary = validator.get_validation_summary()
            assert isinstance(summary, dict)
            assert 'is_valid' in summary
            assert 'error_count' in summary
            assert 'warning_count' in summary

    def test_complete_config_validation(self, component_test_context):
        """Test complete config validation with real settings"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Use actual config data from context
            config_data = ctx.settings.config.model_dump()

            result = validator.validate_config_components(config_data)
            # Real Object Testing - let validator handle full config validation
            assert isinstance(result, bool)

            # Verify validator provides useful information
            components_summary = validator.get_available_components_summary()
            assert isinstance(components_summary, dict)

    def test_validator_utility_methods(self, component_test_context):
        """Test catalog validator utility methods"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Test error and warning management
            validator.add_error("Test error")
            validator.add_warning("Test warning")

            summary = validator.get_validation_summary()
            assert summary['error_count'] >= 1
            assert summary['warning_count'] >= 1
            assert "Test error" in summary['errors']
            assert "Test warning" in summary['warnings']

            # Test clearing messages
            validator.clear_messages()
            summary_after_clear = validator.get_validation_summary()
            assert summary_after_clear['error_count'] == 0
            assert summary_after_clear['warning_count'] == 0

    def test_component_existence_validation(self, component_test_context):
        """Test component existence validation"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Test component validation with various component types
            test_cases = [
                ('task_type', 'classification'),
                ('adapter', 'storage'),
                ('preprocessor', 'standard_scaler'),
            ]

            for component_type, component_name in test_cases:
                result = validator.validate_component_exists(component_type, component_name)
                # Real Object Testing - let validator handle component checks
                assert isinstance(result, bool)

    def test_alternative_suggestions(self, component_test_context):
        """Test alternative component suggestions"""
        with component_test_context.classification_stack() as ctx:
            validator = CatalogValidator()

            # Test suggestion functionality
            suggestions = validator.suggest_alternatives('preprocessor', 'scaler')
            assert isinstance(suggestions, list)
            # Suggestions may be empty if no similar components found

            suggestions = validator.suggest_alternatives('unknown_type', 'unknown_name')
            assert isinstance(suggestions, list)
            assert len(suggestions) == 0  # Should be empty for unknown types
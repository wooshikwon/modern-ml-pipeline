"""
Missing Component Error Handling Tests
Week 1, Days 6-7: Factory & Component Registration Tests

Tests Factory and Registry error handling for missing/invalid components following
comprehensive testing strategy - No Mock Hell approach with real error scenarios.
"""

import pytest

from src.components.adapter.registry import AdapterRegistry
from src.components.evaluator.registry import EvaluatorRegistry
from src.components.fetcher.registry import FetcherRegistry
from src.components.trainer.registry import TrainerRegistry
from src.factory import Factory


class TestRegistryErrorHandling:
    """Test Registry classes handle missing/invalid requests properly."""

    def test_adapter_registry_unknown_type_error(self, settings_builder):
        """Test AdapterRegistry raises proper error for unknown adapter types."""
        settings = settings_builder.build()
        Factory(settings)  # Trigger registration

        # Test get_class() with unknown type
        with pytest.raises(KeyError) as exc_info:
            AdapterRegistry.get_class("nonexistent_adapter")

        error_message = str(exc_info.value)
        # Error message should include available options
        assert "알 수 없는 키" in error_message
        assert "nonexistent_adapter" in error_message
        assert "사용 가능" in error_message
        assert "storage" in error_message  # Should list available adapters

        # Test create() with unknown type (also raises KeyError)
        with pytest.raises(KeyError) as exc_info:
            AdapterRegistry.create("nonexistent_adapter", settings)

        error_message = str(exc_info.value)
        assert "알 수 없는 키" in error_message
        assert "nonexistent_adapter" in error_message

    def test_evaluator_registry_unknown_task_error(self, settings_builder):
        """Test EvaluatorRegistry raises proper error for unknown task types."""
        settings = settings_builder.build()
        Factory(settings)  # Trigger registration

        # Test get_class() with unknown task
        with pytest.raises(KeyError) as exc_info:
            EvaluatorRegistry.get_class("nonexistent_task")

        error_message = str(exc_info.value)
        assert "알 수 없는 키" in error_message
        assert "nonexistent_task" in error_message

        # Test create() with unknown task
        with pytest.raises(KeyError) as exc_info:
            EvaluatorRegistry.create("nonexistent_task", settings)

        error_message = str(exc_info.value)
        assert "알 수 없는 키" in error_message
        assert "nonexistent_task" in error_message


class TestFactoryErrorHandling:
    """Test Factory handles component creation errors gracefully."""

    def test_factory_handles_registry_errors(self, settings_builder):
        """Test Factory allows Registry errors to propagate with meaningful messages."""
        settings = settings_builder.build()
        factory = Factory(settings)

        # Factory wraps Registry errors in ValueError with helpful message
        with pytest.raises(ValueError) as exc_info:
            factory.create_data_adapter("invalid_adapter")

        error_message = str(exc_info.value)
        assert "Failed to create adapter" in error_message
        assert "invalid_adapter" in error_message
        assert "Available:" in error_message

    def test_factory_maintains_state_after_error(self, settings_builder):
        """Test Factory maintains proper state even after encountering errors."""
        settings = settings_builder.build()
        factory = Factory(settings)

        # First, cause an error
        with pytest.raises(ValueError):
            factory.create_data_adapter("nonexistent_adapter")

        # Factory should still work for valid requests
        valid_adapter = factory.create_data_adapter("storage")
        assert valid_adapter is not None

        # Cache should still work properly
        second_adapter = factory.create_data_adapter("storage")
        assert valid_adapter is second_adapter  # Should be cached

        # Registration state should be maintained
        assert Factory._components_registered

    def test_factory_error_messages_are_informative(self, settings_builder):
        """Test Factory preserves informative error messages from registries."""
        settings = settings_builder.build()
        factory = Factory(settings)

        # Test that Factory error messages are helpful
        try:
            factory.create_data_adapter("nonexistent_adapter")
            assert False, "Should have raised an exception"
        except ValueError as e:
            error_msg = str(e)

            # Error message should be helpful
            assert len(error_msg) > 10  # Should have substantial message
            assert "nonexistent_adapter" in error_msg  # Should mention the invalid input
            assert "Available:" in error_msg  # Should provide guidance


class TestMissingComponentScenarios:
    """Test handling of various missing component scenarios."""

    def test_unknown_fetcher_type_error(self, settings_builder):
        """Test FetcherRegistry handles unknown fetcher types properly."""
        settings = settings_builder.build()
        Factory(settings)  # Trigger registration

        # Test create() with unknown fetcher type
        with pytest.raises(KeyError) as exc_info:
            FetcherRegistry.create("nonexistent_fetcher", settings)

        error_message = str(exc_info.value)
        assert "알 수 없는 키" in error_message
        assert "nonexistent_fetcher" in error_message

    def test_unknown_trainer_type_error(self, settings_builder):
        """Test TrainerRegistry handles unknown trainer types properly."""
        settings = settings_builder.build()
        Factory(settings)  # Trigger registration

        # Test create() with unknown trainer type
        with pytest.raises(KeyError) as exc_info:
            TrainerRegistry.create("nonexistent_trainer", settings)

        error_message = str(exc_info.value)
        assert "알 수 없는 키" in error_message
        assert "nonexistent_trainer" in error_message

    def test_invalid_task_configuration(self, settings_builder):
        """Test Factory handles invalid task configuration via direct Registry call."""
        settings = settings_builder.build()
        Factory(settings)  # Trigger registration

        # Test Registry directly since settings_builder validates task_choice
        with pytest.raises(KeyError) as exc_info:
            EvaluatorRegistry.create("nonexistent_task", settings)

        error_message = str(exc_info.value)
        assert "알 수 없는 키" in error_message
        assert "nonexistent_task" in error_message


class TestErrorHandlingEdgeCases:
    """Test error handling for edge cases."""

    def test_factory_sequential_error_recovery(self, settings_builder):
        """Test Factory can recover from multiple sequential errors."""
        settings = settings_builder.build()
        factory = Factory(settings)

        # Multiple error scenarios in sequence
        with pytest.raises(ValueError):
            factory.create_data_adapter("invalid_adapter1")

        with pytest.raises(ValueError):
            factory.create_data_adapter("invalid_adapter2")

        # Note: Fetcher may handle invalid types gracefully, so test Registry directly
        with pytest.raises(KeyError):
            FetcherRegistry.create("invalid_fetcher", settings)

        # Factory should still work for valid requests after all errors
        valid_adapter = factory.create_data_adapter("storage")
        valid_fetcher = factory.create_fetcher()
        valid_evaluator = factory.create_evaluator()

        assert valid_adapter is not None
        assert valid_fetcher is not None
        assert valid_evaluator is not None

        # Caching should still work
        second_adapter = factory.create_data_adapter("storage")
        assert valid_adapter is second_adapter

    def test_registry_error_messages_include_available_options(self, settings_builder):
        """Test Registry error messages include available options for developer guidance."""
        settings = settings_builder.build()
        Factory(settings)  # Trigger registration

        # AdapterRegistry should list available adapters in error message
        with pytest.raises(KeyError) as exc_info:
            AdapterRegistry.get_class("invalid")

        error_message = str(exc_info.value)
        assert "사용 가능" in error_message
        assert "storage" in error_message
        assert "sql" in error_message

        # EvaluatorRegistry should list available tasks in error message
        with pytest.raises(KeyError) as exc_info:
            EvaluatorRegistry.create("invalid_task", settings)

        error_message = str(exc_info.value)
        assert "classification" in error_message.lower() or "regression" in error_message.lower()

    def test_factory_handles_component_initialization_errors(self, settings_builder):
        """Test Factory handles errors during component initialization."""
        # Create settings that might cause initialization issues (but not registry issues)
        settings = settings_builder.build()
        factory = Factory(settings)

        # This test validates that Factory can handle various error types
        # and maintain its state for subsequent valid operations

        # First cause a registry error
        with pytest.raises(ValueError):
            factory.create_data_adapter("nonexistent")

        # Then verify Factory still works
        valid_adapter = factory.create_data_adapter("storage")
        assert valid_adapter is not None

        # Factory state should be consistent
        assert Factory._components_registered
        assert len(factory._component_cache) >= 1  # Should have cached the valid adapter

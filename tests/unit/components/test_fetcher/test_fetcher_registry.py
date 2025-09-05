"""
Unit tests for FetcherRegistry.
Tests registry pattern with self-registration mechanism for feature fetchers.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

from src.components.fetcher.registry import FetcherRegistry
from src.interface import BaseFetcher
from src.settings import Settings


class TestFetcherRegistryBasicOperations:
    """Test FetcherRegistry basic CRUD operations."""
    
    def test_register_valid_fetcher(self):
        """Test registering a valid fetcher class."""
        # Arrange
        class MockFetcher(BaseFetcher):
            def fetch(self, *args, **kwargs):
                return Mock()
        
        # Act
        FetcherRegistry.register("test_fetcher", MockFetcher)
        
        # Assert
        assert "test_fetcher" in FetcherRegistry.fetchers
        assert FetcherRegistry.fetchers["test_fetcher"] == MockFetcher
    
    def test_register_invalid_fetcher_type_error(self):
        """Test registering non-BaseFetcher class raises TypeError."""
        # Arrange
        class InvalidFetcher:
            pass
        
        # Act & Assert
        with pytest.raises(TypeError, match="must be a subclass of BaseFetcher"):
            FetcherRegistry.register("invalid", InvalidFetcher)
    
    def test_get_fetcher_class_existing(self):
        """Test getting existing fetcher class."""
        # Arrange
        class MockFetcher(BaseFetcher):
            def fetch(self, *args, **kwargs):
                return Mock()
        
        FetcherRegistry.register("existing_fetcher", MockFetcher)
        
        # Act
        result = FetcherRegistry.get_fetcher_class("existing_fetcher")
        
        # Assert
        assert result == MockFetcher
    
    def test_get_fetcher_class_nonexistent_value_error(self):
        """Test getting non-existent fetcher raises ValueError with available options."""
        # Act & Assert
        with pytest.raises(ValueError, match="Unknown fetcher type: 'nonexistent'"):
            FetcherRegistry.get_fetcher_class("nonexistent")
    
    def test_get_available_types(self):
        """Test getting all registered fetcher types."""
        # Arrange
        class MockFetcher1(BaseFetcher):
            def fetch(self, *args, **kwargs):
                return Mock()
        
        class MockFetcher2(BaseFetcher):
            def fetch(self, *args, **kwargs):
                return Mock()
        
        FetcherRegistry.register("fetcher1", MockFetcher1)
        FetcherRegistry.register("fetcher2", MockFetcher2)
        
        # Act
        types = FetcherRegistry.get_available_types()
        
        # Assert
        assert "fetcher1" in types
        assert "fetcher2" in types
        assert isinstance(types, list)


class TestFetcherRegistryInstanceCreation:
    """Test FetcherRegistry instance creation functionality."""
    
    def test_create_fetcher_instance(self):
        """Test creating fetcher instance with arguments."""
        # Arrange
        class MockFetcher(BaseFetcher):
            def __init__(self, settings, test_arg=None):
                self.settings = settings
                self.test_arg = test_arg
            
            def fetch(self, *args, **kwargs):
                return Mock()
        
        FetcherRegistry.register("creatable_fetcher", MockFetcher)
        mock_settings = Mock(spec=Settings)
        
        # Act
        instance = FetcherRegistry.create("creatable_fetcher", mock_settings, test_arg="test_value")
        
        # Assert
        assert isinstance(instance, MockFetcher)
        assert instance.settings == mock_settings
        assert instance.test_arg == "test_value"
    
    def test_create_nonexistent_fetcher_error(self):
        """Test creating non-existent fetcher raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="Unknown fetcher type"):
            FetcherRegistry.create("nonexistent_fetcher", Mock())
    
    def test_create_error_message_includes_available_types(self):
        """Test that ValueError includes available fetcher types."""
        # Arrange
        class AvailableFetcher(BaseFetcher):
            def fetch(self, *args, **kwargs):
                return Mock()
        
        FetcherRegistry.register("available_fetcher", AvailableFetcher)
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            FetcherRegistry.create("missing_fetcher", Mock())
        
        error_message = str(exc_info.value)
        assert "available_fetcher" in error_message
        assert "Available types:" in error_message


class TestFetcherRegistryIsolation:
    """Test FetcherRegistry isolation and cleanup mechanisms."""
    
    def test_registry_state_isolation(self):
        """Test that registry state is properly isolated between tests."""
        # Arrange
        initial_count = len(FetcherRegistry.fetchers)
        
        class TestFetcher(BaseFetcher):
            def fetch(self, *args, **kwargs):
                return Mock()
        
        # Act
        FetcherRegistry.register("isolation_test", TestFetcher)
        registered_count = len(FetcherRegistry.fetchers)
        
        # Assert
        assert registered_count == initial_count + 1
        # The clean_registries fixture should restore state after test


class TestFetcherRegistryEdgeCases:
    """Test FetcherRegistry edge cases and error scenarios."""
    
    def test_register_duplicate_fetcher_overwrites(self):
        """Test registering duplicate fetcher type overwrites previous."""
        # Arrange
        class Fetcher1(BaseFetcher):
            def fetch(self, *args, **kwargs):
                return "fetcher1"
        
        class Fetcher2(BaseFetcher):
            def fetch(self, *args, **kwargs):
                return "fetcher2"
        
        # Act
        FetcherRegistry.register("duplicate", Fetcher1)
        first_fetcher = FetcherRegistry.get_fetcher_class("duplicate")
        
        FetcherRegistry.register("duplicate", Fetcher2)
        second_fetcher = FetcherRegistry.get_fetcher_class("duplicate")
        
        # Assert
        assert first_fetcher == Fetcher1
        assert second_fetcher == Fetcher2
        assert first_fetcher != second_fetcher
    
    def test_empty_fetcher_type_registration(self):
        """Test registering with empty string fetcher type."""
        # Arrange
        class EmptyTypeFetcher(BaseFetcher):
            def fetch(self, *args, **kwargs):
                return Mock()
        
        # Act
        FetcherRegistry.register("", EmptyTypeFetcher)
        
        # Assert
        assert "" in FetcherRegistry.fetchers
        assert FetcherRegistry.get_fetcher_class("") == EmptyTypeFetcher


class TestFetcherSelfRegistration:
    """Test fetcher self-registration mechanism."""
    
    def test_pass_through_fetcher_self_registration(self):
        """Test that pass-through fetcher automatically registers itself on import."""
        # Act - Import triggers self-registration
        from src.components.fetcher.modules import pass_through_fetcher
        
        # Assert
        assert "pass_through" in FetcherRegistry.fetchers
        assert FetcherRegistry.get_fetcher_class("pass_through").__name__ == "PassThroughFetcher"
    
    def test_feature_store_fetcher_self_registration(self):
        """Test that feature store fetcher automatically registers itself on import."""
        # Act - Import triggers self-registration  
        from src.components.fetcher.modules import feature_store_fetcher
        
        # Assert
        assert "feature_store" in FetcherRegistry.fetchers
        # Get actual class name to avoid naming assumption issues
        fetcher_class = FetcherRegistry.get_fetcher_class("feature_store")
        assert fetcher_class.__name__ in ["FeatureStoreFetcher", "FeastFetcher"]


class TestFetcherRegistryRobustness:
    """Test FetcherRegistry robustness and error recovery."""
    
    def test_multiple_registrations_stability(self):
        """Test that multiple rapid registrations maintain stability."""
        # Arrange
        fetchers_to_register = []
        for i in range(5):
            class TestFetcher(BaseFetcher):
                def __init__(self, index=i):
                    self.index = index
                def fetch(self, *args, **kwargs):
                    return f"fetcher_{i}"
            fetchers_to_register.append((f"test_{i}", TestFetcher))
        
        # Act
        for fetcher_type, fetcher_class in fetchers_to_register:
            FetcherRegistry.register(fetcher_type, fetcher_class)
        
        # Assert
        for i in range(5):
            fetcher_type = f"test_{i}"
            assert fetcher_type in FetcherRegistry.fetchers
            retrieved_class = FetcherRegistry.get_fetcher_class(fetcher_type)
            instance = FetcherRegistry.create(fetcher_type)
            assert instance.fetch() == f"fetcher_{i}"
    
    def test_registry_consistency_after_errors(self):
        """Test that registry remains consistent after registration errors."""
        # Arrange
        class ValidFetcher(BaseFetcher):
            def fetch(self, *args, **kwargs):
                return Mock()
        
        class InvalidFetcher:
            pass
        
        initial_count = len(FetcherRegistry.fetchers)
        
        # Act
        FetcherRegistry.register("valid", ValidFetcher)
        
        try:
            FetcherRegistry.register("invalid", InvalidFetcher)
        except TypeError:
            pass  # Expected error
        
        # Assert
        assert len(FetcherRegistry.fetchers) == initial_count + 1
        assert "valid" in FetcherRegistry.fetchers
        assert "invalid" not in FetcherRegistry.fetchers
    
    def test_get_available_types_returns_copy(self):
        """Test that get_available_types returns a copy, not reference."""
        # Arrange
        class TestFetcher(BaseFetcher):
            def fetch(self, *args, **kwargs):
                return Mock()
        
        FetcherRegistry.register("copy_test", TestFetcher)
        
        # Act
        types1 = FetcherRegistry.get_available_types()
        types2 = FetcherRegistry.get_available_types()
        
        # Assert
        assert types1 == types2
        assert types1 is not types2  # Different list objects
        assert "copy_test" in types1
        
        # Modify returned list shouldn't affect registry
        types1.append("fake_type")
        original_types = FetcherRegistry.get_available_types()
        assert "fake_type" not in original_types
"""
Unit tests for AdapterRegistry.
Tests registry pattern with self-registration mechanism for data adapters.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

from src.components.adapter.registry import AdapterRegistry
from src.interface.base_adapter import BaseAdapter
from src.settings import Settings


class TestAdapterRegistryBasicOperations:
    """Test AdapterRegistry basic CRUD operations."""
    
    def test_register_valid_adapter(self):
        """Test registering a valid adapter class."""
        # Arrange
        class MockAdapter(BaseAdapter):
            def read(self, *args, **kwargs):
                return Mock()
            def write(self, *args, **kwargs):
                pass
        
        # Act
        AdapterRegistry.register("test_adapter", MockAdapter)
        
        # Assert
        assert "test_adapter" in AdapterRegistry.adapters
        assert AdapterRegistry.adapters["test_adapter"] == MockAdapter
    
    def test_register_invalid_adapter_type_error(self):
        """Test registering non-BaseAdapter class raises TypeError."""
        # Arrange
        class InvalidAdapter:
            pass
        
        # Act & Assert
        with pytest.raises(TypeError, match="어댑터 클래스는 BaseAdapter를 상속해야 합니다"):
            AdapterRegistry.register("invalid", InvalidAdapter)
    
    def test_get_adapter_existing(self):
        """Test getting existing adapter class."""
        # Arrange
        class MockAdapter(BaseAdapter):
            def read(self, *args, **kwargs):
                return Mock()
            def write(self, *args, **kwargs):
                pass
        
        AdapterRegistry.register("existing_adapter", MockAdapter)
        
        # Act
        result = AdapterRegistry.get_adapter("existing_adapter")
        
        # Assert
        assert result == MockAdapter
    
    def test_get_adapter_nonexistent_key_error(self):
        """Test getting non-existent adapter raises KeyError with available options."""
        # Act & Assert
        with pytest.raises(KeyError, match="Unknown adapter type: nonexistent"):
            AdapterRegistry.get_adapter("nonexistent")
    
    def test_list_adapters(self):
        """Test listing all registered adapters."""
        # Arrange
        class MockAdapter1(BaseAdapter):
            def read(self, *args, **kwargs):
                return Mock()
            def write(self, *args, **kwargs):
                pass
        
        class MockAdapter2(BaseAdapter):
            def read(self, *args, **kwargs):
                return Mock()
            def write(self, *args, **kwargs):
                pass
        
        AdapterRegistry.register("adapter1", MockAdapter1)
        AdapterRegistry.register("adapter2", MockAdapter2)
        
        # Act
        adapters = AdapterRegistry.list_adapters()
        
        # Assert
        assert "adapter1" in adapters
        assert "adapter2" in adapters
        assert adapters["adapter1"] == MockAdapter1
        assert adapters["adapter2"] == MockAdapter2
        # Ensure it's a copy, not reference
        assert adapters is not AdapterRegistry.adapters


class TestAdapterRegistryInstanceCreation:
    """Test AdapterRegistry instance creation functionality."""
    
    def test_create_adapter_instance(self):
        """Test creating adapter instance with arguments."""
        # Arrange
        class MockAdapter(BaseAdapter):
            def __init__(self, settings, test_arg=None):
                self.settings = settings
                self.test_arg = test_arg
            
            def read(self, *args, **kwargs):
                return Mock()
            
            def write(self, *args, **kwargs):
                pass
        
        AdapterRegistry.register("creatable_adapter", MockAdapter)
        mock_settings = Mock(spec=Settings)
        
        # Act
        instance = AdapterRegistry.create("creatable_adapter", mock_settings, test_arg="test_value")
        
        # Assert
        assert isinstance(instance, MockAdapter)
        assert instance.settings == mock_settings
        assert instance.test_arg == "test_value"
    
    def test_create_nonexistent_adapter_error(self):
        """Test creating non-existent adapter raises KeyError."""
        # Act & Assert
        with pytest.raises(KeyError):
            AdapterRegistry.create("nonexistent_adapter", Mock())


class TestAdapterRegistryIsolation:
    """Test AdapterRegistry isolation and cleanup mechanisms."""
    
    def test_registry_state_isolation(self):
        """Test that registry state is properly isolated between tests."""
        # Arrange
        initial_count = len(AdapterRegistry.adapters)
        
        class TestAdapter(BaseAdapter):
            def read(self, *args, **kwargs):
                return Mock()
            def write(self, *args, **kwargs):
                pass
        
        # Act
        AdapterRegistry.register("isolation_test", TestAdapter)
        registered_count = len(AdapterRegistry.adapters)
        
        # Assert
        assert registered_count == initial_count + 1
        # The clean_registries fixture should restore state after test


class TestAdapterRegistryEdgeCases:
    """Test AdapterRegistry edge cases and error scenarios."""
    
    def test_register_duplicate_adapter_overwrites(self):
        """Test registering duplicate adapter type overwrites previous."""
        # Arrange
        class Adapter1(BaseAdapter):
            def read(self, *args, **kwargs):
                return "adapter1"
            def write(self, *args, **kwargs):
                pass
        
        class Adapter2(BaseAdapter):
            def read(self, *args, **kwargs):
                return "adapter2"
            def write(self, *args, **kwargs):
                pass
        
        # Act
        AdapterRegistry.register("duplicate", Adapter1)
        first_adapter = AdapterRegistry.get_adapter("duplicate")
        
        AdapterRegistry.register("duplicate", Adapter2)
        second_adapter = AdapterRegistry.get_adapter("duplicate")
        
        # Assert
        assert first_adapter == Adapter1
        assert second_adapter == Adapter2
        assert first_adapter != second_adapter
    
    def test_empty_adapter_type_registration(self):
        """Test registering with empty string adapter type."""
        # Arrange
        class EmptyTypeAdapter(BaseAdapter):
            def read(self, *args, **kwargs):
                return Mock()
            def write(self, *args, **kwargs):
                pass
        
        # Act
        AdapterRegistry.register("", EmptyTypeAdapter)
        
        # Assert
        assert "" in AdapterRegistry.adapters
        assert AdapterRegistry.get_adapter("") == EmptyTypeAdapter


class TestAdapterSelfRegistration:
    """Test adapter self-registration mechanism."""
    
    def test_sql_adapter_self_registration(self):
        """Test that SQL adapter automatically registers itself on import."""
        # Act - Import triggers self-registration
        from src.components.adapter.modules import sql_adapter
        
        # Assert
        assert "sql" in AdapterRegistry.adapters
        assert AdapterRegistry.get_adapter("sql").__name__ == "SqlAdapter"
    
    def test_storage_adapter_self_registration(self):
        """Test that storage adapter automatically registers itself on import."""
        # Act - Import triggers self-registration  
        from src.components.adapter.modules import storage_adapter
        
        # Assert
        assert "storage" in AdapterRegistry.adapters
        assert AdapterRegistry.get_adapter("storage").__name__ == "StorageAdapter"
    
    def test_feature_store_adapter_self_registration(self):
        """Test that feature store adapter automatically registers itself on import."""
        # Act - Import triggers self-registration
        from src.components.adapter.modules import feast_adapter
        
        # Assert
        assert "feature_store" in AdapterRegistry.adapters
        assert AdapterRegistry.get_adapter("feature_store").__name__ == "FeastAdapter"


class TestAdapterRegistryRobustness:
    """Test AdapterRegistry robustness and error recovery."""
    
    def test_multiple_registrations_stability(self):
        """Test that multiple rapid registrations maintain stability."""
        # Arrange
        adapters_to_register = []
        for i in range(10):
            class TestAdapter(BaseAdapter):
                def __init__(self, index=i):
                    self.index = index
                def read(self, *args, **kwargs):
                    return f"adapter_{i}"
                def write(self, *args, **kwargs):
                    pass
            adapters_to_register.append((f"test_{i}", TestAdapter))
        
        # Act
        for adapter_type, adapter_class in adapters_to_register:
            AdapterRegistry.register(adapter_type, adapter_class)
        
        # Assert
        for i in range(10):
            adapter_type = f"test_{i}"
            assert adapter_type in AdapterRegistry.adapters
            retrieved_class = AdapterRegistry.get_adapter(adapter_type)
            instance = AdapterRegistry.create(adapter_type)
            assert instance.read() == f"adapter_{i}"
    
    def test_get_adapter_error_message_includes_available(self):
        """Test that KeyError includes available adapter types."""
        # Arrange
        class AvailableAdapter(BaseAdapter):
            def read(self, *args, **kwargs):
                return Mock()
            def write(self, *args, **kwargs):
                pass
        
        AdapterRegistry.register("available_one", AvailableAdapter)
        
        # Act & Assert
        with pytest.raises(KeyError) as exc_info:
            AdapterRegistry.get_adapter("missing")
        
        error_message = str(exc_info.value)
        assert "available_one" in error_message
        assert "Available:" in error_message
"""
Unit tests for DataHandlerRegistry.
Tests registry pattern with self-registration mechanism for data handlers.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

from src.components.datahandler.registry import DataHandlerRegistry
from src.interface import BaseDataHandler
from src.settings import Settings


class TestDataHandlerRegistryBasicOperations:
    """Test DataHandlerRegistry basic CRUD operations."""
    
    def setup_method(self):
        """Clear registry before each test."""
        DataHandlerRegistry.clear()
    
    def test_register_valid_handler(self):
        """Test registering a valid handler class."""
        # Arrange
        class MockDataHandler(BaseDataHandler):
            def prepare_data(self, df):
                return Mock(), Mock(), {}
            def split_data(self, df):
                return Mock(), Mock()
        
        # Act
        DataHandlerRegistry.register("test_handler", MockDataHandler)
        
        # Assert
        assert "test_handler" in DataHandlerRegistry.handlers
        assert DataHandlerRegistry.handlers["test_handler"] == MockDataHandler
    
    def test_register_invalid_handler_raises_error(self):
        """Test that registering non-BaseDataHandler raises TypeError."""
        # Arrange
        class NotAHandler:
            pass
        
        # Act & Assert
        with pytest.raises(TypeError) as exc_info:
            DataHandlerRegistry.register("invalid", NotAHandler)
        assert "must be a subclass of BaseDataHandler" in str(exc_info.value)
    
    def test_create_existing_handler(self):
        """Test creating an instance of registered handler."""
        # Arrange
        class MockDataHandler(BaseDataHandler):
            def __init__(self, settings):
                super().__init__(settings)
                self.test_attr = "created"
            def prepare_data(self, df):
                return Mock(), Mock(), {}
            def split_data(self, df):
                return Mock(), Mock()
        
        mock_settings = Mock(spec=Settings)
        mock_settings.recipe.data.data_interface = Mock()
        
        DataHandlerRegistry.register("test_handler", MockDataHandler)
        
        # Act
        handler = DataHandlerRegistry.create("test_handler", mock_settings)
        
        # Assert
        assert isinstance(handler, MockDataHandler)
        assert handler.test_attr == "created"
        assert handler.settings == mock_settings
    
    def test_create_nonexistent_handler_raises_error(self):
        """Test that creating non-existent handler raises ValueError."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            DataHandlerRegistry.create("nonexistent", mock_settings)
        
        assert "Unknown handler type: 'nonexistent'" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)


class TestDataHandlerRegistryTaskMapping:
    """Test automatic handler mapping for different task types."""
    
    def setup_method(self):
        """Set up mock handlers for testing."""
        DataHandlerRegistry.clear()
        
        # Mock tabular handler
        class MockTabularHandler(BaseDataHandler):
            def prepare_data(self, df):
                return Mock(), Mock(), {}
            def split_data(self, df):
                return Mock(), Mock()
        
        # Mock timeseries handler
        class MockTimeseriesHandler(BaseDataHandler):
            def prepare_data(self, df):
                return Mock(), Mock(), {}
            def split_data(self, df):
                return Mock(), Mock()
        
        DataHandlerRegistry.register("tabular", MockTabularHandler)
        DataHandlerRegistry.register("timeseries", MockTimeseriesHandler)
    
    def test_get_handler_for_classification_task(self):
        """Test that classification maps to tabular handler."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.recipe.data.data_interface = Mock()
        
        # Act
        handler = DataHandlerRegistry.get_handler_for_task("classification", mock_settings)
        
        # Assert
        assert handler.__class__.__name__ == "MockTabularHandler"
    
    def test_get_handler_for_regression_task(self):
        """Test that regression maps to tabular handler."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.recipe.data.data_interface = Mock()
        
        # Act
        handler = DataHandlerRegistry.get_handler_for_task("regression", mock_settings)
        
        # Assert
        assert handler.__class__.__name__ == "MockTabularHandler"
    
    def test_get_handler_for_clustering_task(self):
        """Test that clustering maps to tabular handler."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.recipe.data.data_interface = Mock()
        
        # Act
        handler = DataHandlerRegistry.get_handler_for_task("clustering", mock_settings)
        
        # Assert
        assert handler.__class__.__name__ == "MockTabularHandler"
    
    def test_get_handler_for_causal_task(self):
        """Test that causal maps to tabular handler."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.recipe.data.data_interface = Mock()
        
        # Act
        handler = DataHandlerRegistry.get_handler_for_task("causal", mock_settings)
        
        # Assert
        assert handler.__class__.__name__ == "MockTabularHandler"
    
    def test_get_handler_for_timeseries_task(self):
        """Test that timeseries maps to timeseries handler."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.recipe.data.data_interface = Mock()
        
        # Act
        handler = DataHandlerRegistry.get_handler_for_task("timeseries", mock_settings)
        
        # Assert
        assert handler.__class__.__name__ == "MockTimeseriesHandler"
    
    def test_get_handler_for_unknown_task_falls_back_to_tabular(self):
        """Test that unknown task types fall back to tabular handler."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.recipe.data.data_interface = Mock()
        
        # Act
        handler = DataHandlerRegistry.get_handler_for_task("unknown_task", mock_settings)
        
        # Assert
        assert handler.__class__.__name__ == "MockTabularHandler"


class TestDataHandlerRegistryUtilityMethods:
    """Test utility methods of DataHandlerRegistry."""
    
    def setup_method(self):
        """Set up test handlers."""
        DataHandlerRegistry.clear()
        
        class MockHandler1(BaseDataHandler):
            def prepare_data(self, df):
                return Mock(), Mock(), {}
            def split_data(self, df):
                return Mock(), Mock()
        
        class MockHandler2(BaseDataHandler):
            def prepare_data(self, df):
                return Mock(), Mock(), {}
            def split_data(self, df):
                return Mock(), Mock()
        
        DataHandlerRegistry.register("handler1", MockHandler1)
        DataHandlerRegistry.register("handler2", MockHandler2)
    
    def test_get_available_handlers(self):
        """Test getting list of available handlers."""
        # Act
        available = DataHandlerRegistry.get_available_handlers()
        
        # Assert
        assert isinstance(available, dict)
        assert "handler1" in available
        assert "handler2" in available
        assert available["handler1"] == "MockHandler1"
        assert available["handler2"] == "MockHandler2"
    
    def test_clear_removes_all_handlers(self):
        """Test that clear removes all registered handlers."""
        # Verify handlers exist
        assert len(DataHandlerRegistry.handlers) == 2
        
        # Act
        DataHandlerRegistry.clear()
        
        # Assert
        assert len(DataHandlerRegistry.handlers) == 0
        assert DataHandlerRegistry.get_available_handlers() == {}
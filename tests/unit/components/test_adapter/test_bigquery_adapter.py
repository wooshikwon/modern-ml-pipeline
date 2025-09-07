"""
Unit tests for BigQueryAdapter.
Tests BigQuery adapter functionality with pandas-gbq integration and security features.
"""

import pytest
import pandas as pd
import sys
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from src.components.adapter.modules.bigquery_adapter import BigQueryAdapter
from src.interface.base_adapter import BaseAdapter
from src.settings import Settings


@pytest.fixture(scope="module")
def mock_pandas_gbq_module():
    """Mock pandas-gbq module for all tests."""
    # Create mock pandas_gbq module
    mock_module = MagicMock()
    mock_module.to_gbq = MagicMock()
    
    # Add to sys.modules before any imports
    sys.modules['pandas_gbq'] = mock_module
    
    yield mock_module
    
    # Cleanup after tests
    if 'pandas_gbq' in sys.modules:
        del sys.modules['pandas_gbq']


@pytest.fixture(autouse=True)
def mock_pandas_gbq(mock_pandas_gbq_module):
    """Reset mock pandas-gbq for each test."""
    # Reset the mock before each test
    mock_pandas_gbq_module.to_gbq.reset_mock()
    yield mock_pandas_gbq_module


class TestBigQueryAdapterInitialization:
    """Test BigQueryAdapter initialization and configuration."""
    
    def test_bigquery_adapter_inherits_base_adapter(self):
        """Test that BigQueryAdapter properly inherits from BaseAdapter."""
        # Arrange & Act
        mock_settings = Mock(spec=Settings)
        adapter = BigQueryAdapter(mock_settings)
        
        # Assert
        assert isinstance(adapter, BaseAdapter)
        assert isinstance(adapter, BigQueryAdapter)
    
    def test_init_with_project_id_and_location_kwargs(self):
        """Test initialization with project_id and location in kwargs."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        project_id = "test-project-123"
        location = "us-central1"
        
        # Act
        adapter = BigQueryAdapter(
            mock_settings, 
            project_id=project_id, 
            location=location
        )
        
        # Assert
        assert adapter.settings == mock_settings
        assert adapter._project_id == project_id
        assert adapter._location == location
    
    def test_init_without_kwargs(self):
        """Test initialization without project_id and location."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        
        # Act
        adapter = BigQueryAdapter(mock_settings)
        
        # Assert
        assert adapter.settings == mock_settings
        assert adapter._project_id is None
        assert adapter._location is None
    
    def test_init_handles_exception_gracefully(self):
        """Test initialization handles exceptions gracefully."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        
        # Act - Should not raise exception even if kwargs processing fails
        adapter = BigQueryAdapter(mock_settings)
        
        # Assert
        assert isinstance(adapter, BigQueryAdapter)


class TestBigQueryAdapterReadFunctionality:
    """Test BigQueryAdapter read functionality."""
    
    def test_read_raises_not_implemented_error(self, mock_pandas_gbq):
        """Test that read method raises NotImplementedError."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        adapter = BigQueryAdapter(mock_settings)
        
        # Act & Assert
        with pytest.raises(NotImplementedError, match="BigQueryAdapter.read는 현재 구현되지 않았습니다"):
            adapter.read("SELECT * FROM test_table")


class TestBigQueryAdapterWriteFunctionality:
    """Test BigQueryAdapter write functionality."""
    
    @patch('src.components.adapter.modules.bigquery_adapter.logger')
    def test_write_success_with_all_options(self, mock_logger, mock_pandas_gbq):
        """Test successful write operation with all options."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        adapter = BigQueryAdapter(mock_settings)
        
        test_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        target = "dataset.table"
        options = {
            'project_id': 'test-project',
            'location': 'us-central1',
            'if_exists': 'replace'
        }
        
        # Act
        adapter.write(test_df, target, options)
        
        # Assert
        mock_pandas_gbq.to_gbq.assert_called_once_with(
            dataframe=test_df,
            destination_table=target,
            project_id='test-project',
            if_exists='replace',
            location='us-central1'
        )
        
        # Verify logging
        assert mock_logger.info.call_count == 2
        mock_logger.info.assert_any_call(
            f"BigQuery에 쓰기 시작: target={target}, if_exists=replace, "
            f"project_id=test-project, location=us-central1"
        )
        mock_logger.info.assert_any_call(f"BigQuery 쓰기 완료: rows={len(test_df)}")
    
    @patch('src.components.adapter.modules.bigquery_adapter.logger')
    def test_write_with_options_priority_over_kwargs(self, mock_logger, mock_pandas_gbq):
        """Test write where options have priority over kwargs."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        adapter = BigQueryAdapter(mock_settings)
        
        test_df = pd.DataFrame({'col1': [1, 2, 3]})
        target = "dataset.table"
        options = {'project_id': 'options-project'}
        
        # Act
        adapter.write(
            test_df, 
            target, 
            options, 
            project_id='kwargs-project',
            location='kwargs-location'
        )
        
        # Assert - options should have priority over kwargs
        mock_pandas_gbq.to_gbq.assert_called_once_with(
            dataframe=test_df,
            destination_table=target,
            project_id='options-project',  # options wins over kwargs
            if_exists='append',  # default
            location='kwargs-location'  # location from kwargs since not in options
        )
    
    @patch('src.components.adapter.modules.bigquery_adapter.logger')
    def test_write_with_instance_variables(self, mock_logger, mock_pandas_gbq):
        """Test write using instance variables for project_id and location."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        adapter = BigQueryAdapter(
            mock_settings, 
            project_id='instance-project',
            location='instance-location'
        )
        
        test_df = pd.DataFrame({'col1': [1, 2, 3]})
        target = "dataset.table"
        
        # Act
        adapter.write(test_df, target)
        
        # Assert
        mock_pandas_gbq.to_gbq.assert_called_once_with(
            dataframe=test_df,
            destination_table=target,
            project_id='instance-project',
            if_exists='append',  # default
            location='instance-location'
        )
    
    @patch('src.components.adapter.modules.bigquery_adapter.logger')
    def test_write_with_default_if_exists(self, mock_logger, mock_pandas_gbq):
        """Test write with default if_exists='append'."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        adapter = BigQueryAdapter(mock_settings)
        
        test_df = pd.DataFrame({'col1': [1]})
        target = "dataset.table"
        options = {'project_id': 'test-project'}
        
        # Act
        adapter.write(test_df, target, options)
        
        # Assert
        mock_pandas_gbq.to_gbq.assert_called_once_with(
            dataframe=test_df,
            destination_table=target,
            project_id='test-project',
            if_exists='append',  # default value
            location=None
        )


class TestBigQueryAdapterErrorHandling:
    """Test BigQueryAdapter error handling scenarios."""
    
    def test_write_missing_project_id_raises_value_error(self, mock_pandas_gbq):
        """Test that missing project_id raises ValueError."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        adapter = BigQueryAdapter(mock_settings)
        
        test_df = pd.DataFrame({'col1': [1, 2, 3]})
        target = "dataset.table"
        # No project_id in options, kwargs, or instance variables
        
        # Act & Assert
        with pytest.raises(ValueError, match="BigQuery project_id가 필요합니다"):
            adapter.write(test_df, target)
    
    def test_write_pandas_gbq_import_error(self):
        """Test ImportError when pandas-gbq is not available."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        
        # Temporarily remove pandas-gbq from sys.modules to simulate import error
        original_module = sys.modules.pop('pandas_gbq', None)
        
        try:
            adapter = BigQueryAdapter(mock_settings)
            test_df = pd.DataFrame({'col1': [1, 2, 3]})
            target = "dataset.table"
            options = {'project_id': 'test-project'}
            
            # Act & Assert
            with pytest.raises(ImportError, match="pandas-gbq가 필요합니다"):
                adapter.write(test_df, target, options)
        finally:
            # Restore the mock module
            if original_module is not None:
                sys.modules['pandas_gbq'] = original_module
            else:
                # Re-add our mock module
                mock_module = MagicMock()
                mock_module.to_gbq = MagicMock()
                sys.modules['pandas_gbq'] = mock_module
    
    def test_read_pandas_gbq_import_error_in_read(self):
        """Test ImportError handling in read method.""" 
        # Arrange
        mock_settings = Mock(spec=Settings)
        
        # Temporarily remove pandas-gbq from sys.modules to simulate import error
        original_module = sys.modules.pop('pandas_gbq', None)
        
        try:
            adapter = BigQueryAdapter(mock_settings)
            
            # Act & Assert
            with pytest.raises(ImportError, match="pandas-gbq가 필요합니다"):
                adapter.read("SELECT * FROM test")
        finally:
            # Restore the mock module
            if original_module is not None:
                sys.modules['pandas_gbq'] = original_module
            else:
                # Re-add our mock module
                mock_module = MagicMock()
                mock_module.to_gbq = MagicMock()
                sys.modules['pandas_gbq'] = mock_module


class TestBigQueryAdapterRegistration:
    """Test BigQueryAdapter registry integration."""
    
    def test_adapter_registry_registration(self):
        """Test that BigQueryAdapter is registered in AdapterRegistry."""
        # Arrange
        from src.components.adapter.registry import AdapterRegistry
        
        # Act
        registered_adapters = AdapterRegistry.list_adapters()
        
        # Assert
        assert "bigquery" in registered_adapters
        
        # Verify we can create the adapter through registry
        mock_settings = Mock(spec=Settings)
        adapter = AdapterRegistry.create("bigquery", mock_settings)
        assert isinstance(adapter, BigQueryAdapter)
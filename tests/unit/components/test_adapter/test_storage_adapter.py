"""
Unit tests for StorageAdapter.
Tests storage adapter functionality with local/cloud storage support.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path

from src.components.adapter.modules.storage_adapter import StorageAdapter
from src.interface.base_adapter import BaseAdapter
from src.settings import Settings


class TestStorageAdapterInitialization:
    """Test StorageAdapter initialization."""
    
    def test_storage_adapter_inherits_base_adapter(self):
        """Test that StorageAdapter properly inherits from BaseAdapter."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': {}})
        }
        
        # Act
        adapter = StorageAdapter(mock_settings)
        
        # Assert
        assert isinstance(adapter, BaseAdapter)
        assert isinstance(adapter, StorageAdapter)
    
    def test_init_with_storage_config_success(self):
        """Test successful initialization with storage configuration."""
        # Arrange
        storage_options = {
            'aws_access_key_id': 'test_key',
            'aws_secret_access_key': 'test_secret'
        }
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': storage_options})
        }
        
        # Act
        adapter = StorageAdapter(mock_settings)
        
        # Assert
        assert adapter.storage_options == storage_options
        assert hasattr(adapter, 'settings')
    
    def test_init_missing_storage_config_uses_defaults(self):
        """Test initialization uses defaults when storage config is missing."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {}  # No 'storage' config
        
        # Act
        with patch('src.components.adapter.modules.storage_adapter.logger') as mock_logger:
            adapter = StorageAdapter(mock_settings)
        
        # Assert
        assert adapter.storage_options == {}
        mock_logger.warning.assert_called_with(
            "Storage 어댑터 설정을 찾을 수 없습니다: 'storage'. 기본값 사용."
        )
    
    def test_init_config_access_error_uses_defaults(self):
        """Test initialization handles config access errors gracefully."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={})  # Missing storage_options key
        }
        
        # Act
        adapter = StorageAdapter(mock_settings)
        
        # Assert
        assert adapter.storage_options == {}


class TestStorageAdapterReadOperations:
    """Test storage read operations."""
    
    @patch('pandas.read_csv')
    def test_read_csv_file_success(self, mock_read_csv):
        """Test successful CSV file reading."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': {'key': 'value'}})
        }
        adapter = StorageAdapter(mock_settings)
        
        expected_df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        mock_read_csv.return_value = expected_df
        uri = "s3://bucket/data.csv"
        
        # Act
        result = adapter.read(uri)
        
        # Assert
        assert result.equals(expected_df)
        mock_read_csv.assert_called_once_with(
            uri, 
            storage_options={'key': 'value'}
        )
    
    @patch('pandas.read_parquet')
    def test_read_parquet_file_success(self, mock_read_parquet):
        """Test successful Parquet file reading."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': {}})
        }
        adapter = StorageAdapter(mock_settings)
        
        expected_df = pd.DataFrame({'num': [1, 2, 3], 'text': ['x', 'y', 'z']})
        mock_read_parquet.return_value = expected_df
        uri = "gs://bucket/data.parquet"
        
        # Act
        result = adapter.read(uri)
        
        # Assert
        assert result.equals(expected_df)
        mock_read_parquet.assert_called_once_with(
            uri,
            storage_options={}
        )
    
    @patch('pandas.read_csv')
    def test_read_with_additional_kwargs(self, mock_read_csv):
        """Test read method passes through additional kwargs."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': {}})
        }
        adapter = StorageAdapter(mock_settings)
        
        expected_df = pd.DataFrame({'data': [1, 2]})
        mock_read_csv.return_value = expected_df
        uri = "data.csv"
        
        # Act
        result = adapter.read(
            uri,
            sep=';',
            encoding='utf-8',
            skiprows=1
        )
        
        # Assert
        assert result.equals(expected_df)
        mock_read_csv.assert_called_once_with(
            uri,
            storage_options={},
            sep=';',
            encoding='utf-8',
            skiprows=1
        )
    
    @patch('pandas.read_parquet')
    def test_read_file_extension_detection(self, mock_read_parquet):
        """Test file format detection by extension."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': {}})
        }
        adapter = StorageAdapter(mock_settings)
        
        expected_df = pd.DataFrame({'test': [1]})
        mock_read_parquet.return_value = expected_df
        
        test_extensions = [
            "data.parquet",
            "data.PARQUET", 
            "data.pq",
            "s3://bucket/file.parquet",
            "/local/path/file.parquet"
        ]
        
        # Act & Assert
        for uri in test_extensions:
            result = adapter.read(uri)
            assert result.equals(expected_df)
        
        assert mock_read_parquet.call_count == len(test_extensions)
    
    def test_read_error_handling(self):
        """Test read operation error handling."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': {}})
        }
        adapter = StorageAdapter(mock_settings)
        
        # Act & Assert
        with patch('pandas.read_csv', side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError, match="File not found"):
                adapter.read("missing.csv")


class TestStorageAdapterWriteOperations:
    """Test storage write operations."""
    
    @patch('pandas.DataFrame.to_csv')
    def test_write_csv_file_success(self, mock_to_csv):
        """Test successful CSV file writing."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': {}})
        }
        adapter = StorageAdapter(mock_settings)
        
        df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        uri = "output.csv"
        
        # Act
        adapter.write(df, uri)
        
        # Assert
        mock_to_csv.assert_called_once_with(uri, index=False)
    
    @patch('pandas.DataFrame.to_parquet')
    def test_write_parquet_file_success(self, mock_to_parquet):
        """Test successful Parquet file writing."""
        # Arrange
        storage_options = {'aws_access_key_id': 'test'}
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': storage_options})
        }
        adapter = StorageAdapter(mock_settings)
        
        df = pd.DataFrame({'num': [1, 2, 3]})
        uri = "s3://bucket/output.parquet"
        
        # Act
        adapter.write(df, uri, compression='snappy')
        
        # Assert
        mock_to_parquet.assert_called_once_with(
            uri,
            storage_options=storage_options,
            compression='snappy'
        )
    
    @patch('pandas.DataFrame.to_csv')
    @patch('pathlib.Path.mkdir')
    def test_write_local_file_creates_directory(self, mock_mkdir, mock_to_csv):
        """Test that local file writing creates parent directories."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': {}})
        }
        adapter = StorageAdapter(mock_settings)
        
        df = pd.DataFrame({'data': [1]})
        uri = "/path/to/output.csv"
        
        # Act
        adapter.write(df, uri)
        
        # Assert
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_to_csv.assert_called_once_with(uri, index=False)
    
    @patch('pandas.DataFrame.to_csv')
    @patch('pathlib.Path.mkdir')
    def test_write_file_uri_creates_directory(self, mock_mkdir, mock_to_csv):
        """Test that file:// URI writing creates parent directories."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': {}})
        }
        adapter = StorageAdapter(mock_settings)
        
        df = pd.DataFrame({'data': [1]})
        uri = "file:///path/to/output.csv"
        
        # Act
        adapter.write(df, uri)
        
        # Assert
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_to_csv.assert_called_once_with(uri, index=False)
    
    @patch('pandas.DataFrame.to_parquet')
    def test_write_cloud_storage_no_directory_creation(self, mock_to_parquet):
        """Test that cloud storage URIs don't trigger directory creation."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': {}})
        }
        adapter = StorageAdapter(mock_settings)
        
        df = pd.DataFrame({'data': [1]})
        cloud_uris = [
            "s3://bucket/path/file.parquet",
            "gs://bucket/path/file.parquet",
            "azure://container/path/file.parquet"
        ]
        
        # Act & Assert
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            for uri in cloud_uris:
                adapter.write(df, uri)
                mock_mkdir.assert_not_called()
        
        assert mock_to_parquet.call_count == len(cloud_uris)
    
    def test_write_error_handling(self):
        """Test write operation error handling."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': {}})
        }
        adapter = StorageAdapter(mock_settings)
        
        df = pd.DataFrame({'data': [1]})
        
        # Act & Assert
        with patch('pandas.DataFrame.to_csv', side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError, match="Access denied"):
                adapter.write(df, "protected.csv")


class TestStorageAdapterFileTypeDetection:
    """Test file type detection logic."""
    
    def test_csv_detection_case_insensitive(self):
        """Test CSV detection is case insensitive."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': {}})
        }
        adapter = StorageAdapter(mock_settings)
        
        csv_variants = [
            "data.csv",
            "data.CSV", 
            "Data.Csv",
            "/path/file.csv",
            "s3://bucket/data.CSV"
        ]
        
        # Act & Assert
        with patch('pandas.read_csv', return_value=pd.DataFrame()) as mock_read_csv:
            for uri in csv_variants:
                adapter.read(uri)
            assert mock_read_csv.call_count == len(csv_variants)
    
    def test_parquet_default_for_non_csv(self):
        """Test that non-CSV files default to Parquet reading."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': {}})
        }
        adapter = StorageAdapter(mock_settings)
        
        non_csv_files = [
            "data.parquet",
            "data.pq",
            "data.json",  # Not CSV -> defaults to Parquet
            "data.txt",   # Not CSV -> defaults to Parquet
            "data"        # No extension -> defaults to Parquet
        ]
        
        # Act & Assert
        with patch('pandas.read_parquet', return_value=pd.DataFrame()) as mock_read_parquet:
            for uri in non_csv_files:
                adapter.read(uri)
            assert mock_read_parquet.call_count == len(non_csv_files)


class TestStorageAdapterIntegration:
    """Test StorageAdapter integration scenarios."""
    
    def test_storage_adapter_with_various_kwargs(self):
        """Test StorageAdapter handles various initialization kwargs."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': {}})
        }
        
        # Act - Test that StorageAdapter can be initialized normally
        adapter = StorageAdapter(mock_settings)
        
        # Assert
        assert adapter.settings == mock_settings
        assert adapter.storage_options == {}
        assert hasattr(adapter, 'settings')
    
    def test_storage_adapter_logging_behavior(self):
        """Test that StorageAdapter properly logs operations."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': {}})
        }
        adapter = StorageAdapter(mock_settings)
        
        with patch('src.components.adapter.modules.storage_adapter.logger') as mock_logger:
            df = pd.DataFrame({'data': [1]})
            uri = "test.csv"
            
            with patch('pandas.read_csv', return_value=df):
                adapter.read(uri)
            
            with patch('pandas.DataFrame.to_csv'):
                adapter.write(df, uri)
        
        # Assert
        mock_logger.info.assert_called()
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("StorageAdapter read from" in log for log in log_calls)
        assert any("StorageAdapter write to" in log for log in log_calls)
    
    def test_complex_storage_options_handling(self):
        """Test complex storage options are properly handled."""
        # Arrange
        complex_storage_options = {
            'aws_access_key_id': 'test_key',
            'aws_secret_access_key': 'test_secret',
            'aws_session_token': 'temp_token',
            'region_name': 'us-west-2',
            'endpoint_url': 'https://custom-s3.example.com'
        }
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': complex_storage_options})
        }
        adapter = StorageAdapter(mock_settings)
        
        # Act & Assert
        with patch('pandas.read_parquet') as mock_read_parquet:
            adapter.read("s3://bucket/data.parquet", columns=['col1', 'col2'])
            mock_read_parquet.assert_called_once_with(
                "s3://bucket/data.parquet",
                storage_options=complex_storage_options,
                columns=['col1', 'col2']
            )


class TestStorageAdapterSelfRegistration:
    """Test StorageAdapter self-registration mechanism."""
    
    def test_storage_adapter_self_registration(self):
        """Test that StorageAdapter registers itself in AdapterRegistry."""
        # Act - Import triggers self-registration
        from src.components.adapter.modules import storage_adapter
        from src.components.adapter.registry import AdapterRegistry
        
        # Assert
        assert "storage" in AdapterRegistry.adapters
        assert AdapterRegistry.adapters["storage"] == StorageAdapter
        
        # Verify can create instance through registry
        mock_settings = Mock(spec=Settings)
        mock_settings.data_adapters = Mock()
        mock_settings.data_adapters.adapters = {
            'storage': Mock(config={'storage_options': {}})
        }
        instance = AdapterRegistry.create("storage", mock_settings)
        assert isinstance(instance, StorageAdapter)
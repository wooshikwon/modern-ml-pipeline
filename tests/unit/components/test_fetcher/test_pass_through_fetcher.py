"""
Unit tests for PassThroughFetcher.
Tests pass-through fetcher functionality that bypasses feature augmentation.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from src.components.fetcher.modules.pass_through_fetcher import PassThroughFetcher
from src.interface.base_fetcher import BaseFetcher


class TestPassThroughFetcherInitialization:
    """Test PassThroughFetcher initialization."""
    
    def test_pass_through_fetcher_inherits_base_fetcher(self):
        """Test that PassThroughFetcher properly inherits from BaseFetcher."""
        # Act
        fetcher = PassThroughFetcher()
        
        # Assert
        assert isinstance(fetcher, BaseFetcher)
        assert isinstance(fetcher, PassThroughFetcher)
    
    def test_init_no_arguments_required(self):
        """Test that PassThroughFetcher can be initialized without arguments."""
        # Act
        fetcher = PassThroughFetcher()
        
        # Assert
        assert fetcher is not None
        assert hasattr(fetcher, 'fetch')


class TestPassThroughFetcherFetchOperations:
    """Test pass-through fetch operations."""
    
    def test_fetch_train_mode_success(self):
        """Test successful fetch operation in train mode."""
        # Arrange
        fetcher = PassThroughFetcher()
        input_df = pd.DataFrame({
            'id': [1, 2, 3],
            'feature1': ['a', 'b', 'c'],
            'feature2': [10, 20, 30]
        })
        
        # Act
        with patch('src.components.fetcher.modules.pass_through_fetcher.logger') as mock_logger:
            result = fetcher.fetch(input_df, run_mode="train")
        
        # Assert
        pd.testing.assert_frame_equal(result, input_df)
        mock_logger.info.assert_called_once_with(
            "피처 증강을 건너뜁니다. ('passthrough' 모드 또는 레시피에 fetcher 미정의)"
        )
    
    def test_fetch_batch_mode_success(self):
        """Test successful fetch operation in batch mode."""
        # Arrange
        fetcher = PassThroughFetcher()
        input_df = pd.DataFrame({
            'entity_id': [100, 200],
            'timestamp': ['2023-01-01', '2023-01-02'],
            'value': [1.5, 2.5]
        })
        
        # Act
        result = fetcher.fetch(input_df, run_mode="batch")
        
        # Assert
        pd.testing.assert_frame_equal(result, input_df)
        # PassThroughFetcher returns the same DataFrame object (no copy)
        assert result is input_df  # Same object reference for efficiency
        assert result.equals(input_df)  # Same content
    
    def test_fetch_serving_mode_success(self):
        """Test successful fetch operation in serving mode."""
        # Arrange
        fetcher = PassThroughFetcher()
        input_df = pd.DataFrame({'user_id': [42], 'session': ['abc123']})
        
        # Act
        with patch('src.components.fetcher.modules.pass_through_fetcher.logger') as mock_logger:
            result = fetcher.fetch(input_df, run_mode="serving")
        
        # Assert
        pd.testing.assert_frame_equal(result, input_df)
        mock_logger.info.assert_called_once()
    
    def test_fetch_default_mode_success(self):
        """Test fetch operation with default batch mode."""
        # Arrange
        fetcher = PassThroughFetcher()
        input_df = pd.DataFrame({'col1': [1], 'col2': [2]})
        
        # Act
        result = fetcher.fetch(input_df)  # No run_mode specified
        
        # Assert
        pd.testing.assert_frame_equal(result, input_df)
    
    def test_fetch_invalid_run_mode_raises_error(self):
        """Test that invalid run_mode raises ValueError."""
        # Arrange
        fetcher = PassThroughFetcher()
        input_df = pd.DataFrame({'data': [1, 2, 3]})
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid run_mode 'invalid'. Valid modes: \\['train', 'batch', 'serving'\\]"):
            fetcher.fetch(input_df, run_mode="invalid")
    
    def test_fetch_empty_dataframe_success(self):
        """Test fetch operation with empty DataFrame."""
        # Arrange
        fetcher = PassThroughFetcher()
        empty_df = pd.DataFrame()
        
        # Act
        result = fetcher.fetch(empty_df, run_mode="train")
        
        # Assert
        pd.testing.assert_frame_equal(result, empty_df)
        assert len(result) == 0
    
    def test_fetch_preserves_dataframe_structure(self):
        """Test that fetch preserves DataFrame structure and metadata."""
        # Arrange
        fetcher = PassThroughFetcher()
        input_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })
        input_df.index = ['row1', 'row2', 'row3']
        
        # Act
        result = fetcher.fetch(input_df, run_mode="batch")
        
        # Assert
        pd.testing.assert_frame_equal(result, input_df)
        assert list(result.columns) == list(input_df.columns)
        assert list(result.index) == list(input_df.index)
        assert result.dtypes.equals(input_df.dtypes)


class TestPassThroughFetcherValidation:
    """Test PassThroughFetcher validation and error handling."""
    
    def test_fetch_all_valid_run_modes(self):
        """Test that all valid run modes are accepted."""
        # Arrange
        fetcher = PassThroughFetcher()
        input_df = pd.DataFrame({'test': [1]})
        valid_modes = ["train", "batch", "serving"]
        
        # Act & Assert
        for mode in valid_modes:
            result = fetcher.fetch(input_df, run_mode=mode)
            pd.testing.assert_frame_equal(result, input_df)
    
    def test_fetch_case_sensitive_run_mode(self):
        """Test that run_mode validation is case-sensitive."""
        # Arrange
        fetcher = PassThroughFetcher()
        input_df = pd.DataFrame({'test': [1]})
        
        # Act & Assert
        with pytest.raises(ValueError):
            fetcher.fetch(input_df, run_mode="TRAIN")  # Uppercase should fail
        
        with pytest.raises(ValueError):
            fetcher.fetch(input_df, run_mode="Batch")  # Title case should fail


class TestPassThroughFetcherLogging:
    """Test PassThroughFetcher logging behavior."""
    
    def test_fetch_logs_passthrough_message(self):
        """Test that fetch operation logs appropriate message."""
        # Arrange
        fetcher = PassThroughFetcher()
        input_df = pd.DataFrame({'data': [1, 2, 3]})
        
        # Act
        with patch('src.components.fetcher.modules.pass_through_fetcher.logger') as mock_logger:
            fetcher.fetch(input_df, run_mode="train")
        
        # Assert
        mock_logger.info.assert_called_once_with(
            "피처 증강을 건너뜁니다. ('passthrough' 모드 또는 레시피에 fetcher 미정의)"
        )
    
    def test_fetch_logs_for_all_run_modes(self):
        """Test that logging occurs for all valid run modes."""
        # Arrange
        fetcher = PassThroughFetcher()
        input_df = pd.DataFrame({'test': [1]})
        valid_modes = ["train", "batch", "serving"]
        
        # Act & Assert
        for mode in valid_modes:
            with patch('src.components.fetcher.modules.pass_through_fetcher.logger') as mock_logger:
                fetcher.fetch(input_df, run_mode=mode)
                mock_logger.info.assert_called_once()


class TestPassThroughFetcherSelfRegistration:
    """Test PassThroughFetcher self-registration mechanism."""
    
    def test_pass_through_fetcher_self_registration(self):
        """Test that PassThroughFetcher registers itself in FetcherRegistry."""
        # Act - Import triggers self-registration
        from src.components.fetcher.modules import pass_through_fetcher
        from src.components.fetcher.registry import FetcherRegistry
        
        # Assert
        assert "pass_through" in FetcherRegistry.fetchers
        assert FetcherRegistry.fetchers["pass_through"] == PassThroughFetcher
        
        # Verify can create instance through registry
        instance = FetcherRegistry.create("pass_through")
        assert isinstance(instance, PassThroughFetcher)


class TestPassThroughFetcherIntegration:
    """Test PassThroughFetcher integration scenarios."""
    
    def test_pass_through_fetcher_with_complex_dataframe(self):
        """Test PassThroughFetcher with complex DataFrame structures."""
        # Arrange
        fetcher = PassThroughFetcher()
        complex_df = pd.DataFrame({
            'user_id': [1001, 1002, 1003],
            'session_id': ['sess_001', 'sess_002', 'sess_003'],
            'timestamp': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 11:00:00', '2023-01-01 12:00:00']),
            'feature_vector': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            'metadata': [{'source': 'web'}, {'source': 'mobile'}, {'source': 'api'}]
        })
        
        # Act
        result = fetcher.fetch(complex_df, run_mode="batch")
        
        # Assert
        pd.testing.assert_frame_equal(result, complex_df)
        # Verify complex data types are preserved
        assert result['timestamp'].dtype == complex_df['timestamp'].dtype
        assert result['feature_vector'].iloc[0] == [0.1, 0.2, 0.3]
        assert result['metadata'].iloc[0] == {'source': 'web'}
    
    def test_pass_through_fetcher_performance_characteristics(self):
        """Test that PassThroughFetcher has minimal performance impact."""
        # Arrange
        fetcher = PassThroughFetcher()
        large_df = pd.DataFrame({
            'id': range(1000),
            'feature1': range(1000, 2000),
            'feature2': [f'value_{i}' for i in range(1000)]
        })
        
        # Act
        import time
        start_time = time.time()
        result = fetcher.fetch(large_df, run_mode="batch")
        end_time = time.time()
        
        # Assert
        pd.testing.assert_frame_equal(result, large_df)
        # Should complete very quickly (less than 1 second for 1000 rows)
        assert (end_time - start_time) < 1.0
        assert len(result) == 1000
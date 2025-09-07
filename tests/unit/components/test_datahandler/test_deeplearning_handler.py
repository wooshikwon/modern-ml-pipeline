"""
Unit tests for DeepLearningDataHandler.
Tests deep learning specific data processing (sequence generation, 3D transformations).
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from src.components.datahandler.modules.deeplearning_handler import DeepLearningDataHandler
from src.interface import BaseDataHandler
from src.settings import Settings
from tests.helpers.dataframe_builder import DataFrameBuilder
from tests.helpers.recipe_builder import RecipeBuilder
from tests.helpers.config_builder import ConfigBuilder


class TestDeepLearningHandlerInitialization:
    """Test DeepLearningDataHandler initialization."""
    
    def test_deeplearning_handler_inherits_base_datahandler(self):
        """Test DeepLearningDataHandler properly inherits BaseDataHandler."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_data_interface.task_type = "timeseries"
        mock_data_interface.sequence_length = 30
        mock_data_interface.use_gpu = True
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        # Act
        handler = DeepLearningDataHandler(mock_settings)
        
        # Assert
        assert isinstance(handler, BaseDataHandler)
        assert isinstance(handler, DeepLearningDataHandler)
        assert hasattr(handler, 'prepare_data')
        assert hasattr(handler, 'validate_data')
    
    def test_deeplearning_handler_initialization_with_explicit_values(self):
        """Test DeepLearningDataHandler initialization with explicit values."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_data_interface.task_type = "classification"
        mock_data_interface.sequence_length = 50
        mock_data_interface.use_gpu = False
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        # Act
        handler = DeepLearningDataHandler(mock_settings)
        
        # Assert
        assert handler.settings == mock_settings
        assert handler.data_interface == mock_data_interface
        assert handler.task_type == "classification"
        assert handler.sequence_length == 50  # explicit value
        assert handler.use_gpu == False  # explicit value
    
    @patch('src.components.datahandler.modules.deeplearning_handler.logger')
    def test_deeplearning_handler_initialization_logging(self, mock_logger):
        """Test proper logging during initialization."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_data_interface.task_type = "timeseries"
        mock_data_interface.sequence_length = 50
        mock_data_interface.use_gpu = False
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        # Act
        handler = DeepLearningDataHandler(mock_settings)
        
        # Assert
        mock_logger.info.assert_any_call("🧠 DeepLearning DataHandler 초기화")
        mock_logger.info.assert_any_call("   Task Type: timeseries")
        mock_logger.info.assert_any_call("   Sequence Length: 50")
        mock_logger.info.assert_any_call("   Use GPU: False")


class TestDeepLearningHandlerValidation:
    """Test DeepLearningDataHandler data validation."""
    
    def test_validate_data_basic_validation(self):
        """Test basic data validation functionality."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_data_interface.task_type = "classification"
        mock_data_interface.target_column = "target"
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        handler = DeepLearningDataHandler(mock_settings)
        
        # Mock parent validate_data
        with patch.object(BaseDataHandler, 'validate_data', return_value=True):
            test_df = pd.DataFrame({
                'feature1': [1, 2, 3, 4, 5],
                'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
                'target': [0, 1, 0, 1, 1]
            })
            
            # Act
            result = handler.validate_data(test_df)
            
            # Assert
            assert result is True
    
    def test_validate_data_missing_target_column(self):
        """Test validation fails when target column is missing."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_data_interface.task_type = "regression"
        mock_data_interface.target_column = "missing_target"
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        handler = DeepLearningDataHandler(mock_settings)
        
        # Mock parent validate_data
        with patch.object(BaseDataHandler, 'validate_data', return_value=True):
            test_df = pd.DataFrame({
                'feature1': [1, 2, 3, 4, 5],
                'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
            })
            
            # Act & Assert
            with pytest.raises(ValueError, match="Target 컬럼 'missing_target'을 찾을 수 없습니다"):
                handler.validate_data(test_df)
    
    @patch('src.components.datahandler.modules.deeplearning_handler.logger')
    def test_validate_data_timeseries_missing_timestamp(self, mock_logger):
        """Test timeseries validation fails when timestamp column is missing."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_data_interface.task_type = "timeseries"
        mock_data_interface.target_column = "target"
        mock_data_interface.timestamp_column = None  # Missing timestamp
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        handler = DeepLearningDataHandler(mock_settings)
        
        # Mock parent validate_data
        with patch.object(BaseDataHandler, 'validate_data', return_value=True):
            test_df = pd.DataFrame({
                'feature1': [1, 2, 3, 4, 5],
                'target': [0.1, 0.2, 0.3, 0.4, 0.5]
            })
            
            # Act & Assert
            with pytest.raises(ValueError, match="TimeSeries task에 필요한 timestamp_column"):
                handler.validate_data(test_df)
    
    @patch('src.components.datahandler.modules.deeplearning_handler.logger')
    def test_validate_data_timeseries_datetime_conversion(self, mock_logger):
        """Test timestamp column is properly converted to datetime."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_data_interface.task_type = "timeseries"
        mock_data_interface.target_column = "target"
        mock_data_interface.timestamp_column = "timestamp"
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        handler = DeepLearningDataHandler(mock_settings)
        handler.sequence_length = 2  # Small sequence for testing
        
        # Mock parent validate_data
        with patch.object(BaseDataHandler, 'validate_data', return_value=True):
            test_df = pd.DataFrame({
                'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],
                'feature1': [1, 2, 3],
                'target': [0.1, 0.2, 0.3]
            })
            
            # Act
            result = handler.validate_data(test_df)
            
            # Assert
            assert result is True
            assert pd.api.types.is_datetime64_any_dtype(test_df['timestamp'])
            mock_logger.info.assert_any_call("Timestamp 컬럼을 datetime으로 변환했습니다: timestamp")
    
    def test_validate_data_timeseries_insufficient_data(self):
        """Test validation fails when there's insufficient data for sequences."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_data_interface.task_type = "timeseries"
        mock_data_interface.target_column = "target"
        mock_data_interface.timestamp_column = "timestamp"
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        handler = DeepLearningDataHandler(mock_settings)
        handler.sequence_length = 30  # Default sequence length
        
        # Mock parent validate_data
        with patch.object(BaseDataHandler, 'validate_data', return_value=True):
            # Only 5 rows, but need 31 (sequence_length + 1)
            test_df = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=5),
                'feature1': [1, 2, 3, 4, 5],
                'target': [0.1, 0.2, 0.3, 0.4, 0.5]
            })
            
            # Act & Assert
            with pytest.raises(ValueError, match="시퀀스 생성을 위해 최소 31개 행이 필요합니다"):
                handler.validate_data(test_df)


class TestDeepLearningHandlerTimeseriesProcessing:
    """Test DeepLearning Handler timeseries sequence processing."""
    
    def test_prepare_timeseries_sequences_basic(self):
        """Test basic timeseries sequence generation."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_data_interface.task_type = "timeseries"
        mock_data_interface.target_column = "target"
        mock_data_interface.timestamp_column = "timestamp"
        mock_data_interface.entity_columns = None
        mock_data_interface.feature_columns = None  # Auto-select
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        handler = DeepLearningDataHandler(mock_settings)
        handler.sequence_length = 3
        
        # Mock parent validate_data
        with patch.object(handler, 'validate_data', return_value=True):
            test_df = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=6),
                'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                'target': [10, 20, 30, 40, 50, 60]
            })
            
            # Act
            X_result, y_result, metadata = handler.prepare_data(test_df)
            
            # Assert - Focus on basic functionality
            assert isinstance(X_result, pd.DataFrame)
            assert isinstance(y_result, pd.Series)
            assert len(X_result) == len(y_result)  # Same number of samples
            assert len(X_result) > 0  # Should produce some sequences
            
            # Check that metadata contains shape information
            assert 'data_shape' in metadata
            data_shape = metadata['data_shape']
            assert len(data_shape) == 2  # (n_samples, flattened_features)
            # For timeseries, should have > 0 samples and features for sequence_length * n_features  
            assert data_shape[0] > 0  # Should have some samples
            assert data_shape[1] > 0  # Should have some features
    
    @patch('src.components.datahandler.modules.deeplearning_handler.logger')
    def test_prepare_timeseries_sequences_with_explicit_features(self, mock_logger):
        """Test timeseries processing with explicitly specified feature columns."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_data_interface.task_type = "timeseries"
        mock_data_interface.target_column = "target"
        mock_data_interface.timestamp_column = "timestamp"
        mock_data_interface.entity_columns = None
        mock_data_interface.feature_columns = ["feature1"]  # Explicit selection
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        handler = DeepLearningDataHandler(mock_settings)
        handler.sequence_length = 2
        
        with patch.object(handler, 'validate_data', return_value=True):
            test_df = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=5),
                'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
                'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],  # Should be ignored
                'target': [10, 20, 30, 40, 50]
            })
            
            # Act
            X_result, y_result, metadata = handler.prepare_data(test_df)
            
            # Assert
            assert len(X_result) == 3  # sequences: [0,1]->2, [1,2]->3, [2,3]->4
            assert X_result.shape[1] == 2  # seq_len * n_features = 2 * 1
            assert list(X_result.columns) == ['seq0_feat0', 'seq1_feat0']
            mock_logger.info.assert_any_call("📈 TimeSeries feature columns (1): ['feature1']")
    
    @patch('src.components.datahandler.modules.deeplearning_handler.logger')
    def test_prepare_timeseries_sequences_with_missing_values(self, mock_logger):
        """Test sequence generation with missing values."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_data_interface.task_type = "timeseries"
        mock_data_interface.target_column = "target"
        mock_data_interface.timestamp_column = "timestamp"
        mock_data_interface.entity_columns = None
        mock_data_interface.feature_columns = None
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        handler = DeepLearningDataHandler(mock_settings)
        handler.sequence_length = 2
        
        with patch.object(handler, 'validate_data', return_value=True):
            test_df = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=5),
                'feature1': [1.0, np.nan, 3.0, 4.0, 5.0],  # Missing value at index 1
                'target': [10, 20, 30, 40, 50]
            })
            
            # Act
            X_result, y_result, metadata = handler.prepare_data(test_df)
            
            # Assert - Focus on that missing values are handled
            assert isinstance(X_result, pd.DataFrame)
            assert isinstance(y_result, pd.Series)
            # Missing values should reduce the number of valid sequences
            assert len(X_result) < 3  # Less than the maximum possible sequences
    
    @patch('src.components.datahandler.modules.deeplearning_handler.logger')
    def test_prepare_timeseries_sequences_missing_value_warning(self, mock_logger):
        """Test warning is logged for features with high missing value ratio."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_data_interface.task_type = "timeseries"
        mock_data_interface.target_column = "target"
        mock_data_interface.timestamp_column = "timestamp"
        mock_data_interface.entity_columns = None
        mock_data_interface.feature_columns = None
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        handler = DeepLearningDataHandler(mock_settings)
        handler.sequence_length = 2
        
        with patch.object(handler, 'validate_data', return_value=True):
            # Create data with >5% missing values
            test_df = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=10),
                'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                'feature2': [1.0, np.nan, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # 20% missing
                'target': list(range(10))
            })
            
            # Act
            X_result, y_result, metadata = handler.prepare_data(test_df)
            
            # Assert
            mock_logger.warning.assert_any_call("⚠️  결측치가 많은 feature 컬럼이 발견되었습니다:")
            # Check if feature2 warning is logged
            calls = [call for call in mock_logger.warning.call_args_list if 'feature2' in str(call)]
            assert len(calls) > 0


class TestDeepLearningHandlerTabularProcessing:
    """Test DeepLearning Handler tabular data processing."""
    
    def test_prepare_tabular_data_classification(self):
        """Test tabular data preparation for classification."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_data_interface.task_type = "classification"
        mock_data_interface.target_column = "target"
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        handler = DeepLearningDataHandler(mock_settings)
        
        with patch.object(handler, 'validate_data', return_value=True):
            # Mock the _prepare_tabular_data method call
            with patch.object(handler, '_prepare_tabular_data', return_value=(
                pd.DataFrame({'feature': [1, 2, 3]}),
                pd.Series([0, 1, 0]),
                {'tabular_metadata': 'test'}
            )) as mock_prepare:
                
                test_df = pd.DataFrame({
                    'feature1': [1, 2, 3],
                    'target': [0, 1, 0]
                })
                
                # Act
                X_result, y_result, metadata = handler.prepare_data(test_df)
                
                # Assert
                mock_prepare.assert_called_once_with(test_df)
                assert isinstance(X_result, pd.DataFrame)
                assert isinstance(y_result, pd.Series)
    
    def test_prepare_tabular_data_regression(self):
        """Test tabular data preparation for regression."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_data_interface.task_type = "regression"
        mock_data_interface.target_column = "target"
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        handler = DeepLearningDataHandler(mock_settings)
        
        with patch.object(handler, 'validate_data', return_value=True):
            with patch.object(handler, '_prepare_tabular_data', return_value=(
                pd.DataFrame({'feature': [1.0, 2.0, 3.0]}),
                pd.Series([0.1, 0.2, 0.3]),
                {'tabular_metadata': 'test'}
            )) as mock_prepare:
                
                test_df = pd.DataFrame({
                    'feature1': [1.0, 2.0, 3.0],
                    'target': [0.1, 0.2, 0.3]
                })
                
                # Act
                X_result, y_result, metadata = handler.prepare_data(test_df)
                
                # Assert
                mock_prepare.assert_called_once_with(test_df)


class TestDeepLearningHandlerErrorHandling:
    """Test DeepLearning Handler error scenarios."""
    
    def test_prepare_data_unsupported_task_type(self):
        """Test error when task type is not supported."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_data_interface.task_type = "unsupported_task"
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        handler = DeepLearningDataHandler(mock_settings)
        
        with patch.object(handler, 'validate_data', return_value=True):
            test_df = pd.DataFrame({'feature1': [1, 2, 3], 'target': [0, 1, 0]})
            
            # Act & Assert
            with pytest.raises(ValueError, match="DeepLearning handler에서 지원하지 않는 task: unsupported_task"):
                handler.prepare_data(test_df)
    
    def test_prepare_timeseries_feature_column_overlap_error(self):
        """Test error when feature columns overlap with forbidden columns."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_data_interface.task_type = "timeseries"
        mock_data_interface.target_column = "target"
        mock_data_interface.timestamp_column = "timestamp"
        mock_data_interface.entity_columns = ["entity"]
        mock_data_interface.feature_columns = ["target", "feature1"]  # target in features!
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        handler = DeepLearningDataHandler(mock_settings)
        handler.sequence_length = 2
        
        with patch.object(handler, 'validate_data', return_value=True):
            test_df = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=5),
                'feature1': [1, 2, 3, 4, 5],
                'entity': ['A', 'A', 'A', 'A', 'A'],
                'target': [10, 20, 30, 40, 50]
            })
            
            # Act & Assert
            with pytest.raises(ValueError, match="feature_columns에 금지된 컬럼이 포함되어 있습니다: \\['target'\\]"):
                handler.prepare_data(test_df)
    
    def test_timeseries_datetime_conversion_error(self):
        """Test error when timestamp column cannot be converted to datetime."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_data_interface.task_type = "timeseries"
        mock_data_interface.target_column = "target"
        mock_data_interface.timestamp_column = "timestamp"
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        handler = DeepLearningDataHandler(mock_settings)
        handler.sequence_length = 2
        
        # Mock parent validate_data
        with patch.object(BaseDataHandler, 'validate_data', return_value=True):
            test_df = pd.DataFrame({
                'timestamp': ['invalid_date', 'also_invalid', 'not_date'],
                'feature1': [1, 2, 3],
                'target': [10, 20, 30]
            })
            
            # Mock pd.to_datetime to raise an exception
            with patch('pandas.to_datetime', side_effect=Exception("Cannot parse")):
                # Act & Assert
                with pytest.raises(ValueError, match="Timestamp 컬럼 'timestamp'을 datetime으로 변환할 수 없습니다"):
                    handler.validate_data(test_df)
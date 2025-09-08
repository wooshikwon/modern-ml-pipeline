"""
Unit tests for TimeseriesDataHandler.
Tests timeseries-specific data processing including temporal features and time-based splitting.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from src.components.datahandler.modules.timeseries_handler import TimeseriesDataHandler
from src.interface import BaseDataHandler
from src.settings import Settings
from tests.helpers.dataframe_builder import DataFrameBuilder
from tests.helpers.recipe_builder import RecipeBuilder


class TestTimeseriesDataHandlerBasicFunctionality:
    """Test basic TimeseriesDataHandler functionality."""
    
    def test_timeseries_handler_inherits_base_datahandler(self):
        """Test TimeseriesDataHandler properly inherits BaseDataHandler."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe.data.data_interface = Mock()
        
        # Act
        handler = TimeseriesDataHandler(mock_settings)
        
        # Assert
        assert isinstance(handler, BaseDataHandler)
        assert hasattr(handler, 'prepare_data')
        assert hasattr(handler, 'split_data')
        assert hasattr(handler, 'validate_data')
    
    def test_timeseries_handler_initialization(self):
        """Test TimeseriesDataHandler initialization."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        # Act
        handler = TimeseriesDataHandler(mock_settings)
        
        # Assert
        assert handler.settings == mock_settings
        assert handler.data_interface == mock_data_interface


class TestTimeseriesDataHandlerValidation:
    """Test timeseries data validation functionality."""
    
    def setup_method(self):
        """Set up test data."""
        base_time = datetime(2024, 1, 1)
        self.valid_timeseries_df = pd.DataFrame({
            'timestamp': [base_time + timedelta(days=i) for i in range(10)],
            'feature_0': np.random.randn(10),
            'feature_1': np.random.randn(10),
            'target': np.random.randn(10)
        })
        
        self.settings = MagicMock()
        recipe = RecipeBuilder.build(
            task_choice="timeseries",
            **{
                'data.data_interface.target_column': 'target',
                'data.data_interface.timestamp_column': 'timestamp'
            }
        )
        self.settings.recipe = recipe
    
    def test_validate_data_valid_timeseries(self):
        """Test validation passes for valid timeseries data."""
        # Arrange
        handler = TimeseriesDataHandler(self.settings)
        
        # Act & Assert
        assert handler.validate_data(self.valid_timeseries_df) == True
    
    def test_validate_data_missing_timestamp_column_raises_error(self):
        """Test validation fails when timestamp column is missing."""
        # Arrange
        handler = TimeseriesDataHandler(self.settings)
        df_no_timestamp = self.valid_timeseries_df.drop(columns=['timestamp'])
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            handler.validate_data(df_no_timestamp)
        assert "Timestamp 컬럼 'timestamp'을 찾을 수 없습니다" in str(exc_info.value)
    
    def test_validate_data_missing_target_column_raises_error(self):
        """Test validation fails when target column is missing."""
        # Arrange
        handler = TimeseriesDataHandler(self.settings)
        df_no_target = self.valid_timeseries_df.drop(columns=['target'])
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            handler.validate_data(df_no_target)
        assert "Target 컬럼 'target'을 찾을 수 없습니다" in str(exc_info.value)
    
    @patch('src.components.datahandler.modules.timeseries_handler.logger')
    def test_validate_data_converts_string_timestamp(self, mock_logger):
        """Test validation converts string timestamps to datetime."""
        # Arrange
        handler = TimeseriesDataHandler(self.settings)
        df_string_timestamp = self.valid_timeseries_df.copy()
        df_string_timestamp['timestamp'] = df_string_timestamp['timestamp'].astype(str)
        
        # Act
        result = handler.validate_data(df_string_timestamp)
        
        # Assert
        assert result == True
        mock_logger.info.assert_called()
        assert pd.api.types.is_datetime64_any_dtype(df_string_timestamp['timestamp'])
    
    def test_validate_data_invalid_timestamp_raises_error(self):
        """Test validation fails for non-convertible timestamp data."""
        # Arrange
        handler = TimeseriesDataHandler(self.settings)
        df_invalid_timestamp = self.valid_timeseries_df.copy()
        df_invalid_timestamp['timestamp'] = ['invalid', 'timestamp', 'data'] * 4  # 12 entries, but we need 10
        df_invalid_timestamp = df_invalid_timestamp.iloc[:10]  # Take only first 10 rows
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            handler.validate_data(df_invalid_timestamp)
        assert "datetime으로 변환할 수 없습니다" in str(exc_info.value)


class TestTimeseriesDataHandlerSplitData:
    """Test timeseries data splitting functionality."""
    
    def setup_method(self):
        """Set up test timeseries data."""
        base_time = datetime(2024, 1, 1)
        self.timeseries_df = pd.DataFrame({
            'timestamp': [base_time + timedelta(days=i) for i in range(100)],
            'feature_0': np.random.randn(100),
            'target': np.random.randn(100)
        })
        
        # Shuffle to test sorting
        self.timeseries_df = self.timeseries_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        self.settings = MagicMock()
        recipe = RecipeBuilder.build(
            task_choice="timeseries",
            **{
                'data.data_interface.target_column': 'target',
                'data.data_interface.timestamp_column': 'timestamp'
            }
        )
        self.settings.recipe = recipe
    
    @patch('src.components.datahandler.modules.timeseries_handler.logger')
    def test_split_data_temporal_ordering_maintained(self, mock_logger):
        """Test that temporal ordering is maintained in split."""
        # Arrange
        handler = TimeseriesDataHandler(self.settings)
        
        # Act
        train_df, test_df = handler.split_data(self.timeseries_df)
        
        # Assert
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert len(train_df) + len(test_df) == len(self.timeseries_df)
        
        # Check temporal ordering
        assert train_df['timestamp'].is_monotonic_increasing
        assert test_df['timestamp'].is_monotonic_increasing
        
        # Check that test period comes after train period
        assert train_df['timestamp'].max() <= test_df['timestamp'].min()
        
        # Check approximate 80/20 split
        assert len(train_df) == 80
        assert len(test_df) == 20
        
        # Verify logging was called
        mock_logger.info.assert_called()
    
    def test_split_data_temporal_continuity(self):
        """Test temporal continuity between train and test sets."""
        # Arrange
        handler = TimeseriesDataHandler(self.settings)
        
        # Act
        train_df, test_df = handler.split_data(self.timeseries_df)
        
        # Assert - Test set should start right after train set ends
        train_end_time = train_df['timestamp'].max()
        test_start_time = test_df['timestamp'].min()
        
        # For daily data, test should start the day after train ends
        expected_test_start = train_end_time + timedelta(days=1)
        assert test_start_time == expected_test_start


class TestTimeseriesDataHandlerPrepareData:
    """Test timeseries data preparation functionality."""
    
    def setup_method(self):
        """Set up test timeseries data."""
        base_time = datetime(2024, 1, 1)
        self.timeseries_df = pd.DataFrame({
            'timestamp': [base_time + timedelta(days=i) for i in range(30)],
            'user_id': ['user_1'] * 30,  # Entity column
            'feature_0': np.random.randn(30),
            'feature_1': np.random.randn(30),
            'target': np.random.randn(30)
        })
        
        self.settings = MagicMock()
        recipe = RecipeBuilder.build(
            task_choice="timeseries",
            **{
                'data.data_interface.target_column': 'target',
                'data.data_interface.timestamp_column': 'timestamp',
                'data.data_interface.entity_columns': ['user_id']
            }
        )
        self.settings.recipe = recipe
    
    @patch('src.components.datahandler.modules.timeseries_handler.logger')
    def test_prepare_data_basic_functionality(self, mock_logger):
        """Test basic data preparation for timeseries."""
        # Arrange
        handler = TimeseriesDataHandler(self.settings)
        
        # Act
        X, y, additional_data = handler.prepare_data(self.timeseries_df)
        
        # Assert
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(additional_data, dict)
        
        # Check that timestamp is in additional_data
        assert 'timestamp' in additional_data
        assert len(additional_data['timestamp']) == len(self.timeseries_df)
        
        # Check that entity, timestamp, and target are excluded from features
        assert 'user_id' not in X.columns
        assert 'timestamp' not in X.columns
        assert 'target' not in X.columns
        
        # Check that target is correct
        assert y.name == 'target'
        assert len(y) == len(self.timeseries_df)
    
    @patch('src.components.datahandler.modules.timeseries_handler.logger')
    def test_prepare_data_generates_time_features(self, mock_logger):
        """Test that time-based features are generated."""
        # Arrange
        handler = TimeseriesDataHandler(self.settings)
        
        # Act
        X, y, additional_data = handler.prepare_data(self.timeseries_df)
        
        # Assert - Check for generated time features
        expected_time_features = ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend']
        for feature in expected_time_features:
            assert feature in X.columns, f"Missing time feature: {feature}"
        
        # Check for lag features (at least some should exist)
        lag_features = [col for col in X.columns if 'lag' in col]
        assert len(lag_features) > 0, "No lag features generated"
        
        # Check for rolling features
        rolling_features = [col for col in X.columns if 'rolling' in col]
        assert len(rolling_features) > 0, "No rolling features generated"
    
    def test_prepare_data_with_explicit_feature_columns(self):
        """Test data preparation with explicitly specified feature columns."""
        # Arrange
        recipe = RecipeBuilder.build(
            task_choice="timeseries",
            **{
                'data.data_interface.target_column': 'target',
                'data.data_interface.timestamp_column': 'timestamp',
                'data.data_interface.entity_columns': ['user_id'],
                'data.data_interface.feature_columns': ['feature_0', 'feature_1']
            }
        )
        self.settings.recipe = recipe
        
        handler = TimeseriesDataHandler(self.settings)
        
        # Act
        X, y, additional_data = handler.prepare_data(self.timeseries_df)
        
        # Assert - Should only include explicitly specified features
        assert set(X.columns) == {"feature_0", "feature_1"}
    
    def test_prepare_data_feature_columns_validation_fails(self):
        """Test that forbidden columns in feature_columns raise error."""
        # Arrange
        recipe = RecipeBuilder.build(
            task_choice="timeseries",
            **{
                'data.data_interface.target_column': 'target',
                'data.data_interface.timestamp_column': 'timestamp',
                'data.data_interface.entity_columns': ['user_id'],
                'data.data_interface.feature_columns': ['feature_0', 'timestamp', 'target']  # Including forbidden columns
            }
        )
        self.settings.recipe = recipe
        
        handler = TimeseriesDataHandler(self.settings)
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            handler.prepare_data(self.timeseries_df)
        assert "금지된 컬럼이 포함되어 있습니다" in str(exc_info.value)
        assert "timestamp, target, entity 컬럼은 feature로 사용할 수 없습니다" in str(exc_info.value)


class TestTimeseriesDataHandlerFeatureGeneration:
    """Test timeseries feature generation functionality."""
    
    def setup_method(self):
        """Set up test data for feature generation."""
        base_time = datetime(2024, 1, 1)
        self.timeseries_df = pd.DataFrame({
            'timestamp': [base_time + timedelta(days=i) for i in range(20)],
            'target': list(range(20))  # Sequential values for easy testing
        })
        
        self.settings = MagicMock()
        recipe = RecipeBuilder.build(
            task_choice="timeseries",
            **{
                'data.data_interface.target_column': 'target',
                'data.data_interface.timestamp_column': 'timestamp'
            }
        )
        self.settings.recipe = recipe
    
    @patch('src.components.datahandler.modules.timeseries_handler.logger')
    def test_generate_time_features(self, mock_logger):
        """Test generation of basic time features."""
        # Arrange
        handler = TimeseriesDataHandler(self.settings)
        
        # Act
        result_df = handler._generate_time_features(self.timeseries_df)
        
        # Assert - Basic time features
        assert 'year' in result_df.columns
        assert 'month' in result_df.columns
        assert 'day' in result_df.columns
        assert 'dayofweek' in result_df.columns
        assert 'quarter' in result_df.columns
        assert 'is_weekend' in result_df.columns
        
        # Check specific values
        assert result_df['year'].iloc[0] == 2024
        assert result_df['month'].iloc[0] == 1
        assert result_df['day'].iloc[0] == 1
        
        # Check weekend flag (January 1, 2024 was a Monday, so not weekend)
        assert result_df['is_weekend'].iloc[0] == 0
    
    @patch('src.components.datahandler.modules.timeseries_handler.logger')
    def test_generate_lag_features(self, mock_logger):
        """Test generation of lag features."""
        # Arrange
        handler = TimeseriesDataHandler(self.settings)
        
        # Act
        result_df = handler._generate_time_features(self.timeseries_df)
        
        # Assert - Lag features
        expected_lags = [1, 2, 3, 7, 14]
        for lag in expected_lags:
            lag_col = f'target_lag_{lag}'
            assert lag_col in result_df.columns, f"Missing lag feature: {lag_col}"
            
            # Check lag values (for sequential data)
            if lag < len(result_df):
                expected_value = lag - 1 if lag - 1 >= 0 else None
                actual_value = result_df[lag_col].iloc[lag]
                if expected_value is not None:
                    assert actual_value == expected_value, f"Incorrect lag value for lag {lag}"
    
    @patch('src.components.datahandler.modules.timeseries_handler.logger')
    def test_generate_rolling_features(self, mock_logger):
        """Test generation of rolling features."""
        # Arrange
        handler = TimeseriesDataHandler(self.settings)
        
        # Act
        result_df = handler._generate_time_features(self.timeseries_df)
        
        # Assert - Rolling features
        expected_windows = [3, 7, 14]
        for window in expected_windows:
            mean_col = f'target_rolling_mean_{window}'
            std_col = f'target_rolling_std_{window}'
            
            assert mean_col in result_df.columns, f"Missing rolling mean feature: {mean_col}"
            assert std_col in result_df.columns, f"Missing rolling std feature: {std_col}"
            
            # Check rolling mean calculation (for sequential data)
            if window < len(result_df):
                # For position 'window', rolling mean should be mean of [0, 1, ..., window-1]
                expected_mean = np.mean(range(window))
                actual_mean = result_df[mean_col].iloc[window]
                if not pd.isna(actual_mean):
                    assert abs(actual_mean - expected_mean) < 1e-10, f"Incorrect rolling mean for window {window}"


class TestTimeseriesDataHandlerUtilityMethods:
    """Test utility methods of TimeseriesDataHandler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.settings = MagicMock()
        recipe = RecipeBuilder.build(
            task_choice="timeseries",
            **{
                'data.data_interface.target_column': 'target',
                'data.data_interface.timestamp_column': 'timestamp',
                'data.data_interface.entity_columns': ['user_id']
            }
        )
        self.settings.recipe = recipe
    
    @patch('src.components.datahandler.modules.timeseries_handler.logger')
    def test_check_missing_values_warning_with_missing_data(self, mock_logger):
        """Test missing values warning functionality."""
        # Arrange
        handler = TimeseriesDataHandler(self.settings)
        
        # Create dataframe with missing values
        df_with_missing = pd.DataFrame({
            'feature_0': [1, 2, np.nan, 4, 5] * 10,  # 20% missing
            'feature_1': [1, 2, 3, 4, 5] * 10        # No missing
        })
        
        # Act
        handler._check_missing_values_warning(df_with_missing, threshold=0.1)
        
        # Assert
        mock_logger.warning.assert_called()
        warning_calls = mock_logger.warning.call_args_list
        assert any("결측치가 많은 컬럼이 발견되었습니다" in str(call) for call in warning_calls)
    
    @patch('src.components.datahandler.modules.timeseries_handler.logger')
    def test_check_missing_values_no_warning_with_clean_data(self, mock_logger):
        """Test no warning with clean data."""
        # Arrange
        handler = TimeseriesDataHandler(self.settings)
        
        # Create clean dataframe
        df_clean = pd.DataFrame({
            'feature_0': [1, 2, 3, 4, 5] * 10,
            'feature_1': [1, 2, 3, 4, 5] * 10
        })
        
        # Act
        handler._check_missing_values_warning(df_clean, threshold=0.05)
        
        # Assert
        mock_logger.info.assert_called()
        info_calls = mock_logger.info.call_args_list
        assert any("모든 특성 컬럼의 결측치 비율이" in str(call) for call in info_calls)
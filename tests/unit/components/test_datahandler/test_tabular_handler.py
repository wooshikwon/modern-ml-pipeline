"""
Unit tests for TabularDataHandler.
Tests traditional tabular ML data processing (classification, regression, clustering, causal).
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from src.components.datahandler.modules.tabular_handler import TabularDataHandler
from src.interface import BaseDataHandler
from src.settings import Settings
from tests.helpers.dataframe_builder import DataFrameBuilder
from tests.helpers.recipe_builder import RecipeBuilder
from tests.helpers.config_builder import ConfigBuilder


class TestTabularDataHandlerBasicFunctionality:
    """Test basic TabularDataHandler functionality."""
    
    def test_tabular_handler_inherits_base_datahandler(self):
        """Test TabularDataHandler properly inherits BaseDataHandler."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe.data.data_interface = Mock()
        
        # Act
        handler = TabularDataHandler(mock_settings)
        
        # Assert
        assert isinstance(handler, BaseDataHandler)
        assert hasattr(handler, 'prepare_data')
        assert hasattr(handler, 'split_data')
        assert hasattr(handler, 'validate_data')
    
    def test_tabular_handler_initialization(self):
        """Test TabularDataHandler initialization."""
        # Arrange
        mock_settings = MagicMock()
        mock_data_interface = Mock()
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        # Act
        handler = TabularDataHandler(mock_settings)
        
        # Assert
        assert handler.settings == mock_settings
        assert handler.data_interface == mock_data_interface


class TestTabularDataHandlerSplitData:
    """Test data splitting functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.classification_df = DataFrameBuilder.build_classification_data(
            n_samples=100, n_classes=3, random_state=42
        )
        self.regression_df = DataFrameBuilder.build_regression_data(
            n_samples=100, random_state=42
        )
    
    def test_split_data_classification_with_stratify(self):
        """Test classification data splitting with stratification."""
        # Arrange
        recipe = RecipeBuilder.build_recipe(
            task_choice="classification",
            target_column="target"
        )
        settings = MagicMock()
        settings.recipe = recipe
        
        handler = TabularDataHandler(settings)
        
        # Act
        train_df, test_df = handler.split_data(self.classification_df)
        
        # Assert
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert len(train_df) + len(test_df) == len(self.classification_df)
        assert len(train_df) > len(test_df)  # Should be ~80/20 split
        
        # Check that all classes are represented in both sets
        train_classes = set(train_df['target'].unique())
        test_classes = set(test_df['target'].unique())
        original_classes = set(self.classification_df['target'].unique())
        assert train_classes == original_classes
        assert test_classes.issubset(original_classes)  # Test may not have all classes due to small size
    
    def test_split_data_classification_insufficient_samples_no_stratify(self):
        """Test classification with insufficient samples falls back to random split."""
        # Arrange - Create data with very few samples per class
        small_df = pd.DataFrame({
            'feature_0': [1, 2, 3, 4],
            'target': [0, 1, 0, 1]  # Only 2 samples per class
        })
        
        recipe = RecipeBuilder.build_recipe(
            task_choice="classification",
            target_column="target"
        )
        settings = MagicMock()
        settings.recipe = recipe
        
        handler = TabularDataHandler(settings)
        
        # Act
        train_df, test_df = handler.split_data(small_df)
        
        # Assert
        assert len(train_df) + len(test_df) == len(small_df)
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
    
    def test_split_data_regression_no_stratify(self):
        """Test regression data splitting without stratification."""
        # Arrange
        recipe = RecipeBuilder.build_recipe(
            task_choice="regression",
            target_column="target"
        )
        settings = MagicMock()
        settings.recipe = recipe
        
        handler = TabularDataHandler(settings)
        
        # Act
        train_df, test_df = handler.split_data(self.regression_df)
        
        # Assert
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert len(train_df) + len(test_df) == len(self.regression_df)
        assert len(train_df) > len(test_df)  # Should be ~80/20 split
    
    def test_split_data_causal_with_treatment_stratify(self):
        """Test causal data splitting with treatment stratification."""
        # Arrange
        causal_df = pd.DataFrame({
            'feature_0': np.random.randn(100),
            'feature_1': np.random.randn(100),
            'treatment': np.random.choice([0, 1], 100),  # Binary treatment
            'target': np.random.randn(100)
        })
        
        recipe = RecipeBuilder.build_recipe(
            task_choice="causal",
            target_column="target",
            treatment_column="treatment"
        )
        settings = MagicMock()
        settings.recipe = recipe
        
        handler = TabularDataHandler(settings)
        
        # Act
        train_df, test_df = handler.split_data(causal_df)
        
        # Assert
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert len(train_df) + len(test_df) == len(causal_df)
        
        # Check treatment distribution
        train_treatments = set(train_df['treatment'].unique())
        test_treatments = set(test_df['treatment'].unique())
        original_treatments = set(causal_df['treatment'].unique())
        assert train_treatments == original_treatments
        assert test_treatments.issubset(original_treatments)


class TestTabularDataHandlerPrepareData:
    """Test data preparation functionality."""
    
    def setup_method(self):
        """Set up test data with entity columns."""
        self.test_df = DataFrameBuilder.build_classification_data(
            n_samples=50, n_features=3, add_entity_column=True, random_state=42
        )
        # Add some missing values for testing
        self.test_df.loc[0:4, 'feature_0'] = np.nan
    
    def test_prepare_data_classification_basic(self):
        """Test basic data preparation for classification."""
        # Arrange
        recipe = RecipeBuilder.build_recipe(
            task_choice="classification",
            target_column="target",
            entity_columns=["user_id"]
        )
        settings = MagicMock()
        settings.recipe = recipe
        
        handler = TabularDataHandler(settings)
        
        # Act
        X, y, additional_data = handler.prepare_data(self.test_df)
        
        # Assert
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(additional_data, dict)
        
        # Check that entity column is excluded from features
        assert 'user_id' not in X.columns
        assert 'target' not in X.columns
        
        # Check target
        assert y.name == 'target'
        assert len(y) == len(self.test_df)
        
        # Check additional data is empty (no special data for classification)
        assert additional_data == {}
    
    def test_prepare_data_with_explicit_feature_columns(self):
        """Test data preparation with explicitly specified feature columns."""
        # Arrange
        recipe = RecipeBuilder.build_recipe(
            task_choice="classification",
            target_column="target",
            entity_columns=["user_id"],
            feature_columns=["feature_0", "feature_1"]
        )
        settings = MagicMock()
        settings.recipe = recipe
        
        handler = TabularDataHandler(settings)
        
        # Act
        X, y, additional_data = handler.prepare_data(self.test_df)
        
        # Assert
        assert list(X.columns) == ["feature_0", "feature_1"]
        assert len(X.columns) == 2
    
    def test_prepare_data_clustering_no_target(self):
        """Test data preparation for clustering (no target)."""
        # Arrange
        recipe = RecipeBuilder.build_recipe(
            task_choice="clustering",
            entity_columns=["user_id"]
        )
        settings = MagicMock()
        settings.recipe = recipe
        
        handler = TabularDataHandler(settings)
        
        # Act
        X, y, additional_data = handler.prepare_data(self.test_df)
        
        # Assert
        assert isinstance(X, pd.DataFrame)
        assert y is None  # Clustering has no target
        assert 'user_id' not in X.columns
        assert 'target' in X.columns  # Target becomes a feature for clustering
    
    def test_prepare_data_causal_with_treatment(self):
        """Test data preparation for causal inference with treatment column."""
        # Arrange
        causal_df = pd.DataFrame({
            'user_id': range(50),
            'feature_0': np.random.randn(50),
            'feature_1': np.random.randn(50),
            'treatment': np.random.choice([0, 1], 50),
            'target': np.random.randn(50)
        })
        
        recipe = RecipeBuilder.build_recipe(
            task_choice="causal",
            target_column="target",
            treatment_column="treatment",
            entity_columns=["user_id"]
        )
        settings = MagicMock()
        settings.recipe = recipe
        
        handler = TabularDataHandler(settings)
        
        # Act
        X, y, additional_data = handler.prepare_data(causal_df)
        
        # Assert
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(additional_data, dict)
        
        # Check that treatment is in additional_data but not in features
        assert 'treatment' not in X.columns
        assert 'treatment' in additional_data
        assert len(additional_data['treatment']) == len(causal_df)
        
        # Check entity column excluded
        assert 'user_id' not in X.columns
    
    @patch('src.components.datahandler.modules.tabular_handler.logger')
    def test_prepare_data_missing_values_warning(self, mock_logger):
        """Test that missing values trigger warning."""
        # Arrange - Create data with >5% missing values
        df_with_missing = self.test_df.copy()
        df_with_missing.loc[0:10, 'feature_0'] = np.nan  # ~20% missing
        
        recipe = RecipeBuilder.build_recipe(
            task_choice="classification",
            target_column="target"
        )
        settings = MagicMock()
        settings.recipe = recipe
        
        handler = TabularDataHandler(settings)
        
        # Act
        X, y, additional_data = handler.prepare_data(df_with_missing)
        
        # Assert
        mock_logger.warning.assert_called()
        # Check that warning was called for missing values
        warning_calls = [call for call in mock_logger.warning.call_args_list 
                        if 'missing' in str(call).lower()]
        assert len(warning_calls) > 0


class TestTabularDataHandlerUtilityMethods:
    """Test utility methods of TabularDataHandler."""
    
    def test_get_exclude_columns_with_entity_columns(self):
        """Test _get_exclude_columns includes entity columns."""
        # Arrange
        recipe = RecipeBuilder.build_recipe(
            task_choice="classification",
            entity_columns=["user_id", "session_id"]
        )
        settings = MagicMock()
        settings.recipe = recipe
        
        handler = TabularDataHandler(settings)
        
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'session_id': [10, 20, 30],
            'feature_0': [1.0, 2.0, 3.0],
            'target': [0, 1, 0]
        })
        
        # Act
        exclude_cols = handler._get_exclude_columns(df)
        
        # Assert
        assert 'user_id' in exclude_cols
        assert 'session_id' in exclude_cols
        assert 'feature_0' not in exclude_cols
    
    def test_get_exclude_columns_nonexistent_entity_ignored(self):
        """Test that non-existent entity columns are ignored."""
        # Arrange
        recipe = RecipeBuilder.build_recipe(
            task_choice="classification",
            entity_columns=["user_id", "nonexistent_col"]
        )
        settings = MagicMock()
        settings.recipe = recipe
        
        handler = TabularDataHandler(settings)
        
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'feature_0': [1.0, 2.0, 3.0],
            'target': [0, 1, 0]
        })
        
        # Act
        exclude_cols = handler._get_exclude_columns(df)
        
        # Assert
        assert 'user_id' in exclude_cols
        assert 'nonexistent_col' not in exclude_cols
        assert len(exclude_cols) == 1
    
    @patch('src.components.datahandler.modules.tabular_handler.logger')
    def test_check_missing_values_warning_threshold(self, mock_logger):
        """Test missing values warning with different thresholds."""
        # Arrange
        recipe = RecipeBuilder.build_recipe(task_choice="classification")
        settings = MagicMock()
        settings.recipe = recipe
        
        handler = TabularDataHandler(settings)
        
        # Create data with 20% missing values in one column
        df = pd.DataFrame({
            'feature_0': [1, 2, np.nan, 4, 5] * 20,  # 20% missing
            'feature_1': [1, 2, 3, 4, 5] * 20       # No missing
        })
        
        # Act - Test with 10% threshold (should trigger warning)
        handler._check_missing_values_warning(df, threshold=0.1)
        
        # Assert
        mock_logger.warning.assert_called()
        
        # Reset mock
        mock_logger.reset_mock()
        
        # Act - Test with 30% threshold (should not trigger warning)
        handler._check_missing_values_warning(df, threshold=0.3)
        
        # Assert
        mock_logger.info.assert_called()  # Should call info instead of warning
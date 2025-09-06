"""
Unit tests for data_handler module.
Tests data splitting and preparation functionality with various edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock

from src.components.trainer.modules.data_handler import (
    split_data,
    prepare_training_data, 
    _get_exclude_columns,
    _get_stratify_column_data
)
from tests.helpers.builders import TrainerDataBuilder, SettingsBuilder


class TestSplitData:
    """Test split_data() function."""
    
    def test_split_classification_with_stratify(self):
        """Test classification task with stratified split."""
        # Arrange: Balanced classification data
        df = TrainerDataBuilder.build_data_for_task_type(
            "classification", n_samples=100, add_entity_columns=["user_id"]
        )
        settings = SettingsBuilder.build_classification_config()
        
        # Act: Split data
        train_df, test_df = split_data(df, settings)
        
        # Assert: Check split results
        assert len(train_df) + len(test_df) == len(df)
        assert len(test_df) == int(len(df) * 0.2)  # 20% test size
        
        # Check stratification worked (similar class distribution)
        train_dist = train_df['target'].value_counts(normalize=True).sort_index()
        test_dist = test_df['target'].value_counts(normalize=True).sort_index()
        assert np.allclose(train_dist.values, test_dist.values, atol=0.1)

    def test_split_classification_small_dataset_no_stratify(self):
        """Test classification with small dataset that can't be stratified."""
        # Arrange: Small dataset that can't be stratified
        df = TrainerDataBuilder.build_edge_case_small_dataset("classification", n_samples=3)
        settings = SettingsBuilder.build_classification_config()
        
        # Act: Split data  
        train_df, test_df = split_data(df, settings)
        
        # Assert: Should still split without error
        assert len(train_df) + len(test_df) == len(df)
        assert len(test_df) >= 1  # At least 1 sample in test
        
    def test_split_classification_single_class(self):
        """Test single class classification data."""
        # Arrange: Single class data
        df = TrainerDataBuilder.build_edge_case_single_class(n_samples=20)
        settings = SettingsBuilder.build_classification_config()
        
        # Act: Split data
        train_df, test_df = split_data(df, settings)
        
        # Assert: Should split without stratification
        assert len(train_df) + len(test_df) == len(df)
        assert train_df['target'].nunique() == 1
        assert test_df['target'].nunique() == 1

    def test_split_regression_no_stratify(self):
        """Test regression task without stratification."""
        # Arrange: Regression data
        df = TrainerDataBuilder.build_data_for_task_type("regression", n_samples=50)
        settings = SettingsBuilder.build_regression_config()
        
        # Act: Split data
        train_df, test_df = split_data(df, settings)
        
        # Assert: Should split without stratification
        assert len(train_df) + len(test_df) == len(df)
        assert len(test_df) == int(len(df) * 0.2)

    def test_split_causal_with_stratify(self):
        """Test causal task with treatment-based stratify."""
        # Arrange: Causal data with balanced treatment
        df = TrainerDataBuilder.build_data_for_task_type("causal", n_samples=100)
        settings = SettingsBuilder.build_causal_config()
        
        # Act: Split data
        train_df, test_df = split_data(df, settings)
        
        # Assert: Check treatment distribution
        assert len(train_df) + len(test_df) == len(df)
        
        # Check treatment distribution is similar
        train_treatment_ratio = train_df['treatment'].mean()
        test_treatment_ratio = test_df['treatment'].mean()
        assert abs(train_treatment_ratio - test_treatment_ratio) < 0.2

    def test_split_clustering_no_stratify(self):
        """Test clustering task without stratification."""
        # Arrange: Clustering data
        df = TrainerDataBuilder.build_data_for_task_type("clustering", n_samples=60)
        settings = SettingsBuilder.build_clustering_config()
        
        # Act: Split data
        train_df, test_df = split_data(df, settings)
        
        # Assert: Should split without stratification
        assert len(train_df) + len(test_df) == len(df)
        
    def test_split_missing_target_column(self):
        """Test case when target column is missing."""
        # Arrange: Data missing target column
        df = TrainerDataBuilder.build_edge_case_missing_columns("target")
        settings = SettingsBuilder.build_classification_config()
        
        # Act: Split data
        train_df, test_df = split_data(df, settings)
        
        # Assert: Should split without stratification
        assert len(train_df) + len(test_df) == len(df)


class TestGetExcludeColumns:
    """Test _get_exclude_columns() function."""
    
    def test_get_exclude_columns_with_entity_and_timestamp(self):
        """Test excluding entity and timestamp columns."""
        # Arrange: Data with entity and timestamp columns
        df = TrainerDataBuilder.build_mixed_data_types()
        settings = SettingsBuilder.build_config_with_exclusions(
            entity_columns=["user_id"],
            timestamp_column="timestamp"
        )
        
        # Act: Get exclude columns
        exclude_cols = _get_exclude_columns(settings, df)
        
        # Assert: Should include entity and timestamp
        assert "user_id" in exclude_cols
        assert "timestamp" in exclude_cols
        
        
    def test_get_exclude_columns_only_existing_columns(self):
        """Test that non-existing columns are excluded from exclude list."""
        # Arrange: Config with non-existing column excludes
        df = TrainerDataBuilder.build_classification_data(n_samples=10)
        settings = SettingsBuilder.build_config_with_exclusions(
            entity_columns=["user_id", "non_existing_col"],
            timestamp_column="also_non_existing"
        )
        
        # Act: Get exclude columns
        exclude_cols = _get_exclude_columns(settings, df)
        
        # Assert: Should only include existing columns
        assert "user_id" in exclude_cols
        assert "non_existing_col" not in exclude_cols
        assert "also_non_existing" not in exclude_cols
        
    def test_get_exclude_columns_empty_result(self):
        """Test case when no columns should be excluded."""
        # Arrange: Simple data with no excludes
        df = TrainerDataBuilder.build_classification_data(
            n_samples=10, add_entity_column=False
        )
        settings = SettingsBuilder.build_classification_config()
        
        # Act: Get exclude columns  
        exclude_cols = _get_exclude_columns(settings, df)
        
        # Assert: Should return empty list
        assert exclude_cols == []


class TestGetStratifyColumnData:
    """Test _get_stratify_column_data() function."""
    
    def test_get_stratify_classification(self):
        """Test extracting target column for classification."""
        # Arrange: Classification data
        df = TrainerDataBuilder.build_classification_data(n_samples=20)
        data_interface = Mock()
        data_interface.task_type = "classification"
        data_interface.target_column = "target"
        
        # Act: Get stratify column data
        result = _get_stratify_column_data(df, data_interface)
        
        # Assert: Should return target column
        assert result is not None
        assert len(result) == len(df)
        assert result.name == "target"
        
    def test_get_stratify_causal(self):
        """Test extracting treatment column for causal."""
        # Arrange: Causal data
        df = TrainerDataBuilder.build_causal_data(n_samples=30)
        data_interface = Mock()
        data_interface.task_type = "causal"
        data_interface.treatment_column = "treatment"
        
        # Act: Get stratify column data
        result = _get_stratify_column_data(df, data_interface)
        
        # Assert: Should return treatment column
        assert result is not None
        assert len(result) == len(df)
        assert result.name == "treatment"
        
    def test_get_stratify_regression_returns_none(self):
        """Test that regression returns None."""
        # Arrange: Regression data
        df = TrainerDataBuilder.build_regression_data(n_samples=15)
        data_interface = Mock()
        data_interface.task_type = "regression"
        
        # Act: Get stratify column data
        result = _get_stratify_column_data(df, data_interface)
        
        # Assert: Should return None
        assert result is None
        
    def test_get_stratify_missing_column(self):
        """Test case when specified column doesn't exist."""
        # Arrange: Data missing target column
        df = TrainerDataBuilder.build_edge_case_missing_columns("target")
        data_interface = Mock()
        data_interface.task_type = "classification"
        data_interface.target_column = "target"
        
        # Act: Get stratify column data
        result = _get_stratify_column_data(df, data_interface)
        
        # Assert: Should return None
        assert result is None


class TestPrepareTrainingData:
    """Test prepare_training_data() function."""
    
    def test_prepare_classification_data_auto_features(self):
        """Test classification with feature_columns=None for auto selection."""
        # Arrange: Classification data
        df = TrainerDataBuilder.build_data_for_task_type(
            "classification", n_samples=50, add_entity_columns=["user_id"]
        )
        settings = SettingsBuilder.build_config_with_auto_features("classification")
        
        # Act: Prepare training data
        X, y, additional_data = prepare_training_data(df, settings)
        
        # Assert: Check results
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert y.name == "target"
        assert len(X) == len(y) == len(df)
        
        # Should exclude target, entity columns, non-numeric
        assert "target" not in X.columns
        assert "user_id" not in X.columns
        assert all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number)))
        
    def test_prepare_classification_data_explicit_features(self):
        """Test classification with explicitly specified features."""
        # Arrange: Classification data
        df = TrainerDataBuilder.build_mixed_data_types()
        feature_columns = ["feature_numeric_int", "feature_numeric_float"]
        settings = SettingsBuilder.build_config_with_explicit_features(
            "classification", feature_columns
        )
        
        # Act: Prepare training data
        X, y, additional_data = prepare_training_data(df, settings)
        
        # Assert: Should use only specified features
        assert list(X.columns) == feature_columns
        assert len(X) == len(df)
        
    def test_prepare_regression_data(self):
        """Test regression data preparation."""
        # Arrange: Regression data
        df = TrainerDataBuilder.build_data_for_task_type("regression", n_samples=40)
        settings = SettingsBuilder.build_config_with_auto_features("regression")
        
        # Act: Prepare training data
        X, y, additional_data = prepare_training_data(df, settings)
        
        # Assert: Check results
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert y.name == "target"
        assert additional_data == {}
        
    def test_prepare_clustering_data(self):
        """Test clustering data preparation (y=None)."""
        # Arrange: Clustering data
        df = TrainerDataBuilder.build_data_for_task_type("clustering", n_samples=30)
        settings = SettingsBuilder.build_config_with_auto_features("clustering")
        
        # Act: Prepare training data
        X, y, additional_data = prepare_training_data(df, settings)
        
        # Assert: y should be None for clustering
        assert isinstance(X, pd.DataFrame)
        assert y is None
        assert additional_data == {}
        
    def test_prepare_causal_data(self):
        """Test causal data preparation with treatment info."""
        # Arrange: Causal data
        df = TrainerDataBuilder.build_causal_data_with_custom_columns(
            target_column="outcome", treatment_column="treatment", n_samples=60
        )
        settings = SettingsBuilder.build_config_with_auto_features("causal")
        
        # Act: Prepare training data
        X, y, additional_data = prepare_training_data(df, settings)
        
        # Assert: Check additional_data has treatment info
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert "treatment" in additional_data
        assert len(additional_data["treatment"]) == len(df)
        assert "treatment_value" in additional_data
        
        # Should exclude target and treatment from features
        assert "outcome" not in X.columns
        assert "target" not in X.columns
        assert "treatment" not in X.columns
        
    def test_prepare_data_no_numeric_features(self):
        """Test case when no numeric features are available."""
        # Arrange: Data with only non-numeric features
        df = TrainerDataBuilder.build_edge_case_no_numeric_features(n_samples=20)
        settings = SettingsBuilder.build_config_with_auto_features("classification")
        
        # Act: Prepare training data
        X, y, additional_data = prepare_training_data(df, settings)
        
        # Assert: X should be empty DataFrame
        assert isinstance(X, pd.DataFrame)
        assert len(X.columns) == 0
        assert len(X) == len(df)  # Same number of rows
        
    def test_prepare_data_all_features_excluded(self):
        """Test case when all features are excluded."""
        # Arrange: Data where all columns get excluded
        df = TrainerDataBuilder.build_edge_case_all_features_excluded(n_samples=15)
        settings = SettingsBuilder.build_config_with_auto_features("classification")
        
        # Act: Prepare training data
        X, y, additional_data = prepare_training_data(df, settings)
        
        # Assert: X should have no feature columns
        assert isinstance(X, pd.DataFrame)
        assert len(X.columns) == 0
        
    def test_prepare_data_invalid_task_type(self):
        """Test unsupported task_type."""
        # Arrange: Data with invalid task type
        df = TrainerDataBuilder.build_classification_data(n_samples=20)
        settings = SettingsBuilder.build_classification_config()
        settings.recipe.data.data_interface.task_type = "invalid_task"
        
        # Act & Assert: Should raise ValueError
        with pytest.raises(ValueError, match="지원하지 않는 task_type"):
            prepare_training_data(df, settings)
            
    def test_prepare_data_with_nan_values(self):
        """Test data preparation with NaN values."""
        # Arrange: Data with NaN values
        df = TrainerDataBuilder.build_edge_case_nan_values(
            nan_in_features=True, n_samples=25
        )
        settings = SettingsBuilder.build_config_with_auto_features("classification")
        
        # Act: Prepare training data
        X, y, additional_data = prepare_training_data(df, settings)
        
        # Assert: Should handle NaN values (keep them as is)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.isna().any().any()  # Should contain NaN values


class TestIntegrationScenarios:
    """Test complete workflow integration scenarios."""
    
    def test_complete_classification_workflow(self):
        """Test complete classification workflow: split → prepare."""
        # Arrange: Classification data
        df = TrainerDataBuilder.build_data_for_task_type(
            "classification", n_samples=100, add_entity_columns=["user_id"]
        )
        settings = SettingsBuilder.build_config_with_auto_features("classification")
        
        # Act: Split then prepare
        train_df, test_df = split_data(df, settings)
        X_train, y_train, _ = prepare_training_data(train_df, settings)
        X_test, y_test, _ = prepare_training_data(test_df, settings)
        
        # Assert: Check complete workflow
        assert len(X_train) + len(X_test) == len(df)
        assert len(y_train) + len(y_test) == len(df)
        assert list(X_train.columns) == list(X_test.columns)
        
    def test_complete_causal_workflow(self):
        """Test complete causal workflow with treatment data."""
        # Arrange: Causal data
        df = TrainerDataBuilder.build_data_for_task_type("causal", n_samples=80)
        settings = SettingsBuilder.build_config_with_auto_features("causal")
        
        # Act: Split then prepare
        train_df, test_df = split_data(df, settings)
        X_train, y_train, additional_train = prepare_training_data(train_df, settings)
        X_test, y_test, additional_test = prepare_training_data(test_df, settings)
        
        # Assert: Check causal workflow
        assert "treatment" in additional_train
        assert "treatment" in additional_test
        assert len(additional_train["treatment"]) == len(X_train)
        assert len(additional_test["treatment"]) == len(X_test)
        
    def test_edge_case_small_dataset_workflow(self):
        """Test edge case: very small dataset workflow."""
        # Arrange: Very small dataset
        df = TrainerDataBuilder.build_edge_case_small_dataset(
            "classification", n_samples=5
        )
        settings = SettingsBuilder.build_config_with_auto_features("classification")
        
        # Act: Complete workflow
        train_df, test_df = split_data(df, settings)
        X_train, y_train, _ = prepare_training_data(train_df, settings)
        X_test, y_test, _ = prepare_training_data(test_df, settings)
        
        # Assert: Should handle small dataset
        assert len(train_df) >= 1
        assert len(test_df) >= 1
        assert len(X_train) == len(train_df)
        assert len(X_test) == len(test_df)
"""
Classification DataHandler Unit Tests - No Mock Hell Approach
Real data handling, real transformations
Following comprehensive testing strategy document principles
"""

import numpy as np
import pandas as pd

from src.components.datahandler.modules.tabular_handler import TabularDataHandler
from src.components.datahandler.base import BaseDataHandler


class TestClassificationDataHandler:
    """Test TabularDataHandler for classification tasks with real data."""

    def test_datahandler_initialization(self, settings_builder):
        """Test DataHandler initialization for classification."""
        # Given: Valid settings for classification
        settings = settings_builder.with_task("classification").with_target_column("target").build()
        settings.recipe.data.data_interface.feature_columns = ["feat1", "feat2", "feat3"]

        # When: Creating TabularDataHandler
        handler = TabularDataHandler(settings)

        # Then: Handler is properly initialized
        assert isinstance(handler, TabularDataHandler)
        assert isinstance(handler, BaseDataHandler)
        assert handler.data_interface.target_column == "target"

    def test_prepare_data(self, settings_builder, test_data_generator):
        """Test preparing data for classification."""
        # Given: Data and handler
        X, y = test_data_generator.classification_data(n_samples=100, n_features=5)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y
        df["entity_id"] = range(len(df))

        settings = (
            settings_builder.with_task("classification")
            .with_target_column("target")
            .with_entity_columns(["entity_id"])
            .build()
        )
        settings.recipe.data.data_interface.feature_columns = [f"feature_{i}" for i in range(5)]

        handler = TabularDataHandler(settings)

        # When: Preparing data
        X_result, y_result, additional_data = handler.prepare_data(df)

        # Then: Data is prepared correctly
        assert isinstance(X_result, pd.DataFrame)
        assert isinstance(y_result, pd.Series)
        assert len(X_result) == len(df)
        assert len(y_result) == len(df)
        assert "target" not in X_result.columns
        assert "entity_id" not in X_result.columns
        assert all(y_result == y)

    def test_split_data(self, settings_builder, test_data_generator):
        """Test train/test split for classification."""
        # Given: Data with classification target
        X, y = test_data_generator.classification_data(n_samples=100, n_features=3)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y

        settings = settings_builder.with_task("classification").with_target_column("target").build()
        settings.recipe.data.data_interface.feature_columns = [f"feature_{i}" for i in range(3)]

        handler = TabularDataHandler(settings)

        # When: Splitting data
        split_result = handler.split_data(df)

        # Then: Data is split correctly (4-way split)
        assert len(split_result["train"]) == 60  # 60% train (default from SettingsBuilder)
        assert len(split_result["validation"]) == 20  # 20% validation
        assert len(split_result["test"]) == 20  # 20% test
        assert (
            split_result["calibration"] is None or len(split_result["calibration"]) == 0
        )  # 0% calibration
        assert set(split_result["train"].columns) == set(split_result["test"].columns)

    def test_prepare_data_with_auto_feature_selection(self, settings_builder):
        """Test automatic feature selection when feature_columns is None."""
        # Given: Data without explicit feature columns
        df = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_2": [2.0, 3.0, 4.0, 5.0, 6.0],
                "target": [0, 1, 0, 1, 1],
                "entity_id": [1, 2, 3, 4, 5],
            }
        )

        settings = (
            settings_builder.with_task("classification")
            .with_target_column("target")
            .with_entity_columns(["entity_id"])
            .build()
        )
        settings.recipe.data.data_interface.feature_columns = None  # Auto selection

        handler = TabularDataHandler(settings)

        # When: Preparing data with auto feature selection
        X, y, _ = handler.prepare_data(df)

        # Then: Features are auto-selected (excluding target and entity)
        assert "target" not in X.columns
        assert "entity_id" not in X.columns
        assert "feature_1" in X.columns
        assert "feature_2" in X.columns

    def test_prepare_data_with_imbalanced_classes(self, settings_builder):
        """Test preparing imbalanced classification data."""
        # Given: Imbalanced data
        df = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "target": [0] * 90 + [1] * 10,  # 90% class 0, 10% class 1
            }
        )

        settings = settings_builder.with_task("classification").with_target_column("target").build()
        settings.recipe.data.data_interface.feature_columns = ["feature_1", "feature_2"]

        handler = TabularDataHandler(settings)

        # When: Preparing imbalanced data
        X, y, _ = handler.prepare_data(df)

        # Then: Data is prepared correctly
        assert len(X) == len(y)
        assert y.value_counts()[0] == 90
        assert y.value_counts()[1] == 10

    def test_datahandler_with_multiclass(self, settings_builder):
        """Test DataHandler with multiclass classification."""
        # Given: Multiclass data
        df = pd.DataFrame(
            {
                "feature_1": np.random.randn(60),
                "feature_2": np.random.randn(60),
                "target": [0] * 20 + [1] * 20 + [2] * 20,
            }
        )

        settings = settings_builder.with_task("classification").with_target_column("target").build()
        settings.recipe.data.data_interface.feature_columns = ["feature_1", "feature_2"]

        handler = TabularDataHandler(settings)

        # When: Preparing multiclass data
        X, y, _ = handler.prepare_data(df)

        # Then: All classes are preserved
        assert len(np.unique(y)) == 3
        assert all(c in [0, 1, 2] for c in np.unique(y))

    def test_datahandler_preserves_feature_order(self, settings_builder):
        """Test that DataHandler preserves feature column order."""
        # Given: Data with specific column order
        df = pd.DataFrame(
            {
                "z_feature": [1, 2, 3],
                "a_feature": [4, 5, 6],
                "m_feature": [7, 8, 9],
                "target": [0, 1, 0],
            }
        )

        feature_cols = ["z_feature", "a_feature", "m_feature"]

        settings = settings_builder.with_task("classification").with_target_column("target").build()
        settings.recipe.data.data_interface.feature_columns = feature_cols

        handler = TabularDataHandler(settings)

        # When: Preparing data
        X, y, _ = handler.prepare_data(df)

        # Then: Column order is preserved
        assert list(X.columns) == feature_cols

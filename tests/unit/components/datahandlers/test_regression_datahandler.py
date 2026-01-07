"""
Regression DataHandler Unit Tests - No Mock Hell Approach
Real data handling, real transformations
Following comprehensive testing strategy document principles
"""

import numpy as np
import pandas as pd

from src.components.datahandler.modules.tabular_handler import TabularDataHandler
from src.components.datahandler.base import BaseDataHandler


class TestRegressionDataHandler:
    """Test TabularDataHandler for regression tasks with real data."""

    def test_datahandler_initialization_regression(self, settings_builder):
        """Test DataHandler initialization for regression."""
        # Given: Valid settings for regression
        settings = settings_builder.with_task("regression").with_target_column("price").build()
        settings.recipe.data.data_interface.feature_columns = ["sqft", "bedrooms", "bathrooms"]

        # When: Creating TabularDataHandler
        handler = TabularDataHandler(settings)

        # Then: Handler is properly initialized
        assert isinstance(handler, TabularDataHandler)
        assert isinstance(handler, BaseDataHandler)
        assert handler.data_interface.target_column == "price"

    def test_prepare_data_continuous_target(self, settings_builder, test_data_generator):
        """Test preparing data with continuous target variable."""
        # Given: Regression data with continuous target
        X, y = test_data_generator.regression_data(n_samples=100, n_features=4)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y  # Continuous values

        settings = settings_builder.with_task("regression").with_target_column("target").build()
        settings.recipe.data.data_interface.feature_columns = [f"feature_{i}" for i in range(4)]

        handler = TabularDataHandler(settings)

        # When: Preparing data
        X_result, y_result, _ = handler.prepare_data(df)

        # Then: Continuous values are preserved
        assert isinstance(y_result, pd.Series)
        assert y_result.dtype in [np.float32, np.float64]
        assert np.std(y_result) > 0  # Has variation

    def test_prepare_data_with_outliers(self, settings_builder):
        """Test preparing regression data with outliers."""
        # Given: Data with outliers
        df = pd.DataFrame(
            {
                "feature_1": [1, 2, 3, 4, 100],  # 100 is outlier
                "feature_2": [5, 6, 7, 8, 9],
                "target": [10, 20, 30, 40, 500],  # 500 is outlier
            }
        )

        settings = settings_builder.with_task("regression").with_target_column("target").build()
        settings.recipe.data.data_interface.feature_columns = ["feature_1", "feature_2"]

        handler = TabularDataHandler(settings)

        # When: Preparing data with outliers
        X, y, _ = handler.prepare_data(df)

        # Then: Data is prepared (outliers preserved by default)
        assert len(X) == len(df)
        assert max(X["feature_1"]) == 100  # Outlier is preserved by default

    def test_split_data_for_regression(self, settings_builder, test_data_generator):
        """Test train/test split for regression."""
        # Given: Regression data
        X, y = test_data_generator.regression_data(n_samples=100, n_features=3)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y

        settings = settings_builder.with_task("regression").with_target_column("target").build()
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

    def test_prepare_data_zero_variance(self, settings_builder):
        """Test preparing data with zero variance features."""
        # Given: Data with constant feature
        df = pd.DataFrame(
            {
                "constant_feature": [5, 5, 5, 5],
                "variable_feature": [1, 2, 3, 4],
                "target": [10, 20, 30, 40],
            }
        )

        settings = settings_builder.with_task("regression").with_target_column("target").build()
        settings.recipe.data.data_interface.feature_columns = [
            "constant_feature",
            "variable_feature",
        ]

        handler = TabularDataHandler(settings)

        # When: Preparing data
        X, y, _ = handler.prepare_data(df)

        # Then: Zero variance feature is included
        assert X["constant_feature"].std() == 0
        assert X["variable_feature"].std() > 0

    def test_prepare_data_with_skewed_features(self, settings_builder):
        """Test preparing data with skewed features."""
        # Given: Skewed data
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "skewed_feature": np.exp(np.random.randn(50)),  # Log-normal distribution
                "normal_feature": np.random.randn(50),
                "target": np.exp(np.random.randn(50)),  # Also skewed
            }
        )

        settings = settings_builder.with_task("regression").with_target_column("target").build()
        settings.recipe.data.data_interface.feature_columns = ["skewed_feature", "normal_feature"]

        handler = TabularDataHandler(settings)

        # When: Preparing skewed data
        X, y, _ = handler.prepare_data(df)

        # Then: Skewed data is preserved for transformation
        from scipy import stats

        skewness = stats.skew(X["skewed_feature"])
        assert abs(skewness) > 1  # Highly skewed

    def test_preserve_data_types_regression(self, settings_builder):
        """Test that DataHandler preserves appropriate data types."""
        # Given: Mixed type data
        df = pd.DataFrame(
            {
                "int_feature": [1, 2, 3, 4],
                "float_feature": [1.5, 2.5, 3.5, 4.5],
                "target": [10.1, 20.2, 30.3, 40.4],
            }
        )

        settings = settings_builder.with_task("regression").with_target_column("target").build()
        settings.recipe.data.data_interface.feature_columns = ["int_feature", "float_feature"]

        handler = TabularDataHandler(settings)

        # When: Preparing data
        X, y, _ = handler.prepare_data(df)

        # Then: Numeric types are preserved
        assert pd.api.types.is_numeric_dtype(X["int_feature"])
        assert pd.api.types.is_numeric_dtype(X["float_feature"])
        assert pd.api.types.is_numeric_dtype(y)

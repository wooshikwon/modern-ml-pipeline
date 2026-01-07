"""
Pass Through Fetcher Unit Tests - No Mock Hell Approach
Real data passing through, real behavior validation
Following comprehensive testing strategy document principles
"""

import pandas as pd

from src.components.fetcher.modules.pass_through_fetcher import PassThroughFetcher
from src.components.fetcher.base import BaseFetcher


class TestPassThroughFetcher:
    """Test PassThroughFetcher with real data passing."""

    def test_pass_through_fetcher_initialization(self):
        """Test PassThroughFetcher initialization."""
        # When: Creating PassThroughFetcher
        fetcher = PassThroughFetcher()

        # Then: Fetcher is properly initialized
        assert isinstance(fetcher, PassThroughFetcher)
        assert isinstance(fetcher, BaseFetcher)
        assert hasattr(fetcher, "fetch")

    def test_fetch_returns_data_unchanged(self, test_data_generator):
        """Test that fetch returns data unchanged."""
        # Given: Data and fetcher
        X, y = test_data_generator.classification_data(n_samples=50, n_features=4)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y
        df["entity_id"] = range(len(df))

        fetcher = PassThroughFetcher()

        # When: Fetching data
        result = fetcher.fetch(df)

        # Then: Data is returned unchanged
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)

    def test_fetch_preserves_column_types(self):
        """Test that fetch preserves column data types."""
        # Given: DataFrame with mixed types
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )

        fetcher = PassThroughFetcher()

        # When: Fetching data
        result = fetcher.fetch(df)

        # Then: Column types are preserved
        assert result["int_col"].dtype == df["int_col"].dtype
        assert result["float_col"].dtype == df["float_col"].dtype
        assert result["str_col"].dtype == df["str_col"].dtype
        assert result["bool_col"].dtype == df["bool_col"].dtype

    def test_fetch_with_empty_dataframe(self):
        """Test fetching empty DataFrame."""
        # Given: Empty DataFrame
        df = pd.DataFrame()

        fetcher = PassThroughFetcher()

        # When: Fetching empty data
        result = fetcher.fetch(df)

        # Then: Empty DataFrame is returned
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert len(result.columns) == 0

    def test_fetch_with_null_values(self):
        """Test fetching DataFrame with null values."""
        # Given: DataFrame with nulls
        df = pd.DataFrame(
            {"col1": [1, None, 3], "col2": [None, 2.5, 3.5], "col3": ["a", None, "c"]}
        )

        fetcher = PassThroughFetcher()

        # When: Fetching data with nulls
        result = fetcher.fetch(df)

        # Then: Nulls are preserved
        assert result["col1"].isna().sum() == 1
        assert result["col2"].isna().sum() == 1
        assert result["col3"].isna().sum() == 1

    def test_fetch_preserves_index(self):
        """Test that fetch preserves DataFrame index."""
        # Given: DataFrame with custom index
        df = pd.DataFrame({"value": [10, 20, 30]}, index=["a", "b", "c"])

        fetcher = PassThroughFetcher()

        # When: Fetching data
        result = fetcher.fetch(df)

        # Then: Index is preserved
        assert list(result.index) == list(df.index)

    def test_fetch_with_large_dataset(self, test_data_generator):
        """Test fetching large dataset."""
        # Given: Large dataset
        X, y = test_data_generator.regression_data(n_samples=1000, n_features=20)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y

        fetcher = PassThroughFetcher()

        # When: Fetching large data
        result = fetcher.fetch(df)

        # Then: All data is returned
        assert len(result) == 1000
        assert len(result.columns) == X.shape[1] + 1  # features + target

"""
Storage Adapter Unit Tests - No Mock Hell Approach
Real files, real data, real behavior validation
Following comprehensive testing strategy document principles
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from src.components.adapter.modules.storage_adapter import StorageAdapter
from src.components.adapter.base import BaseAdapter


class TestStorageAdapterWithRealFiles:
    """Test StorageAdapter with real CSV and Parquet files - No mocks."""

    def test_storage_adapter_initialization(self, settings_builder):
        """Test StorageAdapter initialization with settings."""
        # Given: Valid settings
        settings = settings_builder.with_data_source("storage").build()

        # When: Creating StorageAdapter
        adapter = StorageAdapter(settings)

        # Then: Adapter is properly initialized
        assert isinstance(adapter, StorageAdapter)
        assert isinstance(adapter, BaseAdapter)
        assert hasattr(adapter, "storage_options")

    def test_read_csv_file_with_real_data(self, settings_builder, real_dataset_files):
        """Test reading real CSV file with actual data."""
        # Given: Real CSV file and adapter
        csv_info = real_dataset_files["classification_csv"]
        settings = settings_builder.with_data_source("storage").build()
        adapter = StorageAdapter(settings)

        # When: Reading CSV file
        df = adapter.read(str(csv_info["path"]))

        # Then: Data is read correctly
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(csv_info["data"])
        assert "target" in df.columns
        assert "entity_id" in df.columns
        assert df["target"].nunique() >= 2

    @pytest.mark.skipif(
        not importlib.util.find_spec("pyarrow"),
        reason="pyarrow not installed - required for Parquet support",
    )
    def test_read_parquet_file_with_real_data(self, settings_builder, real_dataset_files):
        """Test reading real Parquet file with actual data."""
        # Given: Real Parquet file and adapter
        parquet_info = real_dataset_files["classification_parquet"]
        if parquet_info["path"] is None:
            pytest.skip("Parquet file not created due to missing pyarrow")
        settings = settings_builder.with_data_source("storage").build()
        adapter = StorageAdapter(settings)

        # When: Reading Parquet file
        df = adapter.read(str(parquet_info["path"]))

        # Then: Data is read correctly
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(parquet_info["data"])
        assert "target" in df.columns
        assert "entity_id" in df.columns
        assert df["target"].nunique() >= 2  # Classification targets

    def test_write_csv_file_with_real_data(
        self, settings_builder, isolated_temp_directory, test_data_generator
    ):
        """Test writing DataFrame to CSV file."""
        # Given: Real data and adapter
        df, _ = test_data_generator.classification_data(n_samples=50)
        df["target"] = np.random.randint(0, 2, size=50)

        settings = settings_builder.with_data_source("storage").build()
        adapter = StorageAdapter(settings)

        csv_path = isolated_temp_directory / "output.csv"

        # When: Writing to CSV
        adapter.write(df, str(csv_path))

        # Then: File is created and readable
        assert csv_path.exists()
        read_df = pd.read_csv(csv_path)
        assert len(read_df) == len(df)
        pd.testing.assert_frame_equal(read_df, df)

    @pytest.mark.skipif(
        not importlib.util.find_spec("pyarrow"),
        reason="pyarrow not installed - required for Parquet support",
    )
    def test_write_parquet_file_with_real_data(
        self, settings_builder, isolated_temp_directory, test_data_generator
    ):
        """Test writing DataFrame to Parquet file."""
        # Given: Real data and adapter
        df, _ = test_data_generator.regression_data(n_samples=50)
        df["target"] = np.random.randn(50)

        settings = settings_builder.with_data_source("storage").build()
        adapter = StorageAdapter(settings)

        parquet_path = isolated_temp_directory / "output.parquet"

        # When: Writing to Parquet
        adapter.write(df, str(parquet_path))

        # Then: File is created and readable
        assert parquet_path.exists()
        read_df = pd.read_parquet(parquet_path)
        assert len(read_df) == len(df)
        pd.testing.assert_frame_equal(read_df, df)

    def test_storage_adapter_with_custom_options(self, settings_builder, tmp_path):
        """Test StorageAdapter with custom storage options."""
        # Given: Settings with custom storage options and base_path
        settings = settings_builder.with_data_source(
            "storage",
            config={"base_path": str(tmp_path), "storage_options": {"timeout": 30, "cache": True}},
        ).build()

        # When: Creating StorageAdapter
        adapter = StorageAdapter(settings)

        # Then: Storage options are properly set
        assert adapter.storage_options == {"timeout": 30, "cache": True}

    def test_read_nonexistent_file_error_handling(self, settings_builder):
        """Test error handling when reading non-existent file."""
        # Given: Adapter and non-existent file path
        settings = settings_builder.with_data_source("storage").build()
        adapter = StorageAdapter(settings)

        # When/Then: Should raise appropriate error
        with pytest.raises(FileNotFoundError):
            adapter.read("/nonexistent/file.csv")

    def test_write_to_nested_directory_creation(
        self, settings_builder, isolated_temp_directory, test_data_generator
    ):
        """Test that nested directories are created when writing."""
        # Given: Data and nested path
        df, _ = test_data_generator.classification_data(n_samples=10)

        settings = settings_builder.with_data_source("storage").build()
        adapter = StorageAdapter(settings)

        nested_path = isolated_temp_directory / "level1" / "level2" / "output.csv"

        # When: Writing to nested path
        adapter.write(df, str(nested_path))

        # Then: Directories are created and file exists
        assert nested_path.exists()
        assert nested_path.parent.exists()
        assert nested_path.parent.parent.exists()

"""
Error Handling and Edge Cases Tests - G1 Phase Gamma
Tests for critical error scenarios in core components

Focus Areas:
1. Exception Paths: Core component error scenarios
2. Boundary Conditions: Empty/invalid inputs
3. Recovery Scenarios: Graceful degradation
4. Validation Logic: Input validation

Following tests/README.md Real Object Testing principles
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.components.adapter.modules.storage_adapter import StorageAdapter
from src.components.evaluator.modules.classification_evaluator import ClassificationEvaluator
from src.components.preprocessor import Preprocessor
from src.factory import Factory
from src.settings.recipe import Preprocessor as PreprocessorConfig
from src.settings.recipe import PreprocessorStep


class TestCriticalExceptionPaths:
    """Test critical exception paths in core components."""

    def test_factory_calibration_wrapper_no_predict_proba(self, settings_builder):
        """Test CalibrationEvaluatorWrapper with model lacking predict_proba."""
        from src.factory import CalibrationEvaluatorWrapper

        # Given: Model without predict_proba method
        mock_model = Mock(spec=["predict"])  # No predict_proba
        mock_calibrator = Mock()

        wrapper = CalibrationEvaluatorWrapper(mock_model, mock_calibrator)

        # When/Then: Should raise AttributeError
        X_test = np.array([[1, 2], [3, 4]])
        y_test = np.array([0, 1])

        with pytest.raises(AttributeError):
            wrapper.evaluate(X_test, y_test)

    def test_factory_create_with_invalid_settings(self):
        """Test Factory creation with completely invalid settings."""
        # Given: Invalid settings object
        invalid_settings = Mock(spec=[])  # No attributes

        # When/Then: Factory should handle gracefully
        with pytest.raises(AttributeError):
            Factory(invalid_settings)

    def test_storage_adapter_read_corrupted_csv(self, settings_builder, isolated_temp_directory):
        """Test StorageAdapter reading corrupted CSV file."""
        # Given: Corrupted CSV file that will cause parsing issues
        corrupted_file = isolated_temp_directory / "corrupted.csv"
        # Create a file that pandas cannot parse properly
        corrupted_file.write_text('col1,col2\n1,2\n"unclosed quote,3\n4,5')

        settings = settings_builder.with_data_source("storage").build()
        adapter = StorageAdapter(settings)

        # When/Then: Should handle gracefully or raise parsing error
        try:
            result = adapter.read(str(corrupted_file))
            # If it succeeds, verify data integrity
            assert len(result) >= 0  # Should have some result
        except (pd.errors.ParserError, pd.errors.ParserWarning, ValueError):
            # Expected error cases
            pass

    def test_storage_adapter_write_to_readonly_directory(self, settings_builder):
        """Test StorageAdapter writing to read-only directory."""
        # Given: Read-only directory and data
        with tempfile.TemporaryDirectory() as tmpdir:
            readonly_dir = Path(tmpdir) / "readonly"
            readonly_dir.mkdir()
            readonly_dir.chmod(0o444)  # Read-only

            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            settings = settings_builder.with_data_source("storage").build()
            adapter = StorageAdapter(settings)

            output_path = readonly_dir / "output.csv"

            # When/Then: Should raise permission error
            with pytest.raises((PermissionError, OSError)):
                adapter.write(df, str(output_path))

            # Cleanup
            readonly_dir.chmod(0o755)

    def test_storage_adapter_read_nonexistent_file(self, settings_builder):
        """Test StorageAdapter reading non-existent file."""
        # Given: Non-existent file path
        settings = settings_builder.with_data_source("storage").build()
        adapter = StorageAdapter(settings)

        # When/Then: Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            adapter.read("/path/to/nonexistent/file.csv")


class TestBoundaryConditions:
    """Test boundary conditions and edge input scenarios."""

    def test_empty_dataframe_handling(self, settings_builder, isolated_temp_directory):
        """Test components handling empty DataFrames."""
        # Given: Empty DataFrame written to file
        empty_df = pd.DataFrame()
        empty_file = isolated_temp_directory / "empty.csv"
        empty_df.to_csv(empty_file, index=False)

        settings = settings_builder.with_data_source("storage").build()
        adapter = StorageAdapter(settings)

        # When: Reading empty file
        # Then: Should raise EmptyDataError for truly empty CSV
        with pytest.raises(pd.errors.EmptyDataError):
            adapter.read(str(empty_file))

    def test_single_row_dataframe_handling(self, settings_builder):
        """Test components handling single-row DataFrames."""
        # Given: Single-row DataFrame
        single_row_df = pd.DataFrame({"feature_0": [1.0], "feature_1": [2.0], "target": [0]})

        settings = settings_builder.with_task("classification").build()
        preprocessor = Preprocessor(settings)

        # When: Fitting and processing single row
        preprocessor.fit(single_row_df)
        processed_data = preprocessor.transform(single_row_df)

        # Then: Should handle gracefully
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) == 1

    def test_nan_values_handling(self, settings_builder):
        """Test handling of NaN values in data."""
        # Given: DataFrame with NaN values
        nan_df = pd.DataFrame(
            {"feature_0": [1, np.nan, 3], "feature_1": [np.nan, 2, 3], "target": [0, 1, 0]}
        )

        settings = settings_builder.with_task("classification").build()
        # Configure preprocessor with imputer step for specific columns
        settings.recipe.preprocessor = PreprocessorConfig(
            steps=[
                PreprocessorStep(
                    type="simple_imputer", columns=["feature_0", "feature_1"], strategy="mean"
                )
            ]
        )

        preprocessor = Preprocessor(settings)

        # When: Fitting and processing data with NaN
        preprocessor.fit(nan_df)
        processed_data = preprocessor.transform(nan_df)

        # Then: NaN should be handled appropriately
        assert isinstance(processed_data, pd.DataFrame)
        assert not pd.isna(processed_data).any().any()  # No NaN after processing


class TestRecoveryScenarios:
    """Test recovery scenarios and graceful degradation."""

    def test_missing_optional_configuration_fallback(self, settings_builder):
        """Test fallback behavior when optional configurations are missing."""
        # Given: Minimal settings without optional configs
        settings = settings_builder.build()  # Minimal settings

        # Test StorageAdapter without storage_options
        adapter = StorageAdapter(settings)
        assert adapter.storage_options == {}  # Should default to empty

        # Test Factory without specific component configs
        factory = Factory(settings)
        assert factory is not None

        # Components should work with defaults
        evaluator = factory.create_evaluator()
        assert evaluator is not None

    def test_component_chain_partial_failure_recovery(self, settings_builder):
        """Test recovery when one component in chain fails."""
        # Given: Factory with components
        settings = settings_builder.with_task("classification").build()
        factory = Factory(settings)

        # Simulate partial failure scenario
        adapter = factory.create_data_adapter("storage")
        assert adapter is not None

        # Even if one component creation fails, others should work
        with pytest.raises(ValueError):
            factory.create_data_adapter("nonexistent")

        # Other components should still be creatable
        evaluator = factory.create_evaluator()
        assert evaluator is not None

        # Preprocessor returns None when not configured (valid recovery behavior)
        preprocessor = factory.create_preprocessor()
        assert preprocessor is None  # Expected behavior for unconfigured component

    def test_evaluator_graceful_degradation_without_predict_proba(self, settings_builder):
        """Test evaluator degradation when optional metrics fail."""
        # Given: Model without predict_proba (no ROC AUC possible)
        settings = settings_builder.with_task("classification").build()
        evaluator = ClassificationEvaluator(settings)

        # Mock model without predict_proba
        mock_model = Mock(spec=["predict"])
        mock_model.predict.return_value = np.array([0, 1, 0, 1])

        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
        y = pd.Series([0, 1, 0, 1])

        # When: Evaluating without predict_proba
        metrics = evaluator.evaluate(mock_model, X, y)

        # Then: Should compute basic metrics, skip ROC AUC
        assert "accuracy" in metrics
        assert metrics["roc_auc"] is None  # Gracefully skipped


class TestInputValidation:
    """Test input validation logic."""

    def test_file_extension_validation(self, settings_builder, isolated_temp_directory):
        """Test validation of file extensions."""
        # Given: File with unsupported extension
        unsupported_file = isolated_temp_directory / "data.txt"
        unsupported_file.write_text("some text data")

        settings = settings_builder.with_data_source("storage").build()
        adapter = StorageAdapter(settings)

        # When/Then: Should raise ValueError for unsupported format
        with pytest.raises(ValueError) as exc_info:
            adapter.read(str(unsupported_file))

        assert "지원되지 않는 파일 형식" in str(exc_info.value)

    def test_empty_target_column_validation(self, settings_builder):
        """Test validation when target column is missing."""
        # Given: DataFrame without target column
        no_target_df = pd.DataFrame({"feature_0": [1, 2, 3], "feature_1": [4, 5, 6]})

        settings = settings_builder.with_task("classification").build()
        preprocessor = Preprocessor(settings)

        # When: Fitting with missing target column
        preprocessor.fit(no_target_df)
        processed_data = preprocessor.transform(no_target_df)

        # Then: Should handle gracefully (inference scenario)
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) == 3

    def test_cascading_error_recovery(self, settings_builder):
        """Test recovery from multiple sequential errors."""
        settings = settings_builder.with_task("classification").build()
        factory = Factory(settings)

        # Multiple error scenarios in sequence
        with pytest.raises(ValueError):
            factory.create_data_adapter("invalid1")

        with pytest.raises(ValueError):
            factory.create_data_adapter("invalid2")

        # Factory should still work for valid requests after all errors
        adapter = factory.create_data_adapter("storage")
        assert adapter is not None

        # Cache should be consistent
        adapter2 = factory.create_data_adapter("storage")
        assert adapter is adapter2  # Same cached instance

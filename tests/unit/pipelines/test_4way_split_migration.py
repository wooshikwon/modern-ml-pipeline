"""
4-way Split Migration Test

Verifies TabularDataHandler implements standardized 4-way split (train/val/test/calibration)
with no data leakage between splits.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from mmp.components.datahandler.modules.tabular_handler import TabularDataHandler


@pytest.fixture
def sample_data():
    """100-row sample dataset."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
            "entity_id": range(100),
        }
    )


@pytest.fixture
def mock_settings():
    """Mock settings for TabularDataHandler."""
    settings = Mock()
    settings.recipe.task_choice = "classification"
    settings.recipe.data.split.train = 0.6
    settings.recipe.data.split.validation = 0.2
    settings.recipe.data.split.test = 0.1
    settings.recipe.data.split.calibration = 0.1
    settings.recipe.data.data_interface.target_column = "target"
    settings.recipe.data.data_interface.timestamp_column = "timestamp"
    settings.recipe.data.data_interface.entity_columns = ["entity_id"]
    settings.recipe.data.data_interface.feature_columns = None
    settings.recipe.evaluation.random_state = 42
    settings.recipe.model.computed = {"seed": 42}
    return settings


class TestTabular4WaySplit:
    """TabularDataHandler 4-way split correctness."""

    def test_returns_10_elements_with_calibration(self, sample_data, mock_settings):
        """split_and_prepare returns 10-tuple including calibration data."""
        handler = TabularDataHandler(mock_settings)
        result = handler.split_and_prepare(sample_data)

        assert len(result) == 10
        (
            X_train, y_train, add_train,
            X_val, y_val, add_val,
            X_test, y_test, add_test,
            calibration_data,
        ) = result

        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0
        assert calibration_data is not None

        X_calib, y_calib, add_calib = calibration_data
        assert len(X_calib) > 0

    def test_no_data_leakage_between_splits(self, sample_data, mock_settings):
        """All four splits have disjoint indices."""
        handler = TabularDataHandler(mock_settings)
        result = handler.split_and_prepare(sample_data)

        X_train, _, _, X_val, _, _, X_test, _, _, calibration_data = result
        X_calib = calibration_data[0]

        index_sets = [
            set(X_train.index),
            set(X_val.index),
            set(X_test.index),
            set(X_calib.index),
        ]

        for i in range(len(index_sets)):
            for j in range(i + 1, len(index_sets)):
                assert index_sets[i].isdisjoint(index_sets[j]), (
                    f"Split {i} and {j} share indices"
                )

    @pytest.mark.parametrize(
        "calib_ratio,expect_calibration",
        [
            (0.1, True),
            (0.0, False),
        ],
    )
    def test_calibration_presence(
        self, sample_data, mock_settings, calib_ratio, expect_calibration
    ):
        """Calibration data is present only when calibration ratio > 0."""
        mock_settings.recipe.data.split.calibration = calib_ratio
        if calib_ratio == 0.0:
            mock_settings.recipe.data.split.test = 0.2  # redistribute

        handler = TabularDataHandler(mock_settings)
        result = handler.split_and_prepare(sample_data)
        calibration_data = result[9]

        if expect_calibration:
            assert calibration_data is not None
        else:
            assert calibration_data is None

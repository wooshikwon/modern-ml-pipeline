"""
Standard mock objects for tests.

Extracted from individual test files to eliminate duplication.
Import directly: `from tests.fixtures.mocks import make_mock_sklearn_model, ...`
Or use via conftest.py fixtures.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd


def make_mock_sklearn_model(predict_return=None, predict_proba_return=None):
    """Create a standard mock sklearn model.

    Args:
        predict_return: Value for model.predict(). Defaults to np.array([0, 1, 0]).
        predict_proba_return: Value for model.predict_proba(). Defaults to 3-class probabilities.
    """
    model = Mock()
    model.predict.return_value = (
        predict_return if predict_return is not None else np.array([0, 1, 0])
    )
    model.predict_proba.return_value = (
        predict_proba_return
        if predict_proba_return is not None
        else np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
    )
    return model


def make_mock_passthrough_fetcher():
    """Create a mock fetcher that augments data with extra columns."""
    fetcher = Mock()

    def fetch_side_effect(df, run_mode="batch"):
        augmented = df.copy()
        augmented["augmented_feature_1"] = [0.5] * len(df)
        augmented["augmented_feature_2"] = [100] * len(df)
        return augmented

    fetcher.fetch.side_effect = fetch_side_effect
    fetcher._fetcher_config = {"entity_columns": ["user_id"]}
    return fetcher


def make_mock_failing_fetcher(error_message="Feature Store unavailable"):
    """Create a mock fetcher that raises on fetch()."""
    fetcher = Mock()
    fetcher.fetch.side_effect = Exception(error_message)
    fetcher._fetcher_config = {"entity_columns": ["user_id"]}
    return fetcher


def make_mock_model_with_schema(
    entity_columns=None,
    feature_columns=None,
    target_column="target",
    trained_fetcher=None,
):
    """Create a mock MLflow model with data_interface_schema.

    Args:
        entity_columns: List of entity column names.
        feature_columns: List of feature column names.
        target_column: Target column name.
        trained_fetcher: Optional trained fetcher mock.
    """
    model = Mock()
    wrapped_model = Mock()
    wrapped_model.data_interface_schema = {
        "entity_columns": entity_columns or [],
        "feature_columns": feature_columns or [],
        "target_column": target_column,
    }
    wrapped_model.trained_fetcher = trained_fetcher
    model.unwrap_python_model.return_value = wrapped_model
    model.predict.return_value = pd.DataFrame({"prediction": [0]})
    return model

"""
API required columns validation tests.

Tests mmp.serving.validators.validate_required_columns directly:
- Feature Store fetcher: only entity_columns required
- pass_through fetcher: all feature_columns required
- Missing columns detected correctly
"""

from unittest.mock import Mock

import pandas as pd
import pytest

from mmp.serving.validators import validate_required_columns


def _make_model(entity_columns, feature_columns, fetcher_config=None):
    """Create mock model with data_interface_schema and optional fetcher."""
    model = Mock()
    wrapped = Mock()
    wrapped.data_interface_schema = {
        "entity_columns": entity_columns,
        "feature_columns": feature_columns,
        "target_column": "target",
    }

    if fetcher_config is not None:
        fetcher = Mock()
        fetcher._fetcher_config = fetcher_config
        wrapped.trained_fetcher = fetcher
    else:
        wrapped.trained_fetcher = None

    model.unwrap_python_model.return_value = wrapped
    model.metadata = None  # no MLflow signature
    return model


@pytest.mark.parametrize(
    "entity_cols,feature_cols,fetcher_config,input_cols,should_raise",
    [
        # Feature Store: only entity_columns required -- all present
        (
            ["user_id", "product_id"],
            ["f1", "f2", "fs_f1"],
            {"entity_columns": ["user_id", "product_id"]},
            ["user_id", "product_id"],
            False,
        ),
        # Feature Store: entity column missing
        (
            ["user_id", "product_id"],
            ["f1", "f2"],
            {"entity_columns": ["user_id", "product_id"]},
            ["user_id"],  # product_id missing
            True,
        ),
        # pass_through: all present -- no error
        (
            ["user_id"],
            ["f1", "f2"],
            None,
            ["user_id", "f1", "f2"],
            False,
        ),
        # pass_through with no schema: no required cols -- passes
        (
            [],
            [],
            None,
            ["anything"],
            False,
        ),
    ],
    ids=[
        "fs_all_entities_present",
        "fs_entity_missing",
        "passthrough_all_present",
        "empty_schema_passes",
    ],
)
def test_validate_required_columns(
    entity_cols, feature_cols, fetcher_config, input_cols, should_raise
):
    model = _make_model(entity_cols, feature_cols, fetcher_config)
    input_df = pd.DataFrame({col: [1] for col in input_cols})

    if should_raise:
        with pytest.raises(Exception):  # HTTPException(422)
            validate_required_columns(input_df, model)
    else:
        # Should not raise
        validate_required_columns(input_df, model)

"""
PyfuncWrapper fetcher integration tests.

- predict() calls fetcher.fetch() with correct run_mode
- Augmented features are passed to model
- Graceful degradation on fetcher failure
- Works without fetcher
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from mmp.utils.integrations.pyfunc_wrapper import PyfuncWrapper


@pytest.fixture
def mock_model():
    model = Mock()
    model.predict.return_value = np.array([0, 1, 0])
    return model


@pytest.fixture
def mock_fetcher():
    fetcher = Mock()

    def fetch_side_effect(df, run_mode="batch"):
        augmented = df.copy()
        augmented["aug_f1"] = 0.5
        augmented["aug_f2"] = 100
        return augmented

    fetcher.fetch.side_effect = fetch_side_effect
    fetcher._fetcher_config = {"entity_columns": ["user_id"]}
    return fetcher


@pytest.fixture
def wrapper_with_fetcher(component_test_context, mock_model, mock_fetcher):
    with component_test_context.classification_stack() as ctx:
        return PyfuncWrapper(
            settings=ctx.settings,
            trained_model=mock_model,
            trained_fetcher=mock_fetcher,
            data_interface_schema={
                "feature_columns": ["f1", "f2", "aug_f1", "aug_f2"],
                "target_column": "target",
                "entity_columns": ["user_id"],
            },
        )


@pytest.fixture
def input_df():
    return pd.DataFrame({"user_id": [1, 2, 3], "f1": [1.0, 2.0, 3.0], "f2": [0.5, 1.5, 2.5]})


class TestFetcherIntegration:
    @pytest.mark.parametrize("run_mode", ["batch", "serving"])
    def test_fetcher_called_with_run_mode(
        self, wrapper_with_fetcher, mock_fetcher, input_df, run_mode
    ):
        """predict() passes run_mode to fetcher.fetch()."""
        params = {"run_mode": run_mode} if run_mode != "batch" else None
        wrapper_with_fetcher.predict(context=None, model_input=input_df, params=params)

        mock_fetcher.fetch.assert_called_once()
        assert mock_fetcher.fetch.call_args[1]["run_mode"] == run_mode

    def test_augmented_features_passed_to_model(self, wrapper_with_fetcher, mock_model, input_df):
        """Model receives augmented columns from fetcher."""
        wrapper_with_fetcher.predict(context=None, model_input=input_df)

        model_input = mock_model.predict.call_args[0][0]
        assert "aug_f1" in model_input.columns
        assert "aug_f2" in model_input.columns


class TestFetcherGracefulDegradation:
    def test_continues_on_fetcher_failure(self, component_test_context):
        """predict() succeeds even when fetcher raises."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1])

        failing_fetcher = Mock()
        failing_fetcher.fetch.side_effect = Exception("Feature Store down")
        failing_fetcher._fetcher_config = {"entity_columns": ["user_id"]}

        with component_test_context.classification_stack() as ctx:
            wrapper = PyfuncWrapper(
                settings=ctx.settings,
                trained_model=mock_model,
                trained_fetcher=failing_fetcher,
                data_interface_schema={
                    "feature_columns": ["f1", "f2"],
                    "target_column": "target",
                },
            )

        input_df = pd.DataFrame({"user_id": [1, 2], "f1": [1.0, 2.0], "f2": [0.5, 1.5]})
        result = wrapper.predict(context=None, model_input=input_df)
        assert len(result) == 2


class TestWithoutFetcher:
    def test_predict_works_without_fetcher(self, component_test_context):
        """predict() works when trained_fetcher is None."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0])

        with component_test_context.classification_stack() as ctx:
            wrapper = PyfuncWrapper(
                settings=ctx.settings,
                trained_model=mock_model,
                trained_fetcher=None,
                data_interface_schema={
                    "feature_columns": ["f1", "f2"],
                    "target_column": "target",
                },
            )

        input_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.5, 1.5, 2.5]})
        result = wrapper.predict(context=None, model_input=input_df)
        assert len(result) == 3
        assert wrapper.trained_fetcher is None

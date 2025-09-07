import pytest
from src.settings.config import Config, Environment, DataSource, FeatureStore, Output, OutputTarget


def _base_config_kwargs():
    return {
        "environment": Environment(name="local"),
        "mlflow": None,
        "data_source": DataSource(name="Local", adapter_type="storage", config={}),
        "feature_store": FeatureStore(provider="none"),
        "serving": None,
        "artifact_store": None,
    }


def test_output_schema_storage_targets_parse():
    cfg = Config(
        **_base_config_kwargs(),
        output=Output(
            inference=OutputTarget(
                name="InferenceOutput",
                enabled=True,
                adapter_type="storage",
                config={"base_path": "./artifacts/predictions"},
            ),
            preprocessed=OutputTarget(
                name="PreprocessedOutput",
                enabled=False,
                adapter_type="storage",
                config={"base_path": "./artifacts/preprocessed"},
            ),
        ),
    )

    assert cfg.output is not None
    assert cfg.output.inference.enabled is True
    assert cfg.output.inference.adapter_type == "storage"
    assert cfg.output.inference.config["base_path"].endswith("predictions")

    assert cfg.output.preprocessed.enabled is False
    assert cfg.output.preprocessed.adapter_type == "storage"


def test_output_schema_sql_and_bigquery_targets_parse():
    cfg = Config(
        **_base_config_kwargs(),
        output=Output(
            inference=OutputTarget(
                name="InferencePG",
                enabled=True,
                adapter_type="sql",
                config={"table": "predictions_local"},
            ),
            preprocessed=OutputTarget(
                name="PreprocBQ",
                enabled=True,
                adapter_type="bigquery",
                config={
                    "project_id": "proj",
                    "dataset_id": "ds",
                    "table": "tbl",
                    "location": "US",
                },
            ),
        ),
    )

    assert cfg.output.inference.adapter_type == "sql"
    assert cfg.output.inference.config["table"] == "predictions_local"

    assert cfg.output.preprocessed.adapter_type == "bigquery"
    assert cfg.output.preprocessed.config["project_id"] == "proj"
    assert cfg.output.preprocessed.config["dataset_id"] == "ds"
    assert cfg.output.preprocessed.config["table"] == "tbl" 
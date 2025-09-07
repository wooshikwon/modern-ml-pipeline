import types
from contextlib import contextmanager
from unittest.mock import MagicMock
import pandas as pd
import mlflow

from src.pipelines.inference_pipeline import run_inference_pipeline
from src.settings.loader import Settings
from src.settings.config import Config, Environment, DataSource, FeatureStore, Output, OutputTarget


@contextmanager
def _dummy_run_ctx(*args, **kwargs):
    class Info:
        run_id = "dummy-run-id"
    class Run:
        info = Info()
    yield Run()


class DummyRecipe:
    class Model:
        computed = {"seed": 42}
    class Data:
        class Loader:
            source_uri = "./data/input.parquet"
        loader = Loader()
    model = Model()
    data = Data()


def _base_settings_with_output(output: Output) -> Settings:
    cfg = Config(
        environment=Environment(name="local"),
        mlflow=None,
        data_source=DataSource(name="Local", adapter_type="storage", config={}),
        feature_store=FeatureStore(provider="none"),
        serving=None,
        artifact_store=None,
        output=output,
    )
    recipe = types.SimpleNamespace(model=types.SimpleNamespace(computed={"seed": 42}), data=types.SimpleNamespace(loader=types.SimpleNamespace(source_uri="./data/input.parquet")))
    settings = types.SimpleNamespace(config=cfg, recipe=recipe)
    return settings


def test_inference_output_storage(monkeypatch):
    df = pd.DataFrame({"a": [1, 2]})

    # Mock Factory and adapters
    factory_mock = MagicMock()
    storage_adapter = MagicMock()
    storage_adapter.read.return_value = df
    factory_mock.create_data_adapter.side_effect = lambda t=None: storage_adapter

    # Patch Factory and start_run in module
    monkeypatch.setattr("src.pipelines.inference_pipeline.Factory", lambda s: factory_mock)
    monkeypatch.setattr("src.pipelines.inference_pipeline.start_run", _dummy_run_ctx)

    # Mock model
    class DummyModel:
        def predict(self, X):
            return df
        def unwrap_python_model(self):
            return types.SimpleNamespace(loader_sql_snapshot="SELECT 1")
    monkeypatch.setattr("mlflow.pyfunc.load_model", lambda uri: DummyModel())

    out = Output(
        inference=OutputTarget(name="out", enabled=True, adapter_type="storage", config={"base_path": "./artifacts/predictions"}),
        preprocessed=OutputTarget(name="out2", enabled=False, adapter_type="storage", config={}),
    )
    settings = _base_settings_with_output(out)

    # Run
    run_inference_pipeline(settings, run_id="r1", data_path="./data/input.parquet")

    # Assert storage write called
    assert storage_adapter.write.called


def test_inference_output_disabled(monkeypatch):
    df = pd.DataFrame({"a": [1, 2]})
    factory_mock = MagicMock()
    storage_adapter = MagicMock()
    storage_adapter.read.return_value = df
    factory_mock.create_data_adapter.side_effect = lambda t=None: storage_adapter
    monkeypatch.setattr("src.pipelines.inference_pipeline.Factory", lambda s: factory_mock)
    monkeypatch.setattr("src.pipelines.inference_pipeline.start_run", _dummy_run_ctx)

    class DummyModel:
        def predict(self, X):
            return df
        def unwrap_python_model(self):
            return types.SimpleNamespace(loader_sql_snapshot="SELECT 1")
    monkeypatch.setattr("mlflow.pyfunc.load_model", lambda uri: DummyModel())

    out = Output(
        inference=OutputTarget(name="out", enabled=False, adapter_type="storage", config={}),
        preprocessed=OutputTarget(name="out2", enabled=False, adapter_type="storage", config={}),
    )
    settings = _base_settings_with_output(out)

    run_inference_pipeline(settings, run_id="r1", data_path="./data/input.parquet")

    # 저장 비활성화 → write가 호출되지 않음
    assert not storage_adapter.write.called 
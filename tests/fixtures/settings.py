"""
Test settings utilities: TestDataGenerator and SettingsBuilder.

Extracted from conftest.py to reduce its size.
Import these via conftest.py fixtures (settings_builder, test_data_generator).
"""

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from mmp.settings import (
    Calibration,
    Config,
    Data,
    DataInterface,
    DataSource,
    DataSplit,
    Environment,
    Evaluation,
    FeatureStore,
    Fetcher,
    HyperparametersTuning,
    Loader,
    MLflow,
    Model,
    Recipe,
    Settings,
)
from mmp.settings.config import Output, OutputTarget
from mmp.settings.recipe import Metadata


class TestDataGenerator:
    """Generate realistic test datasets for different ML tasks."""

    @staticmethod
    def classification_data(
        n_samples: int = 100, n_features: int = 5, n_classes: int = 2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate classification dataset."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features - 1),
            n_redundant=0,
            n_classes=n_classes,
            random_state=random_state,
        )

        feature_cols = [f"feature_{i+1}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df["entity_id"] = range(1, n_samples + 1)

        return df, y

    @staticmethod
    def regression_data(
        n_samples: int = 100, n_features: int = 5, random_state: int = 42
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate regression dataset."""
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features - 1),
            noise=0.1,
            random_state=random_state,
        )

        feature_cols = [f"feature_{i+1}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df["entity_id"] = range(1, n_samples + 1)

        return df, y

    @staticmethod
    def timeseries_data(
        n_samples: int = 100, n_features: int = 3, random_state: int = 42
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate time series dataset."""
        np.random.seed(random_state)

        time_index = pd.date_range("2023-01-01", periods=n_samples, freq="D")
        trend = np.linspace(0, 10, n_samples)
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 7)

        features_data = {}
        for i in range(n_features):
            noise = np.random.normal(0, 0.5, n_samples)
            features_data[f"feature_{i+1}"] = trend * (i + 1) + seasonal + noise

        df = pd.DataFrame(features_data)
        df["timestamp"] = time_index
        df["entity_id"] = range(1, n_samples + 1)

        y = df[[f"feature_{i+1}" for i in range(n_features)]].sum(axis=1) + np.random.normal(
            0, 1, n_samples
        )

        return df, y.values


class SettingsBuilder:
    """Builder pattern for creating test Settings objects."""

    def __init__(self):
        self._environment = Environment(name="test")
        self._mlflow = MLflow(
            tracking_uri=f"file://{Path(tempfile.gettempdir()) / 'mmp_default_mlruns'}",
            experiment_name="test_experiment",
        )
        self._data_source = DataSource(
            name="test_storage",
            adapter_type="storage",
            config={"base_path": "tests/fixtures/data", "storage_options": {}},
        )
        self._feature_store = FeatureStore(provider="none")
        self._output = Output(
            inference=OutputTarget(
                name="test_output",
                adapter_type="storage",
                config={"base_path": "tests/fixtures/output"},
            )
        )

        self._task_choice = "classification"
        self._model = Model(
            class_path="sklearn.ensemble.RandomForestClassifier",
            library="sklearn",
            hyperparameters=HyperparametersTuning(
                tuning_enabled=False, values={"n_estimators": 10, "random_state": 42}
            ),
            calibration=None,
        )
        self._data = Data(
            loader=Loader(source_uri="test_data.csv"),
            data_interface=DataInterface(target_column="target", entity_columns=["entity_id"]),
            fetcher=Fetcher(type="pass_through", feature_views=None),
            split=DataSplit(train=0.6, validation=0.2, test=0.2, calibration=0.0),
        )
        self._evaluation = Evaluation(metrics=["accuracy", "roc_auc"], random_state=42)

    def with_environment(self, name: str) -> "SettingsBuilder":
        """Set environment name."""
        self._environment = Environment(name=name)
        return self

    def with_mlflow(self, tracking_uri: str, experiment_name: str) -> "SettingsBuilder":
        """Set MLflow configuration."""
        self._mlflow = MLflow(tracking_uri=tracking_uri, experiment_name=experiment_name)
        return self

    def with_data_source(
        self, adapter_type: str, name: str = "test_source", config: Dict[str, Any] = None
    ) -> "SettingsBuilder":
        """Set data source configuration."""
        if config is None:
            if adapter_type == "storage":
                from mmp.settings.config import LocalFilesConfig

                config = LocalFilesConfig(base_path="tests/fixtures/data", storage_options={})
            elif adapter_type == "sql":
                from mmp.settings.config import PostgreSQLConfig

                config = PostgreSQLConfig(
                    connection_uri="postgresql://test:test@localhost:5432/test", query_timeout=300
                )
            else:
                config = {}

        self._data_source = DataSource(name=name, adapter_type=adapter_type, config=config)
        return self

    def with_data_path(self, path: str) -> "SettingsBuilder":
        """Set data file path."""
        self._data = Data(
            loader=Loader(source_uri=path),
            data_interface=self._data.data_interface,
            fetcher=self._data.fetcher,
            split=self._data.split,
        )
        return self

    def with_task(self, task_type: str) -> "SettingsBuilder":
        """Set task type (classification, regression, timeseries, etc.)."""
        self._task_choice = task_type

        if task_type == "clustering":
            self._data = Data(
                loader=self._data.loader,
                data_interface=DataInterface(
                    target_column=None,
                    entity_columns=self._data.data_interface.entity_columns,
                ),
                fetcher=self._data.fetcher,
                split=self._data.split,
            )
        elif task_type == "timeseries":
            self._data = Data(
                loader=self._data.loader,
                data_interface=DataInterface(
                    target_column=self._data.data_interface.target_column,
                    entity_columns=self._data.data_interface.entity_columns,
                    timestamp_column="timestamp",
                ),
                fetcher=self._data.fetcher,
                split=self._data.split,
            )
        return self

    def with_timestamp_column(self, name: str) -> "SettingsBuilder":
        """Set timestamp column for timeseries tasks."""
        if not name:
            raise ValueError("timestamp_column must be non-empty")
        self._data.data_interface.timestamp_column = name
        return self

    def with_treatment_column(self, name: str) -> "SettingsBuilder":
        """Set treatment column for causal tasks."""
        if not name:
            raise ValueError("treatment_column must be non-empty")
        self._data.data_interface.treatment_column = name
        return self

    def with_model(
        self, class_path: str, library: str = None, hyperparameters: Dict[str, Any] = None
    ) -> "SettingsBuilder":
        """Set model configuration."""
        if library is None:
            if "sklearn" in class_path:
                library = "sklearn"
            elif "xgboost" in class_path:
                library = "xgboost"
            elif "lightgbm" in class_path:
                library = "lightgbm"
            else:
                library = "unknown"

        if hyperparameters is None:
            if "LinearRegression" in class_path:
                hyperparams = {}
            else:
                hyperparams = {"random_state": 42}
        else:
            hyperparams = hyperparameters

        self._model = Model(
            class_path=class_path,
            library=library,
            hyperparameters=HyperparametersTuning(tuning_enabled=False, values=hyperparams),
            calibration=self._model.calibration,
        )
        return self

    def with_hyperparameter_tuning(
        self,
        enabled: bool = True,
        metric: str = "accuracy",
        direction: str = "maximize",
        n_trials: int = 10,
    ) -> "SettingsBuilder":
        """Enable hyperparameter tuning."""
        if enabled:
            self._model.hyperparameters = HyperparametersTuning(
                tuning_enabled=True,
                optimization_metric=metric,
                direction=direction,
                n_trials=n_trials,
                fixed={"random_state": 42},
                tunable={
                    "n_estimators": {"type": "int", "low": 10, "high": 100},
                    "max_depth": {"type": "int", "low": 3, "high": 10},
                },
            )
        else:
            self._model.hyperparameters.tuning_enabled = False
        return self

    def with_calibration(self, enabled: bool = True, method: str = "beta") -> "SettingsBuilder":
        """Set calibration configuration."""
        if enabled:
            self._model.calibration = Calibration(enabled=True, method=method)
        else:
            self._model.calibration = None
        return self

    def with_target_column(self, target_col: str) -> "SettingsBuilder":
        """Set target column name."""
        self._data.data_interface.target_column = target_col
        return self

    def with_entity_columns(self, entity_cols: list) -> "SettingsBuilder":
        """Set entity columns."""
        self._data.data_interface.entity_columns = entity_cols
        return self

    def with_feature_store(self, enabled: bool = True) -> "SettingsBuilder":
        """Enable/disable feature store."""
        if enabled:
            from mmp.settings.config import FeastConfig

            self._feature_store = FeatureStore(
                provider="feast",
                feast_config=FeastConfig(
                    project="test_project",
                    registry="registry.db",
                    online_store={"type": "sqlite", "path": "test_online_store.db"},
                    offline_store={"type": "file", "path": "./feature_repo"},
                ),
            )
            self._data.fetcher = Fetcher(type="feature_store", feature_views={})
        else:
            self._feature_store = FeatureStore(provider="none")
            self._data.fetcher = Fetcher(type="pass_through", feature_views=None)
        return self

    def with_data_split(
        self,
        train: float = 0.6,
        validation: float = 0.2,
        test: float = 0.2,
        calibration: float = 0.0,
        strategy: str = "random",
        temporal_column: str = None,
    ) -> "SettingsBuilder":
        """Set data split ratios and strategy. Default: 60/20/20/0, random."""
        self._data.split = DataSplit(
            strategy=strategy,
            temporal_column=temporal_column,
            train=train,
            validation=validation,
            test=test,
            calibration=calibration,
        )
        return self

    def build(self) -> Settings:
        """Build the Settings object."""
        config = Config(
            environment=self._environment,
            mlflow=self._mlflow,
            data_source=self._data_source,
            feature_store=self._feature_store,
            output=self._output,
        )

        recipe = Recipe(
            name="test_recipe",
            task_choice=self._task_choice,
            model=self._model,
            data=self._data,
            evaluation=self._evaluation,
            metadata=Metadata(
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                description="Test recipe generated by SettingsBuilder",
            ),
        )

        return Settings(config=config, recipe=recipe)

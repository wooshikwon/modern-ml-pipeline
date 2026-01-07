"""
Modern ML Pipeline - Core Test Fixtures
No Mock Hell: Real objects with test data, minimal mocking, fast execution
Based on comprehensive testing strategy document
"""

import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

# Import Settings system
from src.settings import (
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
from src.settings.config import Output, OutputTarget
from src.settings.recipe import Metadata

# from src.settings.factory import RecipeFactory, ConfigFactory  # deprecated


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY MANAGEMENT & CLEANUP (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════


def _global_kill_enabled() -> bool:
    """Return True if global kill behavior is explicitly enabled via env var."""
    return os.getenv("MMP_ENABLE_GLOBAL_KILL", "0") == "1"


def cleanup_mlflow_processes():
    """Clean up any MLflow server processes to prevent memory leaks"""
    import signal
    import subprocess

    try:
        # Find MLflow server processes
        result = subprocess.run(
            ["pgrep", "-f", "mlflow.server"], capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                try:
                    pid_int = int(pid.strip())
                    os.kill(pid_int, signal.SIGTERM)
                except (ValueError, ProcessLookupError, PermissionError):
                    # Process already dead or permission denied
                    pass

        # Also try pkill as backup
        try:
            subprocess.run(["pkill", "-f", "mlflow.server"], capture_output=True, timeout=3)
            subprocess.run(["pkill", "-f", "uvicorn.*mlflow"], capture_output=True, timeout=3)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    except (subprocess.TimeoutExpired, FileNotFoundError):
        # pgrep not available or timeout - skip cleanup
        pass


def finalize_coverage():
    """Combine coverage files and clean up temporary files"""
    import shutil
    import time

    tmp_coverage_dir = Path(".tmp_coverage")

    if tmp_coverage_dir.exists():
        # Small delay to ensure all coverage processes are done
        time.sleep(0.5)

        try:
            # Try to find any main coverage file in temp directory
            main_coverage = tmp_coverage_dir / ".coverage"
            if main_coverage.exists():
                # Simply move the main coverage file
                try:
                    shutil.move(str(main_coverage), ".coverage")
                except (FileNotFoundError, PermissionError):
                    pass

            # Clean up temporary directory (best effort)
            try:
                shutil.rmtree(tmp_coverage_dir)
            except (FileNotFoundError, PermissionError, OSError):
                # If cleanup fails, schedule delayed cleanup
                try:
                    import threading

                    def delayed_cleanup():
                        time.sleep(2)
                        try:
                            shutil.rmtree(tmp_coverage_dir)
                        except:
                            pass

                    threading.Thread(target=delayed_cleanup, daemon=True).start()
                except:
                    pass

        except Exception:
            # Best effort cleanup - don't fail the tests
            pass


def pytest_sessionstart(session):
    """Called after the Session object has been created"""
    if _global_kill_enabled():
        cleanup_mlflow_processes()


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished, before returning exit status"""
    finalize_coverage()
    if _global_kill_enabled():
        cleanup_mlflow_processes()


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC TESTING ENHANCEMENT (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def ensure_deterministic_execution():
    """모든 테스트에서 재현 가능성 보장 - OOM 방지를 위해 PyTorch import 제거"""
    # Set all random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Set environment variable for Python hash seed
    os.environ["PYTHONHASHSEED"] = "42"

    # Yield to run the test
    yield


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL FIXTURES (AS PER PLANNING DOCUMENT)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def test_data_directory():
    """Provides path to test data directory."""
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture(scope="session")
def test_configs_directory():
    """Provides path to test configuration files."""
    return Path(__file__).parent / "fixtures" / "configs"


@pytest.fixture(scope="session")
def test_recipes_directory():
    """Provides path to test recipe files."""
    return Path(__file__).parent / "fixtures" / "recipes"


@pytest.fixture(scope="session")
def test_expected_directory():
    """Provides path to expected outputs directory."""
    return Path(__file__).parent / "fixtures" / "expected"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TEST DATA GENERATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


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

        # Create realistic column names
        feature_cols = [f"feature_{i+1}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df["entity_id"] = range(1, n_samples + 1)  # Add entity column

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

        # Generate time series with trend and seasonality
        time_index = pd.date_range("2023-01-01", periods=n_samples, freq="D")

        # Base trend
        trend = np.linspace(0, 10, n_samples)

        # Seasonal pattern
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 7)  # Weekly

        # Generate features
        features_data = {}
        for i in range(n_features):
            noise = np.random.normal(0, 0.5, n_samples)
            features_data[f"feature_{i+1}"] = trend * (i + 1) + seasonal + noise

        df = pd.DataFrame(features_data)
        df["timestamp"] = time_index
        df["entity_id"] = range(1, n_samples + 1)

        # Target is sum of features with some noise
        y = df[[f"feature_{i+1}" for i in range(n_features)]].sum(axis=1) + np.random.normal(
            0, 1, n_samples
        )

        return df, y.values


@pytest.fixture(scope="session")
def test_data_generator():
    """Provide TestDataGenerator for all tests."""
    return TestDataGenerator()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SETTINGS BUILDER PATTERN
# ═══════════════════════════════════════════════════════════════════════════════


class SettingsBuilder:
    """Builder pattern for creating test Settings objects."""

    def __init__(self):
        # Default minimal config
        self._environment = Environment(name="test")
        # Default MLflow to isolated file store. Individual tests can override via fixtures.
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

        # Default minimal recipe
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

    # Environment configuration
    def with_environment(self, name: str) -> "SettingsBuilder":
        """Set environment name."""
        self._environment = Environment(name=name)
        return self

    # MLflow configuration
    def with_mlflow(self, tracking_uri: str, experiment_name: str) -> "SettingsBuilder":
        """Set MLflow configuration."""
        self._mlflow = MLflow(tracking_uri=tracking_uri, experiment_name=experiment_name)
        return self

    # Data source configuration
    def with_data_source(
        self, adapter_type: str, name: str = "test_source", config: Dict[str, Any] = None
    ) -> "SettingsBuilder":
        """Set data source configuration."""
        if config is None:
            # Provide default configs based on adapter_type
            if adapter_type == "storage":
                from src.settings.config import LocalFilesConfig

                config = LocalFilesConfig(base_path="tests/fixtures/data", storage_options={})
            elif adapter_type == "sql":
                from src.settings.config import PostgreSQLConfig

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
            split=self._data.split,  # DataSplit 필드 보존
        )
        return self

    # Task configuration
    def with_task(self, task_type: str) -> "SettingsBuilder":
        """Set task type (classification, regression, timeseries, etc.)."""
        self._task_choice = task_type

        # For clustering (unsupervised), target_column must be None
        if task_type == "clustering":
            self._data = Data(
                loader=self._data.loader,
                data_interface=DataInterface(
                    target_column=None,  # Clustering has no target
                    entity_columns=self._data.data_interface.entity_columns,
                ),
                fetcher=self._data.fetcher,
                split=self._data.split,  # Preserve existing split configuration
            )
        # For timeseries, timestamp_column is required
        elif task_type == "timeseries":
            self._data = Data(
                loader=self._data.loader,
                data_interface=DataInterface(
                    target_column=self._data.data_interface.target_column,
                    entity_columns=self._data.data_interface.entity_columns,
                    timestamp_column="timestamp",  # Default timestamp column for testing
                ),
                fetcher=self._data.fetcher,
                split=self._data.split,  # Preserve existing split configuration
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

    # Model configuration
    def with_model(
        self, class_path: str, library: str = None, hyperparameters: Dict[str, Any] = None
    ) -> "SettingsBuilder":
        """Set model configuration."""
        # Infer library from class_path if not provided
        if library is None:
            if "sklearn" in class_path:
                library = "sklearn"
            elif "xgboost" in class_path:
                library = "xgboost"
            elif "lightgbm" in class_path:
                library = "lightgbm"
            else:
                library = "unknown"

        # Default hyperparameters based on model type
        if hyperparameters is None:
            if "LinearRegression" in class_path:
                hyperparams = {}  # LinearRegression doesn't accept random_state
            else:
                hyperparams = {"random_state": 42}
        else:
            hyperparams = hyperparameters

        self._model = Model(
            class_path=class_path,
            library=library,
            hyperparameters=HyperparametersTuning(tuning_enabled=False, values=hyperparams),
            calibration=self._model.calibration,  # 기존 calibration 설정 유지
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
            from src.settings import Calibration

            self._model.calibration = Calibration(enabled=True, method=method)
        else:
            self._model.calibration = None
        return self

    # Data interface configuration
    def with_target_column(self, target_col: str) -> "SettingsBuilder":
        """Set target column name."""
        self._data.data_interface.target_column = target_col
        return self

    def with_entity_columns(self, entity_cols: list) -> "SettingsBuilder":
        """Set entity columns."""
        self._data.data_interface.entity_columns = entity_cols
        return self

    # Feature store configuration
    def with_feature_store(self, enabled: bool = True) -> "SettingsBuilder":
        """Enable/disable feature store."""
        if enabled:
            # Provide minimal feast_config to satisfy validation
            from src.settings.config import FeastConfig

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
    ) -> "SettingsBuilder":
        """Set data split ratios. Default: 60/20/20/0"""
        self._data.split = DataSplit(
            train=train, validation=validation, test=test, calibration=calibration
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


@pytest.fixture(scope="function")
def settings_builder():
    """Provide SettingsBuilder for creating test Settings."""
    return SettingsBuilder()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. COMPUTED FIELD HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def add_model_computed():
    """
    Model에 computed 필드를 추가하는 헬퍼 fixture.

    SettingsFactory가 동적으로 추가하는 computed 필드를
    단위 테스트에서 시뮬레이션하기 위한 헬퍼.

    Usage:
        settings = add_model_computed(settings, run_name="test_run")
    """

    def _add_computed(
        settings: Settings, run_name: str = None, environment: str = None, seed: int = 42
    ) -> Settings:
        from datetime import datetime

        # SettingsFactory와 동일한 로직
        if not hasattr(settings.recipe.model, "computed"):
            settings.recipe.model.computed = {}

        # 기본값 설정
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"test_run_{timestamp}"

        if environment is None:
            environment = settings.config.environment.name

        settings.recipe.model.computed.update(
            {
                "run_name": run_name,
                "environment": environment,
                "seed": seed,
                "recipe_file": "test_recipe.yaml",
            }
        )

        return settings

    return _add_computed


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ISOLATED ENVIRONMENT FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="function")
def isolated_temp_directory():
    """Provide clean temporary directory for each test."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="function")
def isolated_mlflow_tracking(isolated_temp_directory):
    """Provide isolated MLflow tracking URI for each test."""
    tracking_uri = f"file://{isolated_temp_directory}/mlruns"
    return tracking_uri


@pytest.fixture(scope="function")
def test_data_files(isolated_temp_directory, test_data_generator):
    """Create test data files in isolated directory."""
    data_dir = isolated_temp_directory / "data"
    data_dir.mkdir()

    # Generate and save classification data
    X_cls, y_cls = test_data_generator.classification_data(n_samples=50)
    cls_data = X_cls.copy()
    cls_data["target"] = y_cls
    cls_path = data_dir / "classification_data.csv"
    cls_data.to_csv(cls_path, index=False)

    # Generate and save regression data
    X_reg, y_reg = test_data_generator.regression_data(n_samples=50)
    reg_data = X_reg.copy()
    reg_data["target"] = y_reg
    reg_path = data_dir / "regression_data.csv"
    reg_data.to_csv(reg_path, index=False)

    return {
        "classification": cls_path,
        "regression": reg_path,
        "classification_df": cls_data,
        "regression_df": reg_data,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PERFORMANCE BENCHMARKING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="function")
def performance_benchmark():
    """Provide performance benchmarking utilities."""

    class PerformanceBenchmark:
        def __init__(self):
            self.measurements = {}

        def measure_time(self, name: str):
            """Context manager for measuring execution time."""
            return self.Timer(name, self.measurements)

        def assert_time_under(self, name: str, max_seconds: float):
            """Assert that measured time is under threshold."""
            if name not in self.measurements:
                raise ValueError(f"No measurement found for '{name}'")

            actual_time = self.measurements[name]
            assert actual_time < max_seconds, (
                f"Performance test failed: {name} took {actual_time:.3f}s "
                f"but should be under {max_seconds}s"
            )

        def get_measurement(self, name: str) -> float:
            """Get measurement by name."""
            return self.measurements.get(name, 0.0)

        def report_all(self) -> Dict[str, float]:
            """Get all measurements."""
            return self.measurements.copy()

        class Timer:
            def __init__(self, name: str, measurements: Dict[str, float]):
                self.name = name
                self.measurements = measurements
                self.start_time = None

            def __enter__(self):
                self.start_time = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.perf_counter()
                self.measurements[self.name] = end_time - self.start_time

    return PerformanceBenchmark()


# ═══════════════════════════════════════════════════════════════════════════════
# NO MOCK HELL FIXTURES - REAL COMPONENT TESTING
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="function")
def real_dataset_files(isolated_temp_directory, test_data_generator):
    """Create multiple real data files for comprehensive component testing."""
    files = {}

    # Classification dataset with realistic features
    X_cls, y_cls = test_data_generator.classification_data(n_samples=100, n_features=5)
    cls_data = X_cls.copy()
    cls_data["target"] = y_cls
    cls_data["entity_id"] = range(1, 101)  # Add entity column

    # CSV format
    cls_csv_path = isolated_temp_directory / "classification.csv"
    cls_data.to_csv(cls_csv_path, index=False)
    files["classification_csv"] = {"path": cls_csv_path, "data": cls_data, "format": "csv"}

    # Parquet format (only if pyarrow is available)
    try:
        import pyarrow

        cls_parquet_path = isolated_temp_directory / "classification.parquet"
        cls_data.to_parquet(cls_parquet_path)
        files["classification_parquet"] = {
            "path": cls_parquet_path,
            "data": cls_data,
            "format": "parquet",
        }
    except ImportError:
        # Create a dummy entry so tests can skip gracefully
        files["classification_parquet"] = {"path": None, "data": cls_data, "format": "parquet"}

    # Regression dataset
    X_reg, y_reg = test_data_generator.regression_data(n_samples=80, n_features=4)
    reg_data = X_reg.copy()
    reg_data["target"] = y_reg
    reg_data["entity_id"] = range(1, 81)

    reg_path = isolated_temp_directory / "regression.csv"
    reg_data.to_csv(reg_path, index=False)
    files["regression"] = {"path": reg_path, "data": reg_data, "format": "csv"}

    # SQL dataset (SQLite in-memory for fast testing)
    import sqlite3

    db_path = isolated_temp_directory / "test.db"
    conn = sqlite3.connect(db_path)

    # Create classification table
    cls_data.to_sql("classification_table", conn, index=False, if_exists="replace")
    # Create regression table
    reg_data.to_sql("regression_table", conn, index=False, if_exists="replace")

    conn.close()

    files["sql"] = {
        "path": db_path,
        "classification_table": "classification_table",
        "regression_table": "regression_table",
        "classification_data": cls_data,
        "regression_data": reg_data,
    }

    return files


@pytest.fixture(scope="function")
def small_real_models_cache():
    """Cache small, fast real models for performance testing."""
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    # Generate small datasets
    X_cls, y_cls = make_classification(n_samples=50, n_features=4, random_state=42, n_informative=2)
    X_reg, y_reg = make_regression(n_samples=50, n_features=4, random_state=42)

    models = {}

    # Pre-trained classification models
    models["logistic"] = LogisticRegression(random_state=42, max_iter=100, solver="liblinear")
    models["logistic"].fit(X_cls, y_cls)

    models["tree_cls"] = DecisionTreeClassifier(random_state=42, max_depth=3)
    models["tree_cls"].fit(X_cls, y_cls)

    # Pre-trained regression models
    models["linear"] = LinearRegression()
    models["linear"].fit(X_reg, y_reg)

    models["tree_reg"] = DecisionTreeRegressor(random_state=42, max_depth=3)
    models["tree_reg"].fit(X_reg, y_reg)

    return models, {"X_cls": X_cls, "y_cls": y_cls, "X_reg": X_reg, "y_reg": y_reg}


@pytest.fixture(scope="function")
def factory_with_real_storage_adapter(settings_builder, real_dataset_files):
    """Create Factory with real StorageAdapter and real CSV data."""
    from src.factory import Factory

    # Use real CSV file
    cls_info = real_dataset_files["classification_csv"]

    settings = (
        settings_builder.with_data_source("storage")
        .with_data_path(str(cls_info["path"]))
        .with_task("classification")
        .with_model(
            "sklearn.ensemble.RandomForestClassifier",
            hyperparameters={"n_estimators": 10, "random_state": 42},
        )
        .build()
    )

    factory = Factory(settings)
    return factory, cls_info


@pytest.fixture(scope="function")
def factory_with_real_sql_adapter(settings_builder, real_dataset_files):
    """Create Factory with real SQLAdapter and real SQLite database."""
    from src.factory import Factory

    sql_info = real_dataset_files["sql"]

    # Create SQL connection string
    connection_string = f"sqlite:///{sql_info['path']}"

    settings = (
        settings_builder.with_data_source("sql", config={"connection_uri": connection_string})
        .with_data_path(connection_string)
        .with_task("classification")
        .with_model(
            "sklearn.linear_model.LogisticRegression",
            hyperparameters={"random_state": 42, "max_iter": 100},
        )
        .build()
    )

    factory = Factory(settings)
    return factory, sql_info


@pytest.fixture(scope="function")
def fast_factory_setup(settings_builder, small_real_models_cache):
    """Factory setup optimized for speed with small real components."""
    from src.factory import Factory

    models, data = small_real_models_cache

    settings = (
        settings_builder.with_task("classification")
        .with_model(
            "sklearn.linear_model.LogisticRegression",
            hyperparameters={"random_state": 42, "max_iter": 50, "solver": "liblinear"},
        )
        .build()
    )

    factory = Factory(settings)
    return factory, data


@pytest.fixture(scope="function")
def real_component_performance_tracker():
    """Track performance metrics for real component tests."""

    class RealComponentPerformanceTracker:
        def __init__(self):
            self.metrics = {}
            self.thresholds = {
                "adapter_creation": 0.1,  # 100ms
                "model_creation": 0.1,  # 100ms
                "evaluator_creation": 0.05,  # 50ms
                "data_reading": 0.2,  # 200ms
                "model_training": 1.0,  # 1 second
                "evaluation": 0.3,  # 300ms
                "complete_workflow": 2.0,  # 2 seconds
            }

        def measure_time(self, operation_name: str):
            """Context manager to measure operation time."""
            import contextlib
            import time

            @contextlib.contextmanager
            def timer():
                start_time = time.time()
                try:
                    yield
                finally:
                    end_time = time.time()
                    self.metrics[operation_name] = end_time - start_time

            return timer()

        def assert_time_under(self, operation_name: str, threshold: float = None):
            """Assert operation completed within time threshold."""
            actual_time = self.metrics.get(operation_name)
            if actual_time is None:
                raise ValueError(f"No measurement found for operation: {operation_name}")

            threshold = threshold or self.thresholds.get(operation_name, 1.0)

            if actual_time > threshold:
                raise AssertionError(
                    f"Operation '{operation_name}' took {actual_time:.3f}s, "
                    f"which exceeds threshold of {threshold:.3f}s"
                )

        def get_performance_summary(self) -> dict:
            """Get summary of all performance measurements."""
            return {
                "measurements": self.metrics.copy(),
                "thresholds": self.thresholds.copy(),
                "violations": [
                    op for op, time in self.metrics.items() if time > self.thresholds.get(op, 1.0)
                ],
            }

    return RealComponentPerformanceTracker()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PYTEST CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "performance: marks tests as performance benchmarks")
    config.addinivalue_line(
        "markers", "server: marks tests that require exclusive access to server/ports/resources"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark slow tests."""
    for item in items:
        # Mark integration and e2e tests as slow
        if "integration" in str(item.fspath) or "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
        # Ensure server-marked tests acquire global server lock
        if "server" in item.keywords:
            item.add_marker(pytest.mark.usefixtures("server_serial_execution"))


# ═══════════════════════════════════════════════════════════════════════════════
# SERVER/RESOURCE SERIALIZATION LOCK (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def server_serial_execution():
    """Serialize tests marked as 'server' across processes using a filesystem lock.

    This avoids port/MLflow/SQLite conflicts when running in parallel (xdist).
    """
    lock_dir = Path("/tmp/mmp-server.lockdir")
    start_time = time.time()
    # Attempt to acquire lock by creating a directory atomically
    while True:
        try:
            lock_dir.mkdir(mode=0o700)
            break
        except FileExistsError:
            if time.time() - start_time > 300:  # 5 minutes safety timeout
                pytest.skip("Server lock wait timeout exceeded; skipping server-marked test")
            time.sleep(0.2)
    try:
        yield
    finally:
        try:
            lock_dir.rmdir()
        except Exception:
            # Best-effort cleanup
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# 6. COMMON TEST PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="function")
def minimal_classification_settings(settings_builder, test_data_files, isolated_mlflow_tracking):
    """Pre-configured Settings for classification tests."""
    return (
        settings_builder.with_task("classification")
        .with_model("sklearn.ensemble.RandomForestClassifier")
        .with_data_path(str(test_data_files["classification"]))
        .with_target_column("target")
        .with_mlflow(isolated_mlflow_tracking, "test_classification")
        .build()
    )


@pytest.fixture(scope="function")
def minimal_regression_settings(settings_builder, test_data_files, isolated_mlflow_tracking):
    """Pre-configured Settings for regression tests."""
    return (
        settings_builder.with_task("regression")
        .with_model("sklearn.ensemble.RandomForestRegressor")
        .with_data_path(str(test_data_files["regression"]))
        .with_target_column("target")
        .with_mlflow(isolated_mlflow_tracking, "test_regression")
        .build()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST ISOLATION FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="function")
def isolated_working_directory(tmp_path, monkeypatch):
    """
    완전히 격리된 작업 디렉토리를 제공합니다.

    CLI 테스트에서 파일 생성이 메인 프로젝트 디렉토리에 영향을 주지 않도록
    임시 디렉토리로 작업 디렉토리를 변경합니다.

    Args:
        tmp_path: pytest가 제공하는 임시 디렉토리
        monkeypatch: pytest monkeypatch fixture for mocking

    Returns:
        Path: 격리된 임시 작업 디렉토리 경로

    Usage:
        def test_something(isolated_working_directory):
            # 이제 Path.cwd()는 임시 디렉토리를 가리킴
            current_dir = Path.cwd()
            assert current_dir == isolated_working_directory
    """
    # 현재 작업 디렉토리를 임시 디렉토리로 변경
    original_cwd = Path.cwd()
    monkeypatch.chdir(tmp_path)

    # 추가로 Path.cwd()도 임시 디렉토리를 반환하도록 보장
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    yield tmp_path

    # 테스트 후 원래 디렉토리로 복원 (사실상 자동으로 되지만 명시적으로)
    # monkeypatch가 자동으로 복원해주므로 별도 cleanup 불필요


@pytest.fixture(scope="function")
def cli_test_environment(isolated_working_directory):
    """
    CLI 테스트를 위한 완전한 격리 환경을 제공합니다.

    - 격리된 작업 디렉토리
    - 테스트용 config 및 recipe 파일 생성
    - 테스트 데이터 파일 생성

    Returns:
        dict: CLI 테스트에 필요한 파일 경로들
    """
    work_dir = isolated_working_directory

    # 기본 디렉토리 구조 생성
    (work_dir / "configs").mkdir(exist_ok=True)
    (work_dir / "recipes").mkdir(exist_ok=True)
    (work_dir / "data").mkdir(exist_ok=True)

    # 테스트용 설정 파일들 생성
    mlruns_dir = work_dir / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)
    tracking_uri_abs = f"file://{mlruns_dir.as_posix()}"

    test_config = f"""
environment:
  name: "test"
  description: "Test environment"

mlflow:
  tracking_uri: "{tracking_uri_abs}"
  experiment_name: "test_experiment"

data_source:
  name: "test_storage"
  adapter_type: "storage"
  config: {{}}

feature_store:
  provider: "none"
"""

    test_recipe = """
name: "test_recipe"
task_choice: "classification"

model:
  class_path: "sklearn.ensemble.RandomForestClassifier"
  library: "sklearn"
  hyperparameters:
    tuning_enabled: false
    values:
      n_estimators: 10
      random_state: 42

data:
  loader:
    source_uri: "data/test.csv"
  data_interface:
    task_type: "classification"
    target_column: "target"
    entity_columns: ["entity_id"]
  fetcher:
    type: "pass_through"

evaluation:
  metrics: ["accuracy"]
"""

    config_path = work_dir / "configs" / "test.yaml"
    recipe_path = work_dir / "recipes" / "test.yaml"

    config_path.write_text(test_config)
    recipe_path.write_text(test_recipe)

    # 테스트 데이터 생성
    import pandas as pd

    test_data = pd.DataFrame(
        {
            "entity_id": range(1, 51),
            "feature_1": range(50),
            "feature_2": range(50, 100),
            "target": [i % 2 for i in range(50)],
        }
    )
    data_path = work_dir / "data" / "test.csv"
    test_data.to_csv(data_path, index=False)

    return {
        "work_dir": work_dir,
        "config_path": config_path,
        "recipe_path": recipe_path,
        "data_path": data_path,
    }


# ===== Context Fixtures (Phase 1) =====

import pytest

from tests.fixtures.contexts.component_context import ComponentTestContext
from tests.fixtures.contexts.database_context import DatabaseTestContext
from tests.fixtures.contexts.mlflow_context import MLflowTestContext


@pytest.fixture
def mlflow_test_context(isolated_temp_directory, settings_builder, test_data_generator):
    return MLflowTestContext(isolated_temp_directory, settings_builder, test_data_generator)


@pytest.fixture
def serving_test_context(mlflow_test_context, isolated_temp_directory):
    """ServingTestContext fixture for API serving tests"""
    from tests.fixtures.contexts.serving_context import ServingTestContext

    return ServingTestContext(mlflow_test_context, isolated_temp_directory)


@pytest.fixture
def component_test_context(isolated_temp_directory, settings_builder, test_data_generator):
    return ComponentTestContext(isolated_temp_directory, settings_builder, test_data_generator)


@pytest.fixture
def database_test_context(isolated_temp_directory):
    return DatabaseTestContext(isolated_temp_directory)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. FAST-FAIL HELPERS (NETWORK/MLflow)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def make_fail_fast_mlflow(monkeypatch):
    """Return a callable to force MLflow HTTP requests to fail immediately.

    Usage:
        def test_something(make_fail_fast_mlflow):
            make_fail_fast_mlflow()  # apply fail-fast behavior
            ...
    """

    def _apply():
        # Minimize network wait
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "1")
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
        try:
            import requests
            from mlflow.utils import rest_utils

            def _fail_fast(*args, **kwargs):
                raise requests.exceptions.ConnectionError("forced fail-fast for test")

            monkeypatch.setattr(rest_utils, "http_request", _fail_fast, raising=True)
        except Exception:
            # If patching fails (environment differences), rely on env-only fast fail
            pass

    return _apply

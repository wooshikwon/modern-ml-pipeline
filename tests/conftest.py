"""
Modern ML Pipeline - Core Test Fixtures
No Mock Hell: Real objects with test data, minimal mocking, fast execution
"""

import os
import random
import tempfile
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from mmp.settings import Settings

# Re-export SettingsBuilder and TestDataGenerator so existing imports work
from tests.fixtures.settings import SettingsBuilder, TestDataGenerator  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY MANAGEMENT & CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════


def _global_kill_enabled() -> bool:
    """Return True if global kill behavior is explicitly enabled via env var."""
    return os.getenv("MMP_ENABLE_GLOBAL_KILL", "0") == "1"


def cleanup_mlflow_processes():
    """Clean up any MLflow server processes to prevent memory leaks"""
    import signal
    import subprocess

    try:
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
                    pass

        try:
            subprocess.run(["pkill", "-f", "mlflow.server"], capture_output=True, timeout=3)
            subprocess.run(["pkill", "-f", "uvicorn.*mlflow"], capture_output=True, timeout=3)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def finalize_coverage():
    """Combine coverage files and clean up temporary files"""
    import shutil

    tmp_coverage_dir = Path(".tmp_coverage")

    if tmp_coverage_dir.exists():
        time.sleep(0.5)

        try:
            main_coverage = tmp_coverage_dir / ".coverage"
            if main_coverage.exists():
                try:
                    shutil.move(str(main_coverage), ".coverage")
                except (FileNotFoundError, PermissionError):
                    pass

            try:
                shutil.rmtree(tmp_coverage_dir)
            except (FileNotFoundError, PermissionError, OSError):
                try:
                    import threading

                    def delayed_cleanup():
                        time.sleep(2)
                        try:
                            shutil.rmtree(tmp_coverage_dir)
                        except Exception:
                            pass

                    threading.Thread(target=delayed_cleanup, daemon=True).start()
                except Exception:
                    pass

        except Exception:
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
# DETERMINISTIC TESTING
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def ensure_deterministic_execution():
    """Reproducibility for all tests."""
    random.seed(42)
    np.random.seed(42)
    os.environ["PYTHONHASHSEED"] = "42"
    yield


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL FIXTURES
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


@pytest.fixture(scope="session")
def test_data_generator():
    """Provide TestDataGenerator for all tests."""
    return TestDataGenerator()


@pytest.fixture(scope="function")
def settings_builder():
    """Provide SettingsBuilder for creating test Settings."""
    return SettingsBuilder()


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTED FIELD HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def add_model_computed():
    """Helper fixture to add computed fields to Model (simulates SettingsFactory)."""

    def _add_computed(
        settings: Settings, run_name: str = None, environment: str = None, seed: int = 42
    ) -> Settings:
        from datetime import datetime

        if not hasattr(settings.recipe.model, "computed"):
            settings.recipe.model.computed = {}

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
# ISOLATED ENVIRONMENT FIXTURES
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

    X_cls, y_cls = test_data_generator.classification_data(n_samples=50)
    cls_data = X_cls.copy()
    cls_data["target"] = y_cls
    cls_path = data_dir / "classification_data.csv"
    cls_data.to_csv(cls_path, index=False)

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
# PERFORMANCE BENCHMARKING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="function")
def performance_benchmark():
    """Provide performance benchmarking utilities."""

    class PerformanceBenchmark:
        def __init__(self):
            self.measurements = {}

        def measure_time(self, name: str):
            return self.Timer(name, self.measurements)

        def assert_time_under(self, name: str, max_seconds: float):
            if name not in self.measurements:
                raise ValueError(f"No measurement found for '{name}'")
            actual_time = self.measurements[name]
            assert actual_time < max_seconds, (
                f"Performance test failed: {name} took {actual_time:.3f}s "
                f"but should be under {max_seconds}s"
            )

        def get_measurement(self, name: str) -> float:
            return self.measurements.get(name, 0.0)

        def report_all(self) -> Dict[str, float]:
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
# REAL COMPONENT TESTING FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="function")
def real_dataset_files(isolated_temp_directory, test_data_generator):
    """Create multiple real data files for comprehensive component testing."""
    import sqlite3

    files = {}

    X_cls, y_cls = test_data_generator.classification_data(n_samples=100, n_features=5)
    cls_data = X_cls.copy()
    cls_data["target"] = y_cls
    cls_data["entity_id"] = range(1, 101)

    cls_csv_path = isolated_temp_directory / "classification.csv"
    cls_data.to_csv(cls_csv_path, index=False)
    files["classification_csv"] = {"path": cls_csv_path, "data": cls_data, "format": "csv"}

    try:
        import pyarrow  # noqa: F401

        cls_parquet_path = isolated_temp_directory / "classification.parquet"
        cls_data.to_parquet(cls_parquet_path)
        files["classification_parquet"] = {
            "path": cls_parquet_path, "data": cls_data, "format": "parquet",
        }
    except ImportError:
        files["classification_parquet"] = {"path": None, "data": cls_data, "format": "parquet"}

    X_reg, y_reg = test_data_generator.regression_data(n_samples=80, n_features=4)
    reg_data = X_reg.copy()
    reg_data["target"] = y_reg
    reg_data["entity_id"] = range(1, 81)

    reg_path = isolated_temp_directory / "regression.csv"
    reg_data.to_csv(reg_path, index=False)
    files["regression"] = {"path": reg_path, "data": reg_data, "format": "csv"}

    db_path = isolated_temp_directory / "test.db"
    conn = sqlite3.connect(db_path)
    cls_data.to_sql("classification_table", conn, index=False, if_exists="replace")
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

    X_cls, y_cls = make_classification(n_samples=50, n_features=4, random_state=42, n_informative=2)
    X_reg, y_reg = make_regression(n_samples=50, n_features=4, random_state=42)

    models = {}
    models["logistic"] = LogisticRegression(random_state=42, max_iter=100, solver="liblinear")
    models["logistic"].fit(X_cls, y_cls)
    models["tree_cls"] = DecisionTreeClassifier(random_state=42, max_depth=3)
    models["tree_cls"].fit(X_cls, y_cls)
    models["linear"] = LinearRegression()
    models["linear"].fit(X_reg, y_reg)
    models["tree_reg"] = DecisionTreeRegressor(random_state=42, max_depth=3)
    models["tree_reg"].fit(X_reg, y_reg)

    return models, {"X_cls": X_cls, "y_cls": y_cls, "X_reg": X_reg, "y_reg": y_reg}


@pytest.fixture(scope="function")
def factory_with_real_storage_adapter(settings_builder, real_dataset_files):
    """Create Factory with real StorageAdapter and real CSV data."""
    from mmp.factory import Factory

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
    from mmp.factory import Factory

    sql_info = real_dataset_files["sql"]
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
    from mmp.factory import Factory

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
                "adapter_creation": 0.1,
                "model_creation": 0.1,
                "evaluator_creation": 0.05,
                "data_reading": 0.2,
                "model_training": 1.0,
                "evaluation": 0.3,
                "complete_workflow": 2.0,
            }

        def measure_time(self, operation_name: str):
            import contextlib

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
            return {
                "measurements": self.metrics.copy(),
                "thresholds": self.thresholds.copy(),
                "violations": [
                    op for op, t in self.metrics.items() if t > self.thresholds.get(op, 1.0)
                ],
            }

    return RealComponentPerformanceTracker()


# ═══════════════════════════════════════════════════════════════════════════════
# PYTEST CONFIGURATION
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
        if "integration" in str(item.fspath) or "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
        if "server" in item.keywords:
            item.add_marker(pytest.mark.usefixtures("server_serial_execution"))


# ═══════════════════════════════════════════════════════════════════════════════
# SERVER/RESOURCE SERIALIZATION LOCK
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def server_serial_execution():
    """Serialize tests marked as 'server' across processes using a filesystem lock."""
    lock_dir = Path("/tmp/mmp-server.lockdir")
    start_time = time.time()
    while True:
        try:
            lock_dir.mkdir(mode=0o700)
            break
        except FileExistsError:
            if time.time() - start_time > 300:
                pytest.skip("Server lock wait timeout exceeded; skipping server-marked test")
            time.sleep(0.2)
    try:
        yield
    finally:
        try:
            lock_dir.rmdir()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# COMMON TEST PATTERNS
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
    """Provide isolated working directory for CLI tests."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    yield tmp_path


@pytest.fixture(scope="function")
def cli_test_environment(isolated_working_directory):
    """Complete isolated environment for CLI tests."""
    import pandas as pd

    work_dir = isolated_working_directory

    (work_dir / "configs").mkdir(exist_ok=True)
    (work_dir / "recipes").mkdir(exist_ok=True)
    (work_dir / "data").mkdir(exist_ok=True)

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


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

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
# FAST-FAIL HELPERS (NETWORK/MLflow)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def make_fail_fast_mlflow(monkeypatch):
    """Return a callable to force MLflow HTTP requests to fail immediately."""

    def _apply():
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "1")
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
        try:
            import requests
            from mlflow.utils import rest_utils

            def _fail_fast(*args, **kwargs):
                raise requests.exceptions.ConnectionError("forced fail-fast for test")

            monkeypatch.setattr(rest_utils, "http_request", _fail_fast, raising=True)
        except Exception:
            pass

    return _apply

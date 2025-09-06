"""
Global pytest fixtures and configuration for the test suite.
This file provides shared fixtures that can be used across all test modules.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import MagicMock, patch
import tempfile
import shutil

import pytest
import pandas as pd
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.settings import Settings, Config, Recipe
from src.settings.config import Environment, MLflow, DataSource, FeatureStore
from src.settings.recipe import Model, Data, Loader, DataInterface, FeatureView
from src.factory import Factory
from src.components.adapter import AdapterRegistry
from src.components.fetcher import FetcherRegistry
from src.components.evaluator import EvaluatorRegistry
from src.components.preprocessor.registry import PreprocessorStepRegistry
from src.components.trainer.registry import TrainerRegistry
from tests.helpers.builders import (
    ConfigBuilder,
    RecipeBuilder,
    SettingsBuilder,
    DataFrameBuilder,
    FileBuilder,
)


# ============================================================================
# Directory and File System Fixtures
# ============================================================================

@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        
        # Create standard directory structure
        (workspace / "configs").mkdir()
        (workspace / "recipes").mkdir()
        (workspace / "data").mkdir()
        (workspace / "mlruns").mkdir()
        (workspace / "artifacts").mkdir()
        
        # Change to workspace directory
        original_cwd = os.getcwd()
        os.chdir(workspace)
        
        yield workspace
        
        # Restore original directory
        os.chdir(original_cwd)


@pytest.fixture
def sample_csv_data(temp_workspace: Path) -> Path:
    """Create a sample CSV file for testing."""
    # Use FileBuilder to create and cleanup CSV
    with FileBuilder.build_csv_file_context(
        data=DataFrameBuilder.build_classification_data(n_samples=5, n_features=2)
    ) as csv_path:
        # Move file into temp_workspace/data to keep previous semantics
        dest = temp_workspace / "data" / Path(csv_path).name
        dest.parent.mkdir(exist_ok=True, parents=True)
        Path(csv_path).replace(dest)
        yield dest



# ============================================================================
# Factory Fixtures
# ============================================================================

@pytest.fixture
def test_factory(test_settings: Settings) -> Factory:
    """Provide a test Factory instance."""
    return Factory(settings=test_settings)


# ============================================================================
# Component Registry Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def clean_registries():
    """
    Clean component registries before and after each test.
    This ensures test isolation.
    """
    # Store original state
    original_adapters = AdapterRegistry.adapters.copy()
    original_fetchers = FetcherRegistry.fetchers.copy()
    original_evaluators = EvaluatorRegistry.evaluators.copy()
    original_preprocessor_steps = PreprocessorStepRegistry.preprocessor_steps.copy()
    original_trainers = TrainerRegistry.trainers.copy()
    
    yield
    
    # Restore original state
    AdapterRegistry.adapters = original_adapters
    FetcherRegistry.fetchers = original_fetchers
    EvaluatorRegistry.evaluators = original_evaluators
    PreprocessorStepRegistry.preprocessor_steps = original_preprocessor_steps
    TrainerRegistry.trainers = original_trainers


# ============================================================================
# Logger and Environment Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def silence_logger():
    """
    Silence logger output for all tests (autouse).
    This ensures tests run quietly without log spam.
    """
    import logging
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def isolated_env(temp_workspace: Path, monkeypatch) -> Path:
    """
    Provide a completely isolated execution environment.
    Sets environment variables and creates necessary directories.
    """
    # Set test environment variables
    monkeypatch.setenv("ENV_NAME", "test")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(temp_workspace / "mlruns"))
    monkeypatch.setenv("FEAST_REPO_PATH", str(temp_workspace / "feast"))
    monkeypatch.setenv("DATA_PATH", str(temp_workspace / "data"))
    
    # Create additional directories if needed
    (temp_workspace / "logs").mkdir(exist_ok=True)
    (temp_workspace / "models").mkdir(exist_ok=True)
    (temp_workspace / "feast").mkdir(exist_ok=True)
    
    return temp_workspace


# ============================================================================
# Async Fixtures
# ============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Factory Fixtures with Cache Management
# ============================================================================

@pytest.fixture
def factory_with_clean_cache(test_settings: Settings) -> Factory:
    """
    Provide a Factory instance with cleaned cache.
    This ensures no component caching interference between tests.
    """
    factory = Factory(settings=test_settings)
    factory._component_cache.clear()
    return factory


@pytest.fixture
def mock_factory():
    """
    Provide a completely mocked Factory for isolated component tests.
    """
    from unittest.mock import MagicMock
    from tests.helpers.builders import MockBuilder, DataFrameBuilder
    
    factory = MagicMock(spec=Factory)
    
    # Mock component creation methods
    factory.create_model.return_value = MockBuilder.build_mock_model()
    factory.create_data_adapter.return_value = MockBuilder.build_mock_adapter()
    factory.create_fetcher.return_value = MockBuilder.build_mock_fetcher()
    factory.create_evaluator.return_value = MockBuilder.build_mock_evaluator()
    factory.create_preprocessor.return_value = None
    
    return factory


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_mlflow():
    """Mock MLflow for tests."""
    with patch('mlflow.start_run') as mock_start_run, \
         patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.log_metrics') as mock_log_metrics, \
         patch('mlflow.log_params') as mock_log_params, \
         patch('mlflow.log_artifact') as mock_log_artifact, \
         patch('mlflow.pyfunc.save_model') as mock_save_model, \
         patch('mlflow.pyfunc.load_model') as mock_load_model:
        
        # Configure mock behaviors
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        yield {
            'start_run': mock_start_run,
            'log_metric': mock_log_metric,
            'log_metrics': mock_log_metrics,
            'log_params': mock_log_params,
            'log_artifact': mock_log_artifact,
            'save_model': mock_save_model,
            'load_model': mock_load_model,
            'run': mock_run
        }


@pytest.fixture
def mock_database():
    """Mock database connection for tests."""
    from unittest.mock import Mock
    
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    
    return mock_conn


@pytest.fixture
def mock_filesystem():
    """Mock filesystem operations for tests."""
    with patch('builtins.open', create=True) as mock_open, \
         patch('os.path.exists') as mock_exists, \
         patch('os.makedirs') as mock_makedirs:
        
        mock_exists.return_value = True
        yield {
            'open': mock_open,
            'exists': mock_exists,
            'makedirs': mock_makedirs
        }


# ============================================================================
# Data Generation Fixtures
# ============================================================================

# removed: data generation fixtures; use DataFrameBuilder directly in tests


# ============================================================================
# Assertion Helpers
# ============================================================================

@pytest.fixture
def assert_dataframe_equal():
    """Provide a helper function to assert dataframe equality."""
    def _assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, **kwargs):
        pd.testing.assert_frame_equal(df1, df2, **kwargs)
    return _assert_dataframe_equal


@pytest.fixture
def assert_series_equal():
    """Provide a helper function to assert series equality."""
    def _assert_series_equal(s1: pd.Series, s2: pd.Series, **kwargs):
        pd.testing.assert_series_equal(s1, s2, **kwargs)
    return _assert_series_equal


# ============================================================================
# Session-level Fixtures
# ============================================================================

@pytest.fixture(scope='session')
def test_data_dir() -> Path:
    """Provide the path to test data directory."""
    return Path(__file__).parent / 'fixtures' / 'data'


@pytest.fixture(scope='session')
def test_configs_dir() -> Path:
    """Provide the path to test configs directory."""
    return Path(__file__).parent / 'fixtures' / 'configs'


# ============================================================================
# Markers and Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# ============================================================================
# Test Execution Hooks
# ============================================================================

def pytest_runtest_setup(item):
    """Setup for each test."""
    # Skip slow tests in quick mode
    if 'slow' in item.keywords and item.config.getoption("--quick"):
        pytest.skip("skipping slow test in quick mode")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        help="Run quick tests only (skip slow tests)"
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )
    parser.addoption(
        "--e2e",
        action="store_true",
        default=False,
        help="Run end-to-end tests"
    )
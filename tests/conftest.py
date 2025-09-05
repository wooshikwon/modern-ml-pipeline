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
from src.settings.recipe import Model, Data, Loader, EntitySchema, DataInterface
from src.factory import Factory
from src.components.adapter import AdapterRegistry
from src.components.fetcher import FetcherRegistry
from src.components.evaluator import EvaluatorRegistry
from src.components.preprocessor.registry import PreprocessorStepRegistry
from src.components.trainer.registry import TrainerRegistry


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
    data = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'feature2': [1.0, 2.0, 3.0, 4.0, 5.0],
        'target': [0, 1, 0, 1, 0]
    })
    
    csv_path = temp_workspace / "data" / "sample.csv"
    data.to_csv(csv_path, index=False)
    return csv_path


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_config_dict() -> Dict[str, Any]:
    """Provide a test configuration dictionary."""
    return {
        'environment': {
            'name': 'test'
        },
        'mlflow': {
            'tracking_uri': './mlruns',
            'experiment_name': 'test_experiment'
        },
        'data_source': {
            'name': 'test_storage',
            'adapter_type': 'storage',
            'config': {
                'base_path': './data'
            }
        },
        'feature_store': {
            'provider': 'none'
        }
    }


@pytest.fixture
def test_config(test_config_dict: Dict[str, Any]) -> Config:
    """Provide a test Config object."""
    return Config(**test_config_dict)


@pytest.fixture
def test_recipe_dict() -> Dict[str, Any]:
    """Provide a test recipe dictionary."""
    return {
        'name': 'test_recipe',
        'model': {
            'name': 'test_model',
            'class_path': 'sklearn.ensemble.RandomForestClassifier',
            'library': 'sklearn',
            'hyperparameters': {
                'tuning_enabled': False,
                'values': {
                    'n_estimators': 100,
                    'random_state': 42
                }
            },
            'data_interface': {
                'task_type': 'classification',
                'target_column': 'target',
                'feature_columns': ['feature1', 'feature2']
            },
            'loader': {
                'adapter': 'storage',
                'source_uri': './data/sample.csv',
                'entity_schema': {
                    'entity_columns': ['user_id'],
                    'timestamp_column': 'event_timestamp'  # Add required field
                }
            },
            'fetcher': {
                'type': 'pass_through'
            }
        },
        'data': {
            'loader': {
                'adapter': 'storage',
                'source_uri': './data/sample.csv',
                'entity_schema': {
                    'entity_columns': ['user_id'],
                    'timestamp_column': 'event_timestamp'  # Add required field
                }
            },
            'fetcher': {
                'type': 'pass_through'
            },
            'data_interface': {  # Add required field
                'task_type': 'classification',
                'target_column': 'target',
                'feature_columns': ['feature1', 'feature2']
            }
        },
        'evaluation': {  # Add required field
            'metrics': ['accuracy', 'precision', 'recall', 'f1'],
            'split_strategy': 'time_based_split',
            'test_size': 0.2
        }
    }


@pytest.fixture
def test_recipe(test_recipe_dict: Dict[str, Any]) -> Recipe:
    """Provide a test Recipe object."""
    return Recipe(**test_recipe_dict)


@pytest.fixture
def test_settings(test_config: Config, test_recipe: Recipe) -> Settings:
    """Provide a test Settings object."""
    return Settings(config=test_config, recipe=test_recipe)


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


@pytest.fixture
async def async_client(test_settings: Settings):
    """Provide async test client for FastAPI."""
    from httpx import AsyncClient
    from src.serving.router import app
    
    # Setup API context with test settings
    from src.serving._lifespan import setup_api_context
    setup_api_context(test_settings)
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


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

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Generate a sample dataframe for testing."""
    return pd.DataFrame({
        'user_id': range(100),
        'feature1': [i * 0.1 for i in range(100)],
        'feature2': [i * 0.5 for i in range(100)],
        'feature3': [i % 10 for i in range(100)],
        'target': [i % 2 for i in range(100)]
    })


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate classification dataset for testing."""
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42
    )
    
    df_X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    series_y = pd.Series(y, name='target')
    
    return df_X, series_y


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate regression dataset for testing."""
    from sklearn.datasets import make_regression
    
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=3,
        random_state=42
    )
    
    df_X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    series_y = pd.Series(y, name='target')
    
    return df_X, series_y


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
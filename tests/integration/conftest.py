"""
Fixtures for Pipeline Integration Tests.
Provides test infrastructure including temporary MLflow environments, test datasets, and settings.
"""

import pytest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import mlflow
import os

from src.settings import Settings
from tests.helpers.config_builder import ConfigBuilder, SettingsBuilder
from tests.helpers.dataframe_builder import DataFrameBuilder


@pytest.fixture
def tmp_mlflow_tracking():
    """Create temporary MLflow tracking environment."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracking_uri = f"file://{tmp_dir}/mlruns"
        original_uri = mlflow.get_tracking_uri()
        
        # Set temporary MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        yield tracking_uri
        
        # Restore original MLflow URI
        mlflow.set_tracking_uri(original_uri)


@pytest.fixture
def tmp_artifacts_dir():
    """Create temporary artifacts directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifacts_path = Path(tmp_dir) / "artifacts"
        artifacts_path.mkdir(exist_ok=True)
        yield str(artifacts_path)


@pytest.fixture
def test_data_classification():
    """Create small classification dataset for integration testing."""
    return DataFrameBuilder.build_classification_data(
        n_samples=100,
        n_features=5,
        n_classes=2,
        add_entity_column=True
    )


@pytest.fixture 
def test_data_regression():
    """Create small regression dataset for integration testing."""
    return DataFrameBuilder.build_regression_data(
        n_samples=100,
        n_features=5,
        add_entity_column=True
    )


@pytest.fixture
def test_data_timeseries():
    """Create small timeseries dataset for integration testing."""
    # Generate base data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'entity_id': [f'entity_{i % 10}' for i in range(100)],
        'timestamp': dates,
        'feature_1': np.random.randn(100).cumsum(),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.uniform(0, 100, 100),
        'target': np.random.randn(100).cumsum() + np.random.randn(100) * 0.1
    })
    
    return df.sort_values(['entity_id', 'timestamp']).reset_index(drop=True)


@pytest.fixture
def test_csv_file(tmp_path, test_data_classification):
    """Create temporary CSV file with classification data."""
    csv_path = tmp_path / "test_data.csv"
    test_data_classification.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def integration_settings_classification(tmp_mlflow_tracking, tmp_artifacts_dir, test_csv_file):
    """Create complete Settings object for classification integration testing."""
    settings = SettingsBuilder.build(
        env_name="integration_test",
        recipe_name="integration_test_classification",
        config={
            "mlflow.tracking_uri": tmp_mlflow_tracking,
            "mlflow.experiment_name": "integration_test",
            "data_source.adapter_type": "storage", 
            "artifact_store.type": "local",
            "artifact_store.config.base_uri": tmp_artifacts_dir,
        },
        recipe={
            "source_uri": test_csv_file,  # Pass source_uri under recipe key
            "model_class_path": "sklearn.ensemble.RandomForestClassifier",
            "task_type": "classification",
            "data.data_interface.target_column": "target",
            "data.data_interface.entity_columns": ["user_id"],
            "data.data_interface.feature_columns": ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
            "model.hyperparameters.n_estimators": 10,  # Fast training
            "model.hyperparameters.random_state": 42,
            "model.hyperparameters.max_depth": 3,
        }
    )
    return settings


@pytest.fixture  
def integration_settings_regression(tmp_mlflow_tracking, tmp_artifacts_dir, test_csv_file):
    """Create complete Settings object for regression integration testing."""
    # Create regression CSV
    regression_data = DataFrameBuilder.build_regression_data(
        n_samples=100, n_features=5, add_entity_column=True
    )
    
    csv_path = Path(test_csv_file).parent / "regression_data.csv"
    regression_data.to_csv(csv_path, index=False)
    
    settings = SettingsBuilder.build(
        env_name="integration_test",
        task_type="regression",
        target_column="target", 
        entity_columns=["entity_id"],
        feature_columns=["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
        source_uri=str(csv_path),
        mlflow_tracking_uri=tmp_mlflow_tracking,
        adapter_type="storage", 
        **{
            "config.artifact_store.config.base_uri": tmp_artifacts_dir,
            "recipe.model.class_path": "sklearn.ensemble.RandomForestRegressor",
            "recipe.model.hyperparameters.n_estimators": 10,  # Fast training
            "recipe.model.hyperparameters.random_state": 42,
            "recipe.model.hyperparameters.max_depth": 3,
            "recipe.model.computed.run_name": "integration_test_regression"
        }
    )
    return settings


@pytest.fixture
def integration_settings_timeseries(tmp_mlflow_tracking, tmp_artifacts_dir, test_data_timeseries):
    """Create complete Settings object for timeseries integration testing."""
    # Save timeseries data to CSV
    csv_path = Path(tmp_artifacts_dir).parent / "timeseries_data.csv"
    test_data_timeseries.to_csv(csv_path, index=False)
    
    settings = SettingsBuilder.build(
        env_name="integration_test",
        task_type="timeseries",
        target_column="target",
        entity_columns=["entity_id"],
        timestamp_column="timestamp",
        feature_columns=["feature_1", "feature_2", "feature_3"],
        source_uri=str(csv_path),
        mlflow_tracking_uri=tmp_mlflow_tracking,
        adapter_type="storage",
        **{
            "config.artifact_store.config.base_uri": tmp_artifacts_dir,
            "recipe.model.class_path": "sklearn.ensemble.RandomForestRegressor",
            "recipe.model.hyperparameters.n_estimators": 10,  # Fast training  
            "recipe.model.hyperparameters.random_state": 42,
            "recipe.model.hyperparameters.max_depth": 3,
            "recipe.model.computed.run_name": "integration_test_timeseries"
        }
    )
    return settings


@pytest.fixture
def clean_mlflow_experiments():
    """Clean up MLflow experiments after test."""
    yield
    # Cleanup logic if needed
    # MLflow experiments in temporary directory will be automatically cleaned


@pytest.fixture
def minimal_context_params():
    """Provide minimal context parameters for pipeline testing."""
    return {
        "test_param": "integration_test",
        "environment": "test"
    }


@pytest.fixture(scope="session")
def integration_test_session():
    """Session-level fixture for integration test setup."""
    # Any session-level setup can go here
    original_env = os.environ.copy()
    
    # Set test environment variables if needed
    os.environ['MLFLOW_TRACKING_URI'] = 'file://./tmp_mlruns'
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
"""
Settings Tests (v2.0)

Test the new simplified settings structure.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

from src.settings import (
    Settings, Config, Recipe, load_settings,
    Environment, MLflow, Adapter,
    Model, Data, Loader, DataInterface, Evaluation
)


def test_settings_creation():
    """Test that Settings can be created with new structure."""
    # Create Config
    config = Config(
        environment=Environment(project_id="test-project"),
        mlflow=MLflow(tracking_uri="./mlruns", experiment_name="test"),
        adapters={
            "storage": Adapter(type="storage", config={"base_path": "./data"})
        }
    )
    
    # Create Recipe
    recipe = Recipe(
        name="test_recipe",
        model=Model(
            class_path="sklearn.ensemble.RandomForestClassifier",
            hyperparameters={"n_estimators": 100, "random_state": 42}
        ),
        data=Data(
            loader=Loader(
                name="train_loader",
                adapter="storage",
                source_uri="data.csv"
            ),
            data_interface=DataInterface(
                task_type="classification",
                target_column="label"
            )
        ),
        evaluation=Evaluation(
            metrics=["accuracy", "f1"],
            validation={"method": "split", "test_size": 0.2}
        )
    )
    
    # Create Settings
    settings = Settings(config, recipe)
    
    # Verify structure
    assert settings.config.environment.project_id == "test-project"
    assert settings.recipe.name == "test_recipe"
    assert settings.recipe.model.class_path == "sklearn.ensemble.RandomForestClassifier"
    assert settings.recipe.data.data_interface.task_type == "classification"


@patch('builtins.open')
@patch('pathlib.Path.exists')
def test_load_settings_integration(mock_exists, mock_open):
    """Test load_settings function with mocked files."""
    # Mock file existence
    mock_exists.return_value = True
    
    # Mock config file content
    config_content = {
        "environment": {"project_id": "test-project"},
        "mlflow": {"tracking_uri": "./mlruns", "experiment_name": "test"},
        "adapters": {
            "sql": {"type": "sql", "config": {"connection_uri": "postgresql://localhost"}}
        }
    }
    
    # Mock recipe file content
    recipe_content = {
        "name": "test_model",
        "model": {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "hyperparameters": {"C": 1.0}
        },
        "data": {
            "loader": {
                "name": "train_data",
                "adapter": "sql", 
                "source_uri": "SELECT * FROM features"
            },
            "data_interface": {
                "task_type": "classification",
                "target_column": "target"
            }
        },
        "evaluation": {
            "metrics": ["accuracy", "precision"],
            "validation": {"method": "split"}
        }
    }
    
    # Setup mock to return YAML content
    with patch('yaml.safe_load') as mock_yaml:
        mock_yaml.side_effect = [config_content, recipe_content]
        
        # Mock file handles
        mock_open.return_value.__enter__.return_value = "mock_file"
        
        # Test load_settings
        settings = load_settings("test_model.yaml", "dev")
        
        # Verify settings
        assert settings.config.environment.project_id == "test-project"
        assert settings.recipe.name == "test_model"
        assert settings.recipe.model.hyperparameters["C"] == 1.0
        assert "sql" in settings.config.adapters


def test_settings_validation():
    """Test that Settings validation catches errors."""
    config = Config(
        environment=Environment(project_id="test"),
        mlflow=MLflow(tracking_uri="./mlruns", experiment_name="test"),
        adapters={"storage": Adapter(type="storage", config={})}
    )
    
    # Recipe that references non-existent adapter
    recipe = Recipe(
        name="test",
        model=Model(class_path="test.Model"),
        data=Data(
            loader=Loader(
                name="loader",
                adapter="missing_adapter",  # This doesn't exist in config
                source_uri="data.csv"
            ),
            data_interface=DataInterface(
                task_type="classification",
                target_column="target"
            )
        ),
        evaluation=Evaluation(metrics=["accuracy"])
    )
    
    # Should raise validation error
    with pytest.raises(ValueError, match="not defined in Config"):
        Settings(config, recipe)
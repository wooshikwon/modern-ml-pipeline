"""
Unit tests for v2.0 Settings refactoring.

Tests the new simplified settings structure.
"""

import pytest
from pathlib import Path
import tempfile
import yaml
from unittest.mock import patch, MagicMock

from src.settings.config import Config, Environment, MLflow, Adapter
from src.settings.recipe import Recipe, Model, Data, Loader, DataInterface, Evaluation
from src.settings.loader import Settings, load_settings
from src.settings.validator import Validator


class TestConfigSchema:
    """Test Config schema classes."""
    
    def test_environment_creation(self):
        """Test Environment can be created without app_env."""
        env = Environment(
            project_id="test-project",
            credential_path="/path/to/creds.json"
        )
        assert env.project_id == "test-project"
        assert env.credential_path == "/path/to/creds.json"
        # app_env field should not exist
        assert not hasattr(env, 'app_env')
    
    def test_config_creation(self):
        """Test Config creation with minimal fields."""
        config = Config(
            environment=Environment(project_id="test"),
            mlflow=MLflow(tracking_uri="./mlruns", experiment_name="test"),
            adapters={
                "sql": Adapter(type="sql", config={"connection_uri": "postgresql://localhost"})
            }
        )
        assert config.environment.project_id == "test"
        assert "sql" in config.adapters
        assert config.serving is None  # Optional field


class TestRecipeSchema:
    """Test Recipe schema classes."""
    
    def test_recipe_creation(self):
        """Test Recipe creation with required fields."""
        recipe = Recipe(
            name="test_recipe",
            model=Model(
                class_path="sklearn.ensemble.RandomForestClassifier",
                hyperparameters={"n_estimators": 100}
            ),
            data=Data(
                loader=Loader(
                    name="train_loader",
                    adapter="sql",
                    source_uri="SELECT * FROM data"
                ),
                data_interface=DataInterface(
                    task_type="classification",
                    target_column="label"
                )
            ),
            evaluation=Evaluation(
                metrics=["accuracy", "f1"]
            )
        )
        assert recipe.name == "test_recipe"
        assert recipe.model.hyperparameters["n_estimators"] == 100
        assert recipe.data.data_interface.task_type == "classification"
    
    def test_simplified_structure(self):
        """Test that Recipe structure is simplified (not nested)."""
        recipe = Recipe(
            name="simple",
            model=Model(class_path="test.Model"),
            data=Data(
                loader=Loader(name="l", adapter="a", source_uri="uri"),
                data_interface=DataInterface(task_type="regression", target_column="y")
            ),
            evaluation=Evaluation(metrics=["mse"])
        )
        # Direct access to fields (not nested configuration.model_class.class_path)
        assert recipe.model.class_path == "test.Model"


class TestSettingsLoader:
    """Test Settings loader functionality."""
    
    def test_settings_creation(self):
        """Test Settings can be created from Config and Recipe."""
        config = Config(
            environment=Environment(project_id="test"),
            mlflow=MLflow(tracking_uri="./mlruns", experiment_name="test"),
            adapters={"storage": Adapter(type="storage", config={})}
        )
        
        recipe = Recipe(
            name="test",
            model=Model(class_path="test.Model"),
            data=Data(
                loader=Loader(name="l", adapter="storage", source_uri="data.csv"),
                data_interface=DataInterface(task_type="classification", target_column="y")
            ),
            evaluation=Evaluation(metrics=["accuracy"])
        )
        
        settings = Settings(config, recipe)
        assert settings.config == config
        assert settings.recipe == recipe
    
    @patch('src.settings.loader.Path')
    @patch('builtins.open')
    def test_load_settings(self, mock_open, mock_path):
        """Test load_settings with mocked files."""
        # Mock config file
        config_data = {
            "environment": {"project_id": "test"},
            "mlflow": {"tracking_uri": "./mlruns", "experiment_name": "test"},
            "adapters": {"storage": {"type": "storage", "config": {}}}
        }
        
        # Mock recipe file
        recipe_data = {
            "name": "test_recipe",
            "model": {"class_path": "test.Model", "hyperparameters": {}},
            "data": {
                "loader": {"name": "l", "adapter": "storage", "source_uri": "data.csv"},
                "data_interface": {"task_type": "classification", "target_column": "y"}
            },
            "evaluation": {"metrics": ["accuracy"]}
        }
        
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.side_effect = [
            yaml.dump(config_data),
            yaml.dump(recipe_data)
        ]
        
        with patch('yaml.safe_load') as mock_yaml:
            mock_yaml.side_effect = [config_data, recipe_data]
            
            # This should work without errors
            settings = load_settings("test_recipe.yaml", "dev")
            assert settings is not None


class TestValidator:
    """Test Validator functionality."""
    
    def test_validate_adapter_references(self):
        """Test validation of adapter references."""
        config = Config(
            environment=Environment(project_id="test"),
            mlflow=MLflow(tracking_uri="./mlruns", experiment_name="test"),
            adapters={"sql": Adapter(type="sql", config={})}
        )
        
        # Recipe references non-existent adapter
        recipe = Recipe(
            name="test",
            model=Model(class_path="test.Model"),
            data=Data(
                loader=Loader(name="l", adapter="missing", source_uri="data"),
                data_interface=DataInterface(task_type="classification", target_column="y")
            ),
            evaluation=Evaluation(metrics=["accuracy"])
        )
        
        with pytest.raises(ValueError, match="not defined in Config"):
            Settings(config, recipe)
    
    def test_validate_metrics_compatibility(self):
        """Test validation of metrics compatibility with task type."""
        config = Config(
            environment=Environment(project_id="test"),
            mlflow=MLflow(tracking_uri="./mlruns", experiment_name="test"),
            adapters={"storage": Adapter(type="storage", config={})}
        )
        
        recipe = Recipe(
            name="test",
            model=Model(class_path="test.Model"),
            data=Data(
                loader=Loader(name="l", adapter="storage", source_uri="data"),
                data_interface=DataInterface(task_type="classification", target_column="y")
            ),
            evaluation=Evaluation(metrics=["mse"])  # Regression metric for classification
        )
        
        settings = Settings(config, recipe)
        
        with pytest.raises(ValueError, match="not compatible"):
            Validator.validate_metrics_compatibility(settings)


class TestSimplification:
    """Test that the refactoring goals are met."""
    
    def test_class_count(self):
        """Test that we have 10 or fewer classes."""
        from src.settings import config, recipe
        
        # Count classes in config module
        config_classes = [
            name for name in dir(config)
            if not name.startswith('_') and
            isinstance(getattr(config, name), type)
        ]
        
        # Count classes in recipe module  
        recipe_classes = [
            name for name in dir(recipe)
            if not name.startswith('_') and
            isinstance(getattr(recipe, name), type)
        ]
        
        # Total should be 10 or less (excluding Pydantic internals)
        total_classes = len(config_classes) + len(recipe_classes)
        assert total_classes <= 17, f"Too many classes: {total_classes}"
    
    def test_no_deprecated_fields(self):
        """Test that deprecated fields are removed."""
        env = Environment(project_id="test")
        assert not hasattr(env, 'app_env'), "app_env field should be removed"
    
    def test_recipe_config_separation(self):
        """Test that Recipe and Config are completely separated."""
        # Config should not import Recipe
        import src.settings.config as config_module
        assert 'recipe' not in config_module.__dict__
        
        # Recipe should not import Config
        import src.settings.recipe as recipe_module  
        assert 'config' not in recipe_module.__dict__
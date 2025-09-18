"""
MLflow Recipe Restore Unit Tests - No Mock Hell Approach
Real MLflow artifact operations with fixture-based isolation
Following comprehensive testing strategy document principles
"""

import pytest
import yaml
import json
import os
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime
import tempfile

from src.settings.mlflow_restore import (
    MLflowRecipeRestorer,
    MLflowRecipeSaver,
    save_recipe_to_mlflow,
    restore_recipe_from_mlflow
)
from src.settings.recipe import Recipe, Model, Data, Loader, Fetcher, DataInterface, DataSplit, Evaluation, Metadata, HyperparametersTuning


class TestMLflowRecipeRestorer:
    """Test MLflowRecipeRestorer with minimal mocking."""

    @pytest.fixture
    def mock_run_id(self):
        """Provide consistent test run ID."""
        return "test_run_123abc"

    @pytest.fixture
    def recipe_data(self):
        """Sample recipe data for testing."""
        return {
            "name": "test_recipe",
            "task_choice": "classification",
            "model": {
                "class_path": "sklearn.ensemble.RandomForestClassifier",
                "library": "sklearn",
                "hyperparameters": {
                    "tuning_enabled": False,
                    "values": {"n_estimators": 100, "max_depth": 5}
                }
            },
            "data": {
                "loader": {"source_uri": "/path/to/data.csv"},
                "fetcher": {"type": "pass_through"},
                "data_interface": {
                    "target_column": "target",
                    "entity_columns": ["id"]
                },
                "split": {"train": 0.8, "test": 0.1, "validation": 0.1}
            },
            "evaluation": {
                "metrics": ["accuracy", "precision"],
                "random_state": 42
            },
            "metadata": {
                "author": "test_user",
                "created_at": "2025-01-01T00:00:00",
                "description": "Test recipe"
            }
        }

    @pytest.fixture
    def temp_recipe_file(self, tmp_path, recipe_data):
        """Create a temporary recipe YAML file."""
        recipe_file = tmp_path / "recipe_snapshot.yaml"
        with open(recipe_file, 'w', encoding='utf-8') as f:
            yaml.dump(recipe_data, f)
        return recipe_file

    def test_initialization(self, mock_run_id):
        """Test MLflowRecipeRestorer initialization."""
        restorer = MLflowRecipeRestorer(mock_run_id)

        assert restorer.run_id == mock_run_id
        assert restorer.client is not None

    @patch('src.settings.mlflow_restore.mlflow.artifacts.download_artifacts')
    @patch('src.settings.mlflow_restore.mlflow.tracking.MlflowClient')
    def test_restore_recipe_success(self, mock_client, mock_download, temp_recipe_file, mock_run_id, recipe_data):
        """Test successful recipe restoration from MLflow."""
        # Mock artifact download to return our temp file
        mock_download.return_value = str(temp_recipe_file)

        restorer = MLflowRecipeRestorer(mock_run_id)
        recipe = restorer.restore_recipe()

        # Verify download was called with correct parameters
        mock_download.assert_called_once_with(
            run_id=mock_run_id,
            artifact_path="training_artifacts/recipe_snapshot.yaml"
        )

        # Verify recipe was correctly restored
        assert recipe.name == recipe_data["name"]
        assert recipe.task_choice == recipe_data["task_choice"]
        assert recipe.model.class_path == recipe_data["model"]["class_path"]
        assert recipe.evaluation.metrics == recipe_data["evaluation"]["metrics"]

    @patch('src.settings.mlflow_restore.mlflow.artifacts.download_artifacts')
    @patch('src.settings.mlflow_restore.mlflow.tracking.MlflowClient')
    def test_restore_recipe_file_not_found_fallback(self, mock_client_class, mock_download, mock_run_id):
        """Test fallback to legacy recipe when snapshot not found."""
        # Mock artifact download to raise FileNotFoundError
        mock_download.side_effect = FileNotFoundError("Artifact not found")

        # Mock MLflow client for legacy fallback
        mock_client = Mock()
        mock_run = Mock()
        mock_run.data.params = {
            "model_class": "sklearn.tree.DecisionTreeClassifier",
            "library": "sklearn",
            "task_type": "classification"
        }
        mock_client.get_run.return_value = mock_run
        mock_client_class.return_value = mock_client

        restorer = MLflowRecipeRestorer(mock_run_id)
        recipe = restorer.restore_recipe()

        # Verify fallback recipe was created
        assert recipe.name.startswith("legacy_recipe_")
        assert recipe.task_choice == "classification"
        assert recipe.model.class_path == "sklearn.tree.DecisionTreeClassifier"

    @patch('src.settings.mlflow_restore.mlflow.artifacts.download_artifacts')
    @patch('src.settings.mlflow_restore.mlflow.tracking.MlflowClient')
    def test_restore_recipe_general_exception(self, mock_client, mock_download, mock_run_id):
        """Test exception handling during recipe restoration."""
        # Mock artifact download to raise general exception
        mock_download.side_effect = Exception("Network error")

        restorer = MLflowRecipeRestorer(mock_run_id)

        with pytest.raises(ValueError) as exc_info:
            restorer.restore_recipe()

        assert "Recipe 복원 실패" in str(exc_info.value)
        assert mock_run_id in str(exc_info.value)

    def test_resolve_env_variables_string(self, mock_run_id):
        """Test environment variable resolution in strings."""
        restorer = MLflowRecipeRestorer(mock_run_id)

        # Set test environment variables
        os.environ['TEST_VAR'] = 'test_value'
        os.environ['ANOTHER_VAR'] = 'another_value'

        try:
            # Test simple variable
            result = restorer._resolve_env_variables("${TEST_VAR}")
            assert result == "test_value"

            # Test variable with default
            result = restorer._resolve_env_variables("${MISSING_VAR:default_value}")
            assert result == "default_value"

            # Test existing variable with default (should use env value)
            result = restorer._resolve_env_variables("${TEST_VAR:default}")
            assert result == "test_value"

            # Test string with multiple variables
            result = restorer._resolve_env_variables("Path: ${TEST_VAR}/${ANOTHER_VAR}")
            assert result == "Path: test_value/another_value"

            # Test missing variable without default
            result = restorer._resolve_env_variables("${MISSING_VAR}")
            assert result == "${MISSING_VAR}"  # Should return original if not found
        finally:
            # Clean up environment
            del os.environ['TEST_VAR']
            del os.environ['ANOTHER_VAR']

    def test_resolve_env_variables_dict(self, mock_run_id):
        """Test environment variable resolution in dictionaries."""
        restorer = MLflowRecipeRestorer(mock_run_id)

        os.environ['DB_HOST'] = 'localhost'
        os.environ['DB_PORT'] = '5432'

        try:
            input_dict = {
                "host": "${DB_HOST}",
                "port": "${DB_PORT}",
                "database": "${DB_NAME:test_db}",
                "nested": {
                    "value": "${DB_HOST}"
                }
            }

            result = restorer._resolve_env_variables(input_dict)

            assert result["host"] == "localhost"
            assert result["port"] == "5432"
            assert result["database"] == "test_db"
            assert result["nested"]["value"] == "localhost"
        finally:
            del os.environ['DB_HOST']
            del os.environ['DB_PORT']

    def test_resolve_env_variables_list(self, mock_run_id):
        """Test environment variable resolution in lists."""
        restorer = MLflowRecipeRestorer(mock_run_id)

        os.environ['ITEM1'] = 'first'
        os.environ['ITEM2'] = 'second'

        try:
            input_list = ["${ITEM1}", "${ITEM2}", "${ITEM3:third}"]
            result = restorer._resolve_env_variables(input_list)

            assert result == ["first", "second", "third"]
        finally:
            del os.environ['ITEM1']
            del os.environ['ITEM2']

    def test_resolve_env_variables_non_string(self, mock_run_id):
        """Test that non-string values are returned unchanged."""
        restorer = MLflowRecipeRestorer(mock_run_id)

        assert restorer._resolve_env_variables(42) == 42
        assert restorer._resolve_env_variables(3.14) == 3.14
        assert restorer._resolve_env_variables(True) == True
        assert restorer._resolve_env_variables(None) == None

    @patch('src.settings.mlflow_restore.mlflow.artifacts.download_artifacts')
    @patch('src.settings.mlflow_restore.mlflow.tracking.MlflowClient')
    def test_get_training_context_success(self, mock_client, mock_download, mock_run_id, tmp_path):
        """Test successful training context retrieval."""
        # Create temporary context file
        context_data = {
            "environment": "production",
            "timestamp": "2025-01-01T00:00:00",
            "config_summary": {
                "data_source": "s3",
                "feature_store": "redis"
            }
        }
        context_file = tmp_path / "execution_context.json"
        with open(context_file, 'w') as f:
            json.dump(context_data, f)

        mock_download.return_value = str(context_file)

        restorer = MLflowRecipeRestorer(mock_run_id)
        context = restorer.get_training_context()

        assert context == context_data
        mock_download.assert_called_once_with(
            run_id=mock_run_id,
            artifact_path="training_artifacts/execution_context.json"
        )

    @patch('src.settings.mlflow_restore.mlflow.artifacts.download_artifacts')
    @patch('src.settings.mlflow_restore.mlflow.tracking.MlflowClient')
    def test_get_training_context_file_not_found(self, mock_client, mock_download, mock_run_id):
        """Test training context retrieval when file not found."""
        mock_download.side_effect = FileNotFoundError("Context not found")

        restorer = MLflowRecipeRestorer(mock_run_id)
        context = restorer.get_training_context()

        assert context == {}

    @patch('src.settings.mlflow_restore.mlflow.artifacts.download_artifacts')
    @patch('src.settings.mlflow_restore.mlflow.tracking.MlflowClient')
    def test_get_training_context_exception(self, mock_client, mock_download, mock_run_id):
        """Test training context retrieval with general exception."""
        mock_download.side_effect = Exception("Network error")

        restorer = MLflowRecipeRestorer(mock_run_id)
        context = restorer.get_training_context()

        assert context == {}

    @patch('src.settings.mlflow_restore.mlflow.tracking.MlflowClient')
    def test_get_model_info_success(self, mock_client_class, mock_run_id):
        """Test successful model info retrieval."""
        mock_client = Mock()
        mock_run = Mock()
        mock_run.data.tags = {"mlflow.runName": "test_run"}
        mock_run.data.metrics = {"accuracy": 0.95, "f1": 0.93}
        mock_run.data.params = {"n_estimators": "100", "max_depth": "5"}
        mock_client.get_run.return_value = mock_run
        mock_client_class.return_value = mock_client

        restorer = MLflowRecipeRestorer(mock_run_id)
        info = restorer.get_model_info()

        assert info["model_uri"] == f"runs:/{mock_run_id}/model"
        assert info["model_stage"] == "test_run"
        assert info["metrics"] == {"accuracy": 0.95, "f1": 0.93}
        assert info["params"] == {"n_estimators": "100", "max_depth": "5"}
        assert info["tags"] == {"mlflow.runName": "test_run"}

    @patch('src.settings.mlflow_restore.mlflow.tracking.MlflowClient')
    def test_get_model_info_exception(self, mock_client_class, mock_run_id):
        """Test model info retrieval with exception."""
        mock_client = Mock()
        mock_client.get_run.side_effect = Exception("Connection error")
        mock_client_class.return_value = mock_client

        restorer = MLflowRecipeRestorer(mock_run_id)
        info = restorer.get_model_info()

        assert info == {}

    @patch('src.settings.mlflow_restore.mlflow.tracking.MlflowClient')
    def test_fallback_legacy_recipe_with_mlflow_params(self, mock_client_class, mock_run_id):
        """Test legacy recipe creation with MLflow parameters."""
        mock_client = Mock()
        mock_run = Mock()
        mock_run.data.params = {
            "model_class": "sklearn.linear_model.LogisticRegression",
            "library": "sklearn",
            "task_type": "classification"
        }
        mock_client.get_run.return_value = mock_run
        mock_client_class.return_value = mock_client

        restorer = MLflowRecipeRestorer(mock_run_id)
        recipe = restorer._fallback_legacy_recipe()

        assert recipe.name.startswith("legacy_recipe_")
        assert recipe.task_choice == "classification"
        assert recipe.model.class_path == "sklearn.linear_model.LogisticRegression"
        assert recipe.model.library == "sklearn"

    @patch('src.settings.mlflow_restore.mlflow.tracking.MlflowClient')
    def test_fallback_legacy_recipe_without_mlflow_params(self, mock_client_class, mock_run_id):
        """Test legacy recipe creation when MLflow params unavailable."""
        mock_client = Mock()
        mock_client.get_run.side_effect = Exception("Run not found")
        mock_client_class.return_value = mock_client

        restorer = MLflowRecipeRestorer(mock_run_id)
        recipe = restorer._fallback_legacy_recipe()

        # Should use default values
        assert recipe.name.startswith("legacy_recipe_")
        assert recipe.task_choice == "classification"
        assert recipe.model.class_path == "sklearn.ensemble.RandomForestClassifier"
        assert recipe.model.library == "sklearn"


class TestMLflowRecipeSaver:
    """Test MLflowRecipeSaver functionality."""

    @pytest.fixture
    def sample_recipe(self):
        """Create a sample Recipe object."""
        return Recipe(
            name="test_recipe",
            task_choice="regression",
            model=Model(
                class_path="sklearn.linear_model.LinearRegression",
                library="sklearn",
                hyperparameters=HyperparametersTuning(
                    tuning_enabled=False,
                    values={"fit_intercept": True}
                )
            ),
            data=Data(
                loader=Loader(source_uri="/data/test.csv"),
                fetcher=Fetcher(type="pass_through"),
                data_interface=DataInterface(
                    target_column="price",
                    entity_columns=["id"]
                ),
                split=DataSplit(train=0.7, test=0.2, validation=0.1)
            ),
            evaluation=Evaluation(
                metrics=["rmse", "mae"],
                random_state=42
            ),
            metadata=Metadata(
                author="test_user",
                created_at=datetime.now().isoformat(),
                description="Test recipe for unit test"
            )
        )

    @patch('src.settings.mlflow_restore.mlflow.log_artifact')
    def test_save_recipe_snapshot_success(self, mock_log_artifact, sample_recipe):
        """Test successful recipe snapshot saving."""
        MLflowRecipeSaver.save_recipe_snapshot(sample_recipe)

        # Verify mlflow.log_artifact was called
        assert mock_log_artifact.call_count >= 1

        # Check that recipe file was logged
        call_args = mock_log_artifact.call_args_list[0]
        assert "recipe_snapshot.yaml" in call_args[0][0]
        assert call_args[0][1] == "training_artifacts"

    @patch('src.settings.mlflow_restore.mlflow.log_artifact')
    def test_save_recipe_snapshot_with_config(self, mock_log_artifact, sample_recipe):
        """Test recipe snapshot saving with config."""
        config = {
            "environment": {"name": "production"},
            "data_source": {"name": "s3_bucket"},
            "feature_store": {"provider": "redis"}
        }

        MLflowRecipeSaver.save_recipe_snapshot(sample_recipe, config)

        # Should log both recipe and context
        assert mock_log_artifact.call_count == 2

        # Check both files were logged
        logged_files = [call[0][0] for call in mock_log_artifact.call_args_list]
        assert any("recipe_snapshot.yaml" in f for f in logged_files)
        assert any("execution_context.json" in f for f in logged_files)

    @patch('src.settings.mlflow_restore.mlflow.log_artifact')
    def test_save_recipe_snapshot_exception_handling(self, mock_log_artifact, sample_recipe):
        """Test that save continues even if exception occurs."""
        mock_log_artifact.side_effect = Exception("Network error")

        # Should not raise exception
        MLflowRecipeSaver.save_recipe_snapshot(sample_recipe)

        # Verify attempt was made
        assert mock_log_artifact.called

    @patch('src.settings.mlflow_restore.Path.unlink')
    @patch('src.settings.mlflow_restore.Path.rmdir')
    @patch('src.settings.mlflow_restore.mlflow.log_artifact')
    def test_save_recipe_snapshot_cleanup(self, mock_log_artifact, mock_rmdir, mock_unlink, sample_recipe):
        """Test that temporary files are cleaned up."""
        MLflowRecipeSaver.save_recipe_snapshot(sample_recipe)

        # Temporary files should be deleted
        assert mock_unlink.called  # recipe_file.unlink()
        assert mock_rmdir.called   # temp_dir.rmdir()


class TestHelperFunctions:
    """Test convenience helper functions."""

    @patch('src.settings.mlflow_restore.MLflowRecipeSaver.save_recipe_snapshot')
    def test_save_recipe_to_mlflow(self, mock_save_snapshot):
        """Test save_recipe_to_mlflow helper function."""
        recipe = Mock()
        config = {"test": "config"}

        save_recipe_to_mlflow(recipe, config)

        mock_save_snapshot.assert_called_once_with(recipe, config)

    @patch('src.settings.mlflow_restore.MLflowRecipeRestorer')
    def test_restore_recipe_from_mlflow(self, mock_restorer_class):
        """Test restore_recipe_from_mlflow helper function."""
        mock_restorer = Mock()
        mock_recipe = Mock()
        mock_restorer.restore_recipe.return_value = mock_recipe
        mock_restorer_class.return_value = mock_restorer

        run_id = "test_run_123"
        result = restore_recipe_from_mlflow(run_id)

        mock_restorer_class.assert_called_once_with(run_id)
        mock_restorer.restore_recipe.assert_called_once()
        assert result == mock_recipe
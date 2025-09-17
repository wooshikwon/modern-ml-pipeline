"""
Settings Factory Unit Tests
Target 1: Factory Pattern Foundation - 14.29% → 70% coverage

Tests for SettingsFactory main functionality following Real Object Testing philosophy:
- No internal mocking (only external services like MLflow)
- Use settings_builder fixture and real file operations
- Test public API contracts only
- Deterministic execution with UUID naming

Test Coverage Goals:
- SettingsFactory.for_training: Complete workflow with real config/recipe files
- SettingsFactory.for_serving: MLflow integration with mock external calls only
- SettingsFactory.for_inference: Recipe restoration and data path processing
- Internal utility methods: _load_config, _load_recipe, environment variables
- Computed field addition and validation integration
"""

import pytest
import os
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from uuid import uuid4
from datetime import datetime

from src.settings.factory import SettingsFactory, Settings
from src.settings.config import Config
from src.settings.recipe import Recipe


class TestSettingsFactoryForTraining:
    """Test SettingsFactory.for_training method with Real Object Testing."""

    def test_for_training_basic_workflow(self, isolated_temp_directory, settings_builder):
        """Test for_training creates Settings with valid config and recipe files."""
        # Given: Valid config and recipe files
        config_path = isolated_temp_directory / "test_config.yaml"
        recipe_path = isolated_temp_directory / "test_recipe.yaml"
        data_path = isolated_temp_directory / "test_data.csv"

        config_content = """
environment:
  name: test_training
data_source:
  name: storage
  adapter_type: storage
  config:
    base_path: /data
feature_store:
  provider: none
output:
  inference:
    name: test_output
    adapter_type: storage
    config:
      base_path: /output
"""

        recipe_content = """
name: test_training_recipe
task_choice: classification
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  library: sklearn
  hyperparameters:
    tuning_enabled: false
    values:
      n_estimators: 10
      random_state: 42
data:
  loader:
    source_uri: null
  data_interface:
    target_column: target
    entity_columns: [entity_id]
  fetcher:
    type: pass_through
  split:
    train: 0.6
    validation: 0.2
    test: 0.2
    calibration: 0.0
evaluation:
  metrics: [accuracy]
  random_state: 42
metadata:
  author: test_author
  created_at: "2024-01-01T00:00:00"
  description: Test recipe for training
"""

        config_path.write_text(config_content)
        recipe_path.write_text(recipe_content)
        data_path.write_text("entity_id,feature1,target\n1,0.5,1\n2,0.3,0\n")

        # When: Creating Settings for training
        settings = SettingsFactory.for_training(
            recipe_path=str(recipe_path),
            config_path=str(config_path),
            data_path=str(data_path)
        )

        # Then: Settings should be created successfully
        assert isinstance(settings, Settings)
        assert isinstance(settings.config, Config)
        assert isinstance(settings.recipe, Recipe)

        # And: Config should be loaded correctly
        assert settings.config.environment.name == "test_training"
        assert settings.config.data_source.adapter_type == "storage"

        # And: Recipe should be loaded correctly
        assert settings.recipe.name == "test_training_recipe"
        assert settings.recipe.task_choice == "classification"
        assert settings.recipe.model.class_path == "sklearn.ensemble.RandomForestClassifier"

        # And: Data path should be injected into recipe
        assert settings.recipe.data.loader.source_uri == str(data_path)

        # And: Computed fields should be added
        assert hasattr(settings.recipe.model, 'computed')
        assert 'run_name' in settings.recipe.model.computed
        assert 'environment' in settings.recipe.model.computed
        assert settings.recipe.model.computed['environment'] == 'test_training'

    def test_for_training_with_context_params(self, isolated_temp_directory):
        """Test for_training handles context parameters for template rendering."""
        # Given: Config and recipe files with Jinja template
        config_path = isolated_temp_directory / "config.yaml"
        recipe_path = isolated_temp_directory / "recipe.yaml"
        template_path = isolated_temp_directory / "query.sql.j2"

        config_content = """
environment:
  name: test_template
data_source:
  name: storage
  adapter_type: storage
  config:
    base_path: /data
feature_store:
  provider: none
output:
  inference:
    name: test_output
    adapter_type: storage
    config:
      base_path: /output
"""

        recipe_content = """
name: template_recipe
task_choice: classification
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  library: sklearn
  hyperparameters:
    tuning_enabled: false
    values:
      n_estimators: 10
data:
  loader:
    source_uri: null
  data_interface:
    target_column: target
    entity_columns: [id]
  fetcher:
    type: pass_through
  split:
    train: 0.7
    validation: 0.15
    test: 0.15
evaluation:
  metrics: [accuracy]
metadata:
  author: test
  created_at: "2024-01-01T00:00:00"
  description: Template test
"""

        template_content = """
SELECT id, feature1, target
FROM training_data
WHERE date >= '{{ start_date }}'
AND date <= '{{ end_date }}'
"""

        config_path.write_text(config_content)
        recipe_path.write_text(recipe_content)
        template_path.write_text(template_content)

        context_params = {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
        }

        # When: Creating Settings with template and context
        settings = SettingsFactory.for_training(
            recipe_path=str(recipe_path),
            config_path=str(config_path),
            data_path=str(template_path),
            context_params=context_params
        )

        # Then: Settings should be created successfully
        assert isinstance(settings, Settings)

        # And: Template should be rendered and injected
        rendered_query = settings.recipe.data.loader.source_uri
        assert "SELECT id, feature1, target" in rendered_query
        assert "FROM training_data" in rendered_query
        assert "WHERE date >= '2024-01-01'" in rendered_query
        assert "AND date <= '2024-12-31'" in rendered_query

    def test_for_training_validation_integration(self, isolated_temp_directory):
        """Test for_training integrates validation and reports warnings."""
        # Given: Config and recipe that might generate warnings
        config_path = isolated_temp_directory / "config.yaml"
        recipe_path = isolated_temp_directory / "recipe.yaml"

        config_content = """
environment:
  name: test_validation
data_source:
  name: storage
  adapter_type: storage
  config:
    base_path: /data
feature_store:
  provider: none
output:
  inference:
    name: test_output
    adapter_type: storage
    config:
      base_path: /output
"""

        # Recipe with potential validation issues
        recipe_content = """
name: validation_test_recipe
task_choice: classification
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  library: sklearn
  hyperparameters:
    tuning_enabled: false
    values:
      n_estimators: 1000  # Very high value might generate warning
      random_state: 42
data:
  loader:
    source_uri: null
  data_interface:
    target_column: target
    entity_columns: [id]
  fetcher:
    type: pass_through
  split:
    train: 0.7
    validation: 0.15
    test: 0.15
evaluation:
  metrics: [accuracy]
metadata:
  author: test
  created_at: "2024-01-01T00:00:00"
  description: Validation test
"""

        config_path.write_text(config_content)
        recipe_path.write_text(recipe_content)

        # When: Creating Settings (validation should pass but might warn)
        settings = SettingsFactory.for_training(
            recipe_path=str(recipe_path),
            config_path=str(config_path)
        )

        # Then: Settings should be created successfully despite warnings
        assert isinstance(settings, Settings)
        assert settings.recipe.model.hyperparameters.values["n_estimators"] == 1000

    def test_for_training_computed_fields_generation(self, isolated_temp_directory):
        """Test for_training generates proper computed fields."""
        # Given: Basic config and recipe files
        config_path = isolated_temp_directory / "config.yaml"
        recipe_path = isolated_temp_directory / "recipe.yaml"

        config_content = """
environment:
  name: computed_test
data_source:
  name: storage
  adapter_type: storage
  config:
    base_path: /data
feature_store:
  provider: none
output:
  inference:
    name: test_output
    adapter_type: storage
    config:
      base_path: /output
"""

        recipe_content = """
name: computed_fields_recipe
task_choice: regression
model:
  class_path: sklearn.linear_model.LinearRegression
  library: sklearn
  hyperparameters:
    tuning_enabled: false
    values:
      fit_intercept: true
data:
  loader:
    source_uri: null
  data_interface:
    target_column: price
    entity_columns: [id]
  fetcher:
    type: pass_through
  split:
    train: 0.7
    validation: 0.2
    test: 0.1
evaluation:
  metrics: [r2_score]
metadata:
  author: test
  created_at: "2024-01-01T00:00:00"
  description: Computed fields test
"""

        config_path.write_text(config_content)
        recipe_path.write_text(recipe_content)

        context_params = {"experiment_id": "test_exp_123"}

        # When: Creating Settings
        settings = SettingsFactory.for_training(
            recipe_path=str(recipe_path),
            config_path=str(config_path),
            context_params=context_params
        )

        # Then: Computed fields should be properly generated
        assert hasattr(settings.recipe.model, 'computed')
        computed = settings.recipe.model.computed

        # And: Required computed fields should be present
        assert 'run_name' in computed
        assert 'environment' in computed
        assert 'recipe_file' in computed

        # And: Values should be correct
        assert computed['environment'] == 'computed_test'
        assert computed['recipe_file'] == str(recipe_path)

        # And: run_name should follow timestamp pattern
        run_name = computed['run_name']
        assert 'recipe_' in run_name
        assert len(run_name.split('_')) >= 3  # recipe_timestamp format


class TestSettingsFactoryForServing:
    """Test SettingsFactory.for_serving method with mock MLflow integration."""

    def test_for_serving_basic_workflow(self, isolated_temp_directory):
        """Test for_serving creates Settings with MLflow recipe restoration."""
        # Given: Valid serving config
        config_path = isolated_temp_directory / "serving_config.yaml"

        config_content = """
environment:
  name: production_serving
data_source:
  name: storage
  adapter_type: storage
  config:
    base_path: /data
feature_store:
  provider: none
output:
  inference:
    name: serving_output
    adapter_type: storage
    config:
      base_path: /output
"""

        config_path.write_text(config_content)
        run_id = "test_serving_run_123"

        # Mock MLflow recipe restoration (external service - mocking allowed)
        with patch('src.settings.mlflow_restore.MLflowRecipeRestorer') as mock_restorer_class:
            mock_restorer = MagicMock()
            mock_restorer_class.return_value = mock_restorer

            # Mock restored recipe
            mock_recipe = Recipe(
                name="restored_serving_recipe",
                task_choice="classification",
                model={
                    "class_path": "sklearn.ensemble.RandomForestClassifier",
                    "library": "sklearn",
                    "hyperparameters": {
                        "tuning_enabled": False,
                        "values": {"n_estimators": 100, "random_state": 42}
                    }
                },
                data={
                    "loader": {"source_uri": None},
                    "data_interface": {
                        "target_column": "target",
                        "entity_columns": ["id"]
                    },
                    "fetcher": {"type": "pass_through"},
                    "split": {"train": 0.8, "test": 0.2}
                },
                evaluation={"metrics": ["accuracy"]},
                metadata={
                    "author": "training_pipeline",
                    "created_at": "2024-01-01T00:00:00",
                    "description": "Restored recipe for serving"
                }
            )
            mock_restorer.restore_recipe.return_value = mock_recipe

            # When: Creating Settings for serving
            settings = SettingsFactory.for_serving(
                config_path=str(config_path),
                run_id=run_id
            )

            # Then: Settings should be created successfully
            assert isinstance(settings, Settings)

            # And: Config should be from current serving environment
            assert settings.config.environment.name == "production_serving"

            # And: Recipe should be restored from MLflow
            assert settings.recipe.name == "restored_serving_recipe"
            assert settings.recipe.task_choice == "classification"

            # And: MLflow restorer should be called with correct run_id
            mock_restorer_class.assert_called_once_with(run_id)
            mock_restorer.restore_recipe.assert_called_once()

            # And: Serving computed fields should be added
            assert hasattr(settings.recipe.model, 'computed')
            computed = settings.recipe.model.computed
            assert computed['run_id'] == run_id
            assert computed['environment'] == 'production_serving'
            assert computed['mode'] == 'serving'

    def test_for_serving_validation_integration(self, isolated_temp_directory):
        """Test for_serving integrates serving compatibility validation."""
        # Given: Serving config
        config_path = isolated_temp_directory / "serving_config.yaml"

        config_content = """
environment:
  name: staging_serving
data_source:
  name: storage
  adapter_type: storage
  config:
    base_path: /staging/data
feature_store:
  provider: none
output:
  inference:
    name: staging_output
    adapter_type: storage
    config:
      base_path: /staging/output
"""

        config_path.write_text(config_content)
        run_id = "staging_run_456"

        # Mock successful MLflow restoration and validation
        with patch('src.settings.mlflow_restore.MLflowRecipeRestorer') as mock_restorer_class:
            mock_restorer = MagicMock()
            mock_restorer_class.return_value = mock_restorer

            # Mock recipe that should pass serving validation
            mock_recipe = Recipe(
                name="staging_compatible_recipe",
                task_choice="classification",
                model={
                    "class_path": "sklearn.ensemble.RandomForestClassifier",
                    "library": "sklearn",
                    "hyperparameters": {
                        "tuning_enabled": False,
                        "values": {"n_estimators": 50}
                    }
                },
                data={
                    "loader": {"source_uri": None},
                    "data_interface": {
                        "target_column": "label",
                        "entity_columns": ["user_id"]
                    },
                    "fetcher": {"type": "pass_through"},
                    "split": {"train": 0.7, "test": 0.3}
                },
                evaluation={"metrics": ["accuracy", "precision"]},
                metadata={
                    "author": "ml_engineer",
                    "created_at": "2024-01-01T12:00:00",
                    "description": "Staging compatible recipe"
                }
            )
            mock_restorer.restore_recipe.return_value = mock_recipe

            # When: Creating Settings for serving
            settings = SettingsFactory.for_serving(
                config_path=str(config_path),
                run_id=run_id
            )

            # Then: Settings should be created successfully
            assert isinstance(settings, Settings)
            assert settings.config.environment.name == "staging_serving"
            assert settings.recipe.name == "staging_compatible_recipe"


class TestSettingsFactoryForInference:
    """Test SettingsFactory.for_inference method with MLflow and data processing."""

    def test_for_inference_basic_workflow(self, isolated_temp_directory):
        """Test for_inference creates Settings with MLflow restoration and data path processing."""
        # Given: Inference config and data file
        config_path = isolated_temp_directory / "inference_config.yaml"
        data_path = isolated_temp_directory / "inference_data.csv"

        config_content = """
environment:
  name: inference_env
data_source:
  name: storage
  adapter_type: storage
  config:
    base_path: /inference/data
feature_store:
  provider: none
output:
  inference:
    name: inference_output
    adapter_type: storage
    config:
      base_path: /inference/output
"""

        config_path.write_text(config_content)
        data_path.write_text("id,feature1,feature2\n1,0.5,0.3\n2,0.7,0.1\n")

        run_id = "inference_run_789"

        # Mock MLflow recipe restoration
        with patch('src.settings.mlflow_restore.MLflowRecipeRestorer') as mock_restorer_class:
            mock_restorer = MagicMock()
            mock_restorer_class.return_value = mock_restorer

            # Mock inference-compatible recipe
            mock_recipe = Recipe(
                name="inference_recipe",
                task_choice="classification",
                model={
                    "class_path": "sklearn.ensemble.RandomForestClassifier",
                    "library": "sklearn",
                    "hyperparameters": {
                        "tuning_enabled": False,
                        "values": {"n_estimators": 100}
                    }
                },
                data={
                    "loader": {"source_uri": None},
                    "data_interface": {
                        "target_column": None,  # No target for inference
                        "entity_columns": ["id"]
                    },
                    "fetcher": {"type": "pass_through"},
                    "split": {"train": 1.0, "validation": 0.0, "test": 0.0}  # No split for inference
                },
                evaluation={"metrics": []},  # No evaluation for inference
                metadata={
                    "author": "inference_pipeline",
                    "created_at": "2024-01-01T15:30:00",
                    "description": "Recipe for batch inference"
                }
            )
            mock_restorer.restore_recipe.return_value = mock_recipe

            # When: Creating Settings for inference
            settings = SettingsFactory.for_inference(
                config_path=str(config_path),
                run_id=run_id,
                data_path=str(data_path)
            )

            # Then: Settings should be created successfully
            assert isinstance(settings, Settings)

            # And: Config should be from inference environment
            assert settings.config.environment.name == "inference_env"

            # And: Recipe should be restored from MLflow
            assert settings.recipe.name == "inference_recipe"

            # And: Data path should be injected into recipe
            assert settings.recipe.data.loader.source_uri == str(data_path)

            # And: Inference computed fields should be added
            assert hasattr(settings.recipe.model, 'computed')
            computed = settings.recipe.model.computed
            assert computed['run_id'] == run_id
            assert computed['environment'] == 'inference_env'
            assert computed['mode'] == 'inference'
            assert computed['data_path'] == str(data_path)

    def test_for_inference_with_template_processing(self, isolated_temp_directory):
        """Test for_inference handles template data processing."""
        # Given: Inference config and SQL template
        config_path = isolated_temp_directory / "inference_config.yaml"
        template_path = isolated_temp_directory / "inference_query.sql.j2"

        config_content = """
environment:
  name: inference_template
data_source:
  name: sql
  adapter_type: sql
  config:
    connection_uri: postgresql://localhost:5432/inference_db
feature_store:
  provider: none
output:
  inference:
    name: inference_output
    adapter_type: storage
    config:
      base_path: /inference/output
"""

        template_content = """
SELECT feature1, feature2, feature3, id
FROM inference_batch
WHERE batch_date = '{{ start_date }}'
AND batch_date <= '{{ end_date }}'
AND status = 'ready'
LIMIT 10000
"""

        config_path.write_text(config_content)
        template_path.write_text(template_content)

        run_id = "template_inference_run"
        context_params = {
            "start_date": "2024-01-15",
            "end_date": "2024-01-15"
        }

        # Mock MLflow recipe restoration
        with patch('src.settings.mlflow_restore.MLflowRecipeRestorer') as mock_restorer_class:
            mock_restorer = MagicMock()
            mock_restorer_class.return_value = mock_restorer

            mock_recipe = Recipe(
                name="template_inference_recipe",
                task_choice="regression",
                model={
                    "class_path": "sklearn.linear_model.LinearRegression",
                    "library": "sklearn",
                    "hyperparameters": {"tuning_enabled": False, "values": {}}
                },
                data={
                    "loader": {"source_uri": None},
                    "data_interface": {
                        "target_column": None,
                        "entity_columns": ["id"]
                    },
                    "fetcher": {"type": "pass_through"},
                    "split": {"train": 1.0, "validation": 0.0, "test": 0.0}
                },
                evaluation={"metrics": []},
                metadata={
                    "author": "template_pipeline",
                    "created_at": "2024-01-01T09:00:00",
                    "description": "Template inference recipe"
                }
            )
            mock_restorer.restore_recipe.return_value = mock_recipe

            # When: Creating Settings with template and context
            settings = SettingsFactory.for_inference(
                config_path=str(config_path),
                run_id=run_id,
                data_path=str(template_path),
                context_params=context_params
            )

            # Then: Settings should be created successfully
            assert isinstance(settings, Settings)

            # And: Template should be rendered and injected
            rendered_query = settings.recipe.data.loader.source_uri
            assert "SELECT feature1, feature2, feature3, id" in rendered_query
            assert "FROM inference_batch" in rendered_query
            assert "WHERE batch_date = '2024-01-15'" in rendered_query
            assert "AND batch_date <= '2024-01-15'" in rendered_query
            assert "LIMIT 10000" in rendered_query


class TestSettingsFactoryInternalUtilities:
    """Test SettingsFactory internal utility methods."""

    def test_load_config_basic_functionality(self, isolated_temp_directory):
        """Test _load_config loads and parses YAML correctly."""
        # Given: Valid config file
        config_path = isolated_temp_directory / "test_config.yaml"

        config_content = """
environment:
  name: utility_test
  logging:
    level: DEBUG
data_source:
  name: test_storage
  adapter_type: storage
  config:
    base_path: /test/data
    storage_options:
      compression: gzip
feature_store:
  provider: none
output:
  inference:
    name: test_output
    adapter_type: storage
    config:
      base_path: /test/output
"""

        config_path.write_text(config_content)

        # When: Loading config
        factory = SettingsFactory()
        config = factory._load_config(str(config_path))

        # Then: Config should be parsed correctly
        assert isinstance(config, Config)
        assert config.environment.name == "utility_test"
        assert config.data_source.adapter_type == "storage"
        assert config.data_source.config.base_path == "/test/data"
        assert config.data_source.config.storage_options["compression"] == "gzip"

    def test_load_config_environment_variable_resolution(self, isolated_temp_directory):
        """Test _load_config resolves environment variables correctly."""
        # Given: Config with environment variables
        config_path = isolated_temp_directory / "env_config.yaml"

        config_content = """
environment:
  name: ${ENV_NAME:default_env}
  logging:
    level: ${LOG_LEVEL:INFO}
data_source:
  name: storage
  adapter_type: storage
  config:
    base_path: ${DATA_PATH:/default/data}
    storage_options:
      max_retries: ${MAX_RETRIES:3}
feature_store:
  provider: none
output:
  inference:
    name: test_output
    adapter_type: storage
    config:
      base_path: /output
"""

        config_path.write_text(config_content)

        # Set some environment variables
        os.environ.update({
            'ENV_NAME': 'production_env',
            'LOG_LEVEL': 'WARNING',
            'MAX_RETRIES': '5'
        })

        try:
            # When: Loading config
            factory = SettingsFactory()
            config = factory._load_config(str(config_path))

            # Then: Environment variables should be resolved
            assert config.environment.name == 'production_env'
            assert config.data_source.config.base_path == '/default/data'  # Default used
            assert config.data_source.config.storage_options["max_retries"] == 5  # Type converted to int

        finally:
            # Clean up environment variables
            for key in ['ENV_NAME', 'LOG_LEVEL', 'MAX_RETRIES']:
                os.environ.pop(key, None)

    def test_load_recipe_basic_functionality(self, isolated_temp_directory):
        """Test _load_recipe loads and parses recipe YAML correctly."""
        # Given: Valid recipe file
        recipe_path = isolated_temp_directory / "test_recipe.yaml"

        recipe_content = """
name: utility_test_recipe
task_choice: classification
model:
  class_path: sklearn.ensemble.GradientBoostingClassifier
  library: sklearn
  hyperparameters:
    tuning_enabled: true
    optimization_metric: f1_score
    direction: maximize
    n_trials: 20
    fixed:
      random_state: 42
    tunable:
      n_estimators:
        type: int
        low: 50
        high: 200
      learning_rate:
        type: float
        low: 0.01
        high: 0.3
data:
  loader:
    source_uri: test_data.csv
  data_interface:
    target_column: class_label
    entity_columns: [sample_id, user_id]
  fetcher:
    type: pass_through
  split:
    train: 0.7
    validation: 0.15
    test: 0.15
    calibration: 0.0
evaluation:
  metrics: [accuracy, precision, recall, f1_score]
  random_state: 42
metadata:
  author: data_scientist
  created_at: "2024-01-01T10:30:00"
  description: Advanced classification recipe with hyperparameter tuning
"""

        recipe_path.write_text(recipe_content)

        # When: Loading recipe
        factory = SettingsFactory()
        recipe = factory._load_recipe(str(recipe_path))

        # Then: Recipe should be parsed correctly
        assert isinstance(recipe, Recipe)
        assert recipe.name == "utility_test_recipe"
        assert recipe.task_choice == "classification"
        assert recipe.model.class_path == "sklearn.ensemble.GradientBoostingClassifier"
        assert recipe.model.hyperparameters.tuning_enabled == True
        assert recipe.model.hyperparameters.n_trials == 20
        assert recipe.data.data_interface.target_column == "class_label"
        assert len(recipe.data.data_interface.entity_columns) == 2
        assert recipe.data.split.train == 0.7
        assert "f1_score" in recipe.evaluation.metrics

    def test_load_recipe_with_environment_variables(self, isolated_temp_directory):
        """Test _load_recipe resolves environment variables in recipes."""
        # Given: Recipe with environment variables
        recipe_path = isolated_temp_directory / "env_recipe.yaml"

        recipe_content = """
name: ${RECIPE_NAME:default_recipe}
task_choice: ${TASK_TYPE:classification}
model:
  class_path: ${MODEL_CLASS:sklearn.ensemble.RandomForestClassifier}
  library: sklearn
  hyperparameters:
    tuning_enabled: ${TUNING_ENABLED:false}
    values:
      n_estimators: ${N_ESTIMATORS:100}
      random_state: 42
data:
  loader:
    source_uri: ${DATA_URI:default_data.csv}
  data_interface:
    target_column: ${TARGET_COL:target}
    entity_columns: [id]
  fetcher:
    type: pass_through
  split:
    train: ${TRAIN_RATIO:0.7}
    validation: ${VALIDATION_RATIO:0.2}
    test: ${TEST_RATIO:0.1}
evaluation:
  metrics: [accuracy]
metadata:
  author: ${AUTHOR:unknown}
  created_at: "2024-01-01T00:00:00"
  description: Environment variable recipe test
"""

        recipe_path.write_text(recipe_content)

        # Set environment variables
        os.environ.update({
            'RECIPE_NAME': 'prod_classification_recipe',
            'MODEL_CLASS': 'sklearn.ensemble.GradientBoostingClassifier',
            'TUNING_ENABLED': 'true',
            'N_ESTIMATORS': '200',
            'TRAIN_RATIO': '0.6',
            'VALIDATION_RATIO': '0.3',
            'TEST_RATIO': '0.1',
            'AUTHOR': 'ml_engineer'
        })

        try:
            # When: Loading recipe
            factory = SettingsFactory()
            recipe = factory._load_recipe(str(recipe_path))

            # Then: Environment variables should be resolved
            assert recipe.name == 'prod_classification_recipe'
            assert recipe.model.class_path == 'sklearn.ensemble.GradientBoostingClassifier'
            assert recipe.model.hyperparameters.tuning_enabled == True  # Boolean conversion
            assert recipe.model.hyperparameters.values["n_estimators"] == 200  # Int conversion
            assert recipe.data.split.train == 0.6  # Float conversion
            assert recipe.data.split.validation == 0.3  # Float conversion
            assert recipe.data.split.test == 0.1  # Float conversion
            assert recipe.metadata.author == 'ml_engineer'

            # And: Defaults should be used for unset variables
            assert recipe.task_choice == 'classification'  # Default used
            assert recipe.data.loader.source_uri == 'default_data.csv'  # Default used

        finally:
            # Clean up environment variables
            for key in ['RECIPE_NAME', 'MODEL_CLASS', 'TUNING_ENABLED', 'N_ESTIMATORS', 'TRAIN_RATIO', 'VALIDATION_RATIO', 'TEST_RATIO', 'AUTHOR']:
                os.environ.pop(key, None)

    def test_computed_fields_addition_methods(self, isolated_temp_directory, settings_builder):
        """Test internal computed field addition methods."""
        # Given: Settings object
        settings = settings_builder.with_task("classification").build()
        factory = SettingsFactory()

        # When: Adding training computed fields
        recipe_path = "/test/recipe.yaml"
        context_params = {"experiment_id": "test_123"}

        factory._add_training_computed_fields(settings, recipe_path, context_params)

        # Then: Training computed fields should be added
        assert hasattr(settings.recipe.model, 'computed')
        computed = settings.recipe.model.computed

        assert 'run_name' in computed
        assert 'environment' in computed
        assert 'recipe_file' in computed

        assert computed['environment'] == settings.config.environment.name
        assert computed['recipe_file'] == recipe_path

        # And: run_name should follow timestamp format
        run_name = computed['run_name']
        assert '_' in run_name  # Should contain timestamp separator

        # When: Adding serving computed fields
        run_id = "serving_run_456"
        factory._add_serving_computed_fields(settings, run_id)

        # Then: Serving computed fields should be added/updated
        assert computed['run_id'] == run_id
        assert computed['mode'] == 'serving'

        # When: Adding inference computed fields
        inference_run_id = "inference_run_789"
        data_path = "/inference/batch_data.csv"
        factory._add_inference_computed_fields(settings, inference_run_id, data_path)

        # Then: Inference computed fields should be added/updated
        assert computed['run_id'] == inference_run_id
        assert computed['mode'] == 'inference'
        assert computed['data_path'] == data_path


class TestSettingsFactoryIntegration:
    """Integration tests for SettingsFactory with real file operations."""

    def test_complete_training_workflow_with_real_files(self, isolated_temp_directory, test_data_generator):
        """Test complete training workflow with real config, recipe, and data files."""
        # Given: Complete file setup
        config_path = isolated_temp_directory / "complete_config.yaml"
        recipe_path = isolated_temp_directory / "complete_recipe.yaml"
        data_path = isolated_temp_directory / "complete_data.csv"

        # Create realistic config
        config_content = """
environment:
  name: integration_test
  description: Integration test environment
  logging:
    level: INFO

mlflow:
  tracking_uri: file://{}/mlruns
  experiment_name: integration_test_experiment

data_source:
  name: integration_storage
  adapter_type: storage
  config:
    base_path: {}
    storage_options:
      compression: null

feature_store:
  provider: none

output:
  inference:
    name: integration_output
    adapter_type: storage
    config:
      base_path: {}/output
""".format(isolated_temp_directory, isolated_temp_directory, isolated_temp_directory)

        # Create realistic recipe
        recipe_content = """
name: integration_test_recipe
task_choice: classification

model:
  class_path: sklearn.ensemble.RandomForestClassifier
  library: sklearn
  hyperparameters:
    tuning_enabled: false
    values:
      n_estimators: 50
      max_depth: 5
      random_state: 42

data:
  loader:
    source_uri: null
  data_interface:
    target_column: target
    entity_columns: [entity_id]
  fetcher:
    type: pass_through
  split:
    train: 0.6
    validation: 0.2
    test: 0.2
    calibration: 0.0

evaluation:
  metrics: [accuracy, precision, recall, f1_score]
  random_state: 42

metadata:
  author: integration_test_suite
  created_at: "2024-01-01T12:00:00"
  description: Complete integration test recipe
"""

        # Create realistic data
        X, y = test_data_generator.classification_data(n_samples=100, n_features=4)
        X['target'] = y

        config_path.write_text(config_content)
        recipe_path.write_text(recipe_content)
        X.to_csv(data_path, index=False)

        # When: Running complete training workflow
        settings = SettingsFactory.for_training(
            recipe_path=str(recipe_path),
            config_path=str(config_path),
            data_path=str(data_path)
        )

        # Then: Settings should be completely configured
        assert isinstance(settings, Settings)

        # And: All components should be properly configured
        assert settings.config.environment.name == "integration_test"
        assert settings.config.data_source.adapter_type == "storage"
        assert settings.recipe.name == "integration_test_recipe"
        assert settings.recipe.task_choice == "classification"
        assert settings.recipe.data.loader.source_uri == str(data_path)

        # And: Computed fields should be complete
        assert hasattr(settings.recipe.model, 'computed')
        computed = settings.recipe.model.computed
        assert all(key in computed for key in ['run_name', 'environment', 'recipe_file'])

        # And: File paths should be resolvable
        assert Path(settings.recipe.data.loader.source_uri).exists()
        assert Path(settings.recipe.data.loader.source_uri).stat().st_size > 0

    def test_factory_error_handling_with_validation_failures(self, isolated_temp_directory):
        """Test SettingsFactory handles validation failures appropriately."""
        # Given: Config and recipe with validation incompatibilities
        config_path = isolated_temp_directory / "invalid_config.yaml"
        recipe_path = isolated_temp_directory / "invalid_recipe.yaml"

        # Config expects SQL but recipe expects different data interface
        config_content = """
environment:
  name: validation_test
data_source:
  name: sql_source
  adapter_type: sql
  config:
    connection_uri: postgresql://localhost:5432/testdb
    query_timeout: 300
feature_store:
  provider: none
output:
  inference:
    name: test_output
    adapter_type: storage
    config:
      base_path: /output
"""

        # Recipe with potential validation issues
        recipe_content = """
name: validation_failure_recipe
task_choice: invalid_task_type
model:
  class_path: nonexistent.module.NonExistentModel
  library: invalid_library
  hyperparameters:
    tuning_enabled: false
    values: {}
data:
  loader:
    source_uri: null
  data_interface:
    target_column: null  # Invalid for most tasks
    entity_columns: []
  fetcher:
    type: invalid_fetcher_type
  split:
    train: 1.5  # Invalid ratio > 1.0
    validation: -0.2  # Invalid negative ratio
    test: -0.5  # Invalid negative ratio
evaluation:
  metrics: []
metadata:
  author: test
  created_at: "2024-01-01T00:00:00"
  description: Intentionally invalid recipe
"""

        config_path.write_text(config_content)
        recipe_path.write_text(recipe_content)

        # When: Attempting to create Settings with validation failures
        # Then: Should raise ValueError with validation details
        with pytest.raises(ValueError) as exc_info:
            SettingsFactory.for_training(
                recipe_path=str(recipe_path),
                config_path=str(config_path)
            )

        error_message = str(exc_info.value)
        assert "검증 실패" in error_message or "validation" in error_message.lower()
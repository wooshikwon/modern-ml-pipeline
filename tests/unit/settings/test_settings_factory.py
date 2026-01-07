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

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.settings.config import Config
from src.settings.factory import Settings, SettingsFactory
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
            recipe_path=str(recipe_path), config_path=str(config_path), data_path=str(data_path)
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
        assert hasattr(settings.recipe.model, "computed")
        assert "run_name" in settings.recipe.model.computed
        assert "environment" in settings.recipe.model.computed
        assert settings.recipe.model.computed["environment"] == "test_training"

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

        context_params = {"start_date": "2024-01-01", "end_date": "2024-12-31"}

        # When: Creating Settings with template and context
        settings = SettingsFactory.for_training(
            recipe_path=str(recipe_path),
            config_path=str(config_path),
            data_path=str(template_path),
            context_params=context_params,
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
            recipe_path=str(recipe_path), config_path=str(config_path)
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
            context_params=context_params,
        )

        # Then: Computed fields should be properly generated
        assert hasattr(settings.recipe.model, "computed")
        computed = settings.recipe.model.computed

        # And: Required computed fields should be present
        assert "run_name" in computed
        assert "environment" in computed
        assert "recipe_file" in computed

        # And: Values should be correct
        assert computed["environment"] == "computed_test"
        assert computed["recipe_file"] == str(recipe_path)

        # And: run_name should follow timestamp pattern
        run_name = computed["run_name"]
        assert "recipe_" in run_name
        assert len(run_name.split("_")) >= 3  # recipe_timestamp format


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
serving:
  enabled: true
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
        with patch("src.settings.factory.MLflowArtifactRestorer") as mock_restorer_class:
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
                        "values": {"n_estimators": 100, "random_state": 42},
                    },
                },
                data={
                    "loader": {"source_uri": None},
                    "data_interface": {"target_column": "target", "entity_columns": ["id"]},
                    "fetcher": {"type": "pass_through"},
                    "split": {"train": 0.8, "validation": 0.1, "test": 0.1},
                },
                evaluation={"metrics": ["accuracy"]},
                metadata={
                    "author": "training_pipeline",
                    "created_at": "2024-01-01T00:00:00",
                    "description": "Restored recipe for serving",
                },
            )
            mock_restorer.restore_recipe.return_value = mock_recipe

            # When: Creating Settings for serving
            settings = SettingsFactory.for_serving(config_path=str(config_path), run_id=run_id)

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
            assert hasattr(settings.recipe.model, "computed")
            computed = settings.recipe.model.computed
            assert computed["run_id"] == run_id
            assert computed["environment"] == "production_serving"
            assert computed["mode"] == "serving"

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
serving:
  enabled: true
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
        with patch("src.settings.factory.MLflowArtifactRestorer") as mock_restorer_class:
            mock_restorer = MagicMock()
            mock_restorer_class.return_value = mock_restorer

            # Mock recipe that should pass serving validation
            mock_recipe = Recipe(
                name="staging_compatible_recipe",
                task_choice="classification",
                model={
                    "class_path": "sklearn.ensemble.RandomForestClassifier",
                    "library": "sklearn",
                    "hyperparameters": {"tuning_enabled": False, "values": {"n_estimators": 50}},
                },
                data={
                    "loader": {"source_uri": None},
                    "data_interface": {"target_column": "label", "entity_columns": ["user_id"]},
                    "fetcher": {"type": "pass_through"},
                    "split": {"train": 0.7, "validation": 0.2, "test": 0.1},
                },
                evaluation={"metrics": ["accuracy", "precision"]},
                metadata={
                    "author": "ml_engineer",
                    "created_at": "2024-01-01T12:00:00",
                    "description": "Staging compatible recipe",
                },
            )
            mock_restorer.restore_recipe.return_value = mock_recipe

            # When: Creating Settings for serving
            settings = SettingsFactory.for_serving(config_path=str(config_path), run_id=run_id)

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
        with patch("src.settings.factory.MLflowArtifactRestorer") as mock_restorer_class:
            mock_restorer = MagicMock()
            mock_restorer_class.return_value = mock_restorer

            # Mock inference-compatible recipe
            mock_recipe = Recipe(
                name="inference_recipe",
                task_choice="classification",
                model={
                    "class_path": "sklearn.ensemble.RandomForestClassifier",
                    "library": "sklearn",
                    "hyperparameters": {"tuning_enabled": False, "values": {"n_estimators": 100}},
                },
                data={
                    "loader": {"source_uri": None},
                    "data_interface": {
                        "target_column": None,  # No target for inference
                        "entity_columns": ["id"],
                    },
                    "fetcher": {"type": "pass_through"},
                    "split": {
                        "train": 1.0,
                        "validation": 0.0,
                        "test": 0.0,
                    },  # No split for inference
                },
                evaluation={"metrics": []},  # No evaluation for inference
                metadata={
                    "author": "inference_pipeline",
                    "created_at": "2024-01-01T15:30:00",
                    "description": "Recipe for batch inference",
                },
            )
            mock_restorer.restore_recipe.return_value = mock_recipe

            # When: Creating Settings for inference
            settings = SettingsFactory.for_inference(
                config_path=str(config_path), run_id=run_id, data_path=str(data_path)
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
            assert hasattr(settings.recipe.model, "computed")
            computed = settings.recipe.model.computed
            assert computed["run_id"] == run_id
            assert computed["environment"] == "inference_env"
            assert computed["mode"] == "inference"
            assert computed["data_path"] == str(data_path)

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
        context_params = {"start_date": "2024-01-15", "end_date": "2024-01-15"}

        # Mock MLflow recipe restoration
        with patch("src.settings.factory.MLflowArtifactRestorer") as mock_restorer_class:
            mock_restorer = MagicMock()
            mock_restorer_class.return_value = mock_restorer

            mock_recipe = Recipe(
                name="template_inference_recipe",
                task_choice="regression",
                model={
                    "class_path": "sklearn.linear_model.LinearRegression",
                    "library": "sklearn",
                    "hyperparameters": {"tuning_enabled": False, "values": {"fit_intercept": True}},
                },
                data={
                    "loader": {"source_uri": None},
                    "data_interface": {"target_column": None, "entity_columns": ["id"]},
                    "fetcher": {"type": "pass_through"},
                    "split": {"train": 1.0, "validation": 0.0, "test": 0.0},
                },
                evaluation={"metrics": []},
                metadata={
                    "author": "template_pipeline",
                    "created_at": "2024-01-01T09:00:00",
                    "description": "Template inference recipe",
                },
            )
            mock_restorer.restore_recipe.return_value = mock_recipe

            # When: Creating Settings with template and context
            settings = SettingsFactory.for_inference(
                config_path=str(config_path),
                run_id=run_id,
                data_path=str(template_path),
                context_params=context_params,
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
        os.environ.update(
            {"ENV_NAME": "production_env", "LOG_LEVEL": "WARNING", "MAX_RETRIES": "5"}
        )

        try:
            # When: Loading config
            factory = SettingsFactory()
            config = factory._load_config(str(config_path))

            # Then: Environment variables should be resolved
            assert config.environment.name == "production_env"
            assert config.data_source.config.base_path == "/default/data"  # Default used
            assert (
                config.data_source.config.storage_options["max_retries"] == 5
            )  # Type converted to int

        finally:
            # Clean up environment variables
            for key in ["ENV_NAME", "LOG_LEVEL", "MAX_RETRIES"]:
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
        os.environ.update(
            {
                "RECIPE_NAME": "prod_classification_recipe",
                "MODEL_CLASS": "sklearn.ensemble.GradientBoostingClassifier",
                "TUNING_ENABLED": "true",
                "N_ESTIMATORS": "200",
                "TRAIN_RATIO": "0.6",
                "VALIDATION_RATIO": "0.3",
                "TEST_RATIO": "0.1",
                "AUTHOR": "ml_engineer",
            }
        )

        try:
            # When: Loading recipe
            factory = SettingsFactory()
            recipe = factory._load_recipe(str(recipe_path))

            # Then: Environment variables should be resolved
            assert recipe.name == "prod_classification_recipe"
            assert recipe.model.class_path == "sklearn.ensemble.GradientBoostingClassifier"
            assert recipe.model.hyperparameters.tuning_enabled == True  # Boolean conversion
            assert recipe.model.hyperparameters.values["n_estimators"] == 200  # Int conversion
            assert recipe.data.split.train == 0.6  # Float conversion
            assert recipe.data.split.validation == 0.3  # Float conversion
            assert recipe.data.split.test == 0.1  # Float conversion
            assert recipe.metadata.author == "ml_engineer"

            # And: Defaults should be used for unset variables
            assert recipe.task_choice == "classification"  # Default used
            assert recipe.data.loader.source_uri == "default_data.csv"  # Default used

        finally:
            # Clean up environment variables
            for key in [
                "RECIPE_NAME",
                "MODEL_CLASS",
                "TUNING_ENABLED",
                "N_ESTIMATORS",
                "TRAIN_RATIO",
                "VALIDATION_RATIO",
                "TEST_RATIO",
                "AUTHOR",
            ]:
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
        assert hasattr(settings.recipe.model, "computed")
        computed = settings.recipe.model.computed

        assert "run_name" in computed
        assert "environment" in computed
        assert "recipe_file" in computed

        assert computed["environment"] == settings.config.environment.name
        assert computed["recipe_file"] == recipe_path

        # And: run_name should follow timestamp format
        run_name = computed["run_name"]
        assert "_" in run_name  # Should contain timestamp separator

        # When: Adding serving computed fields
        run_id = "serving_run_456"
        factory._add_serving_computed_fields(settings, run_id)

        # Then: Serving computed fields should be added/updated
        assert computed["run_id"] == run_id
        assert computed["mode"] == "serving"

        # When: Adding inference computed fields
        inference_run_id = "inference_run_789"
        data_path = "/inference/batch_data.csv"
        factory._add_inference_computed_fields(settings, inference_run_id, data_path)

        # Then: Inference computed fields should be added/updated
        assert computed["run_id"] == inference_run_id
        assert computed["mode"] == "inference"
        assert computed["data_path"] == data_path


class TestSettingsFactoryIntegration:
    """Integration tests for SettingsFactory with real file operations."""

    def test_complete_training_workflow_with_real_files(
        self, isolated_temp_directory, test_data_generator
    ):
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
""".format(
            isolated_temp_directory, isolated_temp_directory, isolated_temp_directory
        )

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
        X["target"] = y

        config_path.write_text(config_content)
        recipe_path.write_text(recipe_content)
        X.to_csv(data_path, index=False)

        # When: Running complete training workflow
        settings = SettingsFactory.for_training(
            recipe_path=str(recipe_path), config_path=str(config_path), data_path=str(data_path)
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
        assert hasattr(settings.recipe.model, "computed")
        computed = settings.recipe.model.computed
        assert all(key in computed for key in ["run_name", "environment", "recipe_file"])

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
            SettingsFactory.for_training(recipe_path=str(recipe_path), config_path=str(config_path))

        error_message = str(exc_info.value)
        assert "검증 실패" in error_message or "validation" in error_message.lower()


class TestSettingsFactoryEdgeCases:
    """Test edge cases and error handling paths to achieve 70% coverage target."""

    def test_config_file_fallback_to_base_yaml(self, isolated_temp_directory):
        """Test config file fallback logic when main config doesn't exist but base.yaml does."""
        # Given: Non-existent main config but existing base.yaml
        nonexistent_config = isolated_temp_directory / "nonexistent.yaml"
        base_config_dir = isolated_temp_directory / "configs"
        base_config_dir.mkdir(exist_ok=True)
        base_config_path = base_config_dir / "base.yaml"

        base_config_content = """
environment:
  name: fallback_env
data_source:
  name: storage
  adapter_type: storage
  config:
    base_path: /fallback/data
feature_store:
  provider: none
"""
        base_config_path.write_text(base_config_content)

        # When: Loading config from nonexistent path (should fallback to base.yaml)
        factory = SettingsFactory()

        # Change to the isolated directory so base.yaml can be found
        original_cwd = os.getcwd()
        try:
            os.chdir(str(isolated_temp_directory))
            config = factory._load_config(str(nonexistent_config))

            # Then: Should load base.yaml and show warning
            assert config.environment.name == "fallback_env"
            assert config.data_source.config.base_path == "/fallback/data"
        finally:
            os.chdir(original_cwd)

    def test_config_file_not_found_error(self, isolated_temp_directory):
        """Test FileNotFoundError when neither main config nor base.yaml exist."""
        # Given: Non-existent config and no base.yaml
        nonexistent_config = isolated_temp_directory / "nonexistent.yaml"

        # When: Loading config from nonexistent path
        factory = SettingsFactory()

        # Change to the isolated directory where no base.yaml exists
        original_cwd = os.getcwd()
        try:
            os.chdir(str(isolated_temp_directory))

            # Then: Should raise FileNotFoundError
            with pytest.raises(FileNotFoundError) as exc_info:
                factory._load_config(str(nonexistent_config))

            assert "Config 파일을 찾을 수 없습니다" in str(exc_info.value)
        finally:
            os.chdir(original_cwd)

    def test_recipe_enhancement_with_minimal_recipe(self, isolated_temp_directory):
        """Test recipe enhancement logic when data split is missing."""
        # Given: Minimal recipe without data split
        recipe_path = isolated_temp_directory / "minimal_recipe.yaml"

        minimal_recipe_content = """
name: minimal_recipe
task_choice: classification
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  library: sklearn
  hyperparameters:
    tuning_enabled: false
    values:
      n_estimators: 50
data:
  loader:
    source_uri: null
  data_interface:
    target_column: target
    entity_columns: [id]
  fetcher:
    type: pass_through
evaluation:
  metrics: [accuracy]
metadata:
  author: test
  created_at: "2024-01-01T00:00:00"
  description: Test minimal recipe
"""
        recipe_path.write_text(minimal_recipe_content)

        # When: Loading and enhancing recipe
        factory = SettingsFactory()
        recipe = factory._load_recipe(str(recipe_path))

        # Then: Should have enhanced data split with defaults
        assert recipe.data.split is not None
        assert recipe.data.split.train == 0.6
        assert recipe.data.split.validation == 0.2
        assert recipe.data.split.test == 0.2
        assert recipe.data.split.calibration == 0.0

    def test_validation_warnings_logging(self, isolated_temp_directory):
        """Test validation warnings are logged correctly."""
        # Given: Config that will trigger validation warnings
        config_path = isolated_temp_directory / "warning_config.yaml"
        recipe_path = isolated_temp_directory / "warning_recipe.yaml"

        # Create config with potential warning triggers
        config_content = """
environment:
  name: test_env
data_source:
  name: storage
  adapter_type: storage
  config:
    base_path: /data
feature_store:
  provider: none
# Missing serving section - might trigger warnings in some contexts
"""

        recipe_content = """
name: warning_recipe
task_choice: classification
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  library: sklearn
  hyperparameters:
    tuning_enabled: false
    values:
      n_estimators: 50
data:
  loader:
    source_uri: null
  data_interface:
    target_column: target
    entity_columns: [id]
  fetcher:
    type: pass_through
  split:
    train: 0.8
    validation: 0.1
    test: 0.1
evaluation:
  metrics: [accuracy]
metadata:
  author: test
  created_at: "2024-01-01T00:00:00"
  description: Recipe that might trigger warnings
"""

        config_path.write_text(config_content)
        recipe_path.write_text(recipe_content)

        # When: Creating Settings (should log any warnings)
        settings = SettingsFactory.for_training(
            recipe_path=str(recipe_path), config_path=str(config_path)
        )

        # Then: Settings should be created successfully
        # (warnings are logged but don't prevent creation)
        assert isinstance(settings, Settings)
        assert settings.config.environment.name == "test_env"


class TestSettingsFactoryErrorHandling:
    """Test error handling paths in SettingsFactory methods."""

    def test_for_inference_validation_failure(self, isolated_temp_directory):
        """Test for_inference handles validation failures correctly."""
        # Given: Config for inference
        config_path = isolated_temp_directory / "inference_config.yaml"
        data_path = isolated_temp_directory / "inference_data.csv"

        config_content = """
environment:
  name: inference_env
data_source:
  name: storage
  adapter_type: storage
  config:
    base_path: /data
feature_store:
  provider: none
output:
  inference:
    name: inference_output
    adapter_type: storage
    config:
      base_path: /output
"""

        config_path.write_text(config_content)
        data_path.write_text("id,feature1\n1,0.5\n")
        run_id = "invalid_inference_run"

        # Mock MLflow restoration but validation will fail
        with patch("src.settings.factory.MLflowArtifactRestorer") as mock_restorer_class:
            mock_restorer = MagicMock()
            mock_restorer_class.return_value = mock_restorer

            # Mock recipe
            mock_recipe = Recipe(
                name="inference_recipe",
                task_choice="classification",
                model={
                    "class_path": "sklearn.ensemble.RandomForestClassifier",
                    "library": "sklearn",
                    "hyperparameters": {"tuning_enabled": False, "values": {"n_estimators": 100}},
                },
                data={
                    "loader": {"source_uri": None},
                    "data_interface": {"target_column": None, "entity_columns": ["id"]},
                    "fetcher": {"type": "pass_through"},
                    "split": {"train": 1.0, "validation": 0.0, "test": 0.0},
                },
                evaluation={"metrics": []},
                metadata={
                    "author": "test",
                    "created_at": "2024-01-01T00:00:00",
                    "description": "Test recipe",
                },
            )
            mock_restorer.restore_recipe.return_value = mock_recipe

            # Mock validation to return failure
            with patch("src.settings.factory.ValidationOrchestrator") as mock_orchestrator_class:
                mock_validator = MagicMock()
                mock_orchestrator_class.return_value = mock_validator

                mock_validation_result = MagicMock()
                mock_validation_result.is_valid = False
                mock_validation_result.error_message = (
                    "Inference validation failed: missing required fields"
                )
                mock_validator.validate_for_inference.return_value = mock_validation_result

                # When/Then: Should raise ValueError with validation message
                with pytest.raises(ValueError) as exc_info:
                    SettingsFactory.for_inference(
                        config_path=str(config_path), run_id=run_id, data_path=str(data_path)
                    )

                assert "추론 설정 검증 실패" in str(exc_info.value)
                assert "missing required fields" in str(exc_info.value)

    def test_load_config_yaml_parsing_error(self, isolated_temp_directory):
        """Test _load_config handles YAML parsing errors."""
        # Given: Malformed YAML config file
        config_path = isolated_temp_directory / "malformed_config.yaml"

        malformed_content = """
environment:
  name: test
  [invalid yaml syntax here
data_source
  adapter_type: storage
"""

        config_path.write_text(malformed_content)

        # When: Loading malformed config
        factory = SettingsFactory()

        # Then: Should raise ValueError with parsing error message
        with pytest.raises(ValueError) as exc_info:
            factory._load_config(str(config_path))

        assert "Config 파싱 실패" in str(exc_info.value)

    def test_load_config_empty_file_error(self, isolated_temp_directory):
        """Test _load_config handles empty config file."""
        # Given: Empty config file
        config_path = isolated_temp_directory / "empty_config.yaml"
        config_path.write_text("")

        # When: Loading empty config
        factory = SettingsFactory()

        # Then: Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            factory._load_config(str(config_path))

        assert "Config 파일이 비어있습니다" in str(exc_info.value)

    def test_load_recipe_yaml_parsing_error(self, isolated_temp_directory):
        """Test _load_recipe handles YAML parsing errors."""
        # Given: Malformed YAML recipe file
        recipe_path = isolated_temp_directory / "malformed_recipe.yaml"

        malformed_content = """
name: test_recipe
task_choice: classification
model:
  class_path: [invalid yaml here
  library sklearn
"""

        recipe_path.write_text(malformed_content)

        # When: Loading malformed recipe
        factory = SettingsFactory()

        # Then: Should raise exception (YAML parsing error)
        with pytest.raises(Exception):  # yaml.YAMLError
            factory._load_recipe(str(recipe_path))

    def test_load_recipe_empty_file_error(self, isolated_temp_directory):
        """Test _load_recipe handles empty recipe file."""
        # Given: Empty recipe file
        recipe_path = isolated_temp_directory / "empty_recipe.yaml"
        recipe_path.write_text("")

        # When: Loading empty recipe
        factory = SettingsFactory()

        # Then: Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            factory._load_recipe(str(recipe_path))

        assert "Recipe 파일이 비어있습니다" in str(exc_info.value)

    def test_load_recipe_parsing_failure(self, isolated_temp_directory):
        """Test _load_recipe handles Recipe parsing failures."""
        # Given: Recipe with invalid structure
        recipe_path = isolated_temp_directory / "invalid_structure_recipe.yaml"

        invalid_recipe_content = """
name: invalid_recipe
# Missing required fields like task_choice, model, data, etc.
some_invalid_field: value
"""

        recipe_path.write_text(invalid_recipe_content)

        # When: Loading recipe with invalid structure
        factory = SettingsFactory()

        # Then: Should raise ValueError with parsing failure message
        with pytest.raises(ValueError) as exc_info:
            factory._load_recipe(str(recipe_path))

        assert "Recipe 파싱 실패" in str(exc_info.value)

    def test_config_parsing_exception_handling(self, isolated_temp_directory):
        """Test Config object creation exception handling."""
        # Given: Config with valid YAML but invalid for Config class
        config_path = isolated_temp_directory / "invalid_config_structure.yaml"

        config_content = """
# Valid YAML but missing required Config fields
some_field: value
another_field: 123
"""

        config_path.write_text(config_content)

        # When: Loading config with invalid structure
        factory = SettingsFactory()

        # Then: Should raise ValueError with Config parsing failure
        with pytest.raises(ValueError) as exc_info:
            factory._load_config(str(config_path))

        assert "Config 파싱 실패" in str(exc_info.value)


class TestSettingsFactoryJinjaTemplateHandling:
    """Test Jinja template rendering functionality."""

    def test_render_jinja_template_file_not_found(self, isolated_temp_directory):
        """Test _render_jinja_template handles missing template file."""
        # Given: Non-existent template path
        nonexistent_template = isolated_temp_directory / "nonexistent_template.sql.j2"
        context_params = {"param1": "value1"}

        # When: Attempting to render non-existent template
        factory = SettingsFactory()

        # Then: Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError) as exc_info:
            factory._render_jinja_template(str(nonexistent_template), context_params)

        assert "템플릿 파일을 찾을 수 없습니다" in str(exc_info.value)

    def test_render_jinja_template_missing_context_params(self, isolated_temp_directory):
        """Test _render_jinja_template requires context_params for .j2 files."""
        # Given: Jinja template file without context params
        template_path = isolated_temp_directory / "template.sql.j2"
        template_content = "SELECT * FROM table WHERE id = {{ user_id }}"
        template_path.write_text(template_content)

        # When: Rendering .j2 template without context_params
        factory = SettingsFactory()

        # Then: Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            factory._render_jinja_template(str(template_path), None)

        assert "context_params가 필요합니다" in str(exc_info.value)

    def test_render_jinja_template_sql_without_params(self, isolated_temp_directory):
        """Test _render_jinja_template handles .sql files without params."""
        # Given: SQL file (not .j2) without context params
        sql_path = isolated_temp_directory / "query.sql"
        sql_content = "SELECT * FROM table WHERE status = 'active'"
        sql_path.write_text(sql_content)

        # When: Rendering .sql file without context_params
        factory = SettingsFactory()
        result = factory._render_jinja_template(str(sql_path), None)

        # Then: Should return original content
        assert result == sql_content

    def test_render_jinja_template_rendering_failure(self, isolated_temp_directory):
        """Test _render_jinja_template handles rendering failures."""
        # Given: Template with undefined variables
        template_path = isolated_temp_directory / "bad_template.sql.j2"
        template_content = "SELECT * FROM table WHERE id = {{ undefined_variable }}"
        template_path.write_text(template_content)

        context_params = {"defined_variable": "value"}  # Missing undefined_variable

        # Mock the render function to raise error
        with patch(
            "src.utils.template.templating_utils.render_template_from_string"
        ) as mock_render:
            mock_render.side_effect = ValueError("Undefined variable: undefined_variable")

            # When: Rendering template with missing variables
            factory = SettingsFactory()

            # Then: Should raise ValueError with rendering failure message
            with pytest.raises(ValueError) as exc_info:
                factory._render_jinja_template(str(template_path), context_params)

            assert "템플릿 렌더링 실패" in str(exc_info.value)

    def test_process_training_data_path_no_data_path(self, settings_builder):
        """Test _process_training_data_path with no data_path."""
        # Given: Settings with no data_path initially
        settings = settings_builder.with_task("classification").build()
        settings.recipe.data.loader.source_uri = None  # Reset to None
        factory = SettingsFactory()

        # When: Processing with no data_path
        factory._process_training_data_path(settings.recipe, None, None)

        # Then: Should return without modification (no error)
        assert settings.recipe.data.loader.source_uri is None

    def test_process_inference_data_path_no_data_path(self, settings_builder):
        """Test _process_inference_data_path with no data_path."""
        # Given: Settings with no data_path initially
        settings = settings_builder.with_task("classification").build()
        settings.recipe.data.loader.source_uri = None  # Reset to None
        factory = SettingsFactory()

        # When: Processing with no data_path
        factory._process_inference_data_path(settings.recipe, None, None)

        # Then: Should return without modification (no error)
        assert settings.recipe.data.loader.source_uri is None


class TestSettingsFactoryEnvironmentVariables:
    """Test environment variable resolution functionality."""

    def test_resolve_env_variables_complex_types(self):
        """Test _resolve_env_variables handles various data types."""
        factory = SettingsFactory()

        # Test with boolean conversion
        os.environ["TEST_BOOL"] = "true"
        result = factory._resolve_env_variables("${TEST_BOOL}")
        assert result is True

        os.environ["TEST_BOOL"] = "false"
        result = factory._resolve_env_variables("${TEST_BOOL}")
        assert result is False

        # Test with empty string
        os.environ["TEST_EMPTY"] = ""
        result = factory._resolve_env_variables("${TEST_EMPTY}")
        assert result == ""

        # Test with float conversion including scientific notation
        os.environ["TEST_FLOAT"] = "1.5e-3"
        result = factory._resolve_env_variables("${TEST_FLOAT}")
        assert result == 0.0015

        # Clean up
        for key in ["TEST_BOOL", "TEST_EMPTY", "TEST_FLOAT"]:
            os.environ.pop(key, None)

    def test_resolve_env_variables_partial_replacement(self):
        """Test _resolve_env_variables handles partial string replacement."""
        factory = SettingsFactory()

        # Set environment variables
        os.environ["DB_HOST"] = "localhost"
        os.environ["DB_PORT"] = "5432"

        # Test partial replacement in string
        input_str = "postgresql://${DB_HOST}:${DB_PORT}/mydb"
        result = factory._resolve_env_variables(input_str)
        assert result == "postgresql://localhost:5432/mydb"

        # Test with missing variable (should keep original)
        input_str = "path/${MISSING_VAR}/file"
        result = factory._resolve_env_variables(input_str)
        assert result == "path/${MISSING_VAR}/file"

        # Clean up
        os.environ.pop("DB_HOST", None)
        os.environ.pop("DB_PORT", None)

    def test_resolve_env_variables_nested_structures(self):
        """Test _resolve_env_variables handles nested dict and list structures."""
        factory = SettingsFactory()

        os.environ["NESTED_VALUE"] = "resolved_value"
        os.environ["LIST_ITEM"] = "42"

        # Test nested dictionary
        input_dict = {"level1": {"level2": {"value": "${NESTED_VALUE}"}}}
        result = factory._resolve_env_variables(input_dict)
        assert result["level1"]["level2"]["value"] == "resolved_value"

        # Test list with environment variables
        input_list = ["item1", "${LIST_ITEM}", "item3"]
        result = factory._resolve_env_variables(input_list)
        assert result[1] == 42  # Converted to int

        # Clean up
        os.environ.pop("NESTED_VALUE", None)
        os.environ.pop("LIST_ITEM", None)


class TestSettingsFactoryMinimalDefaults:
    """Test minimal recipe/config creation methods."""

    def test_create_minimal_recipe_for_serving(self):
        """Test _create_minimal_recipe_for_serving creates valid minimal recipe."""
        # Given: Factory instance
        factory = SettingsFactory()

        # When: Creating minimal serving recipe
        recipe = factory._create_minimal_recipe_for_serving()

        # Then: Should create valid Recipe object
        assert isinstance(recipe, Recipe)
        assert recipe.name == "serving_recipe"
        assert recipe.task_choice == "classification"
        assert recipe.model.class_path == "sklearn.ensemble.RandomForestClassifier"
        assert recipe.model.hyperparameters.values["n_estimators"] == 100
        assert recipe.data.loader.source_uri is None
        assert recipe.data.fetcher.type == "pass_through"
        assert recipe.metadata.author == "SettingsFactory"

    def test_create_minimal_recipe_for_inference(self):
        """Test _create_minimal_recipe_for_inference creates valid minimal recipe."""
        # Given: Factory instance
        factory = SettingsFactory()

        # When: Creating minimal inference recipe
        recipe = factory._create_minimal_recipe_for_inference()

        # Then: Should create valid Recipe object (same as serving)
        assert isinstance(recipe, Recipe)
        assert recipe.name == "serving_recipe"  # Uses same method internally
        assert recipe.task_choice == "classification"


class TestSettingsFactoryRecipePathHandling:
    """Test recipe path resolution and extension handling."""

    def test_load_recipe_with_extension_addition(self, isolated_temp_directory):
        """Test _load_recipe adds .yaml extension when missing."""
        # Given: Recipe file without extension in name
        recipe_path = isolated_temp_directory / "test_recipe.yaml"

        recipe_content = """
name: extension_test
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
    train: 0.8
    validation: 0.1
    test: 0.1
evaluation:
  metrics: [accuracy]
metadata:
  author: test
  created_at: "2024-01-01T00:00:00"
  description: Extension test recipe
"""

        recipe_path.write_text(recipe_content)

        # When: Loading recipe without extension
        factory = SettingsFactory()
        recipe = factory._load_recipe(str(recipe_path.with_suffix("")))  # Remove .yaml

        # Then: Should still load the recipe
        assert isinstance(recipe, Recipe)
        assert recipe.name == "extension_test"

    def test_load_recipe_yml_extension(self, isolated_temp_directory):
        """Test _load_recipe handles .yml extension."""
        # Given: Recipe file with .yml extension
        recipe_path = isolated_temp_directory / "test_recipe.yml"

        recipe_content = """
name: yml_extension_test
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
  description: YML extension test
"""

        recipe_path.write_text(recipe_content)

        # When: Loading recipe with .yml extension
        factory = SettingsFactory()
        recipe = factory._load_recipe(str(recipe_path.with_suffix("")))  # Without extension

        # Then: Should load the .yml file
        assert isinstance(recipe, Recipe)
        assert recipe.name == "yml_extension_test"

    def test_load_recipe_from_recipes_directory(self, isolated_temp_directory):
        """Test _load_recipe searches in recipes/ directory."""
        # Given: Recipe in recipes/ subdirectory
        recipes_dir = isolated_temp_directory / "recipes"
        recipes_dir.mkdir()
        recipe_path = recipes_dir / "subdir_recipe.yaml"

        recipe_content = """
name: recipes_dir_test
task_choice: classification
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  library: sklearn
  hyperparameters:
    tuning_enabled: false
    values:
      n_estimators: 20
data:
  loader:
    source_uri: null
  data_interface:
    target_column: target
    entity_columns: [id]
  fetcher:
    type: pass_through
  split:
    train: 0.8
    validation: 0.1
    test: 0.1
evaluation:
  metrics: [accuracy]
metadata:
  author: test
  created_at: "2024-01-01T00:00:00"
  description: Recipes directory test
"""

        recipe_path.write_text(recipe_content)

        # Change to the test directory
        original_cwd = os.getcwd()
        try:
            os.chdir(str(isolated_temp_directory))

            # When: Loading recipe by name only (should find in recipes/)
            factory = SettingsFactory()
            recipe = factory._load_recipe("subdir_recipe.yaml")

            # Then: Should find and load from recipes/ directory
            assert isinstance(recipe, Recipe)
            assert recipe.name == "recipes_dir_test"
        finally:
            os.chdir(original_cwd)

    def test_load_recipe_file_not_found(self, isolated_temp_directory):
        """Test _load_recipe raises FileNotFoundError for missing files."""
        # Given: Non-existent recipe path

        # When: Loading non-existent recipe
        factory = SettingsFactory()

        # Change to isolated directory to ensure recipes/ doesn't exist either
        original_cwd = os.getcwd()
        try:
            os.chdir(str(isolated_temp_directory))

            # Then: Should raise FileNotFoundError
            with pytest.raises(FileNotFoundError) as exc_info:
                factory._load_recipe("nonexistent_recipe.yaml")

            assert "Recipe 파일을 찾을 수 없습니다" in str(exc_info.value)
        finally:
            os.chdir(original_cwd)


class TestSettingsFactoryConfigEnhancements:
    """Test config enhancement and default value injection."""

    def test_load_config_with_storage_defaults(self, isolated_temp_directory):
        """Test _load_config adds defaults for storage adapter."""
        # Given: Minimal config without storage config details
        config_path = isolated_temp_directory / "minimal_config.yaml"

        config_content = """
environment:
  name: minimal_env
data_source:
  name: storage
  adapter_type: storage
  # config missing - should be added with defaults
feature_store:
  provider: none
"""

        config_path.write_text(config_content)

        # When: Loading minimal config
        factory = SettingsFactory()
        config = factory._load_config(str(config_path))

        # Then: Should have storage defaults added
        assert config.data_source.config is not None
        assert config.data_source.config.base_path == "."  # 프로젝트 루트 기준
        assert config.data_source.config.storage_options == {}

    def test_load_config_with_output_defaults(self, isolated_temp_directory):
        """Test _load_config adds default output configuration."""
        # Given: Config without output section
        config_path = isolated_temp_directory / "no_output_config.yaml"

        config_content = """
environment:
  name: no_output_env
data_source:
  name: storage
  adapter_type: storage
  config:
    base_path: /data
feature_store:
  provider: none
# output section missing
"""

        config_path.write_text(config_content)

        # When: Loading config without output
        factory = SettingsFactory()
        config = factory._load_config(str(config_path))

        # Then: Should have default output added
        assert config.output is not None
        assert config.output.inference is not None
        assert config.output.inference.name == "default_output"
        assert config.output.inference.adapter_type == "storage"
        assert config.output.inference.config.base_path == "./artifacts"

    def test_load_recipe_metadata_defaults(self, isolated_temp_directory):
        """Test _load_recipe adds default metadata when missing."""
        # Given: Recipe without metadata section
        recipe_path = isolated_temp_directory / "no_metadata_recipe.yaml"

        recipe_content = """
name: no_metadata_recipe
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
    train: 0.8
    validation: 0.1
    test: 0.1
evaluation:
  metrics: [accuracy]
# metadata section missing
"""

        recipe_path.write_text(recipe_content)

        # When: Loading recipe without metadata
        factory = SettingsFactory()
        recipe = factory._load_recipe(str(recipe_path))

        # Then: Should have default metadata added
        assert recipe.metadata is not None
        assert recipe.metadata.author == "CLI Recipe Builder"
        assert recipe.metadata.description == "Auto-filled by SettingsFactory for minimal recipe"
        assert recipe.metadata.created_at is not None


class TestSettingsFactoryBackwardCompatibility:
    """Test backward compatibility function."""

    def test_load_settings_backward_compatibility(self, isolated_temp_directory):
        """Test load_settings() function for backward compatibility."""
        # Given: Valid config and recipe files
        config_path = isolated_temp_directory / "compat_config.yaml"
        recipe_path = isolated_temp_directory / "compat_recipe.yaml"
        data_path = isolated_temp_directory / "compat_data.csv"

        config_content = """
environment:
  name: compat_test
data_source:
  name: storage
  adapter_type: storage
  config:
    base_path: /data
feature_store:
  provider: none
"""

        recipe_content = """
name: compat_recipe
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
    train: 0.8
    validation: 0.1
    test: 0.1
evaluation:
  metrics: [accuracy]
metadata:
  author: compat_test
  created_at: "2024-01-01T00:00:00"
  description: Compatibility test
"""

        config_path.write_text(config_content)
        recipe_path.write_text(recipe_content)
        data_path.write_text("id,feature,target\n1,0.5,1\n")

        # When: Using backward compatibility function
        from src.settings.factory import load_settings

        settings = load_settings(
            recipe_path=str(recipe_path), config_path=str(config_path), data_path=str(data_path)
        )

        # Then: Should work same as SettingsFactory.for_training
        assert isinstance(settings, Settings)
        assert settings.config.environment.name == "compat_test"
        assert settings.recipe.name == "compat_recipe"
        assert settings.recipe.data.loader.source_uri == str(data_path)

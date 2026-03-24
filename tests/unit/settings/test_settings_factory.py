"""
Settings Factory Unit Tests

Tests for SettingsFactory public API:
- for_training: Complete workflow with real config/recipe files
- for_serving: MLflow integration with mock external calls only
- for_inference: Recipe restoration and data path processing
- Environment variable resolution
- Error handling (file not found, malformed YAML, validation failures)
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mmp.settings.config import Config
from mmp.settings.factory import Settings, SettingsFactory
from mmp.settings.recipe import Recipe


# ═══════════════════════════════════════════════════════════════════════════════
# Shared YAML helpers
# ═══════════════════════════════════════════════════════════════════════════════

_MINIMAL_CONFIG = """
environment:
  name: {env_name}
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

_MINIMAL_RECIPE = """
name: {recipe_name}
task_choice: {task_choice}
model:
  class_path: {model_class}
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
evaluation:
  metrics: [accuracy]
metadata:
  author: test
  created_at: "2024-01-01T00:00:00"
  description: Test recipe
"""


def _write_config(path, env_name="test_env"):
    path.write_text(_MINIMAL_CONFIG.format(env_name=env_name))


def _write_recipe(
    path,
    recipe_name="test_recipe",
    task_choice="classification",
    model_class="sklearn.ensemble.RandomForestClassifier",
):
    path.write_text(
        _MINIMAL_RECIPE.format(
            recipe_name=recipe_name, task_choice=task_choice, model_class=model_class
        )
    )


def _mock_restorer(recipe_name="restored_recipe"):
    """Context manager that patches MLflowArtifactRestorer."""
    mock_recipe = Recipe(
        name=recipe_name,
        task_choice="classification",
        model={
            "class_path": "sklearn.ensemble.RandomForestClassifier",
            "library": "sklearn",
            "hyperparameters": {"tuning_enabled": False, "values": {"n_estimators": 100}},
        },
        data={
            "loader": {"source_uri": None},
            "data_interface": {"target_column": "target", "entity_columns": ["id"]},
            "fetcher": {"type": "pass_through"},
            "split": {"train": 0.8, "validation": 0.1, "test": 0.1},
        },
        evaluation={"metrics": ["accuracy"]},
        metadata={"author": "test", "created_at": "2024-01-01T00:00:00", "description": "test"},
    )
    ctx = patch("mmp.settings.factory.MLflowArtifactRestorer")
    mock_class = ctx.__enter__()
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance
    mock_instance.restore_recipe.return_value = mock_recipe
    return ctx, mock_class, mock_instance


# ═══════════════════════════════════════════════════════════════════════════════
# for_training
# ═══════════════════════════════════════════════════════════════════════════════


class TestSettingsFactoryForTraining:
    def test_basic_workflow(self, isolated_temp_directory):
        """for_training creates valid Settings from config + recipe files."""
        config_path = isolated_temp_directory / "config.yaml"
        recipe_path = isolated_temp_directory / "recipe.yaml"
        data_path = isolated_temp_directory / "data.csv"

        _write_config(config_path, env_name="training_env")
        _write_recipe(recipe_path, recipe_name="train_recipe")
        data_path.write_text("entity_id,feature1,target\n1,0.5,1\n2,0.3,0\n")

        settings = SettingsFactory.for_training(
            recipe_path=str(recipe_path), config_path=str(config_path), data_path=str(data_path)
        )

        assert isinstance(settings, Settings)
        assert isinstance(settings.config, Config)
        assert isinstance(settings.recipe, Recipe)
        assert settings.config.environment.name == "training_env"
        assert settings.recipe.name == "train_recipe"
        assert settings.recipe.data.loader.source_uri == str(data_path)
        # Computed fields added
        assert "run_name" in settings.recipe.model.computed
        assert settings.recipe.model.computed["environment"] == "training_env"

    def test_jinja_template_rendering(self, isolated_temp_directory):
        """for_training renders Jinja templates with context_params."""
        config_path = isolated_temp_directory / "config.yaml"
        recipe_path = isolated_temp_directory / "recipe.yaml"
        template_path = isolated_temp_directory / "query.sql.j2"

        # SQL 템플릿이므로 bigquery adapter_type 사용 (data_source 호환성 검증 통과용)
        config_path.write_text(
            "environment:\n"
            "  name: test_env\n"
            "data_source:\n"
            "  name: bigquery\n"
            "  adapter_type: bigquery\n"
            "  config:\n"
            "    connection_uri: bigquery://test\n"
            "    project_id: test-project\n"
            "    dataset_id: test_dataset\n"
            "feature_store:\n"
            "  provider: none\n"
            "output:\n"
            "  inference:\n"
            "    name: test_output\n"
            "    adapter_type: storage\n"
            "    config:\n"
            "      base_path: /output\n"
        )
        _write_recipe(recipe_path)
        template_path.write_text(
            "SELECT * FROM t WHERE date >= '{{ start_date }}' AND date <= '{{ end_date }}'"
        )

        settings = SettingsFactory.for_training(
            recipe_path=str(recipe_path),
            config_path=str(config_path),
            data_path=str(template_path),
            context_params={"start_date": "2024-01-01", "end_date": "2024-12-31"},
        )

        rendered = settings.recipe.data.loader.source_uri
        assert "2024-01-01" in rendered
        assert "2024-12-31" in rendered

    def test_validation_failure_raises(self, isolated_temp_directory):
        """for_training raises ValueError on invalid recipe."""
        config_path = isolated_temp_directory / "config.yaml"
        recipe_path = isolated_temp_directory / "recipe.yaml"

        _write_config(config_path)
        recipe_path.write_text("""
name: bad_recipe
task_choice: invalid_task_type
model:
  class_path: nonexistent.Model
  library: invalid
  hyperparameters:
    tuning_enabled: false
    values: {}
data:
  loader:
    source_uri: null
  data_interface:
    target_column: null
    entity_columns: []
  fetcher:
    type: invalid
  split:
    train: 1.5
    validation: -0.2
    test: -0.5
evaluation:
  metrics: []
metadata:
  author: test
  created_at: "2024-01-01T00:00:00"
  description: bad
""")

        with pytest.raises(ValueError):
            SettingsFactory.for_training(
                recipe_path=str(recipe_path), config_path=str(config_path)
            )


# ═══════════════════════════════════════════════════════════════════════════════
# for_serving
# ═══════════════════════════════════════════════════════════════════════════════


class TestSettingsFactoryForServing:
    def test_basic_workflow(self, isolated_temp_directory):
        """for_serving creates Settings with MLflow-restored recipe."""
        config_path = isolated_temp_directory / "config.yaml"
        config_path.write_text("""
environment:
  name: production
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
    name: out
    adapter_type: storage
    config:
      base_path: /output
""")
        run_id = "serving_run_123"

        with patch("mmp.settings.factory.MLflowArtifactRestorer") as mock_cls:
            mock_restorer = MagicMock()
            mock_cls.return_value = mock_restorer
            mock_restorer.restore_recipe.return_value = Recipe(
                name="restored",
                task_choice="classification",
                model={
                    "class_path": "sklearn.ensemble.RandomForestClassifier",
                    "library": "sklearn",
                    "hyperparameters": {"tuning_enabled": False, "values": {"n_estimators": 100}},
                },
                data={
                    "loader": {"source_uri": None},
                    "data_interface": {"target_column": "target", "entity_columns": ["id"]},
                    "fetcher": {"type": "pass_through"},
                    "split": {"train": 0.8, "validation": 0.1, "test": 0.1},
                },
                evaluation={"metrics": ["accuracy"]},
                metadata={"author": "test", "created_at": "2024-01-01", "description": "test"},
            )

            settings = SettingsFactory.for_serving(config_path=str(config_path), run_id=run_id)

            assert isinstance(settings, Settings)
            assert settings.config.environment.name == "production"
            assert settings.recipe.name == "restored"
            mock_cls.assert_called_once_with(run_id)
            assert settings.recipe.model.computed["mode"] == "serving"


# ═══════════════════════════════════════════════════════════════════════════════
# for_inference
# ═══════════════════════════════════════════════════════════════════════════════


class TestSettingsFactoryForInference:
    def test_basic_workflow(self, isolated_temp_directory):
        """for_inference creates Settings with MLflow restoration + data path injection."""
        config_path = isolated_temp_directory / "config.yaml"
        data_path = isolated_temp_directory / "data.csv"

        _write_config(config_path, env_name="inference_env")
        data_path.write_text("id,feature1\n1,0.5\n2,0.7\n")

        with patch("mmp.settings.factory.MLflowArtifactRestorer") as mock_cls:
            mock_restorer = MagicMock()
            mock_cls.return_value = mock_restorer
            mock_restorer.restore_recipe.return_value = Recipe(
                name="inference_recipe",
                task_choice="classification",
                model={
                    "class_path": "sklearn.ensemble.RandomForestClassifier",
                    "library": "sklearn",
                    "hyperparameters": {"tuning_enabled": False, "values": {"n_estimators": 100}},
                },
                data={
                    "loader": {"source_uri": None},
                    "data_interface": {"target_column": "target", "entity_columns": ["id"]},
                    "fetcher": {"type": "pass_through"},
                    "split": {"train": 1.0, "validation": 0.0, "test": 0.0},
                },
                evaluation={"metrics": []},
                metadata={"author": "test", "description": "test"},
            )

            settings = SettingsFactory.for_inference(
                config_path=str(config_path), run_id="run_789", data_path=str(data_path)
            )

            assert isinstance(settings, Settings)
            assert settings.recipe.data.loader.source_uri == str(data_path)
            assert settings.recipe.model.computed["mode"] == "inference"


# ═══════════════════════════════════════════════════════════════════════════════
# Environment variable resolution
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnvironmentVariableResolution:
    def test_config_resolves_env_vars(self, isolated_temp_directory):
        """_load_config resolves ${VAR:default} syntax."""
        config_path = isolated_temp_directory / "config.yaml"
        config_path.write_text("""
environment:
  name: ${ENV_NAME:default_env}
data_source:
  name: storage
  adapter_type: storage
  config:
    base_path: ${DATA_PATH:/default/data}
feature_store:
  provider: none
output:
  inference:
    name: out
    adapter_type: storage
    config:
      base_path: /output
""")

        os.environ["ENV_NAME"] = "production_env"
        try:
            factory = SettingsFactory()
            config = factory._load_config(str(config_path))
            assert config.environment.name == "production_env"
            assert config.data_source.config.base_path == "/default/data"
        finally:
            os.environ.pop("ENV_NAME", None)

    def test_nested_and_type_conversion(self):
        """resolve_env_variables handles nested dicts, lists, and type conversion."""
        from mmp.settings.env_resolver import resolve_env_variables

        os.environ["TEST_NESTED"] = "resolved"
        os.environ["TEST_INT"] = "42"
        try:
            result = resolve_env_variables(
                {"level1": {"value": "${TEST_NESTED}"}, "items": ["a", "${TEST_INT}"]}
            )
            assert result["level1"]["value"] == "resolved"
            assert result["items"][1] == 42
        finally:
            os.environ.pop("TEST_NESTED", None)
            os.environ.pop("TEST_INT", None)


# ═══════════════════════════════════════════════════════════════════════════════
# Error handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestSettingsFactoryErrors:
    def test_config_file_not_found(self, isolated_temp_directory):
        """FileNotFoundError when config doesn't exist."""
        factory = SettingsFactory()
        original_cwd = os.getcwd()
        try:
            os.chdir(str(isolated_temp_directory))
            with pytest.raises(FileNotFoundError):
                factory._load_config(str(isolated_temp_directory / "nonexistent.yaml"))
        finally:
            os.chdir(original_cwd)

    def test_malformed_yaml(self, isolated_temp_directory):
        """ValueError on malformed YAML."""
        config_path = isolated_temp_directory / "bad.yaml"
        config_path.write_text("environment:\n  name: test\n  [invalid yaml")
        factory = SettingsFactory()
        with pytest.raises(ValueError, match="Config 파싱 실패"):
            factory._load_config(str(config_path))

    def test_empty_config(self, isolated_temp_directory):
        """ValueError on empty config file."""
        config_path = isolated_temp_directory / "empty.yaml"
        config_path.write_text("")
        factory = SettingsFactory()
        with pytest.raises(ValueError, match="Config 파일이 비어있습니다"):
            factory._load_config(str(config_path))

    def test_empty_recipe(self, isolated_temp_directory):
        """ValueError on empty recipe file."""
        recipe_path = isolated_temp_directory / "empty.yaml"
        recipe_path.write_text("")
        factory = SettingsFactory()
        with pytest.raises(ValueError, match="Recipe 파일이 비어있습니다"):
            factory._load_recipe(str(recipe_path))

    def test_recipe_missing_required_fields(self, isolated_temp_directory):
        """ValueError when recipe has invalid structure."""
        recipe_path = isolated_temp_directory / "bad_recipe.yaml"
        recipe_path.write_text("name: bad\nsome_invalid_field: value\n")
        factory = SettingsFactory()
        with pytest.raises(ValueError, match="Recipe 파싱 실패"):
            factory._load_recipe(str(recipe_path))

    def test_jinja_template_file_not_found(self, isolated_temp_directory):
        """FileNotFoundError when template doesn't exist."""
        factory = SettingsFactory()
        with pytest.raises(FileNotFoundError, match="템플릿 파일을 찾을 수 없습니다"):
            factory._render_jinja_template(
                str(isolated_temp_directory / "missing.sql.j2"), {"k": "v"}
            )

    def test_jinja_template_missing_params(self, isolated_temp_directory):
        """ValueError when .j2 template has no context_params."""
        template_path = isolated_temp_directory / "t.sql.j2"
        template_path.write_text("SELECT {{ col }}")
        factory = SettingsFactory()
        with pytest.raises(ValueError, match="context_params가 필요합니다"):
            factory._render_jinja_template(str(template_path), None)


# ═══════════════════════════════════════════════════════════════════════════════
# Backward compatibility
# ═══════════════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    def test_load_settings_function(self, isolated_temp_directory):
        """load_settings() delegates to SettingsFactory.for_training."""
        config_path = isolated_temp_directory / "config.yaml"
        recipe_path = isolated_temp_directory / "recipe.yaml"
        data_path = isolated_temp_directory / "data.csv"

        _write_config(config_path, env_name="compat_test")
        _write_recipe(recipe_path, recipe_name="compat_recipe")
        data_path.write_text("entity_id,feature,target\n1,0.5,1\n")

        from mmp.settings.factory import load_settings

        settings = load_settings(
            recipe_path=str(recipe_path), config_path=str(config_path), data_path=str(data_path)
        )

        assert isinstance(settings, Settings)
        assert settings.config.environment.name == "compat_test"

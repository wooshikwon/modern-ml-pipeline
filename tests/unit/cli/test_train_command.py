"""
Unit Tests for Train Command CLI - No Mock Hell Approach
Following tests/README.md principles: Real components, Context classes, Public API
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import typer
import yaml
from typer.testing import CliRunner

from src.cli.commands.train_command import train_command
from src.settings import SettingsFactory


class TestTrainCommandArgumentParsing:
    """Train command argument parsing tests using real components"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(train_command)

    def test_train_command_with_required_arguments(
        self, component_test_context, isolated_temp_directory, add_model_computed
    ):
        """Test train command with all required arguments using real settings"""
        # Given: Real test data in temp directory
        test_data = pd.DataFrame(
            {
                "feature1": np.random.rand(50),
                "feature2": np.random.rand(50),
                "target": np.random.randint(0, 2, 50),
            }
        )
        data_path = isolated_temp_directory / "train_data.csv"
        test_data.to_csv(data_path, index=False)

        # Given: Real recipe and config files
        recipe_path = isolated_temp_directory / "recipe.yaml"
        recipe_content = {
            "name": "test_recipe",
            "task_choice": "classification",
            "model": {
                "class_path": "sklearn.ensemble.RandomForestClassifier",
                "library": "sklearn",
                "hyperparameters": {
                    "tuning_enabled": False,
                    "values": {"n_estimators": 10, "random_state": 42},
                },
            },
            "data": {
                "loader": {"source_uri": str(data_path)},
                "data_interface": {
                    "target_column": "target",
                    "feature_columns": ["feature1", "feature2"],
                    "entity_columns": [],
                },
                "fetcher": {"type": "pass_through"},
                "split": {"train": 0.6, "validation": 0.2, "test": 0.2},
            },
            "evaluation": {"metrics": ["accuracy", "roc_auc"], "random_state": 42},
            "metadata": {
                "created_at": "2024-01-01 00:00:00",
                "description": "Test recipe for CLI unit test",
            },
        }
        with open(recipe_path, "w") as f:
            yaml.dump(recipe_content, f)

        config_path = isolated_temp_directory / "config.yaml"
        config_content = {
            "environment": {"name": "test"},
            "data_source": {
                "name": "test_storage",
                "adapter_type": "storage",
                "config": {"base_path": str(isolated_temp_directory), "storage_options": {}},
            },
            "feature_store": {"provider": "none"},
            "output": {
                "inference": {
                    "name": "test_output",
                    "enabled": True,
                    "adapter_type": "storage",
                    "config": {"base_path": str(isolated_temp_directory / "output")},
                }
            },
            "mlflow": {
                "tracking_uri": f"file://{isolated_temp_directory}/mlruns",
                "experiment_name": f"test_exp_{isolated_temp_directory.name[:8]}",
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        # When: Execute command with real files
        # Mock only the pipeline execution to avoid full training in unit test
        with patch("src.cli.commands.train_command.run_train_pipeline") as mock_pipeline:
            mock_result = MagicMock()
            mock_result.run_id = "test_run_id"
            mock_result.model_uri = "models:/test_model/1"
            mock_pipeline.return_value = mock_result

            # Patch SettingsFactory to add computed fields
            original_for_training = SettingsFactory.for_training

            def for_training_with_computed(*args, **kwargs):
                settings = original_for_training(*args, **kwargs)
                return add_model_computed(settings)

            with patch.object(SettingsFactory, "for_training", for_training_with_computed):
                result = self.runner.invoke(
                    self.app,
                    [
                        "--recipe",
                        str(recipe_path),
                        "--config",
                        str(config_path),
                        "--data",
                        str(data_path),
                    ],
                )

        # Then: Command executes successfully
        if result.exit_code != 0:
            print(f"Command failed with output:\n{result.output}")
            print(f"Exception: {result.exception}")
        assert result.exit_code == 0

        # Then: Pipeline was called with real settings
        mock_pipeline.assert_called_once()
        call_args = mock_pipeline.call_args
        settings = call_args.kwargs["settings"]

        # Verify settings were created correctly
        assert settings is not None
        assert settings.recipe.model.class_path == "sklearn.ensemble.RandomForestClassifier"
        assert settings.recipe.data.data_interface.target_column == "target"
        assert "feature1" in settings.recipe.data.data_interface.feature_columns

    def test_train_command_with_optional_params(
        self, component_test_context, isolated_temp_directory
    ):
        """Test train command with optional context parameters"""
        # Given: Real test data
        test_data = pd.DataFrame(
            {
                "feature1": np.random.rand(50),
                "feature2": np.random.rand(50),
                "target": np.random.rand(50),  # Regression target
            }
        )
        data_path = isolated_temp_directory / "train_regression.csv"
        test_data.to_csv(data_path, index=False)

        # Given: Recipe with Jinja template placeholders
        recipe_path = isolated_temp_directory / "recipe_template.yaml"
        recipe_content = {
            "name": "test_regression_recipe",
            "task_choice": "regression",
            "model": {
                "class_path": "sklearn.linear_model.LinearRegression",
                "library": "sklearn",
                "hyperparameters": {
                    "tuning_enabled": False,
                    "values": {
                        "fit_intercept": True  # Provide actual hyperparameter for LinearRegression
                    },
                },
            },
            "data": {
                "loader": {"source_uri": str(data_path)},
                "data_interface": {
                    "target_column": "target",
                    "feature_columns": ["feature1", "feature2"],
                    "entity_columns": [],
                },
                "fetcher": {"type": "pass_through"},
                "split": {"train": 0.6, "validation": 0.2, "test": 0.2},
                "context": {"date": "{{ date }}", "version": "{{ version }}"},
            },
            "evaluation": {"metrics": ["rmse", "r2"], "random_state": 42},
            "metadata": {
                "created_at": "2024-01-01 00:00:00",
                "description": "Test recipe for regression",
            },
        }
        with open(recipe_path, "w") as f:
            yaml.dump(recipe_content, f)

        config_path = isolated_temp_directory / "config.yaml"
        config_content = {
            "environment": {"name": "test"},
            "data_source": {
                "name": "test_storage",
                "adapter_type": "storage",
                "config": {"base_path": str(isolated_temp_directory), "storage_options": {}},
            },
            "feature_store": {"provider": "none"},
            "output": {
                "inference": {
                    "name": "test_output",
                    "enabled": True,
                    "adapter_type": "storage",
                    "config": {"base_path": str(isolated_temp_directory / "output")},
                }
            },
            "mlflow": {
                "tracking_uri": f"file://{isolated_temp_directory}/mlruns",
                "experiment_name": f"test_regression_{isolated_temp_directory.name[:8]}",
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        # When: Execute with JSON params
        with patch("src.cli.commands.train_command.run_train_pipeline") as mock_pipeline:
            mock_result = MagicMock()
            mock_result.run_id = "param_run_id"
            mock_result.model_uri = "models:/param_model/1"
            mock_pipeline.return_value = mock_result

            result = self.runner.invoke(
                self.app,
                [
                    "--recipe",
                    str(recipe_path),
                    "--config",
                    str(config_path),
                    "--data",
                    str(data_path),
                    "--params",
                    '{"date": "2024-01-01", "version": 2}',
                ],
            )

        # Then: Command executes successfully with params
        assert result.exit_code == 0

        # Then: Settings were created with context params
        mock_pipeline.assert_called_once()
        call_args = mock_pipeline.call_args
        assert call_args.kwargs["context_params"] == {"date": "2024-01-01", "version": 2}

    def test_train_command_with_missing_file(self, isolated_temp_directory, caplog):
        """Test train command error handling for missing files"""
        # Given: Non-existent file paths
        recipe_path = isolated_temp_directory / "nonexistent_recipe.yaml"
        config_path = isolated_temp_directory / "nonexistent_config.yaml"
        data_path = isolated_temp_directory / "nonexistent_data.csv"

        # When: Execute command with missing files
        import logging

        with caplog.at_level(logging.ERROR):
            result = self.runner.invoke(
                self.app,
                [
                    "--recipe",
                    str(recipe_path),
                    "--config",
                    str(config_path),
                    "--data",
                    str(data_path),
                ],
            )

        # Then: Command fails with appropriate error
        assert result.exit_code != 0
        # Check for error message in log output
        assert "File not found" in caplog.text or "[ERROR]" in caplog.text

    def test_train_command_with_invalid_json_params(
        self, component_test_context, isolated_temp_directory, caplog
    ):
        """Test train command error handling for invalid JSON params"""
        # Given: Valid files
        test_data = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})
        data_path = isolated_temp_directory / "data.csv"
        test_data.to_csv(data_path, index=False)

        recipe_path = isolated_temp_directory / "recipe.yaml"
        recipe_content = {
            "name": "test_recipe",
            "task_choice": "classification",
            "model": {
                "class_path": "sklearn.ensemble.RandomForestClassifier",
                "library": "sklearn",
                "hyperparameters": {"tuning_enabled": False, "values": {"random_state": 42}},
            },
            "data": {
                "loader": {"source_uri": str(data_path)},
                "data_interface": {
                    "target_column": "target",
                    "feature_columns": ["feature1"],
                    "entity_columns": [],
                },
                "fetcher": {"type": "pass_through"},
                "split": {"train": 0.8, "validation": 0.1, "test": 0.1},
            },
            "evaluation": {"metrics": ["accuracy"], "random_state": 42},
            "metadata": {
                "created_at": "2024-01-01 00:00:00",
                "description": "Test recipe for classification",
            },
        }
        with open(recipe_path, "w") as f:
            yaml.dump(recipe_content, f)

        config_path = isolated_temp_directory / "config.yaml"
        config_content = {
            "environment": {"name": "test"},
            "data_source": {
                "name": "test_storage",
                "adapter_type": "storage",
                "config": {"base_path": str(isolated_temp_directory), "storage_options": {}},
            },
            "feature_store": {"provider": "none"},
            "output": {
                "inference": {
                    "name": "test_output",
                    "enabled": True,
                    "adapter_type": "storage",
                    "config": {"base_path": str(isolated_temp_directory / "output")},
                }
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        # When: Execute with invalid JSON
        import logging

        with caplog.at_level(logging.ERROR):
            result = self.runner.invoke(
                self.app,
                [
                    "--recipe",
                    str(recipe_path),
                    "--config",
                    str(config_path),
                    "--data",
                    str(data_path),
                    "--params",
                    "invalid-json-{not-valid}",
                ],
            )

        # Then: Command fails with JSON error
        assert result.exit_code != 0
        # Check for JSON parsing error indicators in log output
        assert "expecting value" in caplog.text.lower() or "Configuration error" in caplog.text

    def test_train_command_with_record_reqs_flag(
        self, component_test_context, isolated_temp_directory
    ):
        """Test train command with requirements recording flag"""
        # Given: Minimal valid setup
        test_data = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})
        data_path = isolated_temp_directory / "data.csv"
        test_data.to_csv(data_path, index=False)

        recipe_path = isolated_temp_directory / "recipe.yaml"
        recipe_content = {
            "name": "test_recipe",
            "task_choice": "classification",
            "model": {
                "class_path": "sklearn.tree.DecisionTreeClassifier",
                "library": "sklearn",
                "hyperparameters": {"tuning_enabled": False, "values": {"random_state": 42}},
            },
            "data": {
                "loader": {"source_uri": str(data_path)},
                "data_interface": {
                    "target_column": "target",
                    "feature_columns": ["feature1"],
                    "entity_columns": [],
                },
                "fetcher": {"type": "pass_through"},
                "split": {"train": 0.8, "validation": 0.1, "test": 0.1},
            },
            "evaluation": {"metrics": ["accuracy"], "random_state": 42},
            "metadata": {
                "created_at": "2024-01-01 00:00:00",
                "description": "Test recipe for classification",
            },
        }
        with open(recipe_path, "w") as f:
            yaml.dump(recipe_content, f)

        config_path = isolated_temp_directory / "config.yaml"
        config_content = {
            "environment": {"name": "test"},
            "data_source": {
                "name": "test_storage",
                "adapter_type": "storage",
                "config": {"base_path": str(isolated_temp_directory), "storage_options": {}},
            },
            "feature_store": {"provider": "none"},
            "output": {
                "inference": {
                    "name": "test_output",
                    "enabled": True,
                    "adapter_type": "storage",
                    "config": {"base_path": str(isolated_temp_directory / "output")},
                }
            },
            "mlflow": {
                "tracking_uri": f"file://{isolated_temp_directory}/mlruns",
                "experiment_name": f"test_reqs_{isolated_temp_directory.name[:8]}",
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        # When: Execute with --record-reqs flag
        with patch("src.cli.commands.train_command.run_train_pipeline") as mock_pipeline:
            mock_result = MagicMock()
            mock_result.run_id = "reqs_run_id"
            mock_result.model_uri = "models:/reqs_model/1"
            mock_pipeline.return_value = mock_result

            result = self.runner.invoke(
                self.app,
                [
                    "--recipe",
                    str(recipe_path),
                    "--config",
                    str(config_path),
                    "--data",
                    str(data_path),
                    "--record-reqs",
                ],
            )

        # Then: Pipeline called with record_requirements=True
        assert result.exit_code == 0
        mock_pipeline.assert_called_once()
        call_args = mock_pipeline.call_args
        assert call_args.kwargs["record_requirements"] == True


class TestTrainCommandIntegration:
    """Integration tests with real pipeline execution"""

    def test_train_command_end_to_end_with_real_pipeline(
        self, mlflow_test_context, isolated_temp_directory
    ):
        """Test full train command execution with real components"""
        # Given: Complete setup with MLflow context
        with mlflow_test_context.for_classification(experiment="cli_train_test") as ctx:
            # Create real training data
            test_data = pd.DataFrame(
                {
                    "feature1": np.random.rand(100),
                    "feature2": np.random.rand(100),
                    "feature3": np.random.rand(100),
                    "target": np.random.randint(0, 2, 100),
                }
            )
            data_path = isolated_temp_directory / "real_train.csv"
            test_data.to_csv(data_path, index=False)

            # Create recipe file
            recipe_path = isolated_temp_directory / "real_recipe.yaml"
            recipe_content = {
                "name": "e2e_test_recipe",
                "task_choice": "classification",
                "model": {
                    "class_path": "sklearn.tree.DecisionTreeClassifier",
                    "library": "sklearn",
                    "hyperparameters": {
                        "tuning_enabled": False,
                        "values": {"max_depth": 3, "random_state": 42},
                    },
                },
                "data": {
                    "loader": {"source_uri": str(data_path)},
                    "data_interface": {
                        "target_column": "target",
                        "feature_columns": ["feature1", "feature2", "feature3"],
                        "entity_columns": [],
                    },
                    "fetcher": {"type": "pass_through"},
                    "split": {"train": 0.7, "validation": 0.15, "test": 0.15},
                },
                "evaluation": {"metrics": ["accuracy", "f1_score"], "random_state": 42},
                "metadata": {"created_at": "2024-01-01 00:00:00", "description": "E2E test recipe"},
            }
            with open(recipe_path, "w") as f:
                yaml.dump(recipe_content, f)

            # Create config file with MLflow settings
            config_path = isolated_temp_directory / "real_config.yaml"
            config_content = {
                "environment": {"name": "test"},
                "data_source": {
                    "name": "test_storage",
                    "adapter_type": "storage",
                    "config": {"base_path": str(isolated_temp_directory), "storage_options": {}},
                },
                "feature_store": {"provider": "none"},
                "output": {
                    "inference": {
                        "name": "test_output",
                        "enabled": True,
                        "adapter_type": "storage",
                        "config": {"base_path": str(isolated_temp_directory / "output")},
                    }
                },
                "mlflow": {"tracking_uri": ctx.mlflow_uri, "experiment_name": ctx.experiment_name},
            }
            with open(config_path, "w") as f:
                yaml.dump(config_content, f)

            # When: Run actual train command
            runner = CliRunner()
            app = typer.Typer()
            app.command()(train_command)

            result = runner.invoke(
                app,
                [
                    "--recipe",
                    str(recipe_path),
                    "--config",
                    str(config_path),
                    "--data",
                    str(data_path),
                ],
            )

            # Then: Training completes successfully
            assert result.exit_code == 0

            # Then: MLflow experiment exists with runs
            assert ctx.experiment_exists()
            assert ctx.get_experiment_run_count() > 0

            # Then: Metrics were logged
            metrics = ctx.get_run_metrics()
            assert metrics is not None
            assert len(metrics) > 0

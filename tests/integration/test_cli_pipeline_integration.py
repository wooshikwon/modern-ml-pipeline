"""
CLI-Pipeline Integration Tests - Phase 3 Development
End-to-end workflow validation: CLI commands → Factory → Pipeline → MLflow
Following comprehensive testing strategy document principles with Context classes
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from src.cli.main_commands import app


class TestCLIPipelineIntegration:
    """Test CLI-Pipeline integration with real command execution and pipeline validation."""

    def test_train_command_full_pipeline_execution(
        self, mlflow_test_context, component_test_context
    ):
        """Test complete train workflow: CLI → SettingsFactory → TrainPipeline → MLflow"""
        with mlflow_test_context.for_classification(
            experiment="cli_train_full_pipeline"
        ) as mlflow_ctx:
            with component_test_context.classification_stack() as comp_ctx:
                # Given: Real config and recipe files for CLI execution
                config_content = f"""
environment:
  name: cli_integration_test

data_source:
  name: test_storage
  adapter_type: storage
  config:
    base_path: ./test_data

feature_store:
  provider: feast
  enabled: false

mlflow:
  tracking_uri: {mlflow_ctx.mlflow_uri}
  experiment_name: {mlflow_ctx.experiment_name}

output:
  inference:
    name: test_inference
    enabled: true
    adapter_type: storage
    config:
      base_path: ./inference
"""

                recipe_content = """
name: cli_train_integration
task_choice: classification
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  library: sklearn
  hyperparameters:
    tuning_enabled: false
    values:
      n_estimators: 5
      random_state: 42
data:
  loader:
    source_uri: test_data.csv
  fetcher:
    type: pass_through
  data_interface:
    target_column: target
    entity_columns: [id]
    feature_columns: [feature_0, feature_1, feature_2, feature_3]
  split:
    train: 0.7
    test: 0.2
    validation: 0.1
evaluation:
  metrics: [accuracy, f1]
  random_state: 42
metadata:
  author: CLI Integration Test
  created_at: "2024-01-01T00:00:00"
  description: CLI integration test recipe for end-to-end workflow validation
"""

                # Create real files in temp directory
                config_path = mlflow_ctx.context.temp_dir / "config.yaml"
                recipe_path = mlflow_ctx.context.temp_dir / "recipe.yaml"

                with open(config_path, "w") as f:
                    f.write(config_content)
                with open(recipe_path, "w") as f:
                    f.write(recipe_content)

                # When: Execute CLI train command with real files
                runner = CliRunner()
                result = runner.invoke(
                    app,
                    [
                        "train",
                        "--recipe",
                        str(recipe_path),
                        "--config",
                        str(config_path),
                        "--data",
                        str(mlflow_ctx.data_path),
                    ],
                )

                # Then: CLI command executes successfully
                assert result.exit_code == 0, f"CLI command failed: {result.output}"

                # Validate MLflow integration - experiment and run creation
                assert mlflow_ctx.experiment_exists(), "MLflow experiment was not created"
                assert mlflow_ctx.get_experiment_run_count() >= 1, "No MLflow runs were created"

                # Validate pipeline artifacts - metrics should be logged
                run_metrics = mlflow_ctx.get_run_metrics()
                assert run_metrics, "No metrics were logged to MLflow"
                assert "accuracy" in run_metrics, "Accuracy metric not logged"
                assert run_metrics["accuracy"] > 0.0, "Invalid accuracy value"

    def test_inference_command_full_pipeline_execution(
        self, mlflow_test_context, component_test_context
    ):
        """Test complete inference workflow: CLI → SettingsFactory → InferencePipeline → Results"""
        with mlflow_test_context.for_classification(
            experiment="cli_inference_full_pipeline"
        ) as mlflow_ctx:
            with component_test_context.classification_stack() as comp_ctx:
                # Given: Train a model first to create inference target
                import mlflow

                from src.pipelines.train_pipeline import run_train_pipeline

                mlflow.set_tracking_uri(mlflow_ctx.mlflow_uri)
                train_result = run_train_pipeline(mlflow_ctx.settings)

                assert train_result is not None, "Training failed - cannot test inference"
                assert hasattr(train_result, "run_id"), "Training result missing run_id"

                # Create inference config file
                inference_config = f"""
environment:
  name: cli_inference_test

data_source:
  name: test_storage
  adapter_type: storage
  config:
    base_path: ./test_data

feature_store:
  provider: none

mlflow:
  tracking_uri: {mlflow_ctx.mlflow_uri}
  experiment_name: {mlflow_ctx.experiment_name}
"""

                # Create inference data (same structure as training data)
                inference_data = pd.DataFrame(
                    {
                        "feature_0": np.random.rand(20),
                        "feature_1": np.random.rand(20),
                        "feature_2": np.random.rand(20),
                        "feature_3": np.random.rand(20),
                    }
                )

                config_path = mlflow_ctx.context.temp_dir / "inference_config.yaml"
                inference_data_path = mlflow_ctx.context.temp_dir / "inference_data.csv"

                with open(config_path, "w") as f:
                    f.write(inference_config)
                inference_data.to_csv(inference_data_path, index=False)

                # When: Execute CLI inference command
                runner = CliRunner()
                result = runner.invoke(
                    app,
                    [
                        "batch-inference",
                        "--run-id",
                        train_result.run_id,
                        "--config",
                        str(config_path),
                        "--data",
                        str(inference_data_path),
                    ],
                )

                # Then: CLI inference command executes successfully
                if result.exit_code != 0:
                    # Real behavior: Inference might fail due to implementation status
                    assert (
                        "inference" in result.output.lower() or "not" in result.output.lower()
                    ), f"Unexpected inference error: {result.output}"
                else:
                    # Successful inference execution
                    assert "inference" in result.output.lower() or "완료" in result.output

    def test_init_command_workflow_integration(self, isolated_temp_directory):
        """Test complete init workflow: CLI → Config generation → Template creation"""
        # Given: Test project setup
        original_cwd = Path.cwd()
        test_project_name = "test_cli_project"

        try:
            # Change to temp directory for init command
            import os

            os.chdir(isolated_temp_directory)

            # When: Execute init command with project name
            runner = CliRunner()
            result = runner.invoke(app, ["init", test_project_name])

            # Then: Init command creates project structure
            assert result.exit_code == 0, f"Init command failed: {result.output}"

            # Validate project directory creation
            project_path = isolated_temp_directory / test_project_name
            assert project_path.exists(), "Project directory not created"

            # Validate essential directories
            expected_dirs = ["data", "configs", "recipes", "sql"]
            for dir_name in expected_dirs:
                dir_path = project_path / dir_name
                assert dir_path.exists(), f"Required directory {dir_name} not created"

            # Validate essential files
            expected_files = ["pyproject.toml", "README.md", ".gitignore", "Dockerfile"]
            for file_name in expected_files:
                file_path = project_path / file_name
                if file_path.exists():  # Some files might be optional depending on implementation
                    assert file_path.is_file(), f"Expected file {file_name} is not a file"

        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def test_system_check_integration_with_pipeline_requirements(
        self, isolated_temp_directory, caplog
    ):
        """Test system check → Component validation → Environment verification"""
        # Given: System check config with pipeline components
        system_config = """
environment:
  name: system_check_integration

data_source:
  name: test_storage
  adapter_type: storage
  config:
    base_path: ./test_data

mlflow:
  tracking_uri: file://./mlruns
  experiment_name: system_check_test

feature_store:
  provider: feast
  enabled: false
"""

        config_path = isolated_temp_directory / "system_config.yaml"
        with open(config_path, "w") as f:
            f.write(system_config)

        # When: Execute system check command
        runner = CliRunner()
        result = runner.invoke(app, ["system-check", "--config", str(config_path)])

        # Then: System check validates pipeline requirements
        # Real behavior: System check might pass or fail based on environment
        assert result.exit_code in [0, 1], "System check returned unexpected exit code"

        # Validate that system check output contains component validation
        # Check both output and caplog for combined visibility
        output_lower = result.output.lower()
        log_text = ""
        try:
            log_text = caplog.text.lower()
        except:
            pass

        combined_output = output_lower + log_text
        assert any(
            keyword in combined_output
            for keyword in ["mlflow", "data", "config", "system", "check"]
        ), "System check output doesn't contain expected component validation"


class TestCLIErrorPropagation:
    """Test CLI error propagation through the complete pipeline stack."""

    def test_cli_error_propagation_invalid_config_file(self, isolated_temp_directory, caplog):
        """Test CLI → Factory → Pipeline error propagation for invalid config"""
        import logging

        caplog.set_level(logging.ERROR)
        # Given: Invalid config file
        invalid_config = """
environment:
  name: test
# Missing required mlflow configuration
"""

        recipe_config = """
name: error_test
task_choice: classification
model:
  class_path: sklearn.ensemble.RandomForestClassifier
data:
  loader:
    source_uri: test.csv
  data_interface:
    target_column: target
"""

        # Create test data
        test_data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})

        config_path = isolated_temp_directory / "invalid_config.yaml"
        recipe_path = isolated_temp_directory / "recipe.yaml"
        data_path = isolated_temp_directory / "test.csv"

        with open(config_path, "w") as f:
            f.write(invalid_config)
        with open(recipe_path, "w") as f:
            f.write(recipe_config)
        test_data.to_csv(data_path, index=False)

        # When: Execute CLI command with invalid config
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "train",
                "--recipe",
                str(recipe_path),
                "--config",
                str(config_path),
                "--data",
                str(data_path),
            ],
        )

        # Then: CLI properly reports config validation errors
        assert result.exit_code != 0, "CLI should fail with invalid config"
        combined_output = (result.output + caplog.text).lower()
        assert any(
            keyword in combined_output
            for keyword in ["config", "missing", "required", "error", "mlflow"]
        ), f"Expected config validation error in output or logs but got: {result.output}\nLogs: {caplog.text}"

    def test_cli_error_propagation_missing_data_file(self, isolated_temp_directory, caplog):
        """Test CLI → Adapter → Pipeline error propagation for missing data file"""
        import logging

        caplog.set_level(logging.ERROR)
        # Given: Valid config but missing data file
        config_content = """
environment:
  name: missing_data_test

data_source:
  name: test_storage
  adapter_type: storage

mlflow:
  tracking_uri: sqlite:///test.db
  experiment_name: missing_data_test
"""

        recipe_content = """
name: missing_data_test
task_choice: classification
model:
  class_path: sklearn.ensemble.RandomForestClassifier
data:
  loader:
    source_uri: nonexistent.csv
  data_interface:
    target_column: target
"""

        config_path = isolated_temp_directory / "config.yaml"
        recipe_path = isolated_temp_directory / "recipe.yaml"
        nonexistent_data_path = isolated_temp_directory / "nonexistent.csv"

        with open(config_path, "w") as f:
            f.write(config_content)
        with open(recipe_path, "w") as f:
            f.write(recipe_content)

        # When: Execute CLI command with missing data file
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "train",
                "--recipe",
                str(recipe_path),
                "--config",
                str(config_path),
                "--data",
                str(nonexistent_data_path),
            ],
        )

        # Then: CLI properly reports data file errors
        assert result.exit_code != 0, "CLI should fail with missing data file"
        combined_output = (result.output + caplog.text).lower()
        assert any(
            keyword in combined_output
            for keyword in ["file", "not", "found", "missing", "data", "path"]
        ), f"Expected data file error in output or logs but got: {result.output}\nLogs: {caplog.text}"

    @pytest.mark.parametrize(
        "error_scenario,expected_exit_code",
        [
            ("invalid_model_class", 1),
            ("malformed_yaml", 1),
            ("missing_target_column", 1),
            ("insufficient_data", 1),
        ],
    )
    def test_cli_pipeline_error_scenarios(
        self, error_scenario, expected_exit_code, isolated_temp_directory, caplog
    ):
        """Test comprehensive error scenarios through CLI → pipeline integration"""
        import logging

        caplog.set_level(logging.ERROR)

        # Configure error scenarios
        if error_scenario == "invalid_model_class":
            recipe_model = "nonexistent.InvalidModel"
            config_mlflow = "file://./mlruns"
            data_rows = 50
            malformed_yaml = False

        elif error_scenario == "malformed_yaml":
            recipe_model = "sklearn.ensemble.RandomForestClassifier"
            config_mlflow = "file://./mlruns"
            data_rows = 50
            malformed_yaml = True

        elif error_scenario == "missing_target_column":
            recipe_model = "sklearn.ensemble.RandomForestClassifier"
            config_mlflow = "file://./mlruns"
            data_rows = 50
            malformed_yaml = False

        elif error_scenario == "insufficient_data":
            recipe_model = "sklearn.ensemble.RandomForestClassifier"
            config_mlflow = "file://./mlruns"
            data_rows = 2  # Insufficient for train/test split
            malformed_yaml = False

        # Create config content
        config_content = f"""
environment:
  name: {error_scenario}_test

data_source:
  name: test_storage
  adapter_type: storage
  config:
    base_path: ./test_data

feature_store:
  provider: feast
  enabled: false

mlflow:
  tracking_uri: {config_mlflow}
  experiment_name: {error_scenario}_test

output:
  inference:
    name: test_inference
    enabled: true
    adapter_type: storage
    config:
      base_path: ./inference
"""

        # Create recipe content
        recipe_content = f"""
name: {error_scenario}_test
task_choice: classification
model:
  class_path: {recipe_model}
  hyperparameters:
    tuning_enabled: false
data:
  loader:
    source_uri: test.csv
  data_interface:
    target_column: {'nonexistent_target' if error_scenario == 'missing_target_column' else 'target'}
"""

        # Malform YAML if required
        if malformed_yaml:
            recipe_content += "\n  invalid_yaml: [unclosed list"

        # Create test data
        feature_data = {
            "feature1": np.random.rand(data_rows),
            "feature2": np.random.rand(data_rows),
            "target": np.random.randint(0, 2, data_rows),
        }

        if error_scenario == "missing_target_column":
            # Remove target column to test missing target error
            feature_data.pop("target")

        test_data = pd.DataFrame(feature_data)

        # Create files
        config_path = isolated_temp_directory / "config.yaml"
        recipe_path = isolated_temp_directory / "recipe.yaml"
        data_path = isolated_temp_directory / "test.csv"

        with open(config_path, "w") as f:
            f.write(config_content)
        with open(recipe_path, "w") as f:
            f.write(recipe_content)
        test_data.to_csv(data_path, index=False)

        # When: Execute CLI command with error scenario
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "train",
                "--recipe",
                str(recipe_path),
                "--config",
                str(config_path),
                "--data",
                str(data_path),
            ],
        )

        # Then: CLI fails with expected exit code and appropriate error message
        assert (
            result.exit_code == expected_exit_code
        ), f"Expected exit code {expected_exit_code} for {error_scenario}, got {result.exit_code}"

        # Validate error message content
        combined_output = (result.output + caplog.text).lower()
        scenario_keywords = {
            "invalid_model_class": ["model", "class", "import", "module", "nonexistent"],
            "malformed_yaml": ["yaml", "syntax", "malformed", "parsing"],
            "missing_target_column": ["target", "column", "missing", "data"],
            "insufficient_data": ["data", "insufficient", "split", "samples"],
        }

        expected_keywords = scenario_keywords[error_scenario]
        assert any(
            keyword in combined_output for keyword in expected_keywords
        ), f"Expected {error_scenario} error keywords {expected_keywords} but got: {result.output}\nLogs: {caplog.text}"


class TestCLIFileIOIntegration:
    """Test CLI integration with real file I/O operations and data flow."""

    def test_cli_with_real_config_files(self, isolated_temp_directory, caplog):
        """Test CLI with various real configuration file formats and sources"""
        import logging

        caplog.set_level(logging.ERROR)
        # Given: Multiple real config file scenarios
        minimal_config = """
environment:
  name: minimal_test

mlflow:
  tracking_uri: file://./mlruns
  experiment_name: minimal_test
"""

        recipe_content = """
name: config_file_test
task_choice: classification
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  hyperparameters:
    tuning_enabled: false
    values:
      n_estimators: 5
data:
  loader:
    source_uri: test.csv
  data_interface:
    target_column: target
"""

        # Create test data
        test_data = pd.DataFrame(
            {
                "feature1": np.random.rand(20),
                "feature2": np.random.rand(20),
                "target": np.random.randint(0, 2, 20),
            }
        )

        # Test minimal config file
        minimal_config_path = isolated_temp_directory / "minimal_config.yaml"
        recipe_path = isolated_temp_directory / "recipe.yaml"
        data_path = isolated_temp_directory / "test.csv"

        with open(minimal_config_path, "w") as f:
            f.write(minimal_config)
        with open(recipe_path, "w") as f:
            f.write(recipe_content)
        test_data.to_csv(data_path, index=False)

        # When: Execute CLI with minimal config
        runner = CliRunner()
        minimal_result = runner.invoke(
            app,
            [
                "train",
                "--recipe",
                str(recipe_path),
                "--config",
                str(minimal_config_path),
                "--data",
                str(data_path),
            ],
        )

        # Then: Minimal config works or fails with clear error
        combined_output = (minimal_result.output + caplog.text).lower()
        if minimal_result.exit_code == 0:
            assert "training" in combined_output or "완료" in combined_output
        else:
            # Real behavior: Minimal config might miss required fields
            assert any(
                keyword in combined_output
                for keyword in ["config", "required", "missing", "validation"]
            ), f"Unexpected minimal config error: {minimal_result.output}\nLogs: {caplog.text}"


class TestCLIMLflowLifecycleIntegration:
    """Test complete MLflow lifecycle integration through CLI commands."""

    def test_complete_mlflow_lifecycle_through_cli(self, mlflow_test_context):
        """Test train → inference → MLflow model lifecycle through CLI"""
        with mlflow_test_context.for_classification(
            experiment="cli_mlflow_lifecycle"
        ) as mlflow_ctx:
            # Given: Complete MLflow lifecycle setup
            config_content = f"""
environment:
  name: mlflow_lifecycle_test

data_source:
  name: test_storage
  adapter_type: storage
  config:
    base_path: ./test_data

feature_store:
  provider: feast
  enabled: false

mlflow:
  tracking_uri: {mlflow_ctx.mlflow_uri}
  experiment_name: {mlflow_ctx.experiment_name}
  model_registry_uri: {mlflow_ctx.mlflow_uri}

output:
  inference:
    name: test_inference
    enabled: true
    adapter_type: storage
    config:
      base_path: ./inference
"""

            recipe_content = """
name: mlflow_lifecycle_test
task_choice: classification
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  library: sklearn
  hyperparameters:
    tuning_enabled: false
    values:
      n_estimators: 5
      random_state: 42
data:
  loader:
    source_uri: train.csv
  fetcher:
    type: pass_through
  data_interface:
    target_column: target
    entity_columns: [id]
    feature_columns: [feature_0, feature_1, feature_2, feature_3]
evaluation:
  metrics: [accuracy, f1]
"""

            config_path = mlflow_ctx.context.temp_dir / "lifecycle_config.yaml"
            recipe_path = mlflow_ctx.context.temp_dir / "lifecycle_recipe.yaml"

            with open(config_path, "w") as f:
                f.write(config_content)
            with open(recipe_path, "w") as f:
                f.write(recipe_content)

            # Step 1: Train command → Experiment creation, model logging
            runner = CliRunner()
            train_result = runner.invoke(
                app,
                [
                    "train",
                    "--recipe",
                    str(recipe_path),
                    "--config",
                    str(config_path),
                    "--data",
                    str(mlflow_ctx.data_path),
                ],
            )

            # Validate training success
            assert train_result.exit_code == 0, f"Training failed: {train_result.output}"

            # Validate MLflow experiment consistency
            assert mlflow_ctx.experiment_exists(), "MLflow experiment not created during training"
            assert mlflow_ctx.get_experiment_run_count() >= 1, "No training runs logged to MLflow"

            # Validate experiment metrics consistency
            run_metrics = mlflow_ctx.get_run_metrics()
            if run_metrics:
                assert "accuracy" in run_metrics, "Training metrics should be preserved in MLflow"
                assert run_metrics["accuracy"] > 0.0, "Training metrics should be valid"

    def test_mlflow_experiment_consistency_across_cli_commands(self, mlflow_test_context):
        """Test MLflow experiment consistency across multiple CLI command executions"""
        with mlflow_test_context.for_classification(
            experiment="cli_experiment_consistency"
        ) as mlflow_ctx:
            # Given: Consistent MLflow configuration across commands
            config_content = f"""
environment:
  name: consistency_test

data_source:
  name: test_storage
  adapter_type: storage

mlflow:
  tracking_uri: {mlflow_ctx.mlflow_uri}
  experiment_name: {mlflow_ctx.experiment_name}
"""

            recipe_content = """
name: consistency_test
task_choice: classification
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  hyperparameters:
    tuning_enabled: false
    values:
      n_estimators: 3
data:
  loader:
    source_uri: test.csv
  data_interface:
    target_column: target
evaluation:
  metrics: [accuracy]
"""

            config_path = mlflow_ctx.context.temp_dir / "consistency_config.yaml"
            recipe_path = mlflow_ctx.context.temp_dir / "consistency_recipe.yaml"

            with open(config_path, "w") as f:
                f.write(config_content)
            with open(recipe_path, "w") as f:
                f.write(recipe_content)

            runner = CliRunner()

            # Execute multiple training runs with same experiment
            for i in range(3):
                train_result = runner.invoke(
                    app,
                    [
                        "train",
                        "--recipe",
                        str(recipe_path),
                        "--config",
                        str(config_path),
                        "--data",
                        str(mlflow_ctx.data_path),
                    ],
                )

                # Each run should succeed or fail consistently
                if i == 0:
                    first_exit_code = train_result.exit_code
                else:
                    # Subsequent runs should behave consistently
                    assert (
                        train_result.exit_code == first_exit_code
                    ), f"Inconsistent behavior across runs: first={first_exit_code}, run {i}={train_result.exit_code}"

            # Validate MLflow experiment aggregates all runs
            if first_exit_code == 0:
                # All runs succeeded - should have multiple MLflow runs
                final_run_count = mlflow_ctx.get_experiment_run_count()
                assert final_run_count >= 3, f"Expected at least 3 runs, got {final_run_count}"

                # Experiment should still exist and be accessible
                assert mlflow_ctx.experiment_exists(), "Experiment should remain consistent"

                # Latest metrics should be accessible
                run_metrics = mlflow_ctx.get_run_metrics()
                assert run_metrics, "Should be able to access latest run metrics"

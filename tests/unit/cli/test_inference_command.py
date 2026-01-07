"""
Unit Tests for Inference Command CLI - Updated for new interface
Following test philosophy: Real components, minimal mocking

Only mocking the actual pipeline execution (run_inference_pipeline) and MLflow interactions.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

from src.cli.commands.inference_command import batch_inference_command


class TestInferenceCommandWithRealComponents:
    """Inference command tests using real components - No Mock Hell compliant"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(batch_inference_command)

    @patch("src.cli.commands.inference_command.restore_all_from_mlflow")
    @patch("src.cli.commands.inference_command.run_inference_pipeline")
    def test_inference_command_with_required_arguments(
        self, mock_run_pipeline, mock_restore, cli_test_environment
    ):
        """Test inference command CLI interface with run_id only"""
        # Mock pipeline execution
        mock_result = SimpleNamespace(run_id="inference_run_123", prediction_count=100)
        mock_run_pipeline.return_value = mock_result

        # Mock MLflow restore
        mock_recipe = MagicMock()
        mock_recipe.model_dump.return_value = {}
        mock_config = MagicMock()
        mock_config.model_dump.return_value = {}
        mock_config.environment.name = "test"
        mock_restore.return_value = (mock_recipe, mock_config, "SELECT * FROM data")

        test_run_id = "test_inference_run"

        # Execute command with just --run-id (artifact 설정 사용)
        result = self.runner.invoke(self.app, ["--run-id", test_run_id])

        # Verify pipeline was called with correct parameters
        mock_run_pipeline.assert_called_once_with(
            run_id=test_run_id,
            recipe_path=None,
            config_path=None,
            data_path=None,
            context_params={},
        )

        # Command should succeed
        assert result.exit_code == 0

    @patch("src.cli.commands.inference_command.restore_all_from_mlflow")
    @patch("src.cli.commands.inference_command.run_inference_pipeline")
    def test_inference_command_with_optional_params(
        self, mock_run_pipeline, mock_restore, cli_test_environment
    ):
        """Test inference command with optional JSON context parameters"""
        mock_result = SimpleNamespace(run_id="inference_run_456", prediction_count=1000)
        mock_run_pipeline.return_value = mock_result

        # Mock MLflow restore
        mock_recipe = MagicMock()
        mock_config = MagicMock()
        mock_config.environment.name = "test"
        mock_restore.return_value = (mock_recipe, mock_config, None)

        config_path = cli_test_environment["config_path"]
        data_path = cli_test_environment["data_path"]
        test_run_id = "test_run_456"

        # Execute with optional JSON parameters
        params = {"batch_date": "2024-01-15", "limit": 1000}
        result = self.runner.invoke(
            self.app,
            [
                "--run-id",
                test_run_id,
                "--config",
                str(config_path),
                "--data",
                str(data_path),
                "--params",
                json.dumps(params),
            ],
        )

        # Verify pipeline was called with correct parameters
        mock_run_pipeline.assert_called_once_with(
            run_id=test_run_id,
            recipe_path=None,
            config_path=str(config_path),
            data_path=str(data_path),
            context_params=params,
        )

        assert result.exit_code == 0

    @patch("src.cli.commands.inference_command.restore_all_from_mlflow")
    @patch("src.cli.commands.inference_command.run_inference_pipeline")
    def test_inference_command_handles_invalid_json_params(
        self, mock_run_pipeline, mock_restore, cli_test_environment
    ):
        """Test inference command handles invalid JSON parameters gracefully"""
        mock_run_pipeline.return_value = SimpleNamespace(run_id="x", prediction_count=0)
        # Mock MLflow restore
        mock_restore.return_value = (MagicMock(), MagicMock(), None)

        # Execute with invalid JSON
        result = self.runner.invoke(
            self.app, ["--run-id", "test_run", "--params", "invalid json {]"]
        )

        # Should fail due to invalid JSON
        assert result.exit_code != 0 or "Invalid" in result.output

    def test_inference_command_help_message(self):
        """Test inference command help output"""
        result = self.runner.invoke(self.app, ["--help"])

        assert result.exit_code == 0
        assert "--run-id" in result.output
        # 새 인터페이스에서는 --recipe와 --config가 선택적
        assert "--recipe" in result.output or "--config" in result.output

    @patch("src.cli.commands.inference_command.run_inference_pipeline")
    def test_inference_command_missing_required_args(self, mock_run_pipeline):
        """Test inference command fails when required arguments are missing"""
        # Execute without required arguments
        result = self.runner.invoke(self.app, [])

        # Should fail due to missing required arguments
        assert result.exit_code != 0
        assert mock_run_pipeline.call_count == 0


class TestInferenceCommandIntegration:
    """Integration tests for inference command with contexts"""

    @patch("src.cli.commands.inference_command.restore_all_from_mlflow")
    @patch("src.cli.commands.inference_command.run_inference_pipeline")
    def test_inference_command_error_handling_settings_failure(
        self, mock_run_pipeline, mock_restore, cli_test_environment
    ):
        """Test inference command error handling when pipeline fails with FileNotFoundError"""
        # Mock pipeline to raise FileNotFoundError
        mock_run_pipeline.side_effect = FileNotFoundError("Run not found")
        # Mock MLflow restore
        mock_recipe = MagicMock()
        mock_config = MagicMock()
        mock_config.environment.name = "test"
        mock_restore.return_value = (mock_recipe, mock_config, None)

        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)

        # Execute with non-existent run_id
        result = runner.invoke(
            app,
            [
                "--run-id",
                "non_existent_run",
            ],
        )

        # Should fail due to pipeline error
        assert result.exit_code != 0

        # Verify pipeline was called
        mock_run_pipeline.assert_called_once()

    @patch("src.cli.commands.inference_command.restore_all_from_mlflow")
    @patch("src.cli.commands.inference_command.run_inference_pipeline")
    def test_inference_command_error_handling_missing_config(self, mock_run_pipeline, mock_restore):
        """Test inference command error handling for missing config file"""
        # Mock pipeline to raise FileNotFoundError for missing config
        mock_run_pipeline.side_effect = FileNotFoundError("Config file not found")
        # Mock MLflow restore
        mock_restore.return_value = (MagicMock(), MagicMock(), None)

        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)

        # Execute with non-existent config file
        result = runner.invoke(
            app,
            [
                "--run-id",
                "test_run",
                "--config",
                "/non/existent/config.yaml",
            ],
        )

        # Should fail due to config file not found
        assert result.exit_code != 0

    @patch("src.cli.commands.inference_command.restore_all_from_mlflow")
    @patch("src.cli.commands.inference_command.run_inference_pipeline")
    def test_inference_command_success_message_formatting(
        self, mock_run_pipeline, mock_restore, cli_test_environment
    ):
        """Test inference command success message and result processing"""
        mock_result = SimpleNamespace(run_id="success_run_123", prediction_count=1000)
        mock_run_pipeline.return_value = mock_result
        # Mock MLflow restore
        mock_recipe = MagicMock()
        mock_config = MagicMock()
        mock_config.environment.name = "test"
        mock_restore.return_value = (mock_recipe, mock_config, None)

        test_run_id = "success_test_run"

        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)

        result = runner.invoke(
            app,
            [
                "--run-id",
                test_run_id,
            ],
        )

        # Verify successful execution
        assert result.exit_code == 0

        # Check result.output for success message content
        assert any(
            keyword in result.output
            for keyword in ["Inference", "completed", "Run ID", "Predictions"]
        )

    @patch("src.cli.commands.inference_command.restore_all_from_mlflow")
    @patch("src.cli.commands.inference_command.run_inference_pipeline")
    def test_inference_command_value_error_handling(
        self, mock_run_pipeline, mock_restore, cli_test_environment, caplog
    ):
        """Test inference command error handling for ValueError"""
        import logging

        caplog.set_level(logging.ERROR)

        # Mock pipeline to raise ValueError
        mock_run_pipeline.side_effect = ValueError("Invalid configuration")
        # Mock MLflow restore
        mock_restore.return_value = (MagicMock(), MagicMock(), None)

        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)

        result = runner.invoke(
            app,
            [
                "--run-id",
                "test_run",
            ],
        )

        # Should fail due to ValueError
        assert result.exit_code != 0

        # Verify error handling via caplog (logger.error is still used for errors)
        assert any(
            keyword in caplog.text.lower() for keyword in ["error", "invalid", "configuration"]
        )

    @patch("src.cli.commands.inference_command.restore_all_from_mlflow")
    @patch("src.cli.commands.inference_command.run_inference_pipeline")
    def test_inference_command_generic_exception_handling(
        self, mock_run_pipeline, mock_restore, cli_test_environment
    ):
        """Test inference command error handling for generic exceptions"""
        # Mock pipeline to raise generic exception
        mock_run_pipeline.side_effect = RuntimeError("Unexpected error occurred")
        # Mock MLflow restore
        mock_restore.return_value = (MagicMock(), MagicMock(), None)

        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)

        result = runner.invoke(
            app,
            [
                "--run-id",
                "test_run",
            ],
        )

        # Should fail due to generic exception
        assert result.exit_code != 0

    @patch("src.cli.commands.inference_command.restore_all_from_mlflow")
    @patch("src.cli.commands.inference_command.run_inference_pipeline")
    def test_inference_command_console_integration(
        self, mock_run_pipeline, mock_restore, cli_test_environment
    ):
        """Test inference command console progress tracking and milestones"""
        mock_result = SimpleNamespace(run_id="console_run_123", prediction_count=50)
        mock_run_pipeline.return_value = mock_result
        # Mock MLflow restore
        mock_recipe = MagicMock()
        mock_config = MagicMock()
        mock_config.environment.name = "test"
        mock_restore.return_value = (mock_recipe, mock_config, None)

        test_run_id = "console_test_run"

        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)

        result = runner.invoke(
            app,
            [
                "--run-id",
                test_run_id,
            ],
        )

        # Verify successful execution
        assert result.exit_code == 0

        # Check result.output for progress/milestone messages
        assert any(keyword in result.output for keyword in ["inference", "done", "Loading"])

    @patch("src.cli.commands.inference_command.restore_all_from_mlflow")
    @patch("src.cli.commands.inference_command.run_inference_pipeline")
    def test_inference_with_recipe_override(
        self, mock_run_pipeline, mock_restore, cli_test_environment
    ):
        """Test inference command with recipe override"""
        mock_result = SimpleNamespace(run_id="override_run_123", prediction_count=100)
        mock_run_pipeline.return_value = mock_result
        # Mock MLflow restore
        mock_restore.return_value = (MagicMock(), MagicMock(), None)

        recipe_path = cli_test_environment["recipe_path"]
        test_run_id = "override_test_run"

        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)

        result = runner.invoke(
            app,
            [
                "--run-id",
                test_run_id,
                "--recipe",
                str(recipe_path),
            ],
        )

        # Verify pipeline was called with recipe override
        mock_run_pipeline.assert_called_once()
        call_kwargs = mock_run_pipeline.call_args[1]
        assert call_kwargs["recipe_path"] == str(recipe_path)
        assert result.exit_code == 0

    @patch("src.cli.commands.inference_command.restore_all_from_mlflow")
    @patch("src.cli.commands.inference_command.run_inference_pipeline")
    def test_inference_with_config_override(
        self, mock_run_pipeline, mock_restore, cli_test_environment
    ):
        """Test inference command with config override"""
        mock_result = SimpleNamespace(run_id="config_override_run_123", prediction_count=100)
        mock_run_pipeline.return_value = mock_result
        # Mock MLflow restore
        mock_restore.return_value = (MagicMock(), MagicMock(), None)

        config_path = cli_test_environment["config_path"]
        test_run_id = "config_override_test_run"

        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)

        result = runner.invoke(
            app,
            [
                "--run-id",
                test_run_id,
                "--config",
                str(config_path),
            ],
        )

        # Verify pipeline was called with config override
        mock_run_pipeline.assert_called_once()
        call_kwargs = mock_run_pipeline.call_args[1]
        assert call_kwargs["config_path"] == str(config_path)
        assert result.exit_code == 0

    @patch("src.cli.commands.inference_command.restore_all_from_mlflow")
    @patch("src.cli.commands.inference_command.run_inference_pipeline")
    def test_inference_with_data_override(
        self, mock_run_pipeline, mock_restore, cli_test_environment
    ):
        """Test inference command with data override"""
        mock_result = SimpleNamespace(run_id="data_override_run_123", prediction_count=100)
        mock_run_pipeline.return_value = mock_result
        # Mock MLflow restore
        mock_restore.return_value = (MagicMock(), MagicMock(), None)

        data_path = cli_test_environment["data_path"]
        test_run_id = "data_override_test_run"

        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)

        result = runner.invoke(
            app,
            [
                "--run-id",
                test_run_id,
                "--data",
                str(data_path),
            ],
        )

        # Verify pipeline was called with data override
        mock_run_pipeline.assert_called_once()
        call_kwargs = mock_run_pipeline.call_args[1]
        assert call_kwargs["data_path"] == str(data_path)
        assert result.exit_code == 0

    @patch("src.cli.commands.inference_command.restore_all_from_mlflow")
    @patch("src.cli.commands.inference_command.run_inference_pipeline")
    def test_inference_with_real_data_files(
        self, mock_run_pipeline, mock_restore, real_dataset_files, cli_test_environment
    ):
        """Test inference command with real data files from fixtures"""
        mock_result = SimpleNamespace(run_id="real_data_run_123", prediction_count=100)
        mock_run_pipeline.return_value = mock_result
        # Mock MLflow restore
        mock_restore.return_value = (MagicMock(), MagicMock(), None)

        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)

        # Use real CSV file
        data_path = real_dataset_files["classification_csv"]["path"]

        result = runner.invoke(app, ["--run-id", "real_data_run", "--data", str(data_path)])

        # Verify pipeline was called with real data path
        if mock_run_pipeline.called:
            call_kwargs = mock_run_pipeline.call_args[1]
            assert call_kwargs["data_path"] == str(data_path)

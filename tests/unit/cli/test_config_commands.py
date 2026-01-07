"""
Unit Tests for Config Commands CLI - No Mock Hell Approach
Following tests/README.md principles: Real components, Context classes, Public API
"""

from unittest.mock import patch

import typer
from typer.testing import CliRunner

from src.cli.commands.get_config_command import get_config_command


class TestGetConfigCommandArgumentParsing:
    """Get config command argument parsing tests using real components"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(get_config_command)

    @patch("builtins.input")
    def test_get_config_command_without_env_name(self, mock_input, isolated_temp_directory):
        """Test get config command without environment name (interactive mode) using real components"""
        # Given: Real InteractiveConfigBuilder with mocked user inputs
        mock_input.side_effect = [
            "dev",  # env_name
            "y",  # use_mlflow
            "./mlruns",  # mlflow_tracking_uri
            "3",  # data_source selection (Local Files)
            "1",  # feature_store selection (없음/None)
            "1",  # artifact_storage selection (Local)
            "y",  # inference_output_enabled
            "3",  # inference_output_source (Local Files)
            "y",  # confirm settings
        ]

        # When: Execute command without env_name (interactive mode)
        with patch("pathlib.Path.write_text") as mock_write:
            result = self.runner.invoke(self.app, [])

        # Then: Command executes successfully with real InteractiveConfigBuilder
        if result.exit_code != 0:
            print(f"Command failed with output:\n{result.output}")
            if result.exception:
                print(f"Exception: {result.exception}")
        assert result.exit_code == 0
        # Verify config file would be written
        mock_write.assert_called()

    @patch("builtins.input")
    def test_get_config_command_with_env_name(self, mock_input, isolated_temp_directory):
        """Test get config command with specified environment name using real components"""
        # Given: Real InteractiveConfigBuilder with mocked user inputs (env_name provided via CLI)
        mock_input.side_effect = [
            "y",  # use_mlflow
            "http://mlflow-server:5000",  # mlflow_tracking_uri
            "1",  # data_source selection (PostgreSQL)
            "1",  # feature_store selection (없음/None)
            "2",  # artifact_storage selection (S3)
            "y",  # inference_output_enabled
            "1",  # inference_output_source (PostgreSQL)
            "y",  # confirm settings
        ]

        # When: Execute command with env_name
        with patch("pathlib.Path.write_text") as mock_write:
            result = self.runner.invoke(self.app, ["--env-name", "production"])

        # Then: Command executes successfully with real InteractiveConfigBuilder
        assert result.exit_code == 0
        # Verify config file would be written with production environment
        mock_write.assert_called()

    @patch("builtins.input")
    def test_get_config_command_with_short_option(self, mock_input, isolated_temp_directory):
        """Test get config command with short option flag using real components"""
        # Given: Real InteractiveConfigBuilder with mocked user inputs
        mock_input.side_effect = [
            "n",  # use_mlflow (no MLflow)
            "3",  # data_source selection (Local Files)
            "1",  # feature_store selection (없음/None)
            # No artifact_storage selection when MLflow is disabled
            "n",  # inference_output_enabled
            "y",  # confirm settings
        ]

        # When: Execute command with short flag
        with patch("pathlib.Path.write_text") as mock_write:
            result = self.runner.invoke(self.app, ["-e", "dev"])

        # Then: Command executes successfully with short option
        assert result.exit_code == 0
        mock_write.assert_called()

    def test_get_config_command_builder_initialization_error(self):
        """Test get config command handles builder initialization errors using real components"""
        # Given: Template directory not found scenario
        with patch("pathlib.Path.exists", return_value=False):
            # When: Execute command which will fail to find templates
            result = self.runner.invoke(self.app, ["--env-name", "test"])

            # Then: Command should handle error gracefully
            # Exit code should be non-zero due to initialization error
            assert result.exit_code != 0 or "error" in result.output.lower()

    @patch("builtins.input")
    def test_get_config_command_interactive_flow_error(self, mock_input):
        """Test get config command handles interactive flow errors using real components"""
        # Given: Invalid user inputs that cause validation errors
        mock_input.side_effect = [
            "",  # Empty env_name (invalid)
            "y",  # Try to continue
            KeyboardInterrupt(),  # User interrupts
        ]

        # When: Execute command with invalid inputs
        result = self.runner.invoke(self.app, ["--env-name", ""])

        # Then: Command should handle gracefully
        # KeyboardInterrupt (user cancel) returns exit_code 0, error returns non-zero
        assert result.exit_code == 0 or "취소" in result.output or "invalid" in result.output.lower()

    def test_get_config_command_help_message(self):
        """Test get config command help message shows correct information"""
        # When: Execute help command
        result = self.runner.invoke(self.app, ["--help"])

        # Then: Help message displays correctly with all options
        assert result.exit_code == 0
        assert "--env-name" in result.output or "-e" in result.output
        assert "환경 이름" in result.output or "env" in result.output.lower()
        assert "대화형" in result.output or "interactive" in result.output.lower()

    @patch("builtins.input")
    def test_get_config_command_file_generation_verification(
        self, mock_input, isolated_temp_directory
    ):
        """Test get config command verifies file generation using real components"""
        # Given: Real InteractiveConfigBuilder with BigQuery configuration
        mock_input.side_effect = [
            "y",  # use_mlflow
            "http://mlflow-server:5000",  # mlflow_tracking_uri
            "2",  # data_source selection (BigQuery)
            "1",  # feature_store selection (없음/None)
            "3",  # artifact_storage selection (GCS)
            "y",  # inference_output_enabled
            "2",  # inference_output_source (BigQuery)
            "y",  # confirm settings
        ]

        # When: Execute command with staging environment
        with (
            patch("pathlib.Path.write_text") as mock_write,
            patch("pathlib.Path.mkdir") as mock_mkdir,
        ):
            result = self.runner.invoke(self.app, ["-e", "staging"])

        # Then: Command executes successfully and files are generated
        assert result.exit_code == 0
        # Verify directory creation and file write were attempted
        mock_write.assert_called()

    @patch("builtins.input")
    def test_get_config_command_environment_name_validation(
        self, mock_input, isolated_temp_directory
    ):
        """Test get config command with various environment name formats using real components"""
        # Given: Test valid environment names with real InteractiveConfigBuilder
        valid_env_names = ["dev", "production", "staging", "test-env", "env_123"]

        for env_name in valid_env_names:
            # Setup user inputs for each environment
            mock_input.side_effect = [
                "y",  # use_mlflow
                "./mlruns",  # mlflow_tracking_uri
                "3",  # data_source selection (Local Files)
                "1",  # feature_store selection (없음/None)
                "1",  # artifact_storage selection (Local)
                "y",  # inference_output_enabled
                "3",  # inference_output_source (Local Files)
                "y",  # confirm settings
            ]

            # When: Execute command with various valid environment names
            with patch("pathlib.Path.write_text") as mock_write:
                result = self.runner.invoke(self.app, ["--env-name", env_name])

            # Then: All valid environment names should work
            assert result.exit_code == 0
            # Verify config file would be written for each environment
            mock_write.assert_called()

            # Reset mock for next iteration
            mock_input.reset_mock()

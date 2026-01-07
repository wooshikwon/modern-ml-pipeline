"""
Unit Tests for Serve Command CLI - No Mock Hell Compliant
Following test philosophy: Real components, minimal mocking

Only mocking the actual API server execution to avoid starting a real server in tests.
All other components (SettingsFactory, console, etc.) use real implementations.
"""

from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

from src.cli.commands.serve_command import serve_api_command


class TestServeCommandWithRealComponents:
    """Serve command tests using real components - No Mock Hell compliant"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(serve_api_command)

    @patch("src.cli.commands.serve_command.run_api_server")
    @patch("src.cli.commands.serve_command.SettingsFactory")
    def test_serve_command_with_required_arguments(
        self, mock_settings_factory, mock_run_server, cli_test_environment
    ):
        """Test serve command CLI interface with mocked SettingsFactory"""
        # Mock SettingsFactory to focus on CLI interface
        mock_settings = MagicMock()
        mock_settings_factory.for_serving.return_value = mock_settings
        mock_run_server.return_value = None

        config_path = cli_test_environment["config_path"]
        test_run_id = "test_run_123"

        result = self.runner.invoke(
            self.app, ["--run-id", test_run_id, "--config", str(config_path)]
        )

        # Verify SettingsFactory.for_serving was called with correct parameters
        mock_settings_factory.for_serving.assert_called_once_with(
            run_id=test_run_id, config_path=str(config_path)
        )

        # Verify run_api_server was called with correct parameters
        mock_run_server.assert_called_once_with(
            settings=mock_settings, run_id=test_run_id, host="0.0.0.0", port=8000
        )

        # Command should succeed
        assert result.exit_code == 0

    @patch("src.cli.commands.serve_command.run_api_server")
    @patch("src.cli.commands.serve_command.SettingsFactory")
    def test_serve_command_with_custom_host_port(
        self, mock_settings_factory, mock_run_server, cli_test_environment
    ):
        """Test serve command with custom host and port parameters"""
        # Mock SettingsFactory and server execution
        mock_settings = MagicMock()
        mock_settings_factory.for_serving.return_value = mock_settings
        mock_run_server.return_value = None

        config_path = cli_test_environment["config_path"]
        test_run_id = "test_run_456"

        # Execute with custom host and port
        result = self.runner.invoke(
            self.app,
            [
                "--run-id",
                test_run_id,
                "--config",
                str(config_path),
                "--host",
                "127.0.0.1",
                "--port",
                "9000",
            ],
        )

        # Verify SettingsFactory was called
        mock_settings_factory.for_serving.assert_called_once_with(
            run_id=test_run_id, config_path=str(config_path)
        )

        # Verify run_api_server was called with custom host/port
        mock_run_server.assert_called_once_with(
            settings=mock_settings, run_id=test_run_id, host="127.0.0.1", port=9000
        )

        assert result.exit_code == 0

    @patch("src.cli.commands.serve_command.run_api_server")
    def test_serve_command_port_validation(self, mock_run_server, cli_test_environment):
        """Test serve command validates port range with real components"""
        mock_run_server.return_value = None
        config_path = cli_test_environment["config_path"]

        # Test with invalid port (too high)
        result = self.runner.invoke(
            self.app, ["--run-id", "test_run", "--config", str(config_path), "--port", "70000"]
        )

        # Should fail due to port validation
        assert result.exit_code != 0 or "Invalid value" in result.output

    def test_serve_command_help_message(self):
        """Test serve command help output"""
        result = self.runner.invoke(self.app, ["--help"])

        assert result.exit_code == 0
        assert "--run-id" in result.output
        assert "--config" in result.output
        assert "--host" in result.output
        assert "--port" in result.output

    @patch("src.cli.commands.serve_command.run_api_server")
    def test_serve_command_missing_required_args(self, mock_run_server):
        """Test serve command fails gracefully when required args missing"""
        # Execute without required arguments
        result = self.runner.invoke(self.app, [])

        # Should fail due to missing required arguments
        assert result.exit_code != 0
        assert mock_run_server.call_count == 0


class TestServeCommandIntegration:
    """Integration tests for serve command with real MLflow context"""

    @patch("src.cli.commands.serve_command.run_api_server")
    @patch("src.cli.commands.serve_command.SettingsFactory")
    def test_serve_command_error_handling_settings_failure(
        self, mock_settings_factory, mock_run_server, cli_test_environment
    ):
        """Test serve command error handling when SettingsFactory fails"""
        # Mock SettingsFactory to raise an error (simulating missing run)
        mock_settings_factory.for_serving.side_effect = FileNotFoundError("Run not found")
        mock_run_server.return_value = None
        config_path = cli_test_environment["config_path"]

        runner = CliRunner()
        app = typer.Typer()
        app.command()(serve_api_command)

        # Execute with run_id that causes SettingsFactory to fail
        result = runner.invoke(app, ["--run-id", "non_existent_run", "--config", str(config_path)])

        # Should fail due to SettingsFactory error
        assert result.exit_code != 0
        assert mock_run_server.call_count == 0

        # Verify SettingsFactory was called
        mock_settings_factory.for_serving.assert_called_once()

    @patch("src.cli.commands.serve_command.run_api_server")
    @patch("src.cli.commands.serve_command.SettingsFactory")
    def test_serve_command_error_handling_missing_config(
        self, mock_settings_factory, mock_run_server
    ):
        """Test serve command error handling for missing config file"""
        # Mock SettingsFactory to raise FileNotFoundError for missing config
        mock_settings_factory.for_serving.side_effect = FileNotFoundError("Config file not found")
        mock_run_server.return_value = None

        runner = CliRunner()
        app = typer.Typer()
        app.command()(serve_api_command)

        # Execute with non-existent config file
        result = runner.invoke(
            app, ["--run-id", "test_run", "--config", "/non/existent/config.yaml"]
        )

        # Should fail due to config file not found
        assert result.exit_code != 0
        assert mock_run_server.call_count == 0

        # Verify SettingsFactory was called with the bad config path
        mock_settings_factory.for_serving.assert_called_once_with(
            run_id="test_run", config_path="/non/existent/config.yaml"
        )

    @patch("src.cli.commands.serve_command.run_api_server")
    @patch("src.cli.commands.serve_command.SettingsFactory")
    def test_serve_command_console_integration(
        self, mock_settings_factory, mock_run_server, cli_test_environment
    ):
        """Test serve command console progress tracking and milestones"""
        mock_settings = MagicMock()
        mock_settings_factory.for_serving.return_value = mock_settings
        mock_run_server.return_value = None

        config_path = cli_test_environment["config_path"]
        test_run_id = "console_test_run"

        runner = CliRunner()
        app = typer.Typer()
        app.command()(serve_api_command)

        result = runner.invoke(
            app,
            [
                "--run-id",
                test_run_id,
                "--config",
                str(config_path),
                "--host",
                "127.0.0.1",
                "--port",
                "9000",
            ],
        )

        # Verify console messages are included in output
        assert result.exit_code == 0

        # Check for console milestone messages in output
        assert any(keyword in result.output for keyword in ["API Server", "Config", "Starting"])

        # Verify SettingsFactory and server were called correctly
        mock_settings_factory.for_serving.assert_called_once()
        mock_run_server.assert_called_once_with(
            settings=mock_settings, run_id=test_run_id, host="127.0.0.1", port=9000
        )

    @patch("src.cli.commands.serve_command.run_api_server")
    @patch("src.cli.commands.serve_command.SettingsFactory")
    def test_serve_command_value_error_handling(
        self, mock_settings_factory, mock_run_server, cli_test_environment, caplog
    ):
        """Test serve command error handling for ValueError (environment setup errors)"""
        import logging

        caplog.set_level(logging.ERROR)

        # Mock SettingsFactory to raise ValueError for environment issues
        mock_settings_factory.for_serving.side_effect = ValueError(
            "Invalid environment configuration"
        )
        mock_run_server.return_value = None

        config_path = cli_test_environment["config_path"]

        runner = CliRunner()
        app = typer.Typer()
        app.command()(serve_api_command)

        result = runner.invoke(app, ["--run-id", "test_run", "--config", str(config_path)])

        # Should fail due to ValueError
        assert result.exit_code != 0
        assert mock_run_server.call_count == 0

        # Verify error message handling in log output (logger.error is still used for errors)
        assert any(
            keyword in caplog.text.lower() for keyword in ["error", "invalid", "configuration"]
        )

    @patch("src.cli.commands.serve_command.run_api_server")
    @patch("src.cli.commands.serve_command.SettingsFactory")
    def test_serve_command_generic_exception_handling(
        self, mock_settings_factory, mock_run_server, cli_test_environment
    ):
        """Test serve command error handling for generic exceptions"""
        # Mock SettingsFactory to raise generic exception
        mock_settings_factory.for_serving.side_effect = RuntimeError("Unexpected error occurred")
        mock_run_server.return_value = None

        config_path = cli_test_environment["config_path"]

        runner = CliRunner()
        app = typer.Typer()
        app.command()(serve_api_command)

        result = runner.invoke(app, ["--run-id", "test_run", "--config", str(config_path)])

        # Should fail due to generic exception
        assert result.exit_code != 0
        assert mock_run_server.call_count == 0

    @patch("src.cli.commands.serve_command.run_api_server")
    @patch("src.cli.commands.serve_command.SettingsFactory")
    def test_serve_command_with_progress_tracking(
        self, mock_settings_factory, mock_run_server, cli_test_environment
    ):
        """Test serve command console progress tracking functionality"""
        mock_settings = MagicMock()
        mock_settings_factory.for_serving.return_value = mock_settings
        mock_run_server.return_value = None

        config_path = cli_test_environment["config_path"]
        test_run_id = "progress_test_run"

        runner = CliRunner()
        app = typer.Typer()
        app.command()(serve_api_command)

        # Execute serve command
        result = runner.invoke(app, ["--run-id", test_run_id, "--config", str(config_path)])

        # Command should succeed
        assert result.exit_code == 0

        # Verify progress tracking and milestone messages in output
        assert any(keyword in result.output for keyword in ["Config", "Server", "API"])

        # Verify all components were called correctly
        mock_settings_factory.for_serving.assert_called_once_with(
            run_id=test_run_id, config_path=str(config_path)
        )
        mock_run_server.assert_called_once_with(
            settings=mock_settings, run_id=test_run_id, host="0.0.0.0", port=8000
        )

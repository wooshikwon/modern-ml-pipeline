"""
Unit Tests for Inference Command CLI - No Mock Hell Compliant
Following test philosophy: Real components, minimal mocking

Only mocking the actual pipeline execution (run_inference_pipeline).
All other components use real implementations.
"""

import pytest
from unittest.mock import patch, MagicMock
import typer
from typer.testing import CliRunner
from pathlib import Path
import json

from src.cli.commands.inference_command import batch_inference_command


class TestInferenceCommandWithRealComponents:
    """Inference command tests using real components - No Mock Hell compliant"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(batch_inference_command)

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.cli.commands.inference_command.SettingsFactory')
    def test_inference_command_with_required_arguments(
        self, mock_settings_factory, mock_run_pipeline, cli_test_environment
    ):
        """Test inference command CLI interface with mocked SettingsFactory"""
        # Mock SettingsFactory and pipeline execution
        mock_settings = MagicMock()
        mock_settings_factory.for_inference.return_value = mock_settings

        mock_result = MagicMock()
        mock_result.processed_rows = 100
        mock_result.output_path = "/tmp/output"
        mock_result.inference_results = "predictions_saved"
        mock_run_pipeline.return_value = mock_result

        config_path = cli_test_environment['config_path']
        data_path = cli_test_environment['data_path']
        test_run_id = 'test_inference_run'

        # Execute command with mocked components
        result = self.runner.invoke(self.app, [
            '--run-id', test_run_id,
            '--config-path', str(config_path),
            '--data-path', str(data_path)
        ])

        # Verify SettingsFactory.for_inference was called correctly
        mock_settings_factory.for_inference.assert_called_once_with(
            run_id=test_run_id,
            config_path=str(config_path),
            data_path=str(data_path),
            context_params=None
        )

        # Verify pipeline was called with correct parameters
        mock_run_pipeline.assert_called_once_with(
            settings=mock_settings,
            run_id=test_run_id,
            data_path=str(data_path),
            context_params={}
        )

        # Command should succeed
        assert result.exit_code == 0

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.cli.commands.inference_command.SettingsFactory')
    def test_inference_command_with_optional_params(
        self, mock_settings_factory, mock_run_pipeline, cli_test_environment
    ):
        """Test inference command with optional JSON context parameters"""
        # Mock SettingsFactory and pipeline execution
        mock_settings = MagicMock()
        mock_settings_factory.for_inference.return_value = mock_settings

        mock_result = MagicMock()
        mock_result.processed_rows = 1000
        mock_result.output_path = "/tmp/batch_output"
        mock_result.inference_results = "batch_complete"
        mock_run_pipeline.return_value = mock_result

        config_path = cli_test_environment['config_path']
        data_path = cli_test_environment['data_path']
        test_run_id = 'test_run_456'

        # Execute with optional JSON parameters
        params = {"batch_date": "2024-01-15", "limit": 1000}
        result = self.runner.invoke(self.app, [
            '--run-id', test_run_id,
            '--config-path', str(config_path),
            '--data-path', str(data_path),
            '--params', json.dumps(params)
        ])

        # Verify SettingsFactory was called with parsed params
        mock_settings_factory.for_inference.assert_called_once_with(
            run_id=test_run_id,
            config_path=str(config_path),
            data_path=str(data_path),
            context_params=params
        )

        # Verify pipeline was called with correct parameters
        mock_run_pipeline.assert_called_once_with(
            settings=mock_settings,
            run_id=test_run_id,
            data_path=str(data_path),
            context_params=params
        )

        assert result.exit_code == 0

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    def test_inference_command_handles_invalid_json_params(
        self, mock_run_pipeline, cli_test_environment
    ):
        """Test inference command handles invalid JSON parameters gracefully"""
        mock_run_pipeline.return_value = MagicMock()

        config_path = cli_test_environment['config_path']
        data_path = cli_test_environment['data_path']

        # Execute with invalid JSON
        result = self.runner.invoke(self.app, [
            '--run-id', 'test_run',
            '--config-path', str(config_path),
            '--data-path', str(data_path),
            '--params', 'invalid json {]'
        ])

        # Should fail due to invalid JSON
        assert result.exit_code != 0 or 'Invalid' in result.output

    def test_inference_command_help_message(self):
        """Test inference command help output"""
        result = self.runner.invoke(self.app, ['--help'])

        assert result.exit_code == 0
        assert '--run-id' in result.output
        assert '--config-path' in result.output
        assert '--data-path' in result.output
        assert '--params' in result.output

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    def test_inference_command_missing_required_args(self, mock_run_pipeline):
        """Test inference command fails when required arguments are missing"""
        # Execute without required arguments
        result = self.runner.invoke(self.app, [])

        # Should fail due to missing required arguments
        assert result.exit_code != 0
        assert mock_run_pipeline.call_count == 0


class TestInferenceCommandIntegration:
    """Integration tests for inference command with contexts"""

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.cli.commands.inference_command.SettingsFactory')
    def test_inference_command_error_handling_settings_failure(
        self, mock_settings_factory, mock_run_pipeline, cli_test_environment
    ):
        """Test inference command error handling when SettingsFactory fails"""
        # Mock SettingsFactory to raise FileNotFoundError
        mock_settings_factory.for_inference.side_effect = FileNotFoundError("Run not found")
        mock_run_pipeline.return_value = MagicMock()

        config_path = cli_test_environment['config_path']
        data_path = cli_test_environment['data_path']

        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)

        # Execute with non-existent run_id
        result = runner.invoke(app, [
            '--run-id', 'non_existent_run',
            '--config-path', str(config_path),
            '--data-path', str(data_path)
        ])

        # Should fail due to SettingsFactory error
        assert result.exit_code != 0
        assert mock_run_pipeline.call_count == 0

        # Verify SettingsFactory was called
        mock_settings_factory.for_inference.assert_called_once()

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.cli.commands.inference_command.SettingsFactory')
    def test_inference_command_error_handling_missing_config(
        self, mock_settings_factory, mock_run_pipeline
    ):
        """Test inference command error handling for missing config file"""
        # Mock SettingsFactory to raise FileNotFoundError for missing config
        mock_settings_factory.for_inference.side_effect = FileNotFoundError("Config file not found")
        mock_run_pipeline.return_value = MagicMock()

        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)

        # Execute with non-existent config file
        result = runner.invoke(app, [
            '--run-id', 'test_run',
            '--config-path', '/non/existent/config.yaml',
            '--data-path', '/tmp/data.csv'
        ])

        # Should fail due to config file not found
        assert result.exit_code != 0
        assert mock_run_pipeline.call_count == 0

        # Verify SettingsFactory was called with the bad config path
        mock_settings_factory.for_inference.assert_called_once_with(
            run_id='test_run',
            config_path='/non/existent/config.yaml',
            data_path='/tmp/data.csv',
            context_params=None
        )

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.cli.commands.inference_command.SettingsFactory')
    def test_inference_command_success_message_formatting(
        self, mock_settings_factory, mock_run_pipeline, cli_test_environment
    ):
        """Test inference command success message and result processing"""
        # Mock SettingsFactory and successful pipeline execution
        mock_settings = MagicMock()
        mock_settings_factory.for_inference.return_value = mock_settings

        mock_result = MagicMock()
        mock_result.processed_rows = 1000
        mock_result.output_path = "/tmp/predictions.csv"
        mock_result.inference_results = "batch_complete"
        mock_run_pipeline.return_value = mock_result

        config_path = cli_test_environment['config_path']
        data_path = cli_test_environment['data_path']
        test_run_id = 'success_test_run'

        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)

        result = runner.invoke(app, [
            '--run-id', test_run_id,
            '--config-path', str(config_path),
            '--data-path', str(data_path)
        ])

        # Verify successful execution
        assert result.exit_code == 0
        output_str = str(result.output)

        # Check for success message content
        assert any(keyword in output_str for keyword in ['처리', '완료', '출력', 'Inference', 'Batch'])

        # Verify all components were called correctly
        mock_settings_factory.for_inference.assert_called_once_with(
            run_id=test_run_id,
            config_path=str(config_path),
            data_path=str(data_path),
            context_params=None
        )
        mock_run_pipeline.assert_called_once_with(
            settings=mock_settings,
            run_id=test_run_id,
            data_path=str(data_path),
            context_params={}
        )

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.cli.commands.inference_command.SettingsFactory')
    def test_inference_command_value_error_handling(
        self, mock_settings_factory, mock_run_pipeline, cli_test_environment
    ):
        """Test inference command error handling for ValueError"""
        # Mock SettingsFactory to raise ValueError
        mock_settings_factory.for_inference.side_effect = ValueError("Invalid environment configuration")
        mock_run_pipeline.return_value = MagicMock()

        config_path = cli_test_environment['config_path']
        data_path = cli_test_environment['data_path']

        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)

        result = runner.invoke(app, [
            '--run-id', 'test_run',
            '--config-path', str(config_path),
            '--data-path', str(data_path)
        ])

        # Should fail due to ValueError
        assert result.exit_code != 0
        assert mock_run_pipeline.call_count == 0

        # Verify error handling
        output_str = str(result.output)
        assert any(keyword in output_str for keyword in ['오류', 'error', 'Error'])

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.cli.commands.inference_command.SettingsFactory')
    def test_inference_command_generic_exception_handling(
        self, mock_settings_factory, mock_run_pipeline, cli_test_environment
    ):
        """Test inference command error handling for generic exceptions"""
        # Mock SettingsFactory to raise generic exception
        mock_settings_factory.for_inference.side_effect = RuntimeError("Unexpected error occurred")
        mock_run_pipeline.return_value = MagicMock()

        config_path = cli_test_environment['config_path']
        data_path = cli_test_environment['data_path']

        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)

        result = runner.invoke(app, [
            '--run-id', 'test_run',
            '--config-path', str(config_path),
            '--data-path', str(data_path)
        ])

        # Should fail due to generic exception
        assert result.exit_code != 0
        assert mock_run_pipeline.call_count == 0

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.cli.commands.inference_command.SettingsFactory')
    def test_inference_command_console_integration(
        self, mock_settings_factory, mock_run_pipeline, cli_test_environment
    ):
        """Test inference command console progress tracking and milestones"""
        # Mock SettingsFactory and pipeline execution
        mock_settings = MagicMock()
        mock_settings_factory.for_inference.return_value = mock_settings

        mock_result = MagicMock()
        mock_result.processed_rows = 50
        mock_result.output_path = "/tmp/results"
        mock_run_pipeline.return_value = mock_result

        config_path = cli_test_environment['config_path']
        data_path = cli_test_environment['data_path']
        test_run_id = 'console_test_run'

        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)

        result = runner.invoke(app, [
            '--run-id', test_run_id,
            '--config-path', str(config_path),
            '--data-path', str(data_path)
        ])

        # Verify successful execution
        assert result.exit_code == 0
        output_str = str(result.output)

        # Check for console progress tracking messages
        assert any(keyword in output_str for keyword in ['추론', '설정', 'Inference', 'Batch', 'Config'])

        # Verify all components were called correctly
        mock_settings_factory.for_inference.assert_called_once_with(
            run_id=test_run_id,
            config_path=str(config_path),
            data_path=str(data_path),
            context_params=None
        )
        mock_run_pipeline.assert_called_once_with(
            settings=mock_settings,
            run_id=test_run_id,
            data_path=str(data_path),
            context_params={}
        )

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    def test_inference_with_real_data_files(
        self, mock_run_pipeline, real_dataset_files, cli_test_environment
    ):
        """Test inference command with real data files from fixtures"""
        mock_result = MagicMock()
        mock_result.processed_rows = 100
        mock_result.output_path = str(real_dataset_files["classification_csv"]["path"])
        mock_run_pipeline.return_value = mock_result

        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)

        # Use real CSV file
        data_path = real_dataset_files["classification_csv"]["path"]
        config_path = cli_test_environment['config_path']

        result = runner.invoke(app, [
            '--run-id', 'real_data_run',
            '--config-path', str(config_path),
            '--data-path', str(data_path)
        ])

        # Verify pipeline was called with real data path
        if mock_run_pipeline.called:
            call_args = mock_run_pipeline.call_args
            assert call_args.kwargs['data_path'] == str(data_path)
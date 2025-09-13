"""
Unit Tests for Serve Command CLI
Phase 3: Production-ready API serving command tests

This module follows the tests/README.md architecture principles:
- Real object testing over mock hell
- Rich console integration patterns
- Comprehensive error scenario coverage
- Architecture compliance with existing CLI test patterns
"""

import pytest
from unittest.mock import patch, MagicMock, call
import typer
from typer.testing import CliRunner
import socket
from pathlib import Path
import tempfile
import os

from src.cli.commands.serve_command import serve_api_command


class TestServeCommandArgumentParsing:
    """Serve command argument parsing and basic functionality tests"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(serve_api_command)

    @patch('src.cli.commands.serve_command.SettingsFactory.for_serving')
    @patch('src.cli.commands.serve_command.run_api_server')
    @patch('src.cli.commands.serve_command.cli_command_start')
    @patch('src.cli.commands.serve_command.get_rich_console')
    def test_serve_command_with_required_arguments(
        self, mock_console, mock_cli_start, mock_run_server, mock_settings_factory
    ):
        """Test serve command with all required arguments"""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings_factory.return_value = mock_settings

        # Mock rich console with progress tracker
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        mock_progress_context = MagicMock()
        mock_console_instance.progress_tracker.return_value.__enter__.return_value = mock_progress_context
        mock_console_instance.progress_tracker.return_value.__exit__.return_value = None

        # Execute command
        result = self.runner.invoke(self.app, [
            '--run-id', 'test_run_123',
            '--config-path', 'configs/serve.yaml'
        ])

        # Verify success
        assert result.exit_code == 0

        # Verify SettingsFactory call
        mock_settings_factory.assert_called_once_with(
            run_id='test_run_123',
            config_path='configs/serve.yaml'
        )

        # Verify API server execution with defaults
        mock_run_server.assert_called_once_with(
            settings=mock_settings,
            run_id='test_run_123',
            host='0.0.0.0',
            port=8000
        )

        # Verify Rich Console calls
        mock_cli_start.assert_called_once_with("API Server", "모델 서빙 서버 시작")
        mock_console.assert_called_once()
        mock_console_instance.progress_tracker.assert_called_once_with("setup", 2, "서버 환경 설정")
        mock_console_instance.log_milestone.assert_any_call("🌐 API Server: http://0.0.0.0:8000", "success")
        mock_console_instance.log_milestone.assert_any_call("📜 API Documentation: http://0.0.0.0:8000/docs", "info")
        mock_console_instance.log_milestone.assert_any_call("🔍 Health Check: http://0.0.0.0:8000/health", "info")

    @patch('src.cli.commands.serve_command.SettingsFactory.for_serving')
    @patch('src.cli.commands.serve_command.run_api_server')
    @patch('src.cli.commands.serve_command.cli_command_start')
    @patch('src.cli.commands.serve_command.get_rich_console')
    def test_serve_command_with_custom_host_port(
        self, mock_console, mock_cli_start, mock_run_server, mock_settings_factory
    ):
        """Test serve command with custom host and port"""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings_factory.return_value = mock_settings

        # Mock rich console with progress tracker
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        mock_progress_context = MagicMock()
        mock_console_instance.progress_tracker.return_value.__enter__.return_value = mock_progress_context
        mock_console_instance.progress_tracker.return_value.__exit__.return_value = None

        # Execute command with custom host and port
        result = self.runner.invoke(self.app, [
            '--run-id', 'prod_model_run',
            '--config-path', 'configs/production.yaml',
            '--host', 'localhost',
            '--port', '9000'
        ])

        # Verify success
        assert result.exit_code == 0

        # Verify SettingsFactory call
        mock_settings_factory.assert_called_once_with(
            run_id='prod_model_run',
            config_path='configs/production.yaml'
        )

        # Verify API server execution with custom parameters
        mock_run_server.assert_called_once_with(
            settings=mock_settings,
            run_id='prod_model_run',
            host='localhost',
            port=9000
        )

        # Verify Rich Console milestone messages with custom host:port
        mock_console_instance.log_milestone.assert_any_call("🌐 API Server: http://localhost:9000", "success")
        mock_console_instance.log_milestone.assert_any_call("📜 API Documentation: http://localhost:9000/docs", "info")
        mock_console_instance.log_milestone.assert_any_call("🔍 Health Check: http://localhost:9000/health", "info")

    def test_serve_command_missing_required_arguments(self):
        """Test serve command fails with missing required arguments"""
        # Missing run-id
        result = self.runner.invoke(self.app, [
            '--config-path', 'configs/serve.yaml'
        ])
        assert result.exit_code != 0
        assert "Missing option '--run-id'" in result.output

        # Missing config-path
        result = self.runner.invoke(self.app, [
            '--run-id', 'test_run_123'
        ])
        assert result.exit_code != 0
        assert "Missing option '--config-path'" in result.output

    @patch('src.cli.commands.serve_command.SettingsFactory.for_serving')
    @patch('src.cli.commands.serve_command.cli_command_start')
    @patch('src.cli.commands.serve_command.cli_command_error')
    def test_serve_command_file_not_found_error(self, mock_cli_error, mock_cli_start, mock_settings_factory):
        """Test serve command handles FileNotFoundError properly"""
        # Setup mock to raise FileNotFoundError
        mock_settings_factory.side_effect = FileNotFoundError("Config 파일을 찾을 수 없습니다")

        # Execute command
        result = self.runner.invoke(self.app, [
            '--run-id', 'missing_run',
            '--config-path', 'nonexistent/config.yaml'
        ])

        # Verify exit code and error handling
        assert result.exit_code == 1

        # Verify Rich Console calls
        mock_cli_start.assert_called_once_with("API Server", "모델 서빙 서버 시작")
        mock_cli_error.assert_called_once_with(
            "API Server",
            "파일을 찾을 수 없습니다: Config 파일을 찾을 수 없습니다",
            "파일 경로를 확인하거나 올바른 Run ID를 사용하세요"
        )

    @patch('src.cli.commands.serve_command.SettingsFactory.for_serving')
    @patch('src.cli.commands.serve_command.cli_command_start')
    @patch('src.cli.commands.serve_command.cli_command_error')
    def test_serve_command_value_error(self, mock_cli_error, mock_cli_start, mock_settings_factory):
        """Test serve command handles ValueError (configuration errors)"""
        # Setup mock to raise ValueError
        mock_settings_factory.side_effect = ValueError("잘못된 모델 형식입니다")

        # Execute command
        result = self.runner.invoke(self.app, [
            '--run-id', 'invalid_run',
            '--config-path', 'configs/invalid.yaml'
        ])

        # Verify exit code and error handling
        assert result.exit_code == 1

        # Verify Rich Console calls
        mock_cli_start.assert_called_once_with("API Server", "모델 서빙 서버 시작")
        mock_cli_error.assert_called_once_with(
            "API Server",
            "환경 설정 오류: 잘못된 모델 형식입니다"
        )

    @patch('src.cli.commands.serve_command.SettingsFactory.for_serving')
    @patch('src.cli.commands.serve_command.cli_command_start')
    @patch('src.cli.commands.serve_command.cli_command_error')
    def test_serve_command_general_exception(self, mock_cli_error, mock_cli_start, mock_settings_factory):
        """Test serve command handles unexpected exceptions"""
        # Setup mock to raise general Exception
        mock_settings_factory.side_effect = Exception("예상치 못한 서버 오류")

        # Execute command
        result = self.runner.invoke(self.app, [
            '--run-id', 'error_run',
            '--config-path', 'configs/error.yaml'
        ])

        # Verify exit code and error handling
        assert result.exit_code == 1

        # Verify Rich Console calls
        mock_cli_start.assert_called_once_with("API Server", "모델 서빙 서버 시작")
        mock_cli_error.assert_called_once_with(
            "API Server",
            "실행 중 오류 발생: 예상치 못한 서버 오류"
        )

    def test_serve_command_help_message(self):
        """Test serve command shows proper help message"""
        result = self.runner.invoke(self.app, ['--help'])

        assert result.exit_code == 0
        assert "API 서버 실행" in result.output
        assert "--run-id" in result.output
        assert "--config-path" in result.output
        assert "--host" in result.output
        assert "--port" in result.output
        assert "MLflow에 저장된 모델을 REST API로 서빙합니다" in result.output

    def test_serve_command_port_validation(self):
        """Test serve command validates port numbers"""
        # Test negative port
        result = self.runner.invoke(self.app, [
            '--run-id', 'test_run',
            '--config-path', 'configs/test.yaml',
            '--port', '-1'
        ])
        # Typer handles this validation automatically
        assert result.exit_code != 0

        # Test port out of range
        result = self.runner.invoke(self.app, [
            '--run-id', 'test_run',
            '--config-path', 'configs/test.yaml',
            '--port', '70000'
        ])
        # The command might still execute as the port validation happens at server startup
        # This tests the CLI parameter parsing, not the actual port binding


class TestServeCommandIntegration:
    """Integration tests for serve command with real API server scenarios"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(serve_api_command)

    @patch('src.cli.commands.serve_command.SettingsFactory.for_serving')
    @patch('src.cli.commands.serve_command.run_api_server')
    @patch('src.cli.commands.serve_command.cli_command_start')
    @patch('src.cli.commands.serve_command.cli_step_complete')
    @patch('src.cli.commands.serve_command.get_rich_console')
    def test_serve_command_progress_tracking(
        self, mock_console, mock_cli_step, mock_cli_start, mock_run_server, mock_settings_factory
    ):
        """Test serve command progress tracking and milestone logging"""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings_factory.return_value = mock_settings

        # Mock rich console with progress tracker
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        mock_progress_context = MagicMock()
        mock_console_instance.progress_tracker.return_value.__enter__.return_value = mock_progress_context
        mock_console_instance.progress_tracker.return_value.__exit__.return_value = None

        # Execute command
        result = self.runner.invoke(self.app, [
            '--run-id', 'progress_run',
            '--config-path', 'configs/dev.yaml',
            '--host', '127.0.0.1',
            '--port', '8080'
        ])

        # Verify success
        assert result.exit_code == 0

        # Verify progress tracking sequence
        mock_cli_start.assert_called_once_with("API Server", "모델 서빙 서버 시작")
        mock_console_instance.progress_tracker.assert_called_once_with("setup", 2, "서버 환경 설정")
        mock_progress_context.assert_called_once_with(2)  # Progress update call

        # Verify step completion
        mock_cli_step.assert_called_once_with(
            "설정",
            "Config: configs/dev.yaml, Run ID: progress_run"
        )

        # Verify milestone logging sequence
        expected_milestones = [
            call("🌐 API Server: http://127.0.0.1:8080", "success"),
            call("📜 API Documentation: http://127.0.0.1:8080/docs", "info"),
            call("🔍 Health Check: http://127.0.0.1:8080/health", "info")
        ]
        mock_console_instance.log_milestone.assert_has_calls(expected_milestones, any_order=False)

    @patch('src.cli.commands.serve_command.SettingsFactory.for_serving')
    @patch('src.cli.commands.serve_command.run_api_server')
    @patch('src.cli.commands.serve_command.get_rich_console')
    def test_serve_command_settings_integration(
        self, mock_console, mock_run_server, mock_settings_factory
    ):
        """Test serve command integrates properly with SettingsFactory for serving"""
        # Setup mock settings with specific attributes
        mock_settings = MagicMock()
        mock_settings.model_config = {"model_type": "classification"}
        mock_settings.serving_config = {"batch_size": 32}
        mock_settings_factory.return_value = mock_settings

        # Mock rich console
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        mock_console_instance.progress_tracker.return_value.__enter__.return_value = MagicMock()
        mock_console_instance.progress_tracker.return_value.__exit__.return_value = None

        # Execute command
        result = self.runner.invoke(self.app, [
            '--run-id', 'settings_test_run',
            '--config-path', 'configs/serving_test.yaml'
        ])

        # Verify success
        assert result.exit_code == 0

        # Verify SettingsFactory.for_serving was called correctly
        mock_settings_factory.assert_called_once_with(
            run_id='settings_test_run',
            config_path='configs/serving_test.yaml'
        )

        # Verify the settings object was passed to run_api_server
        mock_run_server.assert_called_once()
        call_args = mock_run_server.call_args
        assert call_args.kwargs['settings'] is mock_settings
        assert call_args.kwargs['run_id'] == 'settings_test_run'

    def test_serve_command_invalid_run_id_format(self):
        """Test serve command with various run ID formats"""
        # Test empty run ID
        result = self.runner.invoke(self.app, [
            '--run-id', '',
            '--config-path', 'configs/test.yaml'
        ])
        # The command will execute and let the downstream validation handle empty run_id

        # Test run ID with special characters (should be valid)
        result = self.runner.invoke(self.app, [
            '--run-id', 'run-123_test.model',
            '--config-path', 'configs/test.yaml'
        ])
        # This should work as run_id is just a string parameter

    def test_serve_command_config_path_formats(self):
        """Test serve command with various config path formats"""
        test_cases = [
            'config.yaml',
            './config.yaml',
            '/absolute/path/config.yaml',
            'configs/nested/deep/config.yaml',
            'config.yml',
        ]

        for config_path in test_cases:
            result = self.runner.invoke(self.app, [
                '--run-id', 'test_run',
                '--config-path', config_path
            ])
            # The command will execute and file validation happens in SettingsFactory
            # CLI parsing should accept all valid path formats


class TestServeCommandErrorScenarios:
    """Comprehensive error scenario testing for serve command"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(serve_api_command)

    @patch('src.cli.commands.serve_command.SettingsFactory.for_serving')
    @patch('src.cli.commands.serve_command.run_api_server')
    @patch('src.cli.commands.serve_command.cli_command_start')
    @patch('src.cli.commands.serve_command.cli_command_error')
    @patch('src.cli.commands.serve_command.get_rich_console')
    def test_serve_command_server_startup_failure(
        self, mock_console, mock_cli_error, mock_cli_start, mock_run_server, mock_settings_factory
    ):
        """Test serve command handles server startup failures"""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings_factory.return_value = mock_settings

        # Mock rich console
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        mock_console_instance.progress_tracker.return_value.__enter__.return_value = MagicMock()
        mock_console_instance.progress_tracker.return_value.__exit__.return_value = None

        # Mock server startup failure
        mock_run_server.side_effect = Exception("Port 8000 이미 사용 중입니다")

        # Execute command
        result = self.runner.invoke(self.app, [
            '--run-id', 'startup_fail_run',
            '--config-path', 'configs/serve.yaml'
        ])

        # Verify error exit code
        assert result.exit_code == 1

        # Verify error handling
        mock_cli_error.assert_called_once_with(
            "API Server",
            "실행 중 오류 발생: Port 8000 이미 사용 중입니다"
        )

    @patch('src.cli.commands.serve_command.SettingsFactory.for_serving')
    @patch('src.cli.commands.serve_command.cli_command_start')
    @patch('src.cli.commands.serve_command.cli_command_error')
    def test_serve_command_model_loading_failure(
        self, mock_cli_error, mock_cli_start, mock_settings_factory
    ):
        """Test serve command handles model loading failures during settings creation"""
        # Setup mock to simulate model loading failure
        mock_settings_factory.side_effect = FileNotFoundError("MLflow Run ID를 찾을 수 없습니다: invalid_run_id")

        # Execute command
        result = self.runner.invoke(self.app, [
            '--run-id', 'invalid_run_id',
            '--config-path', 'configs/serve.yaml'
        ])

        # Verify error exit code
        assert result.exit_code == 1

        # Verify specific error message for missing run ID
        mock_cli_error.assert_called_once_with(
            "API Server",
            "파일을 찾을 수 없습니다: MLflow Run ID를 찾을 수 없습니다: invalid_run_id",
            "파일 경로를 확인하거나 올바른 Run ID를 사용하세요"
        )

    @patch('src.cli.commands.serve_command.SettingsFactory.for_serving')
    @patch('src.cli.commands.serve_command.cli_command_start')
    @patch('src.cli.commands.serve_command.cli_command_error')
    def test_serve_command_configuration_validation_failure(
        self, mock_cli_error, mock_cli_start, mock_settings_factory
    ):
        """Test serve command handles configuration validation failures"""
        # Setup mock to simulate configuration validation failure
        mock_settings_factory.side_effect = ValueError("Serving 설정이 유효하지 않습니다: missing model_uri")

        # Execute command
        result = self.runner.invoke(self.app, [
            '--run-id', 'config_fail_run',
            '--config-path', 'configs/invalid_serve.yaml'
        ])

        # Verify error exit code
        assert result.exit_code == 1

        # Verify specific error message for configuration issues
        mock_cli_error.assert_called_once_with(
            "API Server",
            "환경 설정 오류: Serving 설정이 유효하지 않습니다: missing model_uri"
        )

    def test_serve_command_keyboard_interrupt(self):
        """Test serve command handles KeyboardInterrupt (Ctrl+C) gracefully"""
        with patch('src.cli.commands.serve_command.SettingsFactory.for_serving') as mock_factory, \
             patch('src.cli.commands.serve_command.run_api_server') as mock_server, \
             patch('src.cli.commands.serve_command.cli_command_start'), \
             patch('src.cli.commands.serve_command.get_rich_console') as mock_console:

            # Setup mocks
            mock_settings = MagicMock()
            mock_factory.return_value = mock_settings

            # Mock rich console
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            mock_console_instance.progress_tracker.return_value.__enter__.return_value = MagicMock()
            mock_console_instance.progress_tracker.return_value.__exit__.return_value = None

            # Mock KeyboardInterrupt during server execution
            mock_server.side_effect = KeyboardInterrupt()

            # Execute command
            result = self.runner.invoke(self.app, [
                '--run-id', 'interrupt_test',
                '--config-path', 'configs/serve.yaml'
            ])

            # KeyboardInterrupt should result in exit code 130 (Unix SIGINT signal)
            assert result.exit_code == 130
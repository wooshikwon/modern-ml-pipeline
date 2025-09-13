"""
Unit Tests for Inference Command CLI  
Days 3-5: CLI argument parsing and validation tests
"""

import pytest
from unittest.mock import patch, MagicMock
import typer
from typer.testing import CliRunner

from src.cli.commands.inference_command import batch_inference_command


class TestInferenceCommandArgumentParsing:
    """Inference command argument parsing tests"""
    
    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(batch_inference_command)
    
    @patch('src.cli.commands.inference_command.SettingsFactory.for_inference')
    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.cli.commands.inference_command.cli_command_start')
    @patch('src.cli.commands.inference_command.cli_command_success')
    def test_inference_command_with_required_arguments(self, mock_cli_success, mock_cli_start, mock_run_pipeline, mock_settings_factory):
        """Test inference command with all required arguments"""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings_factory.return_value = mock_settings

        mock_result = MagicMock()
        mock_result.inference_results = "predictions_saved"
        mock_run_pipeline.return_value = mock_result
        
        # Execute command
        result = self.runner.invoke(self.app, [
            '--run-id', 'abc123def456',
            '--config-path', 'configs/prod.yaml',
            '--data-path', 'data/inference.csv'
        ])
        
        # Verify success
        assert result.exit_code == 0

        # Verify SettingsFactory call
        mock_settings_factory.assert_called_once_with(
            run_id='abc123def456',
            config_path='configs/prod.yaml',
            data_path='data/inference.csv',
            context_params=None
        )

        # Verify pipeline execution
        mock_run_pipeline.assert_called_once_with(
            settings=mock_settings,
            run_id='abc123def456',
            data_path='data/inference.csv',
            context_params={}
        )

        # Verify Rich Console calls
        mock_cli_start.assert_called_once_with("Batch Inference", "배치 추론 파이프라인 실행")
        mock_cli_success.assert_called_once_with("Batch Inference", [
            f"처리된 데이터: {mock_result.processed_rows}행",
            f"출력 경로: {mock_result.output_path}"
        ])
    
    @patch('src.cli.commands.inference_command.SettingsFactory.for_inference')
    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.cli.commands.inference_command.cli_command_start')
    @patch('src.cli.commands.inference_command.cli_command_success')
    def test_inference_command_with_optional_params(self, mock_cli_success, mock_cli_start, mock_run_pipeline, mock_settings_factory):
        """Test inference command with optional context parameters"""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings_factory.return_value = mock_settings

        mock_result = MagicMock()
        mock_result.inference_results = "batch_predictions_complete"
        mock_run_pipeline.return_value = mock_result
        
        # Execute command with JSON params
        result = self.runner.invoke(self.app, [
            '--run-id', 'xyz789abc123', 
            '--config-path', 'configs/inference.yaml',
            '--data-path', 'queries/batch_predict.sql',
            '--params', '{"batch_date": "2024-01-15", "limit": 1000}'
        ])
        
        # Verify success
        assert result.exit_code == 0

        # Verify SettingsFactory call with parsed parameters
        mock_settings_factory.assert_called_once_with(
            run_id='xyz789abc123',
            config_path='configs/inference.yaml',
            data_path='queries/batch_predict.sql',
            context_params={"batch_date": "2024-01-15", "limit": 1000}
        )

        # Verify pipeline execution
        mock_run_pipeline.assert_called_once_with(
            settings=mock_settings,
            run_id='xyz789abc123',
            data_path='queries/batch_predict.sql',
            context_params={"batch_date": "2024-01-15", "limit": 1000}
        )

        # Verify Rich Console calls
        mock_cli_start.assert_called_once_with("Batch Inference", "배치 추론 파이프라인 실행")
        mock_cli_success.assert_called_once_with("Batch Inference", [
            f"처리된 데이터: {mock_result.processed_rows}행",
            f"출력 경로: {mock_result.output_path}"
        ])
    
    def test_inference_command_missing_required_arguments(self):
        """Test inference command fails with missing required arguments"""
        # Missing run-id
        result = self.runner.invoke(self.app, [
            '--config-path', 'configs/test.yaml',
            '--data-path', 'data/inference.csv'
        ])
        assert result.exit_code != 0
        assert "Missing option '--run-id'" in result.output
        
        # Missing config-path
        result = self.runner.invoke(self.app, [
            '--run-id', 'test123',
            '--data-path', 'data/inference.csv'
        ])
        assert result.exit_code != 0
        assert "Missing option '--config-path'" in result.output
        
        # Missing data-path
        result = self.runner.invoke(self.app, [
            '--run-id', 'test123',
            '--config-path', 'configs/test.yaml'
        ])
        assert result.exit_code != 0
        assert "Missing option '--data-path'" in result.output
    
    @patch('src.cli.commands.inference_command.SettingsFactory.for_inference')
    @patch('src.cli.commands.inference_command.cli_command_start')
    @patch('src.cli.commands.inference_command.cli_command_error')
    def test_inference_command_file_not_found_error(self, mock_cli_error, mock_cli_start, mock_settings_factory):
        """Test inference command handles FileNotFoundError"""
        # Setup mock to raise FileNotFoundError
        mock_settings_factory.side_effect = FileNotFoundError("Config 파일을 찾을 수 없습니다")
        
        # Execute command
        result = self.runner.invoke(self.app, [
            '--run-id', 'test123',
            '--config-path', 'nonexistent/config.yaml',
            '--data-path', 'data/inference.csv'
        ])
        
        # Verify exit code and error handling
        assert result.exit_code == 1

        # Verify Rich Console calls
        mock_cli_start.assert_called_once_with("Batch Inference", "배치 추론 파이프라인 실행")
        mock_cli_error.assert_called_once_with(
            "Batch Inference",
            "파일을 찾을 수 없습니다: Config 파일을 찾을 수 없습니다",
            "파일 경로를 확인하거나 올바른 Run ID를 사용하세요"
        )
    
    def test_inference_command_invalid_json_params(self):
        """Test inference command handles invalid JSON in context_params"""
        
        # Execute command with invalid JSON
        result = self.runner.invoke(self.app, [
            '--run-id', 'test123',
            '--config-path', 'configs/test.yaml',
            '--data-path', 'data/inference.csv',
            '--params', '{invalid json syntax}'  # Invalid JSON
        ])
        
        # Should fail due to JSON parsing error
        assert result.exit_code != 0  # Typer will catch JSON parsing error
    
    @patch('src.cli.commands.inference_command.SettingsFactory.for_inference')
    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.cli.commands.inference_command.cli_command_start')
    @patch('src.cli.commands.inference_command.cli_command_success')
    def test_inference_command_short_options(self, mock_cli_success, mock_cli_start, mock_run_pipeline, mock_settings_factory):
        """Test inference command with short option flags"""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings_factory.return_value = mock_settings

        mock_result = MagicMock()
        mock_result.inference_results = "short_test_complete"
        mock_run_pipeline.return_value = mock_result

        # Test short flags work
        result = self.runner.invoke(self.app, [
            '--run-id', 'short123',
            '-c', 'configs/test.yaml',
            '-d', 'data/inference.csv',
            '-p', '{"test": true}'
        ])

        assert result.exit_code == 0

        # Verify SettingsFactory call with short options
        mock_settings_factory.assert_called_once_with(
            run_id='short123',
            config_path='configs/test.yaml',
            data_path='data/inference.csv',
            context_params={"test": True}
        )

        # Verify Rich Console calls
        mock_cli_start.assert_called_once_with("Batch Inference", "배치 추론 파이프라인 실행")
        mock_cli_success.assert_called_once_with("Batch Inference", [
            f"처리된 데이터: {mock_result.processed_rows}행",
            f"출력 경로: {mock_result.output_path}"
        ])
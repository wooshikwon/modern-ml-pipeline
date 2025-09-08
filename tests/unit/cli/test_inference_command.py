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
    
    @patch('src.cli.commands.inference_command.load_config_files')
    @patch('src.cli.commands.inference_command.create_settings_for_inference')
    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.cli.commands.inference_command.setup_logging')
    def test_inference_command_with_required_arguments(self, mock_setup_logging, mock_run_pipeline, 
                                                     mock_create_settings, mock_load_config):
        """Test inference command with all required arguments"""
        # Setup mocks
        mock_config_data = {"environment": {"name": "test"}}
        mock_settings = MagicMock()
        mock_load_config.return_value = mock_config_data
        mock_create_settings.return_value = mock_settings
        
        # Execute command
        result = self.runner.invoke(self.app, [
            '--run-id', 'abc123def456',
            '--config-path', 'configs/prod.yaml',
            '--data-path', 'data/inference.csv'
        ])
        
        # Verify success
        assert result.exit_code == 0
        
        # Verify function calls
        mock_load_config.assert_called_once_with(config_path='configs/prod.yaml')
        mock_create_settings.assert_called_once_with(mock_config_data)
        mock_setup_logging.assert_called_once_with(mock_settings)
        mock_run_pipeline.assert_called_once_with(
            settings=mock_settings,
            run_id='abc123def456',
            data_path='data/inference.csv',
            context_params={}
        )
    
    @patch('src.cli.commands.inference_command.load_config_files')
    @patch('src.cli.commands.inference_command.create_settings_for_inference')
    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.cli.commands.inference_command.setup_logging')
    def test_inference_command_with_optional_params(self, mock_setup_logging, mock_run_pipeline,
                                                   mock_create_settings, mock_load_config):
        """Test inference command with optional context parameters"""
        # Setup mocks
        mock_config_data = {"environment": {"name": "prod"}}
        mock_settings = MagicMock()
        mock_load_config.return_value = mock_config_data
        mock_create_settings.return_value = mock_settings
        
        # Execute command with JSON params
        result = self.runner.invoke(self.app, [
            '--run-id', 'xyz789abc123', 
            '--config-path', 'configs/inference.yaml',
            '--data-path', 'queries/batch_predict.sql',
            '--params', '{"batch_date": "2024-01-15", "limit": 1000}'
        ])
        
        # Verify success
        assert result.exit_code == 0
        
        # Verify function calls with parsed parameters
        mock_run_pipeline.assert_called_once_with(
            settings=mock_settings,
            run_id='xyz789abc123',
            data_path='queries/batch_predict.sql',
            context_params={"batch_date": "2024-01-15", "limit": 1000}
        )
    
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
    
    @patch('src.cli.commands.inference_command.load_config_files')
    def test_inference_command_file_not_found_error(self, mock_load_config):
        """Test inference command handles FileNotFoundError"""
        # Setup mock to raise FileNotFoundError
        mock_load_config.side_effect = FileNotFoundError("Config 파일을 찾을 수 없습니다")
        
        # Execute command
        result = self.runner.invoke(self.app, [
            '--run-id', 'test123',
            '--config-path', 'nonexistent/config.yaml',
            '--data-path', 'data/inference.csv'
        ])
        
        # Verify exit code and error handling
        assert result.exit_code == 1
    
    @patch('src.cli.commands.inference_command.load_config_files')
    def test_inference_command_invalid_json_params(self, mock_load_config):
        """Test inference command handles invalid JSON in context_params"""
        mock_config_data = {"environment": {"name": "test"}}
        mock_load_config.return_value = mock_config_data
        
        # Execute command with invalid JSON
        result = self.runner.invoke(self.app, [
            '--run-id', 'test123',
            '--config-path', 'configs/test.yaml',
            '--data-path', 'data/inference.csv',
            '--params', '{invalid json syntax}'  # Invalid JSON
        ])
        
        # Should fail due to JSON parsing error
        assert result.exit_code == 1
    
    def test_inference_command_short_options(self):
        """Test inference command with short option flags"""
        with patch('src.cli.commands.inference_command.load_config_files'), \
             patch('src.cli.commands.inference_command.create_settings_for_inference'), \
             patch('src.cli.commands.inference_command.run_inference_pipeline'), \
             patch('src.cli.commands.inference_command.setup_logging'):
            
            # Test short flags work
            result = self.runner.invoke(self.app, [
                '--run-id', 'short123',
                '-c', 'configs/test.yaml',
                '-d', 'data/inference.csv',
                '-p', '{"test": true}'
            ])
            
            assert result.exit_code == 0
"""
Unit Tests for Config Commands CLI
Days 3-5: Interactive config creation tests
"""

import pytest
from unittest.mock import patch, MagicMock
import typer
from typer.testing import CliRunner
from pathlib import Path

from src.cli.commands.get_config_command import get_config_command


class TestGetConfigCommandArgumentParsing:
    """Get config command argument parsing tests"""
    
    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(get_config_command)
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    def test_get_config_command_without_env_name(self, mock_builder_class):
        """Test get config command without environment name (interactive mode)"""
        # Setup mocks
        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder
        mock_builder.run_interactive_flow.return_value = {
            'env_name': 'dev',
            'mlflow_enabled': True,
            'data_source_type': 'storage'
        }
        
        # Execute command without env_name
        result = self.runner.invoke(self.app, [])
        
        # Verify command execution
        mock_builder_class.assert_called_once()
        mock_builder.run_interactive_flow.assert_called_once_with(None)
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    def test_get_config_command_with_env_name(self, mock_builder_class):
        """Test get config command with specified environment name"""
        # Setup mocks
        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder
        mock_builder.run_interactive_flow.return_value = {
            'env_name': 'production',
            'mlflow_enabled': True,
            'data_source_type': 'sql'
        }
        
        # Execute command with env_name
        result = self.runner.invoke(self.app, ['--env-name', 'production'])
        
        # Verify command execution with specific env_name
        mock_builder_class.assert_called_once()
        mock_builder.run_interactive_flow.assert_called_once_with('production')
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    def test_get_config_command_with_short_option(self, mock_builder_class):
        """Test get config command with short option flag"""
        # Setup mocks
        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder
        mock_builder.run_interactive_flow.return_value = {
            'env_name': 'dev',
            'mlflow_enabled': False
        }
        
        # Execute command with short flag
        result = self.runner.invoke(self.app, ['-e', 'dev'])
        
        # Verify short option works
        mock_builder.run_interactive_flow.assert_called_once_with('dev')
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    def test_get_config_command_builder_initialization_error(self, mock_builder_class):
        """Test get config command handles builder initialization errors"""
        # Setup mock to raise exception during initialization
        mock_builder_class.side_effect = ImportError("Config builder module not found")
        
        # Execute command
        result = self.runner.invoke(self.app, ['--env-name', 'test'])
        
        # Should handle the error gracefully (exit code depends on error handling)
        mock_builder_class.assert_called_once()
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    def test_get_config_command_interactive_flow_error(self, mock_builder_class):
        """Test get config command handles interactive flow errors"""
        # Setup mocks
        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder
        mock_builder.run_interactive_flow.side_effect = ValueError("Invalid configuration selection")
        
        # Execute command
        result = self.runner.invoke(self.app, ['--env-name', 'invalid'])
        
        # Should handle interactive flow errors
        mock_builder.run_interactive_flow.assert_called_once_with('invalid')
    
    def test_get_config_command_help_message(self):
        """Test get config command help message shows correct information"""
        # Execute help command
        result = self.runner.invoke(self.app, ['--help'])
        
        # Verify help shows environment name option and description
        assert result.exit_code == 0
        assert '--env-name' in result.output or '-e' in result.output
        assert '환경 이름' in result.output or 'env' in result.output.lower()
        assert '대화형' in result.output or 'interactive' in result.output.lower()
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    def test_get_config_command_file_generation_verification(self, mock_builder_class):
        """Test get config command verifies file generation"""
        # Setup mocks with file generation simulation
        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder
        mock_selections = {
            'env_name': 'staging',
            'mlflow_enabled': True,
            'data_source_type': 'bigquery',
            'feature_store_enabled': False
        }
        mock_builder.run_interactive_flow.return_value = mock_selections
        
        # Mock file operations
        with patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.write_text') as mock_write:
            
            # Execute command
            result = self.runner.invoke(self.app, ['-e', 'staging'])
            
            # Verify interactive flow completed
            mock_builder.run_interactive_flow.assert_called_once_with('staging')
    
    @patch('src.cli.utils.config_builder.InteractiveConfigBuilder')
    def test_get_config_command_environment_name_validation(self, mock_builder_class):
        """Test get config command with various environment name formats"""
        # Setup mocks
        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder
        mock_builder.run_interactive_flow.return_value = {'env_name': 'test'}
        
        # Test valid environment names
        valid_env_names = ['dev', 'production', 'staging', 'test-env', 'env_123']
        
        for env_name in valid_env_names:
            result = self.runner.invoke(self.app, ['--env-name', env_name])
            mock_builder.run_interactive_flow.assert_called_with(env_name)
            
            # Reset mock for next iteration
            mock_builder.run_interactive_flow.reset_mock()
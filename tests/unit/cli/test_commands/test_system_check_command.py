"""
Unit tests for system_check_command.
Tests system check command functionality with typer and CLI integration.
"""

import pytest
import typer
import yaml
from typer.testing import CliRunner
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.cli.commands.system_check_command import system_check_command


class TestSystemCheckCommandInitialization:
    """Test system check command initialization and parameter handling."""
    
    def test_system_check_command_exists_and_callable(self):
        """Test that system_check_command is a callable function."""
        assert callable(system_check_command)
        assert hasattr(system_check_command, '__call__')


class TestSystemCheckCommandParameterHandling:
    """Test system check command parameter processing and validation."""
    
    @patch('src.cli.commands.system_check_command.Path')
    @patch('src.cli.commands.system_check_command.SystemChecker')
    @patch('src.cli.commands.system_check_command.load_environment')
    def test_system_check_command_parameter_parsing_success(self, mock_load_env, mock_system_checker, mock_path_class):
        """Test successful parameter parsing for system check."""
        runner = CliRunner()
        
        # Create test app for typer
        app = typer.Typer()
        app.command()(system_check_command)
        
        # Mock Path behavior for .env file
        mock_env_file = Mock()
        mock_env_file.exists.return_value = True
        
        # Mock Path behavior for config file
        mock_config_file = Mock()
        mock_config_file.exists.return_value = True
        
        # Mock Path constructor to return appropriate mocks
        def path_side_effect(path_str):
            if '.env' in str(path_str):
                return mock_env_file
            elif 'configs' in str(path_str):
                return mock_config_file
            return Mock()
        
        mock_path_class.side_effect = path_side_effect
        
        # Mock yaml.safe_load
        test_config = {
            "environment": {"name": "test"},
            "mlflow": {"tracking_uri": "http://localhost:5000"}
        }
        
        # Mock SystemChecker
        mock_checker_instance = Mock()
        mock_checker_instance.run_all_checks.return_value = {"status": "success"}
        mock_system_checker.return_value = mock_checker_instance
        
        with patch('builtins.open', mock_yaml_safe_load=True):
            with patch('yaml.safe_load', return_value=test_config):
                # Act
                result = runner.invoke(app, [
                    "--env-name", "test"
                ])
                
                # Assert
                assert result.exit_code == 0
                mock_load_env.assert_called_once_with("test")
                mock_system_checker.assert_called_once_with(test_config, "test")
                mock_checker_instance.run_all_checks.assert_called_once()
                mock_checker_instance.display_results.assert_called_once_with(show_actionable=False)
    
    @patch('src.cli.commands.system_check_command.Path')
    @patch('src.cli.commands.system_check_command.SystemChecker')
    @patch('src.cli.commands.system_check_command.load_environment')
    def test_system_check_command_with_actionable_flag(self, mock_load_env, mock_system_checker, mock_path_class):
        """Test system check command with actionable flag."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(system_check_command)
        
        # Mock Path behavior 
        mock_env_file = Mock()
        mock_env_file.exists.return_value = True
        mock_config_file = Mock() 
        mock_config_file.exists.return_value = True
        
        def path_side_effect(path_str):
            if '.env' in str(path_str):
                return mock_env_file
            elif 'configs' in str(path_str):
                return mock_config_file
            return Mock()
        
        mock_path_class.side_effect = path_side_effect
        
        test_config = {"environment": {"name": "test"}}
        
        # Mock SystemChecker
        mock_checker_instance = Mock()
        mock_checker_instance.run_all_checks.return_value = {"status": "success"}
        mock_system_checker.return_value = mock_checker_instance
        
        with patch('builtins.open'):
            with patch('yaml.safe_load', return_value=test_config):
                # Act
                result = runner.invoke(app, [
                    "--env-name", "test",
                    "--actionable"
                ])
                
                # Assert
                assert result.exit_code == 0
                mock_checker_instance.display_results.assert_called_once_with(show_actionable=True)


class TestSystemCheckCommandFileHandling:
    """Test system check command file handling scenarios."""
    
    @patch('src.cli.commands.system_check_command.Path')
    def test_system_check_command_no_env_file(self, mock_path_class):
        """Test system check when .env file doesn't exist."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(system_check_command)
        
        # Mock Path behavior - env file doesn't exist, config file exists
        mock_env_file = Mock()
        mock_env_file.exists.return_value = False
        mock_config_file = Mock()
        mock_config_file.exists.return_value = True
        
        def path_side_effect(path_str):
            if '.env' in str(path_str):
                return mock_env_file
            elif 'configs' in str(path_str):
                return mock_config_file
            return Mock()
        
        mock_path_class.side_effect = path_side_effect
        
        test_config = {"environment": {"name": "test"}}
        
        with patch('src.cli.commands.system_check_command.SystemChecker') as mock_system_checker:
            mock_checker_instance = Mock()
            mock_checker_instance.run_all_checks.return_value = {"status": "success"}
            mock_system_checker.return_value = mock_checker_instance
            
            with patch('builtins.open'):
                with patch('yaml.safe_load', return_value=test_config):
                    # Act
                    result = runner.invoke(app, [
                        "--env-name", "test"
                    ])
                    
                    # Assert
                    assert result.exit_code == 0
                    # Should still complete successfully even without .env file
    
    @patch('src.cli.commands.system_check_command.Path')
    def test_system_check_command_config_file_not_found(self, mock_path_class):
        """Test system check when config file doesn't exist."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(system_check_command)
        
        # Mock Path behavior - config file doesn't exist
        mock_env_file = Mock()
        mock_env_file.exists.return_value = False
        mock_config_file = Mock()
        mock_config_file.exists.return_value = False  # Config file missing
        
        def path_side_effect(path_str):
            if '.env' in str(path_str):
                return mock_env_file
            elif 'configs' in str(path_str):
                return mock_config_file
            return Mock()
        
        mock_path_class.side_effect = path_side_effect
        
        # Act
        result = runner.invoke(app, [
            "--env-name", "missing"
        ])
        
        # Assert
        assert result.exit_code == 1  # Should exit with error
    
    @patch('src.cli.commands.system_check_command.Path')
    @patch('src.cli.commands.system_check_command.load_environment')
    def test_system_check_command_env_load_failure(self, mock_load_env, mock_path_class):
        """Test system check when environment loading fails."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(system_check_command)
        
        # Mock Path behavior
        mock_env_file = Mock()
        mock_env_file.exists.return_value = True
        mock_config_file = Mock()
        mock_config_file.exists.return_value = True
        
        def path_side_effect(path_str):
            if '.env' in str(path_str):
                return mock_env_file
            elif 'configs' in str(path_str):
                return mock_config_file
            return Mock()
        
        mock_path_class.side_effect = path_side_effect
        
        # Mock environment loading failure
        mock_load_env.side_effect = Exception("Failed to load environment")
        
        test_config = {"environment": {"name": "test"}}
        
        with patch('src.cli.commands.system_check_command.SystemChecker') as mock_system_checker:
            mock_checker_instance = Mock()
            mock_checker_instance.run_all_checks.return_value = {"status": "success"}
            mock_system_checker.return_value = mock_checker_instance
            
            with patch('builtins.open'):
                with patch('yaml.safe_load', return_value=test_config):
                    # Act
                    result = runner.invoke(app, [
                        "--env-name", "test"
                    ])
                    
                    # Assert
                    assert result.exit_code == 0  # Should continue despite env loading failure


class TestSystemCheckCommandErrorHandling:
    """Test system check command error scenarios."""
    
    def test_system_check_command_keyboard_interrupt(self):
        """Test handling of KeyboardInterrupt."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(system_check_command)
        
        with patch('src.cli.commands.system_check_command.Path') as mock_path_class:
            # Mock Path behavior to trigger processing
            mock_env_file = Mock()
            mock_env_file.exists.return_value = False
            mock_config_file = Mock()
            mock_config_file.exists.return_value = True
            
            def path_side_effect(path_str):
                if '.env' in str(path_str):
                    return mock_env_file
                elif 'configs' in str(path_str):
                    return mock_config_file
                return Mock()
            
            mock_path_class.side_effect = path_side_effect
            
            # Mock yaml loading to raise KeyboardInterrupt
            with patch('builtins.open'):
                with patch('yaml.safe_load', side_effect=KeyboardInterrupt("User interrupted")):
                    # Act
                    result = runner.invoke(app, [
                        "--env-name", "test"
                    ])
                    
                    # Assert
                    assert result.exit_code == 1
    
    @patch('src.cli.commands.system_check_command.Path')
    def test_system_check_command_general_exception(self, mock_path_class):
        """Test handling of general exceptions."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(system_check_command)
        
        # Mock Path behavior
        mock_env_file = Mock()
        mock_env_file.exists.return_value = False
        mock_config_file = Mock()
        mock_config_file.exists.return_value = True
        
        def path_side_effect(path_str):
            if '.env' in str(path_str):
                return mock_env_file
            elif 'configs' in str(path_str):
                return mock_config_file
            return Mock()
        
        mock_path_class.side_effect = path_side_effect
        
        with patch('builtins.open'):
            with patch('yaml.safe_load', side_effect=RuntimeError("Unexpected error")):
                # Act
                result = runner.invoke(app, [
                    "--env-name", "test"
                ])
                
                # Assert
                assert result.exit_code == 1


class TestSystemCheckCommandIntegration:
    """Test system check command integration scenarios."""
    
    @patch('src.cli.commands.system_check_command.Path')
    @patch('src.cli.commands.system_check_command.SystemChecker')
    @patch('src.cli.commands.system_check_command.load_environment')
    def test_system_check_command_complete_workflow(self, mock_load_env, mock_system_checker, mock_path_class):
        """Test complete workflow from parameter parsing to result display."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(system_check_command)
        
        # Mock Path behavior
        mock_env_file = Mock()
        mock_env_file.exists.return_value = True
        mock_config_file = Mock()
        mock_config_file.exists.return_value = True
        
        def path_side_effect(path_str):
            if '.env' in str(path_str):
                return mock_env_file
            elif 'configs' in str(path_str):
                return mock_config_file
            return Mock()
        
        mock_path_class.side_effect = path_side_effect
        
        # Mock comprehensive config
        comprehensive_config = {
            "environment": {"name": "production"},
            "mlflow": {"tracking_uri": "http://mlflow-server:5000"},
            "data": {
                "adapters": {
                    "postgresql": {"host": "db.company.com"},
                    "bigquery": {"project_id": "my-project"}
                }
            },
            "feast": {"registry": "s3://feast-registry/"},
            "monitoring": {"enabled": True}
        }
        
        # Mock SystemChecker with detailed results
        mock_checker_instance = Mock()
        mock_check_results = {
            "mlflow": {"status": "success", "details": "Connected"},
            "postgresql": {"status": "success", "details": "Database accessible"},
            "bigquery": {"status": "warning", "details": "Slow response"},
            "feast": {"status": "error", "details": "Registry not accessible"}
        }
        mock_checker_instance.run_all_checks.return_value = mock_check_results
        mock_system_checker.return_value = mock_checker_instance
        
        with patch('builtins.open'):
            with patch('yaml.safe_load', return_value=comprehensive_config):
                # Act
                result = runner.invoke(app, [
                    "--env-name", "production",
                    "--actionable"
                ])
                
                # Assert - verify entire workflow
                assert result.exit_code == 0
                
                # Verify environment loading
                mock_load_env.assert_called_once_with("production")
                
                # Verify SystemChecker creation and execution
                mock_system_checker.assert_called_once_with(comprehensive_config, "production")
                mock_checker_instance.run_all_checks.assert_called_once()
                mock_checker_instance.display_results.assert_called_once_with(show_actionable=True)
    
    @patch('src.cli.commands.system_check_command.Path') 
    @patch('src.cli.commands.system_check_command.SystemChecker')
    def test_system_check_command_development_workflow(self, mock_system_checker, mock_path_class):
        """Test system check command with typical development workflow."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(system_check_command)
        
        # Mock Path behavior for development setup
        mock_env_file = Mock()
        mock_env_file.exists.return_value = True
        mock_config_file = Mock()
        mock_config_file.exists.return_value = True
        
        def path_side_effect(path_str):
            if '.env' in str(path_str):
                return mock_env_file
            elif 'configs' in str(path_str):
                return mock_config_file
            return Mock()
        
        mock_path_class.side_effect = path_side_effect
        
        # Mock development config
        dev_config = {
            "environment": {"name": "development"},
            "mlflow": {"tracking_uri": "file:///tmp/mlruns"},
            "data": {
                "adapters": {
                    "postgresql": {"host": "localhost", "port": 5432}
                }
            }
        }
        
        # Mock SystemChecker for development
        mock_checker_instance = Mock()
        mock_dev_results = {
            "mlflow": {"status": "success", "details": "Local file store"},
            "postgresql": {"status": "success", "details": "Local database"}
        }
        mock_checker_instance.run_all_checks.return_value = mock_dev_results
        mock_system_checker.return_value = mock_checker_instance
        
        with patch('src.cli.commands.system_check_command.load_environment') as mock_load_env:
            with patch('builtins.open'):
                with patch('yaml.safe_load', return_value=dev_config):
                    # Act
                    result = runner.invoke(app, [
                        "--env-name", "dev"
                    ])
                    
                    # Assert
                    assert result.exit_code == 0
                    mock_system_checker.assert_called_once_with(dev_config, "dev")


class TestSystemCheckCommandEdgeCases:
    """Test system check command edge cases and boundary conditions."""
    
    @patch('src.cli.commands.system_check_command.Path')
    @patch('src.cli.commands.system_check_command.SystemChecker')
    def test_system_check_command_empty_config(self, mock_system_checker, mock_path_class):
        """Test system check with empty configuration."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(system_check_command)
        
        # Mock Path behavior
        mock_env_file = Mock()
        mock_env_file.exists.return_value = False
        mock_config_file = Mock()
        mock_config_file.exists.return_value = True
        
        def path_side_effect(path_str):
            if '.env' in str(path_str):
                return mock_env_file
            elif 'configs' in str(path_str):
                return mock_config_file
            return Mock()
        
        mock_path_class.side_effect = path_side_effect
        
        # Empty config
        empty_config = {}
        
        mock_checker_instance = Mock()
        mock_checker_instance.run_all_checks.return_value = {}
        mock_system_checker.return_value = mock_checker_instance
        
        with patch('builtins.open'):
            with patch('yaml.safe_load', return_value=empty_config):
                # Act
                result = runner.invoke(app, [
                    "--env-name", "minimal"
                ])
                
                # Assert
                assert result.exit_code == 0
                mock_system_checker.assert_called_once_with(empty_config, "minimal")
    
    @patch('src.cli.commands.system_check_command.Path')
    def test_system_check_command_long_env_name(self, mock_path_class):
        """Test system check with very long environment name."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(system_check_command)
        
        long_env_name = "very_long_environment_name_for_testing_edge_cases_and_boundary_conditions"
        
        # Mock Path behavior
        mock_env_file = Mock()
        mock_env_file.exists.return_value = False
        mock_config_file = Mock()
        mock_config_file.exists.return_value = True
        
        def path_side_effect(path_str):
            if '.env' in str(path_str):
                return mock_env_file
            elif 'configs' in str(path_str):
                return mock_config_file
            return Mock()
        
        mock_path_class.side_effect = path_side_effect
        
        test_config = {"environment": {"name": long_env_name}}
        
        with patch('src.cli.commands.system_check_command.SystemChecker') as mock_system_checker:
            mock_checker_instance = Mock()
            mock_checker_instance.run_all_checks.return_value = {"status": "success"}
            mock_system_checker.return_value = mock_checker_instance
            
            with patch('builtins.open'):
                with patch('yaml.safe_load', return_value=test_config):
                    # Act
                    result = runner.invoke(app, [
                        "--env-name", long_env_name
                    ])
                    
                    # Assert - should handle long environment names without issues
                    assert result.exit_code == 0
                    mock_system_checker.assert_called_once_with(test_config, long_env_name)
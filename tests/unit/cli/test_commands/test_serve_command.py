"""
Unit tests for serve_command.
Tests API serving command functionality with typer and CLI integration.
"""

import pytest
import typer
from typer.testing import CliRunner
from unittest.mock import Mock, MagicMock, patch

from src.cli.commands.serve_command import serve_api_command


class TestServeCommandInitialization:
    """Test serve command initialization and parameter handling."""
    
    def test_serve_api_command_exists_and_callable(self):
        """Test that serve_api_command is a callable function."""
        assert callable(serve_api_command)
        assert hasattr(serve_api_command, '__call__')


class TestServeCommandParameterHandling:
    """Test serve command parameter processing and validation."""
    
    def test_serve_api_command_parameter_parsing_success(self):
        """Test successful parameter parsing for API serving."""
        runner = CliRunner()
        
        # Create test app for typer
        app = typer.Typer()
        app.command()(serve_api_command)
        
        with patch('src.cli.commands.serve_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.serve_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.serve_command.setup_logging') as mock_setup_logging:
                    with patch('src.cli.commands.serve_command.run_api_server') as mock_run_server:
                        
                        # Mock dependencies
                        mock_config_data = {"environment": {"name": "test"}}
                        mock_settings = Mock()
                        mock_load_config.return_value = mock_config_data
                        mock_create_settings.return_value = mock_settings
                        
                        # Act
                        result = runner.invoke(app, [
                            "--run-id", "test_model_123",
                            "--config-path", "configs/test.yaml"
                        ])
                        
                        # Assert
                        assert result.exit_code == 0
                        mock_load_config.assert_called_once_with(config_path="configs/test.yaml")
                        mock_create_settings.assert_called_once_with(mock_config_data)
                        mock_setup_logging.assert_called_once_with(mock_settings)
                        mock_run_server.assert_called_once_with(
                            settings=mock_settings,
                            run_id="test_model_123",
                            host="0.0.0.0",  # Default host
                            port=8000  # Default port
                        )
    
    def test_serve_api_command_with_custom_host_port(self):
        """Test serve API command with custom host and port."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(serve_api_command)
        
        with patch('src.cli.commands.serve_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.serve_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.serve_command.setup_logging'):
                    with patch('src.cli.commands.serve_command.run_api_server') as mock_run_server:
                        
                        mock_config_data = {"environment": {"name": "prod"}}
                        mock_settings = Mock()
                        mock_load_config.return_value = mock_config_data
                        mock_create_settings.return_value = mock_settings
                        
                        # Act
                        result = runner.invoke(app, [
                            "--run-id", "prod_model_v1",
                            "--config-path", "configs/production.yaml",
                            "--host", "localhost",
                            "--port", "9000"
                        ])
                        
                        # Assert
                        assert result.exit_code == 0
                        mock_run_server.assert_called_once_with(
                            settings=mock_settings,
                            run_id="prod_model_v1",
                            host="localhost",
                            port=9000
                        )


class TestServeCommandLogging:
    """Test serve command logging functionality."""
    
    def test_serve_api_command_logging_integration(self):
        """Test that logging is properly configured and used."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(serve_api_command)
        
        with patch('src.cli.commands.serve_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.serve_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.serve_command.setup_logging') as mock_setup_logging:
                    with patch('src.cli.commands.serve_command.run_api_server'):
                        with patch('src.cli.commands.serve_command.logger') as mock_logger:
                            
                            mock_config_data = {"environment": {"name": "test"}}
                            mock_settings = Mock()
                            mock_load_config.return_value = mock_config_data
                            mock_create_settings.return_value = mock_settings
                            
                            # Act
                            result = runner.invoke(app, [
                                "--run-id", "test_model_123",
                                "--config-path", "configs/test.yaml",
                                "--host", "0.0.0.0",
                                "--port", "8080"
                            ])
                            
                            # Assert
                            assert result.exit_code == 0
                            mock_setup_logging.assert_called_once_with(mock_settings)
                            
                            # Check that key info is logged
                            mock_logger.info.assert_any_call("Config: configs/test.yaml")
                            mock_logger.info.assert_any_call("Run ID: test_model_123")
                            mock_logger.info.assert_any_call("Server: 0.0.0.0:8080")
                            mock_logger.info.assert_any_call("API Documentation: http://0.0.0.0:8080/docs")


class TestServeCommandErrorHandling:
    """Test serve command error scenarios."""
    
    def test_serve_api_command_file_not_found_error(self):
        """Test handling of FileNotFoundError."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(serve_api_command)
        
        with patch('src.cli.commands.serve_command.load_config_files') as mock_load_config:
            # Mock FileNotFoundError
            mock_load_config.side_effect = FileNotFoundError("Config file not found")
            
            # Act
            result = runner.invoke(app, [
                "--run-id", "test_model_123",
                "--config-path", "nonexistent_config.yaml"
            ])
            
            # Assert
            assert result.exit_code == 1
    
    def test_serve_api_command_value_error(self):
        """Test handling of ValueError during execution."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(serve_api_command)
        
        with patch('src.cli.commands.serve_command.load_config_files') as mock_load_config:
            # Mock ValueError
            mock_load_config.side_effect = ValueError("Invalid configuration")
            
            # Act
            result = runner.invoke(app, [
                "--run-id", "test_model_123",
                "--config-path", "configs/test.yaml"
            ])
            
            # Assert
            assert result.exit_code == 1
    
    def test_serve_api_command_general_exception(self):
        """Test handling of general exceptions."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(serve_api_command)
        
        with patch('src.cli.commands.serve_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.serve_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.serve_command.setup_logging'):
                    with patch('src.cli.commands.serve_command.run_api_server') as mock_run_server:
                        
                        mock_config_data = {"environment": {"name": "test"}}
                        mock_settings = Mock()
                        mock_load_config.return_value = mock_config_data
                        mock_create_settings.return_value = mock_settings
                        
                        # Mock RuntimeError during server startup
                        mock_run_server.side_effect = RuntimeError("API server startup failed")
                        
                        # Act
                        result = runner.invoke(app, [
                            "--run-id", "test_model_123",
                            "--config-path", "configs/test.yaml"
                        ])
                        
                        # Assert
                        assert result.exit_code == 1


class TestServeCommandPortValidation:
    """Test serve command port validation and edge cases."""
    
    def test_serve_api_command_default_host_port(self):
        """Test serve API command with default host and port values."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(serve_api_command)
        
        with patch('src.cli.commands.serve_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.serve_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.serve_command.setup_logging'):
                    with patch('src.cli.commands.serve_command.run_api_server') as mock_run_server:
                        
                        mock_config_data = {"environment": {"name": "test"}}
                        mock_settings = Mock()
                        mock_load_config.return_value = mock_config_data
                        mock_create_settings.return_value = mock_settings
                        
                        # Act - only required parameters, defaults should be used
                        result = runner.invoke(app, [
                            "--run-id", "default_test_model",
                            "--config-path", "configs/default.yaml"
                        ])
                        
                        # Assert
                        assert result.exit_code == 0
                        mock_run_server.assert_called_once_with(
                            settings=mock_settings,
                            run_id="default_test_model",
                            host="0.0.0.0",  # Default
                            port=8000  # Default
                        )
    
    def test_serve_api_command_high_port_number(self):
        """Test serve API command with high port number."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(serve_api_command)
        
        with patch('src.cli.commands.serve_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.serve_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.serve_command.setup_logging'):
                    with patch('src.cli.commands.serve_command.run_api_server') as mock_run_server:
                        
                        mock_config_data = {"environment": {"name": "test"}}
                        mock_settings = Mock()
                        mock_load_config.return_value = mock_config_data
                        mock_create_settings.return_value = mock_settings
                        
                        # Act
                        result = runner.invoke(app, [
                            "--run-id", "high_port_test_model",
                            "--config-path", "configs/test.yaml",
                            "--host", "127.0.0.1",
                            "--port", "65535"  # Maximum valid port
                        ])
                        
                        # Assert
                        assert result.exit_code == 0
                        mock_run_server.assert_called_once_with(
                            settings=mock_settings,
                            run_id="high_port_test_model",
                            host="127.0.0.1",
                            port=65535
                        )


class TestServeCommandIntegration:
    """Test serve command integration scenarios."""
    
    def test_serve_api_command_complete_workflow(self):
        """Test complete workflow from parameter parsing to server startup."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(serve_api_command)
        
        with patch('src.cli.commands.serve_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.serve_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.serve_command.setup_logging') as mock_setup_logging:
                    with patch('src.cli.commands.serve_command.run_api_server') as mock_run_server:
                        
                        # Mock complete workflow
                        mock_config_data = {"environment": {"name": "production"}, "mlflow": {"tracking_uri": "http://mlflow:5000"}}
                        mock_settings = Mock()
                        mock_load_config.return_value = mock_config_data
                        mock_create_settings.return_value = mock_settings
                        
                        # Act
                        result = runner.invoke(app, [
                            "--run-id", "production_model_v3.0",
                            "--config-path", "configs/production.yaml",
                            "--host", "production-api.company.com",
                            "--port", "443"
                        ])
                        
                        # Assert - verify entire workflow
                        assert result.exit_code == 0
                        
                        # Verify config loading
                        mock_load_config.assert_called_once_with(config_path="configs/production.yaml")
                        
                        # Verify settings creation
                        mock_create_settings.assert_called_once_with(mock_config_data)
                        
                        # Verify logging setup
                        mock_setup_logging.assert_called_once_with(mock_settings)
                        
                        # Verify server startup with correct parameters
                        mock_run_server.assert_called_once_with(
                            settings=mock_settings,
                            run_id="production_model_v3.0",
                            host="production-api.company.com",
                            port=443
                        )
    
    def test_serve_api_command_development_setup(self):
        """Test serve API command with typical development setup."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(serve_api_command)
        
        with patch('src.cli.commands.serve_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.serve_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.serve_command.setup_logging'):
                    with patch('src.cli.commands.serve_command.run_api_server') as mock_run_server:
                        with patch('src.cli.commands.serve_command.logger') as mock_logger:
                            
                            # Mock development configuration
                            mock_config_data = {"environment": {"name": "development"}}
                            mock_settings = Mock()
                            mock_load_config.return_value = mock_config_data
                            mock_create_settings.return_value = mock_settings
                            
                            # Act - typical development setup
                            result = runner.invoke(app, [
                                "--run-id", "dev_model_latest",
                                "--config-path", "configs/dev.yaml",
                                "--host", "localhost",
                                "--port", "8080"
                            ])
                            
                            # Assert
                            assert result.exit_code == 0
                            
                            # Verify development-specific logging
                            mock_logger.info.assert_any_call("Config: configs/dev.yaml")
                            mock_logger.info.assert_any_call("Run ID: dev_model_latest")
                            mock_logger.info.assert_any_call("Server: localhost:8080")
                            mock_logger.info.assert_any_call("API Documentation: http://localhost:8080/docs")


class TestServeCommandEdgeCases:
    """Test serve command edge cases and boundary conditions."""
    
    def test_serve_api_command_minimal_required_params(self):
        """Test serve command with only required parameters."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(serve_api_command)
        
        with patch('src.cli.commands.serve_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.serve_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.serve_command.setup_logging'):
                    with patch('src.cli.commands.serve_command.run_api_server') as mock_run_server:
                        
                        mock_config_data = {"environment": {"name": "minimal"}}
                        mock_settings = Mock()
                        mock_load_config.return_value = mock_config_data
                        mock_create_settings.return_value = mock_settings
                        
                        # Act - only required parameters
                        result = runner.invoke(app, [
                            "--run-id", "minimal_model",
                            "--config-path", "configs/minimal.yaml"
                        ])
                        
                        # Assert
                        assert result.exit_code == 0
                        mock_run_server.assert_called_once_with(
                            settings=mock_settings,
                            run_id="minimal_model",
                            host="0.0.0.0",  # Default value
                            port=8000  # Default value
                        )
    
    def test_serve_api_command_long_run_id(self):
        """Test serve API command with very long run ID."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(serve_api_command)
        
        # Create a very long run ID (typical MLflow run IDs can be quite long)
        long_run_id = "a" * 64 + "_very_long_model_name_with_timestamp_20240101120000"
        
        with patch('src.cli.commands.serve_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.serve_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.serve_command.setup_logging'):
                    with patch('src.cli.commands.serve_command.run_api_server') as mock_run_server:
                        
                        mock_config_data = {"environment": {"name": "test"}}
                        mock_settings = Mock()
                        mock_load_config.return_value = mock_config_data
                        mock_create_settings.return_value = mock_settings
                        
                        # Act
                        result = runner.invoke(app, [
                            "--run-id", long_run_id,
                            "--config-path", "configs/test.yaml"
                        ])
                        
                        # Assert - should handle long run IDs without issues
                        assert result.exit_code == 0
                        mock_run_server.assert_called_once_with(
                            settings=mock_settings,
                            run_id=long_run_id,
                            host="0.0.0.0",
                            port=8000
                        )
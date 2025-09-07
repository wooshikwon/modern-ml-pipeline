"""
Unit tests for inference_command.
Tests batch inference command functionality with typer and CLI integration.
"""

import pytest
import json
import typer
from typer.testing import CliRunner
from unittest.mock import Mock, MagicMock, patch

from src.cli.commands.inference_command import batch_inference_command


class TestInferenceCommandInitialization:
    """Test inference command initialization and parameter handling."""
    
    def test_batch_inference_command_exists_and_callable(self):
        """Test that batch_inference_command is a callable function."""
        assert callable(batch_inference_command)
        assert hasattr(batch_inference_command, '__call__')


class TestInferenceCommandParameterHandling:
    """Test inference command parameter processing and validation."""
    
    def test_batch_inference_command_parameter_parsing_success(self):
        """Test successful parameter parsing for batch inference."""
        runner = CliRunner()
        
        # Create test app for typer
        app = typer.Typer()
        app.command()(batch_inference_command)
        
        with patch('src.cli.commands.inference_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.inference_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.inference_command.setup_logging') as mock_setup_logging:
                    with patch('src.cli.commands.inference_command.run_inference_pipeline') as mock_run_inference:
                        
                        # Mock dependencies
                        mock_config_data = {"environment": {"name": "test"}}
                        mock_settings = Mock()
                        mock_load_config.return_value = mock_config_data
                        mock_create_settings.return_value = mock_settings
                        
                        # Act
                        result = runner.invoke(app, [
                            "--run-id", "test_run_123",
                            "--config-path", "configs/test.yaml",
                            "--data-path", "data/inference.csv"
                        ])
                        
                        # Assert
                        assert result.exit_code == 0
                        mock_load_config.assert_called_once_with(config_path="configs/test.yaml")
                        mock_create_settings.assert_called_once_with(mock_config_data)
                        mock_setup_logging.assert_called_once_with(mock_settings)
                        mock_run_inference.assert_called_once_with(
                            settings=mock_settings,
                            run_id="test_run_123",
                            data_path="data/inference.csv",
                            context_params={}
                        )
    
    def test_batch_inference_command_with_context_params(self):
        """Test batch inference command with JSON context parameters."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)
        
        test_params = {"date": "2024-01-01", "version": "v1.0"}
        params_json = json.dumps(test_params)
        
        with patch('src.cli.commands.inference_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.inference_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.inference_command.setup_logging'):
                    with patch('src.cli.commands.inference_command.run_inference_pipeline') as mock_run_inference:
                        
                        mock_config_data = {"environment": {"name": "test"}}
                        mock_settings = Mock()
                        mock_load_config.return_value = mock_config_data
                        mock_create_settings.return_value = mock_settings
                        
                        # Act
                        result = runner.invoke(app, [
                            "--run-id", "test_run_123",
                            "--config-path", "configs/test.yaml",
                            "--data-path", "data/inference.sql",
                            "--params", params_json
                        ])
                        
                        # Assert
                        assert result.exit_code == 0
                        mock_run_inference.assert_called_once_with(
                            settings=mock_settings,
                            run_id="test_run_123",
                            data_path="data/inference.sql",
                            context_params=test_params
                        )


class TestInferenceCommandLogging:
    """Test inference command logging functionality."""
    
    def test_batch_inference_command_logging_integration(self):
        """Test that logging is properly configured and used."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)
        
        with patch('src.cli.commands.inference_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.inference_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.inference_command.setup_logging') as mock_setup_logging:
                    with patch('src.cli.commands.inference_command.run_inference_pipeline'):
                        with patch('src.cli.commands.inference_command.logger') as mock_logger:
                            
                            mock_config_data = {"environment": {"name": "test"}}
                            mock_settings = Mock()
                            mock_load_config.return_value = mock_config_data
                            mock_create_settings.return_value = mock_settings
                            
                            # Act
                            result = runner.invoke(app, [
                                "--run-id", "test_run_123",
                                "--config-path", "configs/test.yaml",
                                "--data-path", "data/inference.csv"
                            ])
                            
                            # Assert
                            assert result.exit_code == 0
                            mock_setup_logging.assert_called_once_with(mock_settings)
                            
                            # Check that key info is logged
                            mock_logger.info.assert_any_call("Config: configs/test.yaml")
                            mock_logger.info.assert_any_call("Data: data/inference.csv")
                            mock_logger.info.assert_any_call("Run ID: test_run_123")
                            mock_logger.info.assert_any_call("✅ 배치 추론이 성공적으로 완료되었습니다.")


class TestInferenceCommandErrorHandling:
    """Test inference command error scenarios."""
    
    def test_batch_inference_command_file_not_found_error(self):
        """Test handling of FileNotFoundError."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)
        
        with patch('src.cli.commands.inference_command.load_config_files') as mock_load_config:
            # Mock FileNotFoundError
            mock_load_config.side_effect = FileNotFoundError("Config file not found")
            
            # Act
            result = runner.invoke(app, [
                "--run-id", "test_run_123",
                "--config-path", "nonexistent_config.yaml",
                "--data-path", "data/inference.csv"
            ])
            
            # Assert
            assert result.exit_code == 1
    
    def test_batch_inference_command_value_error(self):
        """Test handling of ValueError during execution."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)
        
        with patch('src.cli.commands.inference_command.load_config_files') as mock_load_config:
            # Mock ValueError
            mock_load_config.side_effect = ValueError("Invalid configuration")
            
            # Act
            result = runner.invoke(app, [
                "--run-id", "test_run_123",
                "--config-path", "configs/test.yaml",
                "--data-path", "data/inference.csv"
            ])
            
            # Assert
            assert result.exit_code == 1
    
    def test_batch_inference_command_general_exception(self):
        """Test handling of general exceptions."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)
        
        with patch('src.cli.commands.inference_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.inference_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.inference_command.setup_logging'):
                    with patch('src.cli.commands.inference_command.run_inference_pipeline') as mock_run_inference:
                        
                        mock_config_data = {"environment": {"name": "test"}}
                        mock_settings = Mock()
                        mock_load_config.return_value = mock_config_data
                        mock_create_settings.return_value = mock_settings
                        
                        # Mock RuntimeError during pipeline execution
                        mock_run_inference.side_effect = RuntimeError("Inference pipeline execution failed")
                        
                        # Act
                        result = runner.invoke(app, [
                            "--run-id", "test_run_123",
                            "--config-path", "configs/test.yaml",
                            "--data-path", "data/inference.csv"
                        ])
                        
                        # Assert
                        assert result.exit_code == 1


class TestInferenceCommandJSONParameterHandling:
    """Test JSON parameter parsing in inference command."""
    
    def test_batch_inference_command_invalid_json_handling(self):
        """Test handling of invalid JSON in context parameters."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)
        
        with patch('src.cli.commands.inference_command.load_config_files'):
            # Act - provide invalid JSON
            result = runner.invoke(app, [
                "--run-id", "test_run_123",
                "--config-path", "configs/test.yaml",
                "--data-path", "data/inference.csv",
                "--params", "invalid_json_string"
            ])
            
            # Assert
            assert result.exit_code == 1  # Should fail due to JSON parsing error
    
    def test_batch_inference_command_empty_json_params(self):
        """Test handling of empty JSON parameters."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)
        
        with patch('src.cli.commands.inference_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.inference_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.inference_command.setup_logging'):
                    with patch('src.cli.commands.inference_command.run_inference_pipeline') as mock_run_inference:
                        
                        mock_config_data = {"environment": {"name": "test"}}
                        mock_settings = Mock()
                        mock_load_config.return_value = mock_config_data
                        mock_create_settings.return_value = mock_settings
                        
                        # Act
                        result = runner.invoke(app, [
                            "--run-id", "test_run_123", 
                            "--config-path", "configs/test.yaml",
                            "--data-path", "data/inference.csv",
                            "--params", "{}"
                        ])
                        
                        # Assert
                        assert result.exit_code == 0
                        mock_run_inference.assert_called_once_with(
                            settings=mock_settings,
                            run_id="test_run_123",
                            data_path="data/inference.csv",
                            context_params={}
                        )
    
    def test_batch_inference_command_complex_json_params(self):
        """Test handling of complex JSON parameters."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)
        
        complex_params = {
            "date_range": {"start": "2024-01-01", "end": "2024-01-31"},
            "filters": ["category_a", "category_b"],
            "options": {"batch_size": 1000, "use_cache": True}
        }
        params_json = json.dumps(complex_params)
        
        with patch('src.cli.commands.inference_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.inference_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.inference_command.setup_logging'):
                    with patch('src.cli.commands.inference_command.run_inference_pipeline') as mock_run_inference:
                        
                        mock_config_data = {"environment": {"name": "test"}}
                        mock_settings = Mock()
                        mock_load_config.return_value = mock_config_data
                        mock_create_settings.return_value = mock_settings
                        
                        # Act
                        result = runner.invoke(app, [
                            "--run-id", "test_run_123",
                            "--config-path", "configs/test.yaml", 
                            "--data-path", "data/inference.sql.j2",
                            "--params", params_json
                        ])
                        
                        # Assert
                        assert result.exit_code == 0
                        mock_run_inference.assert_called_once_with(
                            settings=mock_settings,
                            run_id="test_run_123",
                            data_path="data/inference.sql.j2",
                            context_params=complex_params
                        )


class TestInferenceCommandIntegration:
    """Test inference command integration scenarios."""
    
    def test_batch_inference_command_complete_workflow(self):
        """Test complete workflow from parameter parsing to pipeline execution."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)
        
        test_params = {"filter_date": "2024-01-01"}
        params_json = json.dumps(test_params)
        
        with patch('src.cli.commands.inference_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.inference_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.inference_command.setup_logging') as mock_setup_logging:
                    with patch('src.cli.commands.inference_command.run_inference_pipeline') as mock_run_inference:
                        
                        # Mock complete workflow
                        mock_config_data = {"environment": {"name": "production"}}
                        mock_settings = Mock()
                        mock_load_config.return_value = mock_config_data
                        mock_create_settings.return_value = mock_settings
                        
                        # Act
                        result = runner.invoke(app, [
                            "--run-id", "production_model_v2.1",
                            "--config-path", "configs/production.yaml",
                            "--data-path", "queries/inference_batch.sql.j2",
                            "--params", params_json
                        ])
                        
                        # Assert - verify entire workflow
                        assert result.exit_code == 0
                        
                        # Verify config loading
                        mock_load_config.assert_called_once_with(config_path="configs/production.yaml")
                        
                        # Verify settings creation
                        mock_create_settings.assert_called_once_with(mock_config_data)
                        
                        # Verify logging setup
                        mock_setup_logging.assert_called_once_with(mock_settings)
                        
                        # Verify pipeline execution with correct parameters
                        mock_run_inference.assert_called_once_with(
                            settings=mock_settings,
                            run_id="production_model_v2.1",
                            data_path="queries/inference_batch.sql.j2",
                            context_params=test_params
                        )


class TestInferenceCommandEdgeCases:
    """Test inference command edge cases and boundary conditions."""
    
    def test_batch_inference_command_minimal_required_params(self):
        """Test inference command with only required parameters."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)
        
        with patch('src.cli.commands.inference_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.inference_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.inference_command.setup_logging'):
                    with patch('src.cli.commands.inference_command.run_inference_pipeline') as mock_run_inference:
                        
                        mock_config_data = {"environment": {"name": "test"}}
                        mock_settings = Mock()
                        mock_load_config.return_value = mock_config_data
                        mock_create_settings.return_value = mock_settings
                        
                        # Act - only required parameters
                        result = runner.invoke(app, [
                            "--run-id", "minimal_test",
                            "--config-path", "configs/minimal.yaml",
                            "--data-path", "data/minimal.csv"
                        ])
                        
                        # Assert
                        assert result.exit_code == 0
                        mock_run_inference.assert_called_once_with(
                            settings=mock_settings,
                            run_id="minimal_test",
                            data_path="data/minimal.csv",
                            context_params={}  # Empty dict when no params provided
                        )
    
    def test_batch_inference_command_empty_string_params(self):
        """Test handling of empty string in context parameters."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(batch_inference_command)
        
        with patch('src.cli.commands.inference_command.load_config_files') as mock_load_config:
            with patch('src.cli.commands.inference_command.create_settings_for_inference') as mock_create_settings:
                with patch('src.cli.commands.inference_command.setup_logging'):
                    with patch('src.cli.commands.inference_command.run_inference_pipeline') as mock_run_inference:
                        
                        mock_config_data = {"environment": {"name": "test"}}
                        mock_settings = Mock()
                        mock_load_config.return_value = mock_config_data
                        mock_create_settings.return_value = mock_settings
                        
                        # Act - empty string params
                        result = runner.invoke(app, [
                            "--run-id", "test_run",
                            "--config-path", "configs/test.yaml",
                            "--data-path", "data/test.csv",
                            "--params", ""  # Empty string
                        ])
                        
                        # Assert - should handle empty string gracefully
                        # Empty string should be treated as None, resulting in empty dict
                        assert result.exit_code == 0
                        mock_run_inference.assert_called_once_with(
                            settings=mock_settings,
                            run_id="test_run", 
                            data_path="data/test.csv",
                            context_params={}
                        )
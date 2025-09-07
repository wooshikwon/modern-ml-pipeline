"""
Unit tests for train_command.
Tests training command functionality with typer and CLI integration.
"""

import pytest
import json
import typer
from typer.testing import CliRunner
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.cli.commands.train_command import train_command


class TestTrainCommandInitialization:
    """Test train command initialization and parameter handling."""
    
    def test_train_command_exists_and_callable(self):
        """Test that train_command is a callable function."""
        assert callable(train_command)
        assert hasattr(train_command, '__call__')


class TestTrainCommandParameterHandling:
    """Test train command parameter processing and validation."""
    
    def test_train_command_parameter_parsing_success(self):
        """Test successful parameter parsing."""
        runner = CliRunner()
        
        # Create test app for typer
        app = typer.Typer()
        app.command()(train_command)
        
        with patch('src.cli.commands.train_command.load_settings') as mock_load_settings:
            with patch('src.cli.commands.train_command.setup_logging') as mock_setup_logging:
                with patch('src.cli.commands.train_command.run_train_pipeline') as mock_run_pipeline:
                    
                    # Mock settings object
                    mock_settings = Mock()
                    mock_settings.recipe.data.loader.source_uri = None
                    mock_settings.recipe.model.computed = {"run_name": "test_run"}
                    mock_load_settings.return_value = mock_settings
                    
                    # Act
                    result = runner.invoke(app, [
                        "--recipe-path", "test_recipe.yaml",
                        "--config-path", "test_config.yaml", 
                        "--data-path", "test_data.csv"
                    ])
                    
                    # Assert
                    assert result.exit_code == 0
                    mock_load_settings.assert_called_once_with("test_recipe.yaml", "test_config.yaml")
                    mock_setup_logging.assert_called_once_with(mock_settings)
                    mock_run_pipeline.assert_called_once_with(settings=mock_settings, context_params=None)
                    assert mock_settings.recipe.data.loader.source_uri == "test_data.csv"
    
    def test_train_command_with_context_params(self):
        """Test train command with JSON context parameters."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(train_command)
        
        test_params = {"date": "2024-01-01", "version": "v1.0"}
        params_json = json.dumps(test_params)
        
        with patch('src.cli.commands.train_command.load_settings') as mock_load_settings:
            with patch('src.cli.commands.train_command.setup_logging'):
                with patch('src.cli.commands.train_command.run_train_pipeline') as mock_run_pipeline:
                    
                    mock_settings = Mock()
                    mock_settings.recipe.data.loader.source_uri = None
                    mock_settings.recipe.model.computed = {"run_name": "test_run"}
                    mock_load_settings.return_value = mock_settings
                    
                    # Act
                    result = runner.invoke(app, [
                        "--recipe-path", "test_recipe.yaml",
                        "--config-path", "test_config.yaml",
                        "--data-path", "test_data.csv",
                        "--params", params_json
                    ])
                    
                    # Assert
                    assert result.exit_code == 0
                    mock_run_pipeline.assert_called_once_with(settings=mock_settings, context_params=test_params)


class TestTrainCommandJinjaTemplateHandling:
    """Test train command Jinja template processing."""
    
    def test_train_command_jinja_template_success(self):
        """Test successful Jinja template processing."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(train_command)
        
        test_params = {"date": "2024-01-01"}
        params_json = json.dumps(test_params)
        rendered_sql = "SELECT * FROM table WHERE date = '2024-01-01'"
        
        with patch('src.cli.commands.train_command.load_settings') as mock_load_settings:
            with patch('src.cli.commands.train_command.setup_logging'):
                with patch('src.cli.commands.train_command.run_train_pipeline'):
                    with patch('src.cli.commands.train_command.logger') as mock_logger:
                        with patch('pathlib.Path') as mock_path_class:
                            with patch('src.utils.system.templating_utils.render_template_from_string') as mock_render_template:
                            
                                # Mock Path behavior
                                mock_template_path = Mock()
                                mock_template_path.exists.return_value = True
                                mock_template_path.read_text.return_value = "SELECT * FROM table WHERE date = '{{ date }}'"
                                mock_path_class.return_value = mock_template_path
                                
                                # Mock template rendering
                                mock_render_template.return_value = rendered_sql
                                
                                mock_settings = Mock()
                                mock_settings.recipe.data.loader.source_uri = None
                                mock_settings.recipe.model.computed = {"run_name": "test_run"}
                                mock_load_settings.return_value = mock_settings
                                
                                # Act
                                result = runner.invoke(app, [
                                    "--recipe-path", "test_recipe.yaml",
                                    "--config-path", "test_config.yaml",
                                    "--data-path", "query_template.sql.j2",
                                    "--params", params_json
                                ])
                                
                                # Assert
                                assert result.exit_code == 0
                                mock_render_template.assert_called_once_with(
                                    "SELECT * FROM table WHERE date = '{{ date }}'", 
                                    test_params
                                )
                                assert mock_settings.recipe.data.loader.source_uri == rendered_sql
                                mock_logger.info.assert_any_call("✅ Jinja 템플릿 렌더링 성공: query_template.sql.j2")
    
    def test_train_command_jinja_template_no_params_error(self):
        """Test error when Jinja template provided without parameters."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(train_command)
        
        with patch('src.cli.commands.train_command.load_settings'):
            with patch('src.cli.commands.train_command.setup_logging'):
                with patch('pathlib.Path') as mock_path_class:
                    # Mock Path behavior
                    mock_template_path = Mock()
                    mock_template_path.exists.return_value = True
                    mock_path_class.return_value = mock_template_path
                    
                    # Act
                    result = runner.invoke(app, [
                        "--recipe-path", "test_recipe.yaml",
                        "--config-path", "test_config.yaml",
                        "--data-path", "query_template.sql.j2"
                        # No --params provided
                    ])
                    
                    # Assert
                    assert result.exit_code == 1  # Should exit with error
    
    def test_train_command_template_file_not_found(self):
        """Test error when template file does not exist."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(train_command)
        
        test_params = {"date": "2024-01-01"}
        params_json = json.dumps(test_params)
        
        with patch('src.cli.commands.train_command.load_settings'):
            with patch('src.cli.commands.train_command.setup_logging'):
                with patch('pathlib.Path') as mock_path_class:
                    # Mock Path behavior - file doesn't exist
                    mock_template_path = Mock()
                    mock_template_path.exists.return_value = False
                    mock_path_class.return_value = mock_template_path
                    
                    # Act
                    result = runner.invoke(app, [
                        "--recipe-path", "test_recipe.yaml",
                        "--config-path", "test_config.yaml", 
                        "--data-path", "nonexistent_template.sql.j2",
                        "--params", params_json
                    ])
                    
                    # Assert
                    assert result.exit_code == 1  # Should exit with error


class TestTrainCommandErrorHandling:
    """Test train command error scenarios."""
    
    def test_train_command_file_not_found_error(self):
        """Test handling of FileNotFoundError."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(train_command)
        
        with patch('src.cli.commands.train_command.load_settings') as mock_load_settings:
            # Mock FileNotFoundError
            mock_load_settings.side_effect = FileNotFoundError("Recipe file not found")
            
            # Act
            result = runner.invoke(app, [
                "--recipe-path", "nonexistent_recipe.yaml",
                "--config-path", "test_config.yaml",
                "--data-path", "test_data.csv"
            ])
            
            # Assert
            assert result.exit_code == 1
    
    def test_train_command_value_error(self):
        """Test handling of ValueError during execution.""" 
        runner = CliRunner()
        app = typer.Typer()
        app.command()(train_command)
        
        with patch('src.cli.commands.train_command.load_settings') as mock_load_settings:
            # Mock ValueError
            mock_load_settings.side_effect = ValueError("Invalid configuration")
            
            # Act
            result = runner.invoke(app, [
                "--recipe-path", "test_recipe.yaml",
                "--config-path", "test_config.yaml",
                "--data-path", "test_data.csv"
            ])
            
            # Assert
            assert result.exit_code == 1
    
    def test_train_command_general_exception(self):
        """Test handling of general exceptions."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(train_command)
        
        with patch('src.cli.commands.train_command.load_settings') as mock_load_settings:
            with patch('src.cli.commands.train_command.setup_logging'):
                with patch('src.cli.commands.train_command.run_train_pipeline') as mock_run_pipeline:
                    
                    mock_settings = Mock()
                    mock_settings.recipe.data.loader.source_uri = None
                    mock_settings.recipe.model.computed = {"run_name": "test_run"}
                    mock_load_settings.return_value = mock_settings
                    
                    # Mock RuntimeError during pipeline execution
                    mock_run_pipeline.side_effect = RuntimeError("Pipeline execution failed")
                    
                    # Act
                    result = runner.invoke(app, [
                        "--recipe-path", "test_recipe.yaml",
                        "--config-path", "test_config.yaml",
                        "--data-path", "test_data.csv"
                    ])
                    
                    # Assert
                    assert result.exit_code == 1


class TestTrainCommandIntegration:
    """Test train command integration scenarios."""
    
    def test_train_command_logging_integration(self):
        """Test that logging is properly configured and used."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(train_command)
        
        with patch('src.cli.commands.train_command.load_settings') as mock_load_settings:
            with patch('src.cli.commands.train_command.setup_logging') as mock_setup_logging:
                with patch('src.cli.commands.train_command.run_train_pipeline'):
                    with patch('src.cli.commands.train_command.logger') as mock_logger:
                        
                        mock_settings = Mock()
                        mock_settings.recipe.data.loader.source_uri = None
                        mock_settings.recipe.model.computed = {"run_name": "test_run"}
                        mock_load_settings.return_value = mock_settings
                        
                        # Act
                        result = runner.invoke(app, [
                            "--recipe-path", "test_recipe.yaml",
                            "--config-path", "test_config.yaml",
                            "--data-path", "test_data.csv"
                        ])
                        
                        # Assert
                        assert result.exit_code == 0
                        mock_setup_logging.assert_called_once_with(mock_settings)
                        
                        # Check that key info is logged
                        expected_calls = [
                            mock_logger.info.assert_any_call("Recipe: test_recipe.yaml"),
                            mock_logger.info.assert_any_call("Config: test_config.yaml"),
                            mock_logger.info.assert_any_call("Data: test_data.csv"),
                            mock_logger.info.assert_any_call("Run Name: test_run"),
                            mock_logger.info.assert_any_call("✅ 학습이 성공적으로 완료되었습니다.")
                        ]
    
    def test_train_command_settings_modification(self):
        """Test that settings are properly modified based on data path."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(train_command)
        
        with patch('src.cli.commands.train_command.load_settings') as mock_load_settings:
            with patch('src.cli.commands.train_command.setup_logging'):
                with patch('src.cli.commands.train_command.run_train_pipeline') as mock_run_pipeline:
                    
                    mock_settings = Mock()
                    mock_settings.recipe.data.loader.source_uri = "original_source"
                    mock_settings.recipe.model.computed = {"run_name": "test_run"}
                    mock_load_settings.return_value = mock_settings
                    
                    # Act
                    result = runner.invoke(app, [
                        "--recipe-path", "test_recipe.yaml",
                        "--config-path", "test_config.yaml",
                        "--data-path", "new_data_path.parquet"
                    ])
                    
                    # Assert
                    assert result.exit_code == 0
                    # Verify settings were modified
                    assert mock_settings.recipe.data.loader.source_uri == "new_data_path.parquet"
                    mock_run_pipeline.assert_called_once_with(settings=mock_settings, context_params=None)


class TestTrainCommandJSONParameterHandling:
    """Test JSON parameter parsing in train command."""
    
    def test_train_command_invalid_json_handling(self):
        """Test handling of invalid JSON in context parameters."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(train_command)
        
        with patch('src.cli.commands.train_command.load_settings'):
            # Act - provide invalid JSON
            result = runner.invoke(app, [
                "--recipe-path", "test_recipe.yaml",
                "--config-path", "test_config.yaml",
                "--data-path", "test_data.csv",
                "--params", "invalid_json_string"
            ])
            
            # Assert
            assert result.exit_code == 1  # Should fail due to JSON parsing error
    
    def test_train_command_empty_json_params(self):
        """Test handling of empty JSON parameters."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(train_command)
        
        with patch('src.cli.commands.train_command.load_settings') as mock_load_settings:
            with patch('src.cli.commands.train_command.setup_logging'):
                with patch('src.cli.commands.train_command.run_train_pipeline') as mock_run_pipeline:
                    
                    mock_settings = Mock()
                    mock_settings.recipe.data.loader.source_uri = None
                    mock_settings.recipe.model.computed = {"run_name": "test_run"}
                    mock_load_settings.return_value = mock_settings
                    
                    # Act
                    result = runner.invoke(app, [
                        "--recipe-path", "test_recipe.yaml",
                        "--config-path", "test_config.yaml",
                        "--data-path", "test_data.csv",
                        "--params", "{}"
                    ])
                    
                    # Assert
                    assert result.exit_code == 0
                    mock_run_pipeline.assert_called_once_with(settings=mock_settings, context_params={})
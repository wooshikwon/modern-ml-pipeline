"""
Unit Tests for Train Command CLI
Days 3-5: CLI argument parsing and validation tests
"""

import pytest
from unittest.mock import patch, MagicMock, call
import typer
from typer.testing import CliRunner

from src.cli.commands.train_command import train_command


class TestTrainCommandArgumentParsing:
    """Train command argument parsing tests"""
    
    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(train_command)
    
    @patch('src.cli.commands.train_command.load_settings')
    @patch('src.cli.commands.train_command.run_train_pipeline')
    @patch('src.cli.commands.train_command.setup_logging')
    def test_train_command_with_required_arguments(self, mock_setup_logging, mock_run_pipeline, mock_load_settings):
        """Test train command with all required arguments"""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.recipe.model.computed = {"run_name": "test_run"}
        mock_load_settings.return_value = mock_settings
        
        # Execute command
        result = self.runner.invoke(self.app, [
            '--recipe-path', 'recipes/test.yaml',
            '--config-path', 'configs/test.yaml', 
            '--data-path', 'data/train.csv'
        ])
        
        # Verify success
        assert result.exit_code == 0
        
        # Verify function calls
        mock_load_settings.assert_called_once_with('recipes/test.yaml', 'configs/test.yaml')
        mock_setup_logging.assert_called_once_with(mock_settings)
        mock_run_pipeline.assert_called_once_with(settings=mock_settings, context_params=None)
        
        # Verify data path injection
        assert mock_settings.recipe.data.loader.source_uri == 'data/train.csv'
        mock_settings.validate_data_source_compatibility.assert_called_once()
    
    @patch('src.cli.commands.train_command.load_settings')
    @patch('src.cli.commands.train_command.run_train_pipeline') 
    @patch('src.cli.commands.train_command.setup_logging')
    def test_train_command_with_optional_params(self, mock_setup_logging, mock_run_pipeline, mock_load_settings):
        """Test train command with optional context parameters"""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.recipe.model.computed = {"run_name": "test_run_with_params"}
        mock_load_settings.return_value = mock_settings
        
        # Execute command with JSON params
        result = self.runner.invoke(self.app, [
            '--recipe-path', 'recipes/advanced.yaml',
            '--config-path', 'configs/prod.yaml',
            '--data-path', 'data/features.parquet',
            '--params', '{"date": "2024-01-01", "version": 2}'
        ])
        
        # Verify success
        assert result.exit_code == 0
        
        # Verify function calls with parsed parameters
        mock_load_settings.assert_called_once_with('recipes/advanced.yaml', 'configs/prod.yaml')
        mock_run_pipeline.assert_called_once_with(
            settings=mock_settings, 
            context_params={"date": "2024-01-01", "version": 2}
        )
    
    def test_train_command_missing_required_arguments(self):
        """Test train command fails with missing required arguments"""
        # Missing recipe-path
        result = self.runner.invoke(self.app, [
            '--config-path', 'configs/test.yaml',
            '--data-path', 'data/train.csv'
        ])
        assert result.exit_code != 0
        assert "Missing option '--recipe-path'" in result.output
        
        # Missing config-path
        result = self.runner.invoke(self.app, [
            '--recipe-path', 'recipes/test.yaml',
            '--data-path', 'data/train.csv'
        ])
        assert result.exit_code != 0
        assert "Missing option '--config-path'" in result.output
        
        # Missing data-path
        result = self.runner.invoke(self.app, [
            '--recipe-path', 'recipes/test.yaml',
            '--config-path', 'configs/test.yaml'
        ])
        assert result.exit_code != 0
        assert "Missing option '--data-path'" in result.output
    
    @patch('src.cli.commands.train_command.load_settings')
    def test_train_command_file_not_found_error(self, mock_load_settings):
        """Test train command handles FileNotFoundError"""
        # Setup mock to raise FileNotFoundError
        mock_load_settings.side_effect = FileNotFoundError("Recipe 파일을 찾을 수 없습니다")
        
        # Execute command
        result = self.runner.invoke(self.app, [
            '--recipe-path', 'nonexistent/recipe.yaml',
            '--config-path', 'configs/test.yaml',
            '--data-path', 'data/train.csv'
        ])
        
        # Verify exit code and error handling
        assert result.exit_code == 1
        
    @patch('src.cli.commands.train_command.load_settings')
    def test_train_command_invalid_json_params(self, mock_load_settings):
        """Test train command handles invalid JSON in context_params"""
        mock_settings = MagicMock()
        mock_load_settings.return_value = mock_settings
        
        # Execute command with invalid JSON
        result = self.runner.invoke(self.app, [
            '--recipe-path', 'recipes/test.yaml',
            '--config-path', 'configs/test.yaml',
            '--data-path', 'data/train.csv',
            '--params', '{"invalid": json}'  # Invalid JSON
        ])
        
        # Should fail due to JSON parsing error
        assert result.exit_code == 1
    
    @patch('src.cli.commands.train_command.load_settings')
    @patch('src.cli.commands.train_command.setup_logging')
    @patch('src.utils.system.templating_utils.render_template_from_string')
    def test_train_command_jinja_template_processing(self, mock_render, mock_setup_logging, mock_load_settings):
        """Test train command with Jinja template data path"""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.recipe.model.computed = {"run_name": "template_run"}
        mock_load_settings.return_value = mock_settings
        mock_render.return_value = "SELECT * FROM table WHERE date = '2024-01-01'"
        
        # Mock template file existence and run_train_pipeline
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value="SELECT * FROM table WHERE date = '{{date}}'"), \
             patch('src.cli.commands.train_command.run_train_pipeline') as mock_run_pipeline:
            
            result = self.runner.invoke(self.app, [
                '--recipe-path', 'recipes/test.yaml',
                '--config-path', 'configs/test.yaml',
                '--data-path', 'queries/dynamic.sql.j2',
                '--params', '{"date": "2024-01-01"}'
            ])
            
            # Verify template was processed
            assert result.exit_code == 0
            mock_render.assert_called_once_with(
                "SELECT * FROM table WHERE date = '{{date}}'", 
                {"date": "2024-01-01"}
            )
    
    def test_train_command_short_options(self):
        """Test train command with short option flags"""
        with patch('src.cli.commands.train_command.load_settings'), \
             patch('src.cli.commands.train_command.run_train_pipeline'), \
             patch('src.cli.commands.train_command.setup_logging'):
            
            # Test short flags work
            result = self.runner.invoke(self.app, [
                '-r', 'recipes/test.yaml',
                '-c', 'configs/test.yaml',
                '-d', 'data/train.csv',
                '-p', '{}'
            ])
            
            assert result.exit_code == 0
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
    
    @patch('src.cli.commands.train_command.SettingsFactory.for_training')
    @patch('src.cli.commands.train_command.run_train_pipeline')
    @patch('src.cli.commands.train_command.cli_command_start')
    @patch('src.cli.commands.train_command.cli_command_success')
    def test_train_command_with_required_arguments(self, mock_cli_success, mock_cli_start, mock_run_pipeline, mock_settings_factory):
        """Test train command with all required arguments"""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.recipe.model.computed = {"run_name": "test_run"}
        mock_settings_factory.return_value = mock_settings

        mock_result = MagicMock()
        mock_result.run_id = "test_run_id"
        mock_result.model_uri = "models:/test_model/1"
        mock_run_pipeline.return_value = mock_result

        # Execute command
        result = self.runner.invoke(self.app, [
            '--recipe-path', 'recipes/test.yaml',
            '--config-path', 'configs/test.yaml',
            '--data-path', 'data/train.csv'
        ])

        # Verify success
        assert result.exit_code == 0

        # Verify SettingsFactory call
        mock_settings_factory.assert_called_once_with(
            recipe_path='recipes/test.yaml',
            config_path='configs/test.yaml',
            data_path='data/train.csv',
            context_params=None
        )

        # Verify pipeline execution
        mock_run_pipeline.assert_called_once_with(
            settings=mock_settings,
            context_params=None,
            record_requirements=False
        )

        # Verify Rich Console calls
        mock_cli_start.assert_called_once_with("Training", "모델 학습 파이프라인 실행")
        mock_cli_success.assert_called_once_with("Training", ["Run ID: test_run_id", "Model URI: models:/test_model/1"])
    
    @patch('src.cli.commands.train_command.SettingsFactory.for_training')
    @patch('src.cli.commands.train_command.run_train_pipeline')
    @patch('src.cli.commands.train_command.cli_command_start')
    @patch('src.cli.commands.train_command.cli_command_success')
    def test_train_command_with_optional_params(self, mock_cli_success, mock_cli_start, mock_run_pipeline, mock_settings_factory):
        """Test train command with optional context parameters"""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.recipe.model.computed = {"run_name": "test_run_with_params"}
        mock_settings_factory.return_value = mock_settings

        mock_result = MagicMock()
        mock_result.run_id = "param_run_id"
        mock_result.model_uri = "models:/param_model/1"
        mock_run_pipeline.return_value = mock_result

        # Execute command with JSON params
        result = self.runner.invoke(self.app, [
            '--recipe-path', 'recipes/advanced.yaml',
            '--config-path', 'configs/prod.yaml',
            '--data-path', 'data/features.parquet',
            '--params', '{"date": "2024-01-01", "version": 2}'
        ])

        # Verify success
        assert result.exit_code == 0

        # Verify SettingsFactory call with parsed parameters
        mock_settings_factory.assert_called_once_with(
            recipe_path='recipes/advanced.yaml',
            config_path='configs/prod.yaml',
            data_path='data/features.parquet',
            context_params={"date": "2024-01-01", "version": 2}
        )

        # Verify pipeline execution
        mock_run_pipeline.assert_called_once_with(
            settings=mock_settings,
            context_params={"date": "2024-01-01", "version": 2},
            record_requirements=False
        )

        # Verify Rich Console calls
        mock_cli_start.assert_called_once_with("Training", "모델 학습 파이프라인 실행")
        mock_cli_success.assert_called_once_with("Training", ["Run ID: param_run_id", "Model URI: models:/param_model/1"])
    
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
    
    @patch('src.cli.commands.train_command.SettingsFactory.for_training')
    @patch('src.cli.commands.train_command.cli_command_start')
    @patch('src.cli.commands.train_command.cli_command_error')
    def test_train_command_file_not_found_error(self, mock_cli_error, mock_cli_start, mock_settings_factory):
        """Test train command handles FileNotFoundError"""
        # Setup mock to raise FileNotFoundError
        mock_settings_factory.side_effect = FileNotFoundError("Recipe 파일을 찾을 수 없습니다")

        # Execute command
        result = self.runner.invoke(self.app, [
            '--recipe-path', 'nonexistent/recipe.yaml',
            '--config-path', 'configs/test.yaml',
            '--data-path', 'data/train.csv'
        ])

        # Verify exit code and error handling
        assert result.exit_code == 1

        # Verify Rich Console calls
        mock_cli_start.assert_called_once_with("Training", "모델 학습 파이프라인 실행")
        mock_cli_error.assert_called_once_with(
            "Training",
            "파일을 찾을 수 없습니다: Recipe 파일을 찾을 수 없습니다",
            "파일 경로를 확인하거나 'mmp get-config/get-recipe'를 실행하세요"
        )
        
    def test_train_command_invalid_json_params(self):
        """Test train command handles invalid JSON in context_params"""
        # Execute command with invalid JSON
        result = self.runner.invoke(self.app, [
            '--recipe-path', 'recipes/test.yaml',
            '--config-path', 'configs/test.yaml',
            '--data-path', 'data/train.csv',
            '--params', '{"invalid": json}'  # Invalid JSON
        ])

        # Should fail due to JSON parsing error
        assert result.exit_code != 0  # Typer will catch JSON parsing error
    
    @patch('src.cli.commands.train_command.SettingsFactory.for_training')
    @patch('src.cli.commands.train_command.run_train_pipeline')
    @patch('src.cli.commands.train_command.cli_command_start')
    @patch('src.cli.commands.train_command.cli_command_success')
    def test_train_command_jinja_template_processing(self, mock_cli_success, mock_cli_start, mock_run_pipeline, mock_settings_factory):
        """Test train command with Jinja template data path"""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings.recipe.model.computed = {"run_name": "template_run"}
        mock_settings_factory.return_value = mock_settings

        mock_result = MagicMock()
        mock_result.run_id = "template_run_id"
        mock_result.model_uri = "models:/template_model/1"
        mock_run_pipeline.return_value = mock_result

        result = self.runner.invoke(self.app, [
            '--recipe-path', 'recipes/test.yaml',
            '--config-path', 'configs/test.yaml',
            '--data-path', 'queries/dynamic.sql.j2',
            '--params', '{"date": "2024-01-01"}'
        ])

        # Verify template processing delegation to SettingsFactory
        assert result.exit_code == 0
        mock_settings_factory.assert_called_once_with(
            recipe_path='recipes/test.yaml',
            config_path='configs/test.yaml',
            data_path='queries/dynamic.sql.j2',
            context_params={"date": "2024-01-01"}
        )

        # Verify Rich Console calls
        mock_cli_start.assert_called_once_with("Training", "모델 학습 파이프라인 실행")
        mock_cli_success.assert_called_once_with("Training", ["Run ID: template_run_id", "Model URI: models:/template_model/1"])
    
    @patch('src.cli.commands.train_command.SettingsFactory.for_training')
    @patch('src.cli.commands.train_command.run_train_pipeline')
    @patch('src.cli.commands.train_command.cli_command_start')
    @patch('src.cli.commands.train_command.cli_command_success')
    def test_train_command_short_options(self, mock_cli_success, mock_cli_start, mock_run_pipeline, mock_settings_factory):
        """Test train command with short option flags"""
        # Setup mocks
        mock_settings = MagicMock()
        mock_settings_factory.return_value = mock_settings

        mock_result = MagicMock()
        mock_result.run_id = "short_run_id"
        mock_result.model_uri = "models:/short_model/1"
        mock_run_pipeline.return_value = mock_result

        # Test short flags work
        result = self.runner.invoke(self.app, [
            '-r', 'recipes/test.yaml',
            '-c', 'configs/test.yaml',
            '-d', 'data/train.csv',
            '-p', '{}'
        ])

        assert result.exit_code == 0

        # Verify SettingsFactory call with short options
        mock_settings_factory.assert_called_once_with(
            recipe_path='recipes/test.yaml',
            config_path='configs/test.yaml',
            data_path='data/train.csv',
            context_params={}
        )

        # Verify Rich Console calls
        mock_cli_start.assert_called_once_with("Training", "모델 학습 파이프라인 실행")
        mock_cli_success.assert_called_once_with("Training", ["Run ID: short_run_id", "Model URI: models:/short_model/1"])
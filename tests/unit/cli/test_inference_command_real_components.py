"""
Unit Tests for Inference Command CLI with Real Components
Following test philosophy: Real Object Testing - using actual SettingsFactory

Only mocking the actual pipeline execution (run_inference_pipeline).
SettingsFactory uses real implementation to test actual component integration.
"""

import pytest
from unittest.mock import patch, MagicMock
import typer
from typer.testing import CliRunner
from pathlib import Path
import tempfile
import yaml
import json
import pandas as pd

from src.cli.commands.inference_command import batch_inference_command
from src.settings import SettingsFactory


class TestInferenceCommandWithRealSettingsFactory:
    """Inference command tests using real SettingsFactory - Real Object Testing"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(batch_inference_command)

    def create_test_config_and_data(self, temp_dir: Path) -> tuple[Path, Path, Path]:
        """Create minimal config, recipe files and test data for testing"""
        # Create config file
        config_path = temp_dir / "config.yaml"
        config_data = {
            'environment': {'name': 'test'},
            'mlflow': {
                'tracking_uri': 'sqlite:///mlruns.db',
                'experiment_name': 'test_experiment'
            },
            'data_source': {
                'name': 'test_storage',
                'adapter_type': 'storage',
                'config': {'base_path': str(temp_dir / 'data')}
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        # Create recipe file
        recipe_path = temp_dir / "recipe.yaml"
        recipe_data = {
            'name': 'test_recipe',
            'task_choice': 'classification',
            'data': {
                'loader': {
                    'name': 'csv',
                    'source_uri': str(temp_dir / 'data' / 'test.csv')
                },
                'target_column': 'target'
            },
            'model': {
                'name': 'sklearn.ensemble.RandomForestClassifier',
                'hyperparameters': {'n_estimators': 10}
            },
            'evaluation': {'k_fold': 3}
        }
        with open(recipe_path, 'w') as f:
            yaml.dump(recipe_data, f)

        # Create test data file
        data_dir = temp_dir / 'data'
        data_dir.mkdir(exist_ok=True)
        data_path = data_dir / 'inference.csv'

        # Create simple test dataframe
        df = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'feature_3': [10, 20, 30, 40, 50],
            'entity_id': [1, 2, 3, 4, 5]
        })
        df.to_csv(data_path, index=False)

        return config_path, recipe_path, data_path

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.settings.factory.load_run_settings')  # Mock MLflow run loading
    def test_inference_command_real_settings_factory_integration(
        self, mock_load_run_settings, mock_run_pipeline
    ):
        """Test inference command with real SettingsFactory to verify actual component integration"""
        # Setup mocks
        mock_result = MagicMock()
        mock_result.processed_rows = 5
        mock_result.output_path = "/tmp/predictions.csv"
        mock_run_pipeline.return_value = mock_result

        # Mock MLflow run settings loading
        mock_recipe = MagicMock()
        mock_recipe.task_choice = 'classification'
        mock_recipe.model = MagicMock()
        mock_recipe.model.name = 'sklearn.ensemble.RandomForestClassifier'
        mock_recipe.model.hyperparameters = {'n_estimators': 10}
        mock_recipe.data = MagicMock()
        mock_recipe.data.loader = MagicMock()
        mock_recipe.data.loader.name = 'csv'
        mock_load_run_settings.return_value = (MagicMock(), mock_recipe)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path, recipe_path, data_path = self.create_test_config_and_data(temp_path)
            test_run_id = 'test_inference_run'

            # Execute inference command with real SettingsFactory
            result = self.runner.invoke(self.app, [
                '--run-id', test_run_id,
                '--config-path', str(config_path),
                '--data-path', str(data_path)
            ])

            # Command should succeed
            assert result.exit_code == 0

            # Verify run_inference_pipeline was called
            mock_run_pipeline.assert_called_once()
            call_args = mock_run_pipeline.call_args

            # Verify settings was created properly
            settings = call_args.kwargs['settings']
            assert settings is not None
            assert call_args.kwargs['run_id'] == test_run_id
            assert call_args.kwargs['data_path'] == str(data_path)
            assert call_args.kwargs['context_params'] == {}

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.settings.factory.load_run_settings')
    def test_inference_command_with_json_params_real_factory(
        self, mock_load_run_settings, mock_run_pipeline
    ):
        """Test inference command with JSON parameters using real SettingsFactory"""
        # Setup mocks
        mock_result = MagicMock()
        mock_result.processed_rows = 100
        mock_result.output_path = "/tmp/batch_output.csv"
        mock_run_pipeline.return_value = mock_result

        # Mock MLflow run settings loading
        mock_recipe = MagicMock()
        mock_recipe.task_choice = 'classification'
        mock_recipe.model = MagicMock()
        mock_recipe.data = MagicMock()
        mock_recipe.data.loader = MagicMock()
        mock_load_run_settings.return_value = (MagicMock(), mock_recipe)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path, recipe_path, data_path = self.create_test_config_and_data(temp_path)
            test_run_id = 'test_params_run'

            # Execute with JSON parameters
            params = {"batch_date": "2024-01-15", "limit": 1000}
            result = self.runner.invoke(self.app, [
                '--run-id', test_run_id,
                '--config-path', str(config_path),
                '--data-path', str(data_path),
                '--params', json.dumps(params)
            ])

            # Command should succeed
            assert result.exit_code == 0

            # Verify run_inference_pipeline was called with parsed params
            mock_run_pipeline.assert_called_once()
            call_args = mock_run_pipeline.call_args
            assert call_args.kwargs['context_params'] == params

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    def test_inference_command_real_factory_file_not_found(
        self, mock_run_pipeline
    ):
        """Test inference command error handling with real SettingsFactory when files are missing"""
        mock_run_pipeline.return_value = MagicMock()
        test_run_id = 'test_run_456'

        # Execute with non-existent files
        result = self.runner.invoke(self.app, [
            '--run-id', test_run_id,
            '--config-path', '/non/existent/config.yaml',
            '--data-path', '/non/existent/data.csv'
        ])

        # Should fail due to file not found
        assert result.exit_code == 1
        assert mock_run_pipeline.call_count == 0

        # Check error message
        output_str = str(result.output)
        assert any(keyword in output_str for keyword in ['찾을 수 없습니다', 'not found', 'Error'])

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    def test_inference_command_real_factory_missing_data_file(
        self, mock_run_pipeline
    ):
        """Test inference command error handling when data file is missing"""
        mock_run_pipeline.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create only config file, no data file
            config_path = temp_path / "config.yaml"
            config_data = {
                'environment': {'name': 'test'},
                'mlflow': {
                    'tracking_uri': 'sqlite:///mlruns.db',
                    'experiment_name': 'test_experiment'
                },
                'data_source': {
                    'name': 'test_storage',
                    'adapter_type': 'storage',
                    'config': {'base_path': str(temp_path / 'data')}
                }
            }
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)

            test_run_id = 'test_run_789'

            # Execute with non-existent data file
            result = self.runner.invoke(self.app, [
                '--run-id', test_run_id,
                '--config-path', str(config_path),
                '--data-path', str(temp_path / 'missing_data.csv')
            ])

            # Should fail due to missing data file
            assert result.exit_code == 1
            assert mock_run_pipeline.call_count == 0

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    def test_inference_command_invalid_json_params_real_factory(
        self, mock_run_pipeline
    ):
        """Test inference command handling of invalid JSON parameters with real factory"""
        mock_run_pipeline.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path, recipe_path, data_path = self.create_test_config_and_data(temp_path)
            test_run_id = 'invalid_json_run'

            # Execute with invalid JSON
            result = self.runner.invoke(self.app, [
                '--run-id', test_run_id,
                '--config-path', str(config_path),
                '--data-path', str(data_path),
                '--params', 'not valid json {]'
            ])

            # Should fail due to invalid JSON
            assert result.exit_code == 1
            assert mock_run_pipeline.call_count == 0

            # Check error message
            output_str = str(result.output)
            assert any(keyword in output_str for keyword in ['오류', 'error', 'Error'])

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.settings.factory.load_run_settings')
    def test_inference_command_progress_tracking_real_factory(
        self, mock_load_run_settings, mock_run_pipeline
    ):
        """Test inference command console progress tracking with real SettingsFactory"""
        # Setup mocks
        mock_result = MagicMock()
        mock_result.processed_rows = 50
        mock_result.output_path = "/tmp/results.csv"
        mock_run_pipeline.return_value = mock_result

        # Mock MLflow run settings loading
        mock_recipe = MagicMock()
        mock_recipe.task_choice = 'classification'
        mock_recipe.model = MagicMock()
        mock_recipe.data = MagicMock()
        mock_recipe.data.loader = MagicMock()
        mock_load_run_settings.return_value = (MagicMock(), mock_recipe)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path, recipe_path, data_path = self.create_test_config_and_data(temp_path)
            test_run_id = 'progress_test_run'

            # Execute inference command
            result = self.runner.invoke(self.app, [
                '--run-id', test_run_id,
                '--config-path', str(config_path),
                '--data-path', str(data_path)
            ])

            # Command should succeed
            assert result.exit_code == 0
            output_str = str(result.output)

            # Check for progress tracking messages
            assert any(keyword in output_str for keyword in ['추론', '설정', 'Inference', 'Config'])

            # Verify settings was created properly by real factory
            mock_run_pipeline.assert_called_once()
            call_args = mock_run_pipeline.call_args
            settings = call_args.kwargs['settings']
            assert settings is not None
            # Settings should have proper structure from real factory
            assert hasattr(settings, 'config')
            assert hasattr(settings, 'recipe')

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.settings.factory.load_run_settings')
    def test_inference_command_result_processing_real_factory(
        self, mock_load_run_settings, mock_run_pipeline
    ):
        """Test inference command result processing with real SettingsFactory"""
        # Setup mocks with different result attributes
        mock_result = MagicMock()
        # Test with processed_rows and output_path attributes
        mock_result.processed_rows = 1000
        mock_result.output_path = "/tmp/final_predictions.csv"
        mock_run_pipeline.return_value = mock_result

        # Mock MLflow run settings loading
        mock_recipe = MagicMock()
        mock_recipe.task_choice = 'classification'
        mock_recipe.model = MagicMock()
        mock_recipe.data = MagicMock()
        mock_recipe.data.loader = MagicMock()
        mock_load_run_settings.return_value = (MagicMock(), mock_recipe)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path, recipe_path, data_path = self.create_test_config_and_data(temp_path)
            test_run_id = 'result_test_run'

            # Execute inference command
            result = self.runner.invoke(self.app, [
                '--run-id', test_run_id,
                '--config-path', str(config_path),
                '--data-path', str(data_path)
            ])

            # Command should succeed
            assert result.exit_code == 0
            output_str = str(result.output)

            # Check for success details in output (lines 87-92 in inference_command.py)
            assert any(keyword in output_str for keyword in ['처리', '완료', '출력'])

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.settings.factory.load_run_settings')
    def test_inference_command_result_without_attributes_real_factory(
        self, mock_load_run_settings, mock_run_pipeline
    ):
        """Test inference command when result doesn't have optional attributes"""
        # Setup mocks without processed_rows or output_path attributes
        mock_result = MagicMock()
        # Remove the attributes to test hasattr checks
        del mock_result.processed_rows
        del mock_result.output_path
        mock_run_pipeline.return_value = mock_result

        # Mock MLflow run settings loading
        mock_recipe = MagicMock()
        mock_recipe.task_choice = 'classification'
        mock_recipe.model = MagicMock()
        mock_recipe.data = MagicMock()
        mock_recipe.data.loader = MagicMock()
        mock_load_run_settings.return_value = (MagicMock(), mock_recipe)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path, recipe_path, data_path = self.create_test_config_and_data(temp_path)
            test_run_id = 'no_attr_run'

            # Execute inference command
            result = self.runner.invoke(self.app, [
                '--run-id', test_run_id,
                '--config-path', str(config_path),
                '--data-path', str(data_path)
            ])

            # Command should succeed even without optional attributes
            assert result.exit_code == 0


class TestInferenceCommandErrorHandlingWithRealComponents:
    """Test error handling with real components to ensure proper integration"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(batch_inference_command)

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    def test_inference_command_yaml_parse_error_real_factory(
        self, mock_run_pipeline
    ):
        """Test inference command handling of YAML parse errors with real SettingsFactory"""
        mock_run_pipeline.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create invalid YAML file
            config_path = temp_path / "bad_yaml.yaml"
            with open(config_path, 'w') as f:
                f.write("invalid: yaml: content: [unclosed")

            # Create valid data file
            data_path = temp_path / 'data.csv'
            pd.DataFrame({'col1': [1, 2, 3]}).to_csv(data_path, index=False)

            test_run_id = 'yaml_error_run'

            # Execute inference command
            result = self.runner.invoke(self.app, [
                '--run-id', test_run_id,
                '--config-path', str(config_path),
                '--data-path', str(data_path)
            ])

            # Should fail due to YAML parse error
            assert result.exit_code == 1
            assert mock_run_pipeline.call_count == 0

            # Check error handling
            output_str = str(result.output)
            assert any(keyword in output_str for keyword in ['오류', 'error', 'Error'])

    @patch('src.cli.commands.inference_command.run_inference_pipeline')
    @patch('src.settings.factory.load_run_settings')
    def test_inference_command_pipeline_runtime_error_real_factory(
        self, mock_load_run_settings, mock_run_pipeline
    ):
        """Test inference command handling of pipeline runtime errors with real SettingsFactory"""
        # Mock pipeline to raise exception
        mock_run_pipeline.side_effect = RuntimeError("Model loading failed")

        # Mock MLflow run settings loading
        mock_recipe = MagicMock()
        mock_recipe.task_choice = 'classification'
        mock_recipe.model = MagicMock()
        mock_recipe.data = MagicMock()
        mock_recipe.data.loader = MagicMock()
        mock_load_run_settings.return_value = (MagicMock(), mock_recipe)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create config file
            config_path = temp_path / "config.yaml"
            config_data = {
                'environment': {'name': 'test'},
                'mlflow': {
                    'tracking_uri': 'sqlite:///mlruns.db',
                    'experiment_name': 'test_experiment'
                },
                'data_source': {
                    'name': 'test_storage',
                    'adapter_type': 'storage',
                    'config': {'base_path': str(temp_path / 'data')}
                }
            }
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)

            # Create data file
            data_dir = temp_path / 'data'
            data_dir.mkdir(exist_ok=True)
            data_path = data_dir / 'test.csv'
            pd.DataFrame({'col1': [1, 2, 3]}).to_csv(data_path, index=False)

            test_run_id = 'runtime_error_run'

            # Execute inference command
            result = self.runner.invoke(self.app, [
                '--run-id', test_run_id,
                '--config-path', str(config_path),
                '--data-path', str(data_path)
            ])

            # Should fail due to runtime error
            assert result.exit_code == 1

            # Verify pipeline was attempted to run
            mock_run_pipeline.assert_called_once()

            # Check error message
            output_str = str(result.output)
            assert any(keyword in output_str for keyword in ['오류', 'error', 'Error'])
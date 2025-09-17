"""
Unit Tests for Serve Command CLI with Real Components
Following test philosophy: Real Object Testing - using actual SettingsFactory

Only mocking the actual API server execution to avoid starting a real server in tests.
SettingsFactory uses real implementation to test actual component integration.
"""

import pytest
from unittest.mock import patch, MagicMock
import typer
from typer.testing import CliRunner
from pathlib import Path
import tempfile
import yaml

from src.cli.commands.serve_command import serve_api_command
from src.settings import SettingsFactory


class TestServeCommandWithRealSettingsFactory:
    """Serve command tests using real SettingsFactory - Real Object Testing"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(serve_api_command)

    def create_test_config_and_recipe(self, temp_dir: Path) -> tuple[Path, Path]:
        """Create minimal config and recipe files for testing"""
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
            },
            'feature_store': {
                'provider': 'none'
            },
            'output': {
                'inference': {
                    'name': 'test_output',
                    'enabled': True,
                    'adapter_type': 'storage',
                    'config': {'base_path': str(temp_dir / 'output')}
                }
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

        return config_path, recipe_path

    @patch('src.cli.commands.serve_command.run_api_server')
    @patch('src.settings.factory.MLflowRecipeRestorer')  # Mock MLflow restorer
    def test_serve_command_real_settings_factory_integration(
        self, mock_restorer_class, mock_run_server
    ):
        """Test serve command with real SettingsFactory to verify actual component integration"""
        # Setup mocks
        mock_run_server.return_value = None

        # Mock MLflow Recipe Restorer
        mock_recipe = MagicMock()
        mock_recipe.task_choice = 'classification'
        mock_recipe.model = MagicMock()
        mock_recipe.model.name = 'sklearn.ensemble.RandomForestClassifier'
        mock_recipe.model.hyperparameters = {'n_estimators': 10}
        mock_restorer = MagicMock()
        mock_restorer.restore_recipe.return_value = mock_recipe
        mock_restorer_class.return_value = mock_restorer

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path, recipe_path = self.create_test_config_and_recipe(temp_path)
            test_run_id = 'test_run_123'

            # Execute serve command with real SettingsFactory
            result = self.runner.invoke(self.app, [
                '--run-id', test_run_id,
                '--config-path', str(config_path)
            ])

            # Command should succeed
            assert result.exit_code == 0

            # Verify run_api_server was called
            mock_run_server.assert_called_once()
            call_args = mock_run_server.call_args

            # Verify settings was created properly
            settings = call_args.kwargs['settings']
            assert settings is not None
            assert call_args.kwargs['run_id'] == test_run_id
            assert call_args.kwargs['host'] == '0.0.0.0'
            assert call_args.kwargs['port'] == 8000

    @patch('src.cli.commands.serve_command.run_api_server')
    def test_serve_command_real_settings_factory_file_not_found(
        self, mock_run_server
    ):
        """Test serve command error handling with real SettingsFactory when files are missing"""
        mock_run_server.return_value = None
        test_run_id = 'test_run_456'

        # Execute with non-existent config file
        result = self.runner.invoke(self.app, [
            '--run-id', test_run_id,
            '--config-path', '/non/existent/config.yaml'
        ])

        # Should fail due to file not found
        assert result.exit_code == 1
        assert mock_run_server.call_count == 0

        # Check error message
        output_str = str(result.output)
        assert any(keyword in output_str for keyword in ['찾을 수 없습니다', 'not found', 'Error'])

    @patch('src.cli.commands.serve_command.run_api_server')
    def test_serve_command_real_settings_factory_invalid_config(
        self, mock_run_server
    ):
        """Test serve command error handling with real SettingsFactory for invalid config"""
        mock_run_server.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create invalid config file (missing required fields)
            config_path = temp_path / "invalid_config.yaml"
            config_data = {
                'environment': {'name': 'test'},
                'feature_store': {'provider': 'none'},
                'output': {
                    'inference': {
                        'name': 'test_output',
                        'enabled': True,
                        'adapter_type': 'storage',
                        'config': {'base_path': str(temp_path / 'output')}
                    }
                }
                # Missing mlflow and data_source
            }
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)

            test_run_id = 'test_run_789'

            # Execute serve command
            result = self.runner.invoke(self.app, [
                '--run-id', test_run_id,
                '--config-path', str(config_path)
            ])

            # Should fail due to validation error
            assert result.exit_code == 1
            assert mock_run_server.call_count == 0

            # Check error message
            output_str = str(result.output)
            assert any(keyword in output_str for keyword in ['오류', 'error', 'Error'])

    @patch('src.cli.commands.serve_command.run_api_server')
    @patch('src.settings.factory.MLflowRecipeRestorer')
    def test_serve_command_with_custom_host_port_real_factory(
        self, mock_restorer_class, mock_run_server
    ):
        """Test serve command with custom host/port using real SettingsFactory"""
        # Setup mocks
        mock_run_server.return_value = None

        # Mock MLflow Recipe Restorer
        mock_recipe = MagicMock()
        mock_recipe.task_choice = 'classification'
        mock_recipe.model = MagicMock()
        mock_recipe.model.name = 'sklearn.linear_model.LogisticRegression'
        mock_restorer = MagicMock()
        mock_restorer.restore_recipe.return_value = mock_recipe
        mock_restorer_class.return_value = mock_restorer

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path, recipe_path = self.create_test_config_and_recipe(temp_path)
            test_run_id = 'custom_host_port_run'

            # Execute with custom host and port
            result = self.runner.invoke(self.app, [
                '--run-id', test_run_id,
                '--config-path', str(config_path),
                '--host', '127.0.0.1',
                '--port', '9000'
            ])

            # Command should succeed
            assert result.exit_code == 0

            # Verify run_api_server was called with custom host/port
            mock_run_server.assert_called_once()
            call_args = mock_run_server.call_args
            assert call_args.kwargs['host'] == '127.0.0.1'
            assert call_args.kwargs['port'] == 9000

    @patch('src.cli.commands.serve_command.run_api_server')
    @patch('src.settings.factory.MLflowRecipeRestorer')
    def test_serve_command_progress_tracking_with_real_factory(
        self, mock_restorer_class, mock_run_server
    ):
        """Test serve command console progress tracking with real SettingsFactory"""
        # Setup mocks
        mock_run_server.return_value = None

        # Mock MLflow Recipe Restorer with proper structure
        mock_recipe = MagicMock()
        mock_recipe.task_choice = 'classification'
        mock_recipe.model = MagicMock()
        mock_recipe.model.name = 'sklearn.svm.SVC'
        mock_restorer = MagicMock()
        mock_restorer.restore_recipe.return_value = mock_recipe
        mock_restorer_class.return_value = mock_restorer

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path, recipe_path = self.create_test_config_and_recipe(temp_path)
            test_run_id = 'progress_test_run'

            # Execute serve command
            result = self.runner.invoke(self.app, [
                '--run-id', test_run_id,
                '--config-path', str(config_path)
            ])

            # Command should succeed
            assert result.exit_code == 0
            output_str = str(result.output)

            # Check for progress tracking messages
            assert any(keyword in output_str for keyword in ['설정', 'Config', 'Server', 'API'])

            # Verify settings was created properly by real factory
            mock_run_server.assert_called_once()
            call_args = mock_run_server.call_args
            settings = call_args.kwargs['settings']
            assert settings is not None
            # Settings should have proper structure from real factory
            assert hasattr(settings, 'config')
            assert hasattr(settings, 'recipe')


class TestServeCommandErrorHandlingWithRealComponents:
    """Test error handling with real components to ensure proper integration"""

    def setup_method(self):
        self.runner = CliRunner()
        self.app = typer.Typer()
        self.app.command()(serve_api_command)

    @patch('src.cli.commands.serve_command.run_api_server')
    def test_serve_command_yaml_parse_error_real_factory(
        self, mock_run_server
    ):
        """Test serve command handling of YAML parse errors with real SettingsFactory"""
        mock_run_server.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create invalid YAML file
            config_path = temp_path / "bad_yaml.yaml"
            with open(config_path, 'w') as f:
                f.write("invalid: yaml: content: [unclosed")

            test_run_id = 'yaml_error_run'

            # Execute serve command
            result = self.runner.invoke(self.app, [
                '--run-id', test_run_id,
                '--config-path', str(config_path)
            ])

            # Should fail due to YAML parse error
            assert result.exit_code == 1
            assert mock_run_server.call_count == 0

            # Check error handling
            output_str = str(result.output)
            assert any(keyword in output_str for keyword in ['오류', 'error', 'Error'])

    @patch('src.cli.commands.serve_command.run_api_server')
    @patch('src.settings.factory.MLflowRecipeRestorer')
    def test_serve_command_server_runtime_error_real_factory(
        self, mock_restorer_class, mock_run_server
    ):
        """Test serve command handling of server runtime errors with real SettingsFactory"""
        # Mock server to raise exception
        mock_run_server.side_effect = RuntimeError("Port already in use")

        # Mock MLflow Recipe Restorer
        mock_recipe = MagicMock()
        mock_recipe.task_choice = 'classification'
        mock_recipe.model = MagicMock()
        mock_restorer = MagicMock()
        mock_restorer.restore_recipe.return_value = mock_recipe
        mock_restorer_class.return_value = mock_restorer

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
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
                },
                'feature_store': {
                    'provider': 'none'
                },
                'output': {
                    'inference': {
                        'name': 'test_output',
                        'enabled': True,
                        'adapter_type': 'storage',
                        'config': {'base_path': str(temp_path / 'output')}
                    }
                }
            }
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)

            test_run_id = 'runtime_error_run'

            # Execute serve command
            result = self.runner.invoke(self.app, [
                '--run-id', test_run_id,
                '--config-path', str(config_path)
            ])

            # Should fail due to runtime error
            assert result.exit_code == 1

            # Verify server was attempted to start
            mock_run_server.assert_called_once()

            # Check error message
            output_str = str(result.output)
            assert any(keyword in output_str for keyword in ['오류', 'error', 'Error'])
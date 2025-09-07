"""
Unit tests for list_commands.
Tests component listing command functionality with typer and CLI integration.
"""

import pytest
import typer
import yaml
from typer.testing import CliRunner
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path

from src.cli.commands.list_commands import (
    list_adapters, list_evaluators, list_preprocessors, list_models, _load_catalog_from_directory
)
from src.utils.system.console_manager import (
    cli_success, cli_error, cli_warning, cli_print, cli_info
)


class TestListCommandsInitialization:
    """Test list commands initialization and basic functionality."""
    
    def test_list_adapters_exists_and_callable(self):
        """Test that list_adapters is a callable function."""
        assert callable(list_adapters)
        assert hasattr(list_adapters, '__call__')
    
    def test_list_evaluators_exists_and_callable(self):
        """Test that list_evaluators is a callable function."""
        assert callable(list_evaluators)
        assert hasattr(list_evaluators, '__call__')
    
    def test_list_preprocessors_exists_and_callable(self):
        """Test that list_preprocessors is a callable function."""
        assert callable(list_preprocessors)
        assert hasattr(list_preprocessors, '__call__')
    
    def test_list_models_exists_and_callable(self):
        """Test that list_models is a callable function."""
        assert callable(list_models)
        assert hasattr(list_models, '__call__')
    
    def test_load_catalog_from_directory_exists_and_callable(self):
        """Test that _load_catalog_from_directory helper function exists."""
        assert callable(_load_catalog_from_directory)
        assert hasattr(_load_catalog_from_directory, '__call__')


class TestListAdaptersCommand:
    """Test list_adapters command functionality."""
    
    @patch('src.cli.commands.list_commands.AdapterRegistry')
    @patch('src.cli.commands.list_commands.cli_success')
    @patch('src.cli.commands.list_commands.cli_print')
    def test_list_adapters_with_available_adapters(self, mock_cli_print, mock_cli_success, mock_adapter_registry):
        """Test list_adapters with available adapters."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(list_adapters)
        
        # Mock AdapterRegistry with available adapters
        mock_adapter_registry.list_adapters.return_value = {
            'postgresql': 'PostgreSQLAdapter',
            'bigquery': 'BigQueryAdapter',
            's3': 'S3Adapter'
        }
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 0
        mock_adapter_registry.list_adapters.assert_called_once()
        
        # Check that adapters are displayed in sorted order
        mock_cli_success.assert_called_once_with("Available Adapters:")
        mock_cli_print.assert_any_call("  - [cyan]bigquery[/cyan]")
        mock_cli_print.assert_any_call("  - [cyan]postgresql[/cyan]")
        mock_cli_print.assert_any_call("  - [cyan]s3[/cyan]")
    
    @patch('src.cli.commands.list_commands.AdapterRegistry')
    @patch('src.cli.commands.list_commands.cli_success')
    @patch('src.cli.commands.list_commands.cli_print')
    def test_list_adapters_with_no_adapters(self, mock_cli_print, mock_cli_success, mock_adapter_registry):
        """Test list_adapters with no available adapters."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(list_adapters)
        
        # Mock AdapterRegistry with no adapters
        mock_adapter_registry.list_adapters.return_value = {}
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 0
        mock_adapter_registry.list_adapters.assert_called_once()
        
        # Check that no adapters message is displayed
        mock_cli_success.assert_called_once_with("Available Adapters:")
        mock_cli_print.assert_called_with("  [dim](No adapters available)[/dim]")
    
    @patch('src.cli.commands.list_commands.AdapterRegistry')
    @patch('src.cli.commands.list_commands.cli_success')
    @patch('src.cli.commands.list_commands.cli_print')
    def test_list_adapters_sorting(self, mock_cli_print, mock_cli_success, mock_adapter_registry):
        """Test that list_adapters sorts adapters alphabetically."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(list_adapters)
        
        # Mock AdapterRegistry with unsorted adapters
        mock_adapter_registry.list_adapters.return_value = {
            'z_adapter': 'ZAdapter',
            'a_adapter': 'AAdapter',
            'm_adapter': 'MAdapter'
        }
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 0
        
        # Verify sorted order
        print_calls = [call[0][0] for call in mock_cli_print.call_args_list if call[0][0].startswith("  - ")]
        expected_order = ["  - [cyan]a_adapter[/cyan]", "  - [cyan]m_adapter[/cyan]", "  - [cyan]z_adapter[/cyan]"]
        assert print_calls == expected_order


class TestListEvaluatorsCommand:
    """Test list_evaluators command functionality."""
    
    @patch('src.cli.commands.list_commands.EvaluatorRegistry')
    @patch('src.cli.commands.list_commands.cli_success')
    @patch('src.cli.commands.list_commands.cli_print')
    def test_list_evaluators_with_available_evaluators(self, mock_cli_print, mock_cli_success, mock_evaluator_registry):
        """Test list_evaluators with available evaluators."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(list_evaluators)
        
        # Mock EvaluatorRegistry with available tasks
        mock_evaluator_registry.get_available_tasks.return_value = [
            'Classification', 'Regression', 'Clustering'
        ]
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 0
        mock_evaluator_registry.get_available_tasks.assert_called_once()
        
        # Check that evaluators are displayed in sorted order
        mock_cli_success.assert_called_once_with("Available Evaluators:")
        mock_cli_print.assert_any_call("  - [cyan]Classification[/cyan]")
        mock_cli_print.assert_any_call("  - [cyan]Clustering[/cyan]")
        mock_cli_print.assert_any_call("  - [cyan]Regression[/cyan]")
    
    @patch('src.cli.commands.list_commands.EvaluatorRegistry')
    @patch('src.cli.commands.list_commands.cli_success')
    @patch('src.cli.commands.list_commands.cli_print')
    def test_list_evaluators_with_no_evaluators(self, mock_cli_print, mock_cli_success, mock_evaluator_registry):
        """Test list_evaluators with no available evaluators."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(list_evaluators)
        
        # Mock EvaluatorRegistry with no tasks
        mock_evaluator_registry.get_available_tasks.return_value = []
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 0
        mock_evaluator_registry.get_available_tasks.assert_called_once()
        
        # Check that no evaluators message is displayed
        mock_cli_success.assert_called_once_with("Available Evaluators:")
        mock_cli_print.assert_called_with("  [dim](No evaluators available)[/dim]")


class TestListPreprocessorsCommand:
    """Test list_preprocessors command functionality."""
    
    @patch('src.cli.commands.list_commands.PreprocessorStepRegistry')
    @patch('src.cli.commands.list_commands.cli_success')
    @patch('src.cli.commands.list_commands.cli_print')
    def test_list_preprocessors_with_available_steps(self, mock_cli_print, mock_cli_success, mock_preprocessor_registry):
        """Test list_preprocessors with available preprocessor steps."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(list_preprocessors)
        
        # Mock PreprocessorStepRegistry with available steps
        mock_preprocessor_registry._steps = {
            'standard_scaler': 'StandardScaler',
            'one_hot_encoder': 'OneHotEncoder',
            'imputer': 'SimpleImputer'
        }
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 0
        
        # Check that preprocessor steps are displayed in sorted order
        mock_cli_success.assert_called_once_with("Available Preprocessor Steps:")
        mock_cli_print.assert_any_call("  - [cyan]imputer[/cyan]")
        mock_cli_print.assert_any_call("  - [cyan]one_hot_encoder[/cyan]")
        mock_cli_print.assert_any_call("  - [cyan]standard_scaler[/cyan]")
    
    @patch('src.cli.commands.list_commands.PreprocessorStepRegistry')
    @patch('src.cli.commands.list_commands.cli_success')
    @patch('src.cli.commands.list_commands.cli_print')
    def test_list_preprocessors_with_no_steps(self, mock_cli_print, mock_cli_success, mock_preprocessor_registry):
        """Test list_preprocessors with no available steps."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(list_preprocessors)
        
        # Mock PreprocessorStepRegistry with no steps
        mock_preprocessor_registry._steps = {}
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 0
        
        # Check that no preprocessor steps message is displayed
        mock_cli_success.assert_called_once_with("Available Preprocessor Steps:")
        mock_cli_print.assert_called_with("  [dim](No preprocessor steps available)[/dim]")


class TestLoadCatalogFromDirectory:
    """Test _load_catalog_from_directory helper function."""
    
    @patch('src.cli.commands.list_commands.Path')
    @patch('src.cli.commands.list_commands.yaml')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_catalog_from_directory_success(self, mock_file_open, mock_yaml, mock_path):
        """Test successful catalog loading from directory structure."""
        # Mock directory structure
        catalog_dir = Mock()
        catalog_dir.exists.return_value = True
        
        # Mock category directories
        classification_dir = Mock()
        classification_dir.is_dir.return_value = True
        classification_dir.name = "classification"
        
        regression_dir = Mock()
        regression_dir.is_dir.return_value = True
        regression_dir.name = "regression"
        
        catalog_dir.iterdir.return_value = [classification_dir, regression_dir]
        
        # Mock YAML files in classification directory
        rf_model_file = Mock()
        rf_model_file.name = "random_forest.yaml"
        svm_model_file = Mock()
        svm_model_file.name = "svm.yaml"
        classification_dir.glob.return_value = [rf_model_file, svm_model_file]
        
        # Mock YAML files in regression directory
        lr_model_file = Mock()
        lr_model_file.name = "linear_regression.yaml"
        regression_dir.glob.return_value = [lr_model_file]
        
        # Mock path construction
        mock_path.return_value.parent.parent.parent = Mock()
        mock_path.return_value.parent.parent.parent.__truediv__ = Mock()
        mock_path.return_value.parent.parent.parent.__truediv__.return_value = catalog_dir
        
        # Mock YAML loading
        mock_yaml_data = [
            {'class_path': 'RandomForestClassifier', 'library': 'scikit-learn'},
            {'class_path': 'SVM', 'library': 'scikit-learn'},
            {'class_path': 'LinearRegression', 'library': 'scikit-learn'}
        ]
        mock_yaml.safe_load.side_effect = mock_yaml_data
        
        # Act
        result = _load_catalog_from_directory()
        
        # Assert
        expected_catalog = {
            'classification': [
                {'class_path': 'RandomForestClassifier', 'library': 'scikit-learn'},
                {'class_path': 'SVM', 'library': 'scikit-learn'}
            ],
            'regression': [
                {'class_path': 'LinearRegression', 'library': 'scikit-learn'}
            ]
        }
        
        assert result == expected_catalog
    
    @patch('src.cli.commands.list_commands.Path')
    def test_load_catalog_from_directory_no_catalog_dir(self, mock_path):
        """Test catalog loading when catalog directory doesn't exist."""
        # Mock directory doesn't exist
        catalog_dir = Mock()
        catalog_dir.exists.return_value = False
        
        mock_path.return_value.parent.parent.parent = Mock()
        mock_path.return_value.parent.parent.parent.__truediv__ = Mock()
        mock_path.return_value.parent.parent.parent.__truediv__.return_value = catalog_dir
        
        # Act
        result = _load_catalog_from_directory()
        
        # Assert
        assert result == {}
    
    @patch('src.cli.commands.list_commands.Path')
    @patch('src.cli.commands.list_commands.logger')
    @patch('src.cli.commands.list_commands.yaml')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_catalog_from_directory_yaml_error(self, mock_file_open, mock_yaml, mock_logger, mock_path):
        """Test catalog loading with YAML parsing error."""
        # Mock directory structure
        catalog_dir = Mock()
        catalog_dir.exists.return_value = True
        
        category_dir = Mock()
        category_dir.is_dir.return_value = True
        category_dir.name = "test_category"
        catalog_dir.iterdir.return_value = [category_dir]
        
        # Mock YAML file that will cause an error
        bad_model_file = Mock()
        bad_model_file.name = "bad_model.yaml"
        category_dir.glob.return_value = [bad_model_file]
        
        mock_path.return_value.parent.parent.parent = Mock()
        mock_path.return_value.parent.parent.parent.__truediv__ = Mock()
        mock_path.return_value.parent.parent.parent.__truediv__.return_value = catalog_dir
        
        # Mock YAML to raise exception
        mock_yaml.safe_load.side_effect = yaml.YAMLError("Invalid YAML")
        
        # Act
        result = _load_catalog_from_directory()
        
        # Assert
        assert result == {'test_category': []}
        mock_logger.warning.assert_called_once()


class TestListModelsCommand:
    """Test list_models command functionality."""
    
    @patch('src.cli.commands.list_commands.Path')
    @patch('src.cli.commands.list_commands._load_catalog_from_directory')
    @patch('src.cli.commands.list_commands.cli_success')
    @patch('src.cli.commands.list_commands.cli_print')
    def test_list_models_from_directory(self, mock_cli_print, mock_cli_success, mock_load_catalog, mock_path):
        """Test list_models loading from directory structure."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(list_models)
        
        # Mock catalog directory exists
        catalog_dir = Mock()
        catalog_dir.exists.return_value = True
        mock_path.return_value.parent.parent.parent = Mock()
        mock_path.return_value.parent.parent.parent.__truediv__ = Mock()
        mock_path.return_value.parent.parent.parent.__truediv__.return_value = catalog_dir
        
        # Mock catalog data
        mock_catalog = {
            'Classification': [
                {'class_path': 'RandomForestClassifier', 'library': 'scikit-learn'},
                {'class_path': 'XGBClassifier', 'library': 'xgboost'}
            ],
            'Regression': [
                {'class_path': 'LinearRegression', 'library': 'scikit-learn'}
            ]
        }
        mock_load_catalog.return_value = mock_catalog
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 0
        mock_load_catalog.assert_called_once()
        
        # Check that models are displayed by category
        mock_cli_success.assert_called_once_with("Available Models from Catalog:")
        mock_cli_print.assert_any_call("\n[bold cyan]--- Classification ---[/bold cyan]")
        mock_cli_print.assert_any_call("  - [green]RandomForestClassifier[/green] [dim](scikit-learn)[/dim]")
        mock_cli_print.assert_any_call("  - [green]XGBClassifier[/green] [dim](xgboost)[/dim]")
        mock_cli_print.assert_any_call("\n[bold cyan]--- Regression ---[/bold cyan]")
        mock_cli_print.assert_any_call("  - [green]LinearRegression[/green] [dim](scikit-learn)[/dim]")
    
    @patch('src.cli.commands.list_commands.Path')
    @patch('src.cli.commands.list_commands.load_model_catalog')
    @patch('src.cli.commands.list_commands.cli_success')
    @patch('src.cli.commands.list_commands.cli_print')
    def test_list_models_fallback_to_catalog_yaml(self, mock_cli_print, mock_cli_success, mock_load_model_catalog, mock_path):
        """Test list_models falling back to catalog.yaml file."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(list_models)
        
        # Mock catalog directory doesn't exist (fallback scenario)
        catalog_dir = Mock()
        catalog_dir.exists.return_value = False
        mock_path.return_value.parent.parent.parent = Mock()
        mock_path.return_value.parent.parent.parent.__truediv__ = Mock()
        mock_path.return_value.parent.parent.parent.__truediv__.return_value = catalog_dir
        
        # Mock fallback catalog data
        mock_catalog = {
            'Clustering': [
                {'class_path': 'KMeans', 'library': 'scikit-learn'}
            ]
        }
        mock_load_model_catalog.return_value = mock_catalog
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 0
        mock_load_model_catalog.assert_called_once()
        
        # Check that fallback catalog is used
        mock_cli_success.assert_called_once_with("Available Models from Catalog:")
        mock_cli_print.assert_any_call("\n[bold cyan]--- Clustering ---[/bold cyan]")
        mock_cli_print.assert_any_call("  - [green]KMeans[/green] [dim](scikit-learn)[/dim]")
    
    @patch('src.cli.commands.list_commands.Path')
    @patch('src.cli.commands.list_commands._load_catalog_from_directory')
    @patch('src.cli.commands.list_commands.load_model_catalog')
    @patch('src.cli.commands.list_commands.cli_error')
    def test_list_models_no_catalog_available(self, mock_cli_error, mock_load_model_catalog, mock_load_catalog, mock_path):
        """Test list_models when no catalog is available."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(list_models)
        
        # Mock catalog directory exists but returns empty catalog
        catalog_dir = Mock()
        catalog_dir.exists.return_value = True
        mock_path.return_value.parent.parent.parent = Mock()
        mock_path.return_value.parent.parent.parent.__truediv__ = Mock()
        mock_path.return_value.parent.parent.parent.__truediv__.return_value = catalog_dir
        
        mock_load_catalog.return_value = {}  # Empty catalog
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 1  # Should exit with error
        mock_cli_error.assert_called_with(
            "src/models/catalog/ 디렉토리나 catalog.yaml 파일을 찾을 수 없거나 내용이 비어있습니다."
        )
    
    @patch('src.cli.commands.list_commands.Path')
    @patch('src.cli.commands.list_commands._load_catalog_from_directory')
    @patch('src.cli.commands.list_commands.cli_success')
    @patch('src.cli.commands.list_commands.cli_print')
    def test_list_models_handles_missing_model_info(self, mock_cli_print, mock_cli_success, mock_load_catalog, mock_path):
        """Test list_models handles models with missing information."""
        runner = CliRunner()
        app = typer.Typer()
        app.command()(list_models)
        
        # Mock catalog directory exists
        catalog_dir = Mock()
        catalog_dir.exists.return_value = True
        mock_path.return_value.parent.parent.parent = Mock()
        mock_path.return_value.parent.parent.parent.__truediv__ = Mock()
        mock_path.return_value.parent.parent.parent.__truediv__.return_value = catalog_dir
        
        # Mock catalog with missing information
        mock_catalog = {
            'Test': [
                {'class_path': 'ModelWithInfo', 'library': 'test-lib'},
                {'class_path': 'ModelMissingLibrary'},  # Missing library
                {'library': 'lib-only'},  # Missing class_path
                {}  # Missing both
            ]
        }
        mock_load_catalog.return_value = mock_catalog
        
        # Act
        result = runner.invoke(app, [])
        
        # Assert
        assert result.exit_code == 0
        
        # Check that models are displayed with fallback values
        mock_cli_print.assert_any_call("  - [green]ModelWithInfo[/green] [dim](test-lib)[/dim]")
        mock_cli_print.assert_any_call("  - [green]ModelMissingLibrary[/green] [dim](Unknown)[/dim]")
        mock_cli_print.assert_any_call("  - [green]Unknown[/green] [dim](lib-only)[/dim]")
        mock_cli_print.assert_any_call("  - [green]Unknown[/green] [dim](Unknown)[/dim]")


class TestListCommandsIntegration:
    """Test list commands integration scenarios."""
    
    @patch('src.cli.commands.list_commands.AdapterRegistry')
    @patch('src.cli.commands.list_commands.EvaluatorRegistry')
    @patch('src.cli.commands.list_commands.PreprocessorStepRegistry')
    @patch('src.cli.commands.list_commands._load_catalog_from_directory')
    @patch('src.cli.commands.list_commands.Path')
    def test_all_list_commands_work_together(self, mock_path, mock_load_catalog, 
                                           mock_preprocessor_registry, mock_evaluator_registry, mock_adapter_registry):
        """Test that all list commands work together without conflicts."""
        
        # Mock all registries
        mock_adapter_registry.list_adapters.return_value = {'postgresql': 'PostgreSQLAdapter'}
        mock_evaluator_registry.get_available_tasks.return_value = ['Classification']
        mock_preprocessor_registry._steps = {'scaler': 'StandardScaler'}
        
        # Mock models catalog
        catalog_dir = Mock()
        catalog_dir.exists.return_value = True
        mock_path.return_value.parent.parent.parent = Mock()
        mock_path.return_value.parent.parent.parent.__truediv__ = Mock()
        mock_path.return_value.parent.parent.parent.__truediv__.return_value = catalog_dir
        
        mock_load_catalog.return_value = {
            'Classification': [{'class_path': 'RandomForest', 'library': 'sklearn'}]
        }
        
        runner = CliRunner()
        
        # Test all commands
        app1 = typer.Typer()
        app1.command()(list_adapters)
        result1 = runner.invoke(app1, [])
        
        app2 = typer.Typer()
        app2.command()(list_evaluators)
        result2 = runner.invoke(app2, [])
        
        app3 = typer.Typer()
        app3.command()(list_preprocessors)
        result3 = runner.invoke(app3, [])
        
        app4 = typer.Typer()
        app4.command()(list_models)
        result4 = runner.invoke(app4, [])
        
        # Assert all commands work
        assert result1.exit_code == 0
        assert result2.exit_code == 0
        assert result3.exit_code == 0
        assert result4.exit_code == 0
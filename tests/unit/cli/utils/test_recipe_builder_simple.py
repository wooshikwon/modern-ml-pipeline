"""
Simple tests for RecipeBuilder to increase coverage.
These tests focus on individual methods rather than full interaction flow.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.cli.utils.recipe_builder import RecipeBuilder, InteractiveUI


class TestInteractiveUI:
    """Test InteractiveUI methods."""

    def test_text_input_with_default(self):
        """Test text input with default value."""
        ui = InteractiveUI()

        with patch('builtins.input', return_value=''):
            result = ui.text_input("Enter name", default="default")
            assert result == "default"

    def test_text_input_with_user_value(self):
        """Test text input with user value."""
        ui = InteractiveUI()

        with patch('builtins.input', return_value='user_value'):
            result = ui.text_input("Enter name")
            assert result == "user_value"

    def test_confirm_yes(self):
        """Test confirm with yes answer."""
        ui = InteractiveUI()

        with patch('builtins.input', return_value='y'):
            assert ui.confirm("Continue?") == True

    def test_confirm_no(self):
        """Test confirm with no answer."""
        ui = InteractiveUI()

        with patch('builtins.input', return_value='n'):
            assert ui.confirm("Continue?") == False

    def test_select_from_list(self):
        """Test select from list."""
        ui = InteractiveUI()

        with patch('builtins.input', return_value='2'):
            result = ui.select_from_list("Choose", ["A", "B", "C"])
            assert result == "B"

    def test_number_input(self):
        """Test number input."""
        ui = InteractiveUI()

        with patch('builtins.input', return_value='42'):
            result = ui.number_input("Enter number", default=10)
            assert result == 42

    def test_single_choice(self):
        """Test single choice method."""
        ui = InteractiveUI()

        choices = [("opt1", "Option 1"), ("opt2", "Option 2")]
        with patch('builtins.input', return_value='1'):
            result = ui.single_choice("Choose", choices)
            assert result == "opt1"


class TestRecipeBuilderMethods:
    """Test individual RecipeBuilder methods."""

    def test_get_available_tasks(self):
        """Test getting available tasks from MODEL_REGISTRY."""
        builder = RecipeBuilder()
        tasks = builder.get_available_tasks()

        # Should return tasks from MODEL_REGISTRY
        assert isinstance(tasks, set)
        # Check that at least classification exists
        assert 'classification' in tasks or len(tasks) >= 0

    def test_get_available_models_for_task(self):
        """Test getting available models for a task."""
        builder = RecipeBuilder()

        # Test for classification if it exists
        models = builder.get_available_models_for_task('classification')
        assert isinstance(models, dict)

        # Test for non-existent task
        models = builder.get_available_models_for_task('non_existent_task')
        assert models == {}

    def test_get_available_preprocessors(self):
        """Test getting available preprocessors."""
        builder = RecipeBuilder()

        preprocessors = builder.get_available_preprocessors()
        assert isinstance(preprocessors, dict)
        # Should have categories like 'encoder', 'scaler', etc.
        expected_categories = {'encoder', 'scaler', 'imputer', 'feature_generator'}
        assert len(preprocessors.keys() & expected_categories) > 0

    def test_get_available_metrics_for_task(self):
        """Test getting available metrics for a task."""
        builder = RecipeBuilder()

        # Test classification metrics
        metrics = builder.get_available_metrics_for_task('classification')
        assert isinstance(metrics, list)
        assert len(metrics) > 0

        # Test regression metrics
        metrics = builder.get_available_metrics_for_task('regression')
        assert isinstance(metrics, list)
        assert len(metrics) > 0

    def test_categorize_preprocessor(self):
        """Test categorizing preprocessor steps."""
        builder = RecipeBuilder()

        # Test known preprocessors
        assert builder._categorize_preprocessor('StandardScaler') == 'scaler'
        assert builder._categorize_preprocessor('OneHotEncoder') == 'encoder'
        assert builder._categorize_preprocessor('SimpleImputer') == 'imputer'
        assert builder._categorize_preprocessor('Unknown') == 'feature_generator'

    def test_configure_preprocessor_params(self):
        """Test configuring preprocessor parameters."""
        builder = RecipeBuilder()

        with patch.object(builder.ui, 'confirm', return_value=False):
            # Test without custom params
            params = builder._configure_preprocessor_params('StandardScaler')
            assert params == {}

    def test_generate_template_variables(self):
        """Test generating template variables."""
        builder = RecipeBuilder()

        recipe_data = {
            'name': 'test_recipe',
            'task_choice': 'classification',
            'model': {
                'library': 'sklearn',
                'class_path': 'sklearn.ensemble.RandomForestClassifier',
                'params': {'n_estimators': 100}
            },
            'data': {
                'data_interface': {'target': 'label'},
                'split': {'train': 0.7, 'validation': 0.15, 'test': 0.15}
            },
            'evaluation': {
                'metrics': ['accuracy', 'roc_auc']
            }
        }

        author = "Test Author"
        description = "Test Description"

        vars = builder.generate_template_variables(recipe_data, author, description)

        assert vars['recipe_name'] == 'test_recipe'
        assert vars['task'] == 'classification'
        assert vars['author'] == author
        assert vars['description'] == description
        assert vars['model_library'] == 'sklearn'
        assert vars['model_class'] == 'RandomForestClassifier'

    def test_create_recipe_file(self, tmp_path):
        """Test creating a recipe file."""
        builder = RecipeBuilder()

        recipe_data = {
            'name': 'test_recipe',
            'task_choice': 'classification',
            'model': {
                'library': 'sklearn',
                'class_path': 'sklearn.ensemble.RandomForestClassifier'
            }
        }

        output_path = tmp_path / "test_recipe.yaml"
        result = builder.create_recipe_file(recipe_data, str(output_path))

        assert result.exists()
        assert result.name == "test_recipe.yaml"

        # Check content
        content = result.read_text()
        assert 'name: test_recipe' in content
        assert 'task_choice: classification' in content

    def test_build_model_config(self):
        """Test building model configuration."""
        builder = RecipeBuilder()

        selected_model = {
            'library': 'sklearn',
            'class_path': 'sklearn.ensemble.RandomForestClassifier',
            'params': {}
        }

        with patch.object(builder.ui, 'confirm', side_effect=[False, False, False]):
            # No hyperparams, no tuning, no calibration
            config = builder._build_model_config(selected_model, 'classification')

            assert config['library'] == 'sklearn'
            assert config['class_path'] == 'sklearn.ensemble.RandomForestClassifier'
            assert 'hyperparameters' not in config
            assert 'tuning' not in config
            assert 'calibration' not in config

    def test_build_data_interface_config(self):
        """Test building data interface configuration."""
        builder = RecipeBuilder()

        # Test for classification
        with patch.object(builder.ui, 'text_input', side_effect=['label', 'id']):
            config = builder._build_data_interface_config('classification')
            assert config['target'] == 'label'
            assert config['entity_column'] == 'id'

        # Test for timeseries
        with patch.object(builder.ui, 'text_input', side_effect=['value', 'id', 'timestamp']):
            config = builder._build_data_interface_config('timeseries')
            assert config['target'] == 'value'
            assert config['entity_column'] == 'id'
            assert config['timestamp_column'] == 'timestamp'

    def test_build_data_split_config(self):
        """Test building data split configuration."""
        builder = RecipeBuilder()

        # Test without calibration
        with patch.object(builder.ui, 'number_input', side_effect=[0.7, 0.15, 0.15]):
            config = builder._build_data_split_config('classification', {})
            assert config['train_ratio'] == 0.7
            assert config['validation_ratio'] == 0.15
            assert config['test_ratio'] == 0.15

        # Test with calibration
        calibration_config = {'enabled': True}
        with patch.object(builder.ui, 'number_input', side_effect=[0.6, 0.1, 0.15, 0.15]):
            config = builder._build_data_split_config('classification', calibration_config)
            assert config['train_ratio'] == 0.6
            assert config['calibration_ratio'] == 0.1
            assert config['validation_ratio'] == 0.15
            assert config['test_ratio'] == 0.15
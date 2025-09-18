"""
Comprehensive Test Suite for Recipe Builder
Following Real Object Testing philosophy

Tests cover:
- RecipeBuilder interactive flow
- Task selection and configuration
- Model selection and hyperparameter configuration
- Data configuration (loader, fetcher, split)
- Validation and error handling
- Recipe file generation
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from typing import Dict, Any, Optional
import yaml

from src.cli.utils.recipe_builder import RecipeBuilder, build_recipe_interactive, create_recipe_file


class MockInteractiveUI:
    """Mock UI for controlled testing of RecipeBuilder."""

    def __init__(self, selections: Dict[str, Any] = None):
        """Initialize mock UI with predefined selections."""
        self.selections = selections or {}
        self.call_history = []

    def show_panel(self, *args, **kwargs):
        """Record panel display calls."""
        self.call_history.append(('show_panel', args, kwargs))

    def show_info(self, message):
        """Record info display calls."""
        self.call_history.append(('show_info', message))

    def show_success(self, message):
        """Record success display calls."""
        self.call_history.append(('show_success', message))

    def show_warning(self, message):
        """Record warning display calls."""
        self.call_history.append(('show_warning', message))

    def show_table(self, headers, rows, title=None):
        """Record table display calls."""
        self.call_history.append(('show_table', headers, rows, title))

    def print_divider(self, text=None):
        """Record divider display calls."""
        self.call_history.append(('print_divider', text))

    def text_input(self, prompt, default=None, validator=None, **kwargs):
        """Return predefined text input."""
        key = prompt.lower()

        # Map prompts to selection keys
        if "recipe 이름" in prompt.lower():
            return self.selections.get('recipe_name', 'test_recipe')
        elif "sql 파일" in prompt.lower() or "데이터 파일" in prompt.lower():
            return self.selections.get('data_path', 'data/train.csv')
        elif "target column" in prompt.lower():
            return self.selections.get('target_column', 'target')
        elif "entity column" in prompt.lower():
            return self.selections.get('entity_column', 'entity_id')
        elif "timestamp column" in prompt.lower():
            return self.selections.get('timestamp_column', 'timestamp')
        elif "작성자" in prompt.lower():
            return self.selections.get('author', 'Test Author')
        elif "설명" in prompt.lower():
            return self.selections.get('description', 'Test recipe description')

        return default or 'test_value'

    def number_input(self, prompt, default=None, min_value=None, max_value=None):
        """Return predefined number input."""
        if "estimators" in prompt.lower():
            return self.selections.get('n_estimators', 100)
        elif "max_depth" in prompt.lower():
            return self.selections.get('max_depth', 10)
        elif "train" in prompt.lower():
            return self.selections.get('train_split', 0.7)
        elif "validation" in prompt.lower():
            return self.selections.get('val_split', 0.15)
        elif "test" in prompt.lower():
            return self.selections.get('test_split', 0.15)
        elif "random_state" in prompt.lower():
            return self.selections.get('random_state', 42)

        return default or 1

    def confirm(self, prompt, default=False, **kwargs):
        """Return predefined confirmation."""
        if "feature store" in prompt.lower():
            return self.selections.get('use_feature_store', False)
        elif "전처리" in prompt.lower() or "preprocessor" in prompt.lower():
            return self.selections.get('use_preprocessor', False)
        elif "캘리브레이션" in prompt.lower() or "calibration" in prompt.lower():
            return self.selections.get('use_calibration', False)
        elif "하이퍼파라미터 튜닝" in prompt.lower():
            return self.selections.get('enable_tuning', False)
        elif "저장" in prompt.lower():
            return self.selections.get('save_recipe', True)

        return default

    def select_from_list(self, prompt, options, **kwargs):
        """Return predefined selection from list."""
        if "task" in prompt.lower():
            return self.selections.get('task_choice', 'classification')
        elif "모델" in prompt.lower() or "model" in prompt.lower():
            return self.selections.get('model_choice', options[0] if options else None)
        elif "fetcher" in prompt.lower():
            return self.selections.get('fetcher_type', 'pass_through')
        elif "평가 지표" in prompt.lower() or "metric" in prompt.lower():
            return self.selections.get('metrics', options[:2] if options else [])

        return options[0] if options else None

    def multi_select(self, prompt, options):
        """Return predefined multi-selection."""
        if "평가 지표" in prompt.lower() or "metric" in prompt.lower():
            metrics = self.selections.get('metrics', ['accuracy', 'roc_auc'])
            if isinstance(metrics, list):
                return metrics
            return [metrics]

        return options[:2] if options else []


class TestRecipeBuilder:
    """Test RecipeBuilder class and its methods."""

    def test_initialization(self):
        """Test RecipeBuilder initialization."""
        builder = RecipeBuilder()

        assert builder.ui is not None
        assert hasattr(builder, 'build_recipe_interactively')
        assert hasattr(builder, 'create_recipe_file')
        assert hasattr(builder, 'get_available_tasks')
        assert hasattr(builder, 'get_available_models_for_task')
        assert hasattr(builder, 'get_available_preprocessors')
        assert hasattr(builder, 'get_available_metrics_for_task')

    def test_build_recipe_classification(self):
        """Test building a classification recipe."""
        builder = RecipeBuilder()

        # Set up mock UI with classification selections
        mock_ui = MockInteractiveUI({
            'recipe_name': 'clf_recipe',
            'task_choice': 'classification',
            'model_choice': 'sklearn.ensemble.RandomForestClassifier',
            'data_path': 'data/train.csv',
            'target_column': 'label',
            'entity_column': 'id',
            'use_feature_store': False,
            'use_preprocessor': False,
            'use_calibration': False,
            'enable_tuning': False,
            'n_estimators': 100,
            'max_depth': 10,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'metrics': ['accuracy', 'roc_auc'],
            'author': 'Test User',
            'description': 'Classification test recipe'
        })

        builder.ui = mock_ui

        recipe = builder.build_recipe_interactively()

        assert recipe is not None
        assert recipe['name'] == 'clf_recipe'
        assert recipe['task_choice'] == 'classification'
        assert recipe['model']['class_path'] == 'sklearn.ensemble.RandomForestClassifier'
        assert recipe['model']['library'] == 'sklearn'
        assert recipe['model']['hyperparameters']['values']['n_estimators'] == 100
        assert recipe['model']['hyperparameters']['values']['max_depth'] == 10
        assert recipe['data']['loader']['source_uri'] == 'data/train.csv'
        assert recipe['data']['data_interface']['target_column'] == 'label'
        assert recipe['data']['data_interface']['entity_columns'] == ['id']
        assert recipe['data']['split']['train'] == 0.7
        assert recipe['data']['split']['validation'] == 0.15
        assert recipe['data']['split']['test'] == 0.15
        assert 'accuracy' in recipe['evaluation']['metrics']
        assert 'roc_auc' in recipe['evaluation']['metrics']

    def test_build_recipe_regression(self):
        """Test building a regression recipe."""
        builder = RecipeBuilder()

        mock_ui = MockInteractiveUI({
            'recipe_name': 'reg_recipe',
            'task_choice': 'regression',
            'model_choice': 'sklearn.linear_model.LinearRegression',
            'data_path': 'data/train.parquet',
            'target_column': 'price',
            'entity_column': 'product_id',
            'use_feature_store': False,
            'metrics': ['rmse', 'mae', 'r2_score']
        })

        builder.ui = mock_ui

        recipe = builder.build_recipe_interactively()

        assert recipe['task_choice'] == 'regression'
        assert recipe['model']['class_path'] == 'sklearn.linear_model.LinearRegression'
        assert recipe['model']['library'] == 'sklearn'
        assert recipe['data']['data_interface']['target_column'] == 'price'
        assert 'rmse' in recipe['evaluation']['metrics']
        assert 'mae' in recipe['evaluation']['metrics']
        assert 'r2_score' in recipe['evaluation']['metrics']

    def test_build_recipe_timeseries(self):
        """Test building a timeseries recipe."""
        builder = RecipeBuilder()

        mock_ui = MockInteractiveUI({
            'recipe_name': 'ts_recipe',
            'task_choice': 'timeseries',
            'model_choice': 'src.models.custom.lstm_timeseries.LSTMTimeSeriesModel',
            'data_path': 'data/timeseries.csv',
            'target_column': 'value',
            'entity_column': 'sensor_id',
            'timestamp_column': 'timestamp',
            'metrics': ['rmse', 'mae']
        })

        builder.ui = mock_ui

        recipe = builder.build_recipe_interactively()

        assert recipe['task_choice'] == 'timeseries'
        assert 'lstm' in recipe['model']['class_path'].lower()
        assert recipe['data']['data_interface']['timestamp_column'] == 'timestamp'

    def test_build_recipe_with_feature_store(self):
        """Test building recipe with feature store configuration."""
        builder = RecipeBuilder()

        mock_ui = MockInteractiveUI({
            'recipe_name': 'fs_recipe',
            'task_choice': 'classification',
            'model_choice': 'sklearn.ensemble.RandomForestClassifier',
            'use_feature_store': True,
            'fetcher_type': 'feast',
            'data_path': 'feast://feature_service',
            'target_column': 'target'
        })

        builder.ui = mock_ui

        recipe = builder.build_recipe_interactively()

        assert recipe['data']['fetcher']['type'] == 'feast'
        assert 'feast://' in recipe['data']['loader']['source_uri']

    def test_build_recipe_with_preprocessor(self):
        """Test building recipe with preprocessor configuration."""
        builder = RecipeBuilder()

        mock_ui = MockInteractiveUI({
            'recipe_name': 'prep_recipe',
            'task_choice': 'classification',
            'use_preprocessor': True
        })

        # Mock preprocessor selection
        with patch.object(builder, '_configure_preprocessor') as mock_config:
            mock_config.return_value = {
                'class_path': 'src.components.preprocessor.TabularPreprocessor',
                'config': {'scaling': 'standard', 'encoding': 'onehot'}
            }

            builder.ui = mock_ui
            recipe = builder.build_recipe_interactively()

            assert recipe['preprocessor'] is not None
            mock_config.assert_called_once()

    def test_build_recipe_with_hyperparameter_tuning(self):
        """Test building recipe with hyperparameter tuning enabled."""
        builder = RecipeBuilder()

        mock_ui = MockInteractiveUI({
            'recipe_name': 'tuning_recipe',
            'task_choice': 'classification',
            'model_choice': 'sklearn.ensemble.RandomForestClassifier',
            'enable_tuning': True,
            'n_trials': 50,
            'optimization_metric': 'roc_auc'
        })

        # Mock tuning configuration
        with patch.object(builder, '_configure_hyperparameter_tuning') as mock_tuning:
            mock_tuning.return_value = {
                'tuning_enabled': True,
                'n_trials': 50,
                'optimization_metric': 'roc_auc',
                'tunable': {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
                    'max_depth': {'type': 'int', 'low': 5, 'high': 30}
                }
            }

            builder.ui = mock_ui
            recipe = builder.build_recipe_interactively()

            assert recipe['model']['hyperparameters']['tuning_enabled'] is True
            mock_tuning.assert_called_once()

    def test_build_recipe_with_calibration(self):
        """Test building recipe with calibration configuration."""
        builder = RecipeBuilder()

        mock_ui = MockInteractiveUI({
            'recipe_name': 'cal_recipe',
            'task_choice': 'classification',
            'use_calibration': True
        })

        # Mock calibration configuration
        with patch.object(builder, '_configure_calibration') as mock_cal:
            mock_cal.return_value = {
                'method': 'isotonic',
                'cv': 3
            }

            builder.ui = mock_ui
            recipe = builder.build_recipe_interactively()

            assert recipe['model']['calibration'] is not None
            mock_cal.assert_called_once()

    def test_save_recipe(self, tmp_path):
        """Test saving recipe to file."""
        builder = RecipeBuilder()

        recipe_data = {
            'name': 'test_recipe',
            'task_choice': 'classification',
            'model': {'class_path': 'sklearn.ensemble.RandomForestClassifier'}
        }

        output_path = tmp_path / "test_recipe.yaml"

        result_path = builder.save_recipe(recipe_data, str(output_path))

        assert result_path.exists()

        # Load and verify saved recipe
        with open(result_path, 'r') as f:
            loaded_recipe = yaml.safe_load(f)

        assert loaded_recipe['name'] == 'test_recipe'
        assert loaded_recipe['task_choice'] == 'classification'

    def test_validate_recipe_structure(self):
        """Test recipe structure validation."""
        builder = RecipeBuilder()

        # Valid recipe
        valid_recipe = {
            'name': 'test',
            'task_choice': 'classification',
            'model': {'class_path': 'sklearn.ensemble.RandomForestClassifier'},
            'data': {'loader': {'source_uri': 'data.csv'}},
            'evaluation': {'metrics': ['accuracy']}
        }

        # Should not raise
        builder._validate_recipe_structure(valid_recipe)

        # Invalid recipe (missing required field)
        invalid_recipe = {
            'name': 'test',
            # Missing task_choice
            'model': {'class_path': 'sklearn.ensemble.RandomForestClassifier'}
        }

        with pytest.raises(ValueError):
            builder._validate_recipe_structure(invalid_recipe)


class TestRecipeBuilderHelperFunctions:
    """Test helper functions for recipe building."""

    def test_build_recipe_interactive(self):
        """Test the build_recipe_interactive function."""
        with patch('src.cli.utils.recipe_builder.RecipeBuilder') as MockBuilder:
            mock_instance = MagicMock()
            mock_instance.build_recipe_interactively.return_value = {'name': 'test'}
            MockBuilder.return_value = mock_instance

            result = build_recipe_interactive()

            assert result == {'name': 'test'}
            mock_instance.build_recipe_interactively.assert_called_once()

    def test_create_recipe_file(self, tmp_path):
        """Test the create_recipe_file function."""
        recipe_data = {
            'name': 'test_recipe',
            'task_choice': 'classification',
            'model': {
                'class_path': 'sklearn.ensemble.RandomForestClassifier',
                'library': 'sklearn'
            }
        }

        output_path = tmp_path / "output_recipe.yaml"

        result_path = create_recipe_file(recipe_data, str(output_path))

        assert result_path.exists()
        assert result_path.name == "output_recipe.yaml"

        # Verify content
        with open(result_path, 'r') as f:
            content = yaml.safe_load(f)

        assert content['name'] == 'test_recipe'

    def test_create_recipe_file_default_path(self, tmp_path):
        """Test create_recipe_file with default output path."""
        recipe_data = {'name': 'test_recipe'}

        with patch('src.cli.utils.recipe_builder.Path.cwd', return_value=tmp_path):
            result_path = create_recipe_file(recipe_data)

            assert result_path.exists()
            assert result_path.parent == tmp_path / 'recipes'

    def test_get_model_options_for_task(self):
        """Test getting model options for different tasks."""
        builder = RecipeBuilder()

        # Classification models
        clf_models = builder._get_model_options('classification')
        assert any('RandomForest' in model for model in clf_models)
        assert any('LogisticRegression' in model for model in clf_models)

        # Regression models
        reg_models = builder._get_model_options('regression')
        assert any('LinearRegression' in model for model in reg_models)
        assert any('RandomForestRegressor' in model for model in reg_models)

        # Timeseries models
        ts_models = builder._get_model_options('timeseries')
        assert any('LSTM' in model for model in ts_models)

    def test_get_metrics_for_task(self):
        """Test getting metrics for different tasks."""
        builder = RecipeBuilder()

        # Classification metrics
        clf_metrics = builder._get_metrics_for_task('classification')
        assert 'accuracy' in clf_metrics
        assert 'roc_auc' in clf_metrics
        assert 'f1_score' in clf_metrics

        # Regression metrics
        reg_metrics = builder._get_metrics_for_task('regression')
        assert 'rmse' in reg_metrics
        assert 'mae' in reg_metrics
        assert 'r2_score' in reg_metrics

        # Timeseries metrics
        ts_metrics = builder._get_metrics_for_task('timeseries')
        assert 'rmse' in ts_metrics
        assert 'mae' in ts_metrics
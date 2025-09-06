"""
Unit tests for OptunaOptimizer.
Tests Optuna hyperparameter optimization functionality with comprehensive mocking.
"""

import pytest
import sys
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

from src.components.trainer.modules.optimizer import OptunaOptimizer
from tests.helpers.builders import TrainerDataBuilder, SettingsBuilder


class TestOptunaOptimizerInitialization:
    """Test OptunaOptimizer initialization."""
    
    @patch('src.components.trainer.modules.optimizer.OptunaOptimizer._create_pruner')
    def test_optimizer_initialization_with_tuning_config(self, mock_create_pruner):
        """Test initialization with tuning-enabled configuration."""
        # Arrange: Create tuning-enabled settings and mock pruner creation
        mock_pruner_instance = Mock()
        mock_create_pruner.return_value = mock_pruner_instance
        
        settings = SettingsBuilder.build_tuning_enabled_config("classification")
        mock_factory_provider = Mock()
        
        # Act: Initialize optimizer
        optimizer = OptunaOptimizer(settings, mock_factory_provider)
        
        # Assert: Check initialization
        assert optimizer.settings == settings
        assert optimizer.factory_provider == mock_factory_provider
        assert optimizer.pruner is not None  # Pruner should be created
        assert optimizer.pruner == mock_pruner_instance
        mock_create_pruner.assert_called_once()
    
    @patch('src.components.trainer.modules.optimizer.OptunaOptimizer._create_pruner')
    def test_optimizer_initialization_with_factory_provider(self, mock_create_pruner):
        """Test that factory_provider is properly stored."""
        # Arrange
        mock_pruner_instance = Mock()
        mock_create_pruner.return_value = mock_pruner_instance
        
        settings = SettingsBuilder.build_tuning_enabled_config("regression")
        mock_factory_provider = Mock(return_value="mock_factory")
        
        # Act
        optimizer = OptunaOptimizer(settings, mock_factory_provider)
        
        # Assert
        assert callable(optimizer.factory_provider)
        assert optimizer.factory_provider() == "mock_factory"

    def test_create_pruner_success(self):
        """Test successful pruner creation."""
        # Arrange
        mock_pruner_instance = Mock()
        mock_median_pruner = Mock(return_value=mock_pruner_instance)
        
        # Create mock optuna modules
        mock_optuna = Mock()
        mock_optuna.pruners.MedianPruner = mock_median_pruner
        
        with patch.dict('sys.modules', {'optuna': mock_optuna, 'optuna.pruners': mock_optuna.pruners}):
            settings = SettingsBuilder.build_tuning_enabled_config()
            mock_factory_provider = Mock()
            
            # Act
            optimizer = OptunaOptimizer(settings, mock_factory_provider)
            
            # Assert
            assert optimizer.pruner == mock_pruner_instance
            mock_median_pruner.assert_called_once()

    def test_create_pruner_import_error(self):
        """Test pruner creation when optuna import fails."""
        # Arrange
        settings = SettingsBuilder.build_tuning_enabled_config()
        mock_factory_provider = Mock()
        
        # Act: OptunaOptimizer will try to import optuna and fail
        optimizer = OptunaOptimizer(settings, mock_factory_provider)
        
        # Assert: Pruner should be None when import fails
        assert optimizer.pruner is None


class TestOptunaOptimizerOptimize:
    """Test OptunaOptimizer optimize method."""
    
    @patch('src.components.trainer.modules.optimizer.OptunaOptimizer._create_pruner')
    def test_optimize_study_creation_and_execution(self, mock_create_pruner):
        """Test study creation and optimization execution with mocks."""
        # Arrange: Setup mocks
        mock_create_pruner.return_value = Mock()
        settings = SettingsBuilder.build_tuning_enabled_config()
        
        # Mock factory and optuna_integration
        mock_factory = Mock()
        mock_optuna_integration = Mock()
        mock_factory.create_optuna_integration.return_value = mock_optuna_integration
        mock_factory_provider = Mock(return_value=mock_factory)
        
        # Mock study
        mock_study = Mock()
        mock_study.best_params = {'n_estimators': 100, 'max_depth': 10}
        mock_study.best_value = 0.95
        mock_study.trials = [Mock(value=0.9), Mock(value=0.95), Mock(value=0.85)]
        for trial in mock_study.trials:
            trial.state.name = 'COMPLETE'
        mock_optuna_integration.create_study.return_value = mock_study
        
        # Mock hyperparameter suggestion
        mock_optuna_integration.suggest_hyperparameters.return_value = {
            'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.1
        }
        
        # Mock training callback
        def mock_training_callback(train_df, params, seed):
            return {'score': 0.95, 'accuracy': 0.95}
        
        # Test data
        train_df = TrainerDataBuilder.build_classification_data(n_samples=50)
        
        # Act: Run optimization
        optimizer = OptunaOptimizer(settings, mock_factory_provider)
        result = optimizer.optimize(train_df, mock_training_callback)
        
        # Assert: Check study creation
        mock_optuna_integration.create_study.assert_called_once()
        study_call_args = mock_optuna_integration.create_study.call_args[1]
        assert 'direction' in study_call_args
        assert 'study_name' in study_call_args
        assert 'pruner' in study_call_args
        
        # Assert: Check optimization execution
        mock_study.optimize.assert_called_once()
        optimize_args = mock_study.optimize.call_args[1]
        assert 'n_trials' in optimize_args
        assert 'timeout' in optimize_args
        assert 'callbacks' in optimize_args
        
        # Assert: Check result structure
        assert result['enabled'] is True
        assert result['engine'] == 'optuna'
        assert result['best_params'] == {'n_estimators': 100, 'max_depth': 10}
        assert result['best_score'] == 0.95
        assert result['total_trials'] == 3
        assert 'optimization_time' in result
        assert isinstance(result['optimization_time'], float)

    @patch('src.components.trainer.modules.optimizer.OptunaOptimizer._create_pruner')
    def test_hyperparameter_space_definition(self, mock_create_pruner):
        """Test hyperparameter space definition verification."""
        # Arrange: Create settings with specific hyperparameter space
        mock_create_pruner.return_value = Mock()
        settings = SettingsBuilder.build_tuning_enabled_config()
        
        # Mock factory and integration
        mock_factory = Mock()
        mock_optuna_integration = Mock()
        mock_factory.create_optuna_integration.return_value = mock_optuna_integration
        mock_factory_provider = Mock(return_value=mock_factory)
        
        # Mock study
        mock_study = Mock()
        mock_study.best_params = {}
        mock_study.best_value = 0.8
        mock_study.trials = []
        
        def mock_optimize(objective, n_trials=None, timeout=None, callbacks=None):
            # Mock the optimize method to call the objective function once
            mock_trial = Mock()
            mock_trial.number = 0
            objective(mock_trial)
            
        mock_study.optimize = mock_optimize
        mock_optuna_integration.create_study.return_value = mock_study
        
        # Set up hyperparameter space in settings  
        tunable_params = {
            'n_estimators': {'type': 'int', 'range': [50, 200]},
            'max_depth': {'type': 'int', 'range': [3, 20]},
            'learning_rate': {'type': 'float', 'range': [0.01, 1.0]}
        }
        settings.recipe.model.hyperparameters.tunable = tunable_params
        
        # Mock training callback
        mock_training_callback = Mock(return_value={'score': 0.8})
        train_df = TrainerDataBuilder.build_classification_data(n_samples=30)
        
        # Act
        optimizer = OptunaOptimizer(settings, mock_factory_provider)
        result = optimizer.optimize(train_df, mock_training_callback)
        
        # Assert: Check that search space includes tunable hyperparameters
        expected_search_space = {
            'n_estimators': {'type': 'int', 'range': [50, 200]},
            'max_depth': {'type': 'int', 'range': [3, 20]},
            'learning_rate': {'type': 'float', 'range': [0.01, 1.0]}
        }
        assert result['search_space'] == expected_search_space
        
        # Assert: Check that suggest_hyperparameters was called with correct tunable params
        mock_optuna_integration.suggest_hyperparameters.assert_called()
        suggest_call_args = mock_optuna_integration.suggest_hyperparameters.call_args[0]
        assert suggest_call_args[1] == tunable_params

    @patch('src.components.trainer.modules.optimizer.OptunaOptimizer._create_pruner')
    def test_objective_function_execution(self, mock_create_pruner):
        """Test objective function execution and score return."""
        # Arrange
        mock_create_pruner.return_value = Mock()
        settings = SettingsBuilder.build_tuning_enabled_config()
        
        # Mock factory
        mock_factory = Mock()
        mock_optuna_integration = Mock()
        mock_factory.create_optuna_integration.return_value = mock_optuna_integration
        mock_factory_provider = Mock(return_value=mock_factory)
        
        # Mock study and trial
        mock_study = Mock()
        mock_trial = Mock()
        mock_trial.number = 5
        mock_study.best_params = {}
        mock_study.best_value = 0.92
        mock_study.trials = []
        mock_optuna_integration.create_study.return_value = mock_study
        
        # Mock hyperparameter suggestion
        suggested_params = {'n_estimators': 150, 'max_depth': 8}
        mock_optuna_integration.suggest_hyperparameters.return_value = suggested_params
        
        # Mock training callback - check it receives correct parameters
        training_results = []
        def mock_training_callback(train_df, params, seed):
            training_results.append({
                'params': params,
                'seed': seed,
                'train_df_shape': train_df.shape
            })
            return {'score': 0.92}
        
        # Capture objective function for testing
        captured_objective = None
        def capture_optimize(objective, **kwargs):
            nonlocal captured_objective
            captured_objective = objective
            return None
        mock_study.optimize.side_effect = capture_optimize
        
        train_df = TrainerDataBuilder.build_regression_data(n_samples=40)
        
        # Act
        optimizer = OptunaOptimizer(settings, mock_factory_provider)
        optimizer.optimize(train_df, mock_training_callback)
        
        # Test the captured objective function
        assert captured_objective is not None
        score = captured_objective(mock_trial)
        
        # Assert: Check objective function behavior
        assert score == 0.92
        assert len(training_results) == 1
        assert training_results[0]['params'] == suggested_params
        assert training_results[0]['seed'] == mock_trial.number
        assert training_results[0]['train_df_shape'] == train_df.shape
        
        # Assert: Check that suggest_hyperparameters was called with trial
        mock_optuna_integration.suggest_hyperparameters.assert_called_with(
            mock_trial, settings.recipe.model.hyperparameters.tunable or {}
        )

    @patch('src.components.trainer.modules.optimizer.OptunaOptimizer._create_pruner')
    def test_optimization_with_pruned_trials(self, mock_create_pruner):
        """Test optimization with pruned trials tracking."""
        # Arrange
        settings = SettingsBuilder.build_tuning_enabled_config()
        
        # Mock factory
        mock_factory = Mock()
        mock_optuna_integration = Mock()
        mock_factory.create_optuna_integration.return_value = mock_optuna_integration
        mock_factory_provider = Mock(return_value=mock_factory)
        
        # Mock study with completed and pruned trials
        mock_study = Mock()
        
        # Create mock trials - some completed, some pruned
        completed_trial1 = Mock(value=0.9)
        completed_trial1.state.name = 'COMPLETE'
        
        completed_trial2 = Mock(value=0.95)
        completed_trial2.state.name = 'COMPLETE'
        
        pruned_trial1 = Mock(value=None)
        pruned_trial1.state.name = 'PRUNED'
        
        pruned_trial2 = Mock(value=None)
        pruned_trial2.state.name = 'PRUNED'
        
        mock_study.trials = [completed_trial1, completed_trial2, pruned_trial1, pruned_trial2]
        mock_study.best_params = {'n_estimators': 120}
        mock_study.best_value = 0.95
        mock_optuna_integration.create_study.return_value = mock_study
        
        # Mock training callback
        mock_training_callback = Mock(return_value={'score': 0.9})
        train_df = TrainerDataBuilder.build_clustering_data(n_samples=60)
        
        # Act
        optimizer = OptunaOptimizer(settings, mock_factory_provider)
        result = optimizer.optimize(train_df, mock_training_callback)
        
        # Assert: Check trial statistics
        assert result['total_trials'] == 4
        assert result['pruned_trials'] == 2
        assert result['optimization_history'] == [0.9, 0.95]  # Only non-None values


class TestOptunaOptimizerErrorHandling:
    """Test OptunaOptimizer error handling."""
    
    @patch('src.components.trainer.modules.optimizer.OptunaOptimizer._create_pruner')
    def test_optimization_factory_provider_error(self, mock_create_pruner):
        """Test error handling when factory_provider fails."""
        # Arrange
        settings = SettingsBuilder.build_tuning_enabled_config()
        
        # Mock factory_provider that raises exception
        def failing_factory_provider():
            raise RuntimeError("Factory creation failed")
        
        train_df = TrainerDataBuilder.build_classification_data(n_samples=20)
        mock_training_callback = Mock()
        
        # Act & Assert
        optimizer = OptunaOptimizer(settings, failing_factory_provider)
        with pytest.raises(RuntimeError, match="Factory creation failed"):
            optimizer.optimize(train_df, mock_training_callback)
    
    @patch('src.components.trainer.modules.optimizer.OptunaOptimizer._create_pruner')
    def test_optimization_training_callback_error(self, mock_create_pruner):
        """Test error handling when training_callback fails."""
        # Arrange
        settings = SettingsBuilder.build_tuning_enabled_config()
        
        # Mock factory
        mock_factory = Mock()
        mock_optuna_integration = Mock()
        mock_factory.create_optuna_integration.return_value = mock_optuna_integration
        mock_factory_provider = Mock(return_value=mock_factory)
        
        # Mock study
        mock_study = Mock()
        mock_study.best_params = {}
        mock_study.best_value = 0.0
        mock_study.trials = []
        mock_optuna_integration.create_study.return_value = mock_study
        
        # Failing training callback
        def failing_training_callback(train_df, params, seed):
            raise ValueError("Training failed")
        
        # Capture objective function
        captured_objective = None
        def capture_optimize(objective, **kwargs):
            nonlocal captured_objective
            captured_objective = objective
        mock_study.optimize.side_effect = capture_optimize
        
        train_df = TrainerDataBuilder.build_regression_data(n_samples=15)
        
        # Act
        optimizer = OptunaOptimizer(settings, mock_factory_provider)
        optimizer.optimize(train_df, failing_training_callback)
        
        # Test that objective function propagates the error
        mock_trial = Mock()
        mock_trial.number = 1
        mock_optuna_integration.suggest_hyperparameters.return_value = {}
        
        with pytest.raises(ValueError, match="Training failed"):
            captured_objective(mock_trial)

    @patch('src.components.trainer.modules.optimizer.OptunaOptimizer._create_pruner')
    def test_optimization_study_creation_error(self, mock_create_pruner):
        """Test error handling when study creation fails."""
        # Arrange
        settings = SettingsBuilder.build_tuning_enabled_config()
        
        # Mock factory with failing optuna_integration
        mock_factory = Mock()
        mock_optuna_integration = Mock()
        mock_optuna_integration.create_study.side_effect = RuntimeError("Study creation failed")
        mock_factory.create_optuna_integration.return_value = mock_optuna_integration
        mock_factory_provider = Mock(return_value=mock_factory)
        
        train_df = TrainerDataBuilder.build_causal_data(n_samples=25)
        mock_training_callback = Mock()
        
        # Act & Assert
        optimizer = OptunaOptimizer(settings, mock_factory_provider)
        with pytest.raises(RuntimeError, match="Study creation failed"):
            optimizer.optimize(train_df, mock_training_callback)


class TestOptunaOptimizerIntegration:
    """Test OptunaOptimizer integration scenarios."""
    
    @patch('src.components.trainer.modules.optimizer.OptunaOptimizer._create_pruner')
    def test_complete_optimization_workflow_classification(self, mock_create_pruner):
        """Test complete optimization workflow for classification task."""
        # Arrange: Full workflow setup
        settings = SettingsBuilder.build_tuning_enabled_config("classification")
        
        # Mock complete factory setup
        mock_factory = Mock()
        mock_optuna_integration = Mock()
        mock_factory.create_optuna_integration.return_value = mock_optuna_integration
        mock_factory_provider = Mock(return_value=mock_factory)
        
        # Mock study with realistic behavior
        mock_study = Mock()
        mock_study.best_params = {'n_estimators': 180, 'max_depth': 12}
        mock_study.best_value = 0.98
        
        # Create realistic trial history
        trial_values = [0.85, 0.90, 0.92, 0.95, 0.98, 0.96]
        mock_trials = []
        for i, value in enumerate(trial_values):
            trial = Mock(value=value)
            trial.state.name = 'COMPLETE' if i < 5 else 'PRUNED'
            mock_trials.append(trial)
        
        mock_study.trials = mock_trials
        mock_optuna_integration.create_study.return_value = mock_study
        
        # Mock hyperparameter suggestions (different for each trial)
        param_suggestions = [
            {'n_estimators': 100, 'max_depth': 8},
            {'n_estimators': 150, 'max_depth': 10},
            {'n_estimators': 180, 'max_depth': 12},
        ]
        mock_optuna_integration.suggest_hyperparameters.side_effect = param_suggestions
        
        # Realistic training callback with score variation
        call_count = 0
        def realistic_training_callback(train_df, params, seed):
            nonlocal call_count
            # Simulate score variation based on parameters
            base_score = 0.85
            param_bonus = (params.get('n_estimators', 100) - 100) / 1000
            score = min(base_score + param_bonus + call_count * 0.02, 0.98)
            call_count += 1
            return {'score': score}
        
        train_df = TrainerDataBuilder.build_data_for_task_type(
            "classification", n_samples=100, add_entity_columns=["user_id"]
        )
        
        # Act: Run complete workflow
        optimizer = OptunaOptimizer(settings, mock_factory_provider)
        result = optimizer.optimize(train_df, realistic_training_callback)
        
        # Assert: Check complete result structure
        assert result['enabled'] is True
        assert result['engine'] == 'optuna'
        assert isinstance(result['best_params'], dict)
        assert isinstance(result['best_score'], (int, float))
        assert isinstance(result['optimization_history'], list)
        assert isinstance(result['total_trials'], int)
        assert isinstance(result['pruned_trials'], int)
        assert isinstance(result['optimization_time'], float)
        assert isinstance(result['search_space'], dict)
        
        # Assert: Check reasonable values
        assert result['best_score'] > 0
        assert result['total_trials'] > 0
        assert result['optimization_time'] >= 0
        assert len(result['optimization_history']) <= result['total_trials']
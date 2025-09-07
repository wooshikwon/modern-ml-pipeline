"""
Tests for optuna_integration - Hyperparameter tuning with Optuna

Comprehensive testing of Optuna integration utilities including:
- Optuna dependency management and import handling
- Logging callbacks for trial progress tracking  
- OptunaIntegration class initialization and configuration
- Study creation with different samplers and pruners
- Hyperparameter suggestion with various parameter types
- Error handling and integration scenarios

Test Categories:
1. TestOptunaRequireImport - Optuna dependency and import management
2. TestOptunaLoggingCallback - Trial logging and progress callbacks
3. TestOptunaIntegrationInit - OptunaIntegration class initialization
4. TestOptunaStudyCreation - Optuna study creation and configuration
5. TestOptunaSuggestions - Hyperparameter suggestion functionality
6. TestOptunaEdgeCases - Error handling and edge case scenarios
"""

import pytest
from unittest.mock import patch, MagicMock, call
import sys
from typing import Dict, Any, Optional

from src.utils.integrations.optuna_integration import (
    _require_optuna,
    logging_callback, 
    OptunaIntegration
)


class TestOptunaRequireImport:
    """Test Optuna dependency and import management."""

    @patch('src.utils.integrations.optuna_integration.optuna', create=True)
    def test_require_optuna_success(self, mock_optuna):
        """Test successful Optuna import."""
        mock_optuna_module = MagicMock()
        mock_optuna_module.__name__ = 'optuna'
        
        with patch.dict('sys.modules', {'optuna': mock_optuna_module}):
            result = _require_optuna()
            
            assert result == mock_optuna_module

    def test_require_optuna_import_error(self):
        """Test ImportError when Optuna is not available."""
        with patch.dict('sys.modules', {}, clear=True):
            # Remove optuna from sys.modules if it exists
            if 'optuna' in sys.modules:
                del sys.modules['optuna']
            
            with pytest.raises(ImportError) as exc_info:
                _require_optuna()
                
            assert "Optuna가 설치되지 않았습니다" in str(exc_info.value)
            assert "hyperparameter tuning을 사용하려면 optuna를 설치하세요" in str(exc_info.value)

    @patch('builtins.__import__', side_effect=ModuleNotFoundError("No module named 'optuna'"))
    def test_require_optuna_module_not_found(self, mock_import):
        """Test handling of ModuleNotFoundError for Optuna."""
        with pytest.raises(ImportError) as exc_info:
            _require_optuna()
            
        assert "Optuna가 설치되지 않았습니다" in str(exc_info.value)

    @patch('builtins.__import__', side_effect=Exception("Unexpected import error"))
    def test_require_optuna_general_exception(self, mock_import):
        """Test handling of general exceptions during Optuna import."""
        with pytest.raises(ImportError) as exc_info:
            _require_optuna()
            
        assert "Optuna가 설치되지 않았습니다" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None


class TestOptunaLoggingCallback:
    """Test trial logging and progress callbacks."""

    @patch('src.utils.integrations.optuna_integration.logger')
    def test_logging_callback_completed_trial(self, mock_logger):
        """Test logging callback for completed trial."""
        mock_study = MagicMock()
        mock_study.best_value = 0.95123
        
        mock_trial = MagicMock()
        mock_trial.number = 42
        mock_trial.value = 0.91456
        
        logging_callback(mock_study, mock_trial)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "Trial 42 완료" in call_args
        assert "현재 점수: 0.91456" in call_args
        assert "최고 점수: 0.95123" in call_args

    @patch('src.utils.integrations.optuna_integration.logger')
    def test_logging_callback_pruned_trial(self, mock_logger):
        """Test logging callback for pruned trial."""
        mock_study = MagicMock()
        mock_study.best_value = 0.95123
        
        mock_trial = MagicMock()
        mock_trial.number = 43
        mock_trial.value = None  # Pruned trial
        
        logging_callback(mock_study, mock_trial)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "Trial 43 완료" in call_args
        assert "현재 점수: N/A (pruned)" in call_args
        assert "최고 점수: 0.95123" in call_args

    @patch('src.utils.integrations.optuna_integration.logger')
    def test_logging_callback_no_best_value(self, mock_logger):
        """Test logging callback when no best value exists yet."""
        mock_study = MagicMock()
        mock_study.best_value = None
        
        mock_trial = MagicMock()
        mock_trial.number = 1
        mock_trial.value = 0.85123
        
        logging_callback(mock_study, mock_trial)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "Trial 1 완료" in call_args
        assert "현재 점수: 0.85123" in call_args
        assert "최고 점수: N/A" in call_args

    @patch('src.utils.integrations.optuna_integration.logger')
    def test_logging_callback_missing_attributes(self, mock_logger):
        """Test logging callback with missing attributes."""
        mock_study = MagicMock()
        del mock_study.best_value  # Simulate missing attribute
        
        mock_trial = MagicMock()
        del mock_trial.number  # Simulate missing attribute
        del mock_trial.value   # Simulate missing attribute
        
        logging_callback(mock_study, mock_trial)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "Trial N/A 완료" in call_args
        assert "현재 점수: N/A (pruned)" in call_args
        assert "최고 점수: N/A" in call_args

    @patch('src.utils.integrations.optuna_integration.logger')
    def test_logging_callback_exception_handling(self, mock_logger):
        """Test logging callback exception handling."""
        mock_study = MagicMock()
        mock_study.best_value = MagicMock(side_effect=Exception("Access error"))
        
        mock_trial = MagicMock()
        
        logging_callback(mock_study, mock_trial)
        
        mock_logger.info.assert_called_once_with(
            "Optuna 로깅 콜백에서 정보를 읽을 수 없습니다."
        )

    @patch('src.utils.integrations.optuna_integration.logger')
    def test_logging_callback_precision_handling(self, mock_logger):
        """Test logging callback with high precision values."""
        mock_study = MagicMock()
        mock_study.best_value = 0.123456789
        
        mock_trial = MagicMock()
        mock_trial.number = 100
        mock_trial.value = 0.987654321
        
        logging_callback(mock_study, mock_trial)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        # Should be formatted to 5 decimal places
        assert "현재 점수: 0.98765" in call_args
        assert "최고 점수: 0.12346" in call_args


class TestOptunaIntegrationInit:
    """Test OptunaIntegration class initialization."""

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters."""
        tuning_config = {'param1': 'value1'}
        seed = 42
        timeout = 300
        n_jobs = 4
        pruning = {'type': 'median'}
        
        integration = OptunaIntegration(
            tuning_config=tuning_config,
            seed=seed,
            timeout=timeout,
            n_jobs=n_jobs,
            pruning=pruning
        )
        
        assert integration.tuning_config == tuning_config
        assert integration.seed == seed
        assert integration.timeout == timeout
        assert integration.n_jobs == n_jobs
        assert integration.pruning == pruning

    def test_init_with_minimal_parameters(self):
        """Test initialization with minimal parameters."""
        tuning_config = {'param1': 'value1'}
        
        integration = OptunaIntegration(tuning_config=tuning_config)
        
        assert integration.tuning_config == tuning_config
        assert integration.seed is None
        assert integration.timeout is None
        assert integration.n_jobs is None
        assert integration.pruning == {}

    def test_init_with_none_pruning(self):
        """Test initialization with None pruning parameter."""
        tuning_config = {'param1': 'value1'}
        
        integration = OptunaIntegration(
            tuning_config=tuning_config,
            pruning=None
        )
        
        assert integration.pruning == {}

    def test_init_with_empty_config(self):
        """Test initialization with empty tuning config."""
        tuning_config = {}
        
        integration = OptunaIntegration(tuning_config=tuning_config)
        
        assert integration.tuning_config == {}

    def test_init_parameter_types(self):
        """Test initialization parameter type handling."""
        # Test with different types
        integration = OptunaIntegration(
            tuning_config="string_config",  # Could be any type
            seed=0,  # Edge case for seed
            timeout=-1,  # Edge case for timeout
            n_jobs=1,  # Minimum n_jobs
            pruning={'complex': {'nested': 'structure'}}
        )
        
        assert integration.tuning_config == "string_config"
        assert integration.seed == 0
        assert integration.timeout == -1
        assert integration.n_jobs == 1
        assert integration.pruning == {'complex': {'nested': 'structure'}}


class TestOptunaStudyCreation:
    """Test Optuna study creation and configuration."""

    @patch('src.utils.integrations.optuna_integration._require_optuna')
    def test_create_study_with_seed(self, mock_require_optuna):
        """Test study creation with seed."""
        mock_optuna = MagicMock()
        mock_tpe_sampler = MagicMock()
        mock_study = MagicMock()
        
        mock_optuna.samplers.TPESampler.return_value = mock_tpe_sampler
        mock_optuna.create_study.return_value = mock_study
        mock_require_optuna.return_value = mock_optuna
        
        integration = OptunaIntegration({'config': 'value'}, seed=42)
        result = integration.create_study('maximize', 'test_study')
        
        # Verify TPESampler called with seed
        mock_optuna.samplers.TPESampler.assert_called_once_with(seed=42)
        
        # Verify create_study called with correct parameters
        mock_optuna.create_study.assert_called_once_with(
            direction='maximize',
            study_name='test_study',
            sampler=mock_tpe_sampler,
            pruner=None
        )
        
        assert result == mock_study

    @patch('src.utils.integrations.optuna_integration._require_optuna')
    def test_create_study_without_seed(self, mock_require_optuna):
        """Test study creation without seed."""
        mock_optuna = MagicMock()
        mock_tpe_sampler = MagicMock()
        mock_study = MagicMock()
        
        mock_optuna.samplers.TPESampler.return_value = mock_tpe_sampler
        mock_optuna.create_study.return_value = mock_study
        mock_require_optuna.return_value = mock_optuna
        
        integration = OptunaIntegration({'config': 'value'})  # No seed
        result = integration.create_study('minimize', 'test_study_2')
        
        # Verify TPESampler called without seed
        mock_optuna.samplers.TPESampler.assert_called_once_with()
        
        # Verify create_study called with correct parameters
        mock_optuna.create_study.assert_called_once_with(
            direction='minimize',
            study_name='test_study_2',
            sampler=mock_tpe_sampler,
            pruner=None
        )
        
        assert result == mock_study

    @patch('src.utils.integrations.optuna_integration._require_optuna')
    def test_create_study_with_pruner(self, mock_require_optuna):
        """Test study creation with pruner."""
        mock_optuna = MagicMock()
        mock_tpe_sampler = MagicMock()
        mock_study = MagicMock()
        mock_pruner = MagicMock()
        
        mock_optuna.samplers.TPESampler.return_value = mock_tpe_sampler
        mock_optuna.create_study.return_value = mock_study
        mock_require_optuna.return_value = mock_optuna
        
        integration = OptunaIntegration({'config': 'value'}, seed=123)
        result = integration.create_study('maximize', 'test_study_3', pruner=mock_pruner)
        
        # Verify create_study called with pruner
        mock_optuna.create_study.assert_called_once_with(
            direction='maximize',
            study_name='test_study_3',
            sampler=mock_tpe_sampler,
            pruner=mock_pruner
        )
        
        assert result == mock_study

    @patch('src.utils.integrations.optuna_integration._require_optuna')
    def test_create_study_optuna_import_error(self, mock_require_optuna):
        """Test study creation when Optuna import fails."""
        mock_require_optuna.side_effect = ImportError("Optuna not available")
        
        integration = OptunaIntegration({'config': 'value'})
        
        with pytest.raises(ImportError):
            integration.create_study('maximize', 'test_study')


class TestOptunaSuggestions:
    """Test hyperparameter suggestion functionality."""

    @patch('src.utils.integrations.optuna_integration._require_optuna')
    def test_suggest_int_parameter(self, mock_require_optuna):
        """Test suggesting integer parameter."""
        mock_trial = MagicMock()
        mock_trial.suggest_int.return_value = 50
        
        integration = OptunaIntegration({'config': 'value'})
        param_space = {
            'n_estimators': {
                'type': 'int',
                'low': 10,
                'high': 100,
                'log': False
            }
        }
        
        result = integration.suggest_hyperparameters(mock_trial, param_space)
        
        mock_trial.suggest_int.assert_called_once_with('n_estimators', 10, 100, log=False)
        assert result == {'n_estimators': 50}

    @patch('src.utils.integrations.optuna_integration._require_optuna')
    def test_suggest_float_parameter(self, mock_require_optuna):
        """Test suggesting float parameter."""
        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 0.05
        
        integration = OptunaIntegration({'config': 'value'})
        param_space = {
            'learning_rate': {
                'type': 'float',
                'low': 0.01,
                'high': 0.1,
                'log': True
            }
        }
        
        result = integration.suggest_hyperparameters(mock_trial, param_space)
        
        mock_trial.suggest_float.assert_called_once_with('learning_rate', 0.01, 0.1, log=True)
        assert result == {'learning_rate': 0.05}

    @patch('src.utils.integrations.optuna_integration._require_optuna')
    def test_suggest_categorical_parameter(self, mock_require_optuna):
        """Test suggesting categorical parameter."""
        mock_trial = MagicMock()
        mock_trial.suggest_categorical.return_value = 'gini'
        
        integration = OptunaIntegration({'config': 'value'})
        param_space = {
            'criterion': {
                'type': 'categorical',
                'choices': ['gini', 'entropy', 'log_loss']
            }
        }
        
        result = integration.suggest_hyperparameters(mock_trial, param_space)
        
        mock_trial.suggest_categorical.assert_called_once_with('criterion', ['gini', 'entropy', 'log_loss'])
        assert result == {'criterion': 'gini'}

    @patch('src.utils.integrations.optuna_integration._require_optuna')
    def test_suggest_mixed_parameters(self, mock_require_optuna):
        """Test suggesting mixed parameter types."""
        mock_trial = MagicMock()
        mock_trial.suggest_int.return_value = 75
        mock_trial.suggest_float.return_value = 0.03
        mock_trial.suggest_categorical.return_value = 'rbf'
        
        integration = OptunaIntegration({'config': 'value'})
        param_space = {
            'n_estimators': {
                'type': 'int',
                'low': 50,
                'high': 100
            },
            'learning_rate': {
                'type': 'float',
                'low': 0.01,
                'high': 0.1
            },
            'kernel': {
                'type': 'categorical',
                'choices': ['linear', 'rbf', 'poly']
            }
        }
        
        result = integration.suggest_hyperparameters(mock_trial, param_space)
        
        assert result == {
            'n_estimators': 75,
            'learning_rate': 0.03,
            'kernel': 'rbf'
        }

    def test_suggest_fixed_parameters(self):
        """Test handling of fixed (non-tunable) parameters."""
        mock_trial = MagicMock()
        
        integration = OptunaIntegration({'config': 'value'})
        param_space = {
            'random_state': 42,
            'verbose': True,
            'model_name': 'RandomForest'
        }
        
        result = integration.suggest_hyperparameters(mock_trial, param_space)
        
        # No suggest methods should be called for fixed parameters
        assert not hasattr(mock_trial, 'suggest_int') or not mock_trial.suggest_int.called
        assert result == {
            'random_state': 42,
            'verbose': True,
            'model_name': 'RandomForest'
        }

    @patch('src.utils.integrations.optuna_integration._require_optuna')
    def test_suggest_unknown_parameter_type(self, mock_require_optuna):
        """Test handling of unknown parameter type."""
        mock_trial = MagicMock()
        
        integration = OptunaIntegration({'config': 'value'})
        param_space = {
            'unknown_param': {
                'type': 'unknown_type',
                'value': 'some_value'
            }
        }
        
        result = integration.suggest_hyperparameters(mock_trial, param_space)
        
        # Should return the dict as-is for unknown types
        assert result == {
            'unknown_param': {
                'type': 'unknown_type',
                'value': 'some_value'
            }
        }

    @patch('src.utils.integrations.optuna_integration._require_optuna')
    def test_suggest_parameters_with_log_flag_variations(self, mock_require_optuna):
        """Test parameter suggestion with different log flag values."""
        mock_trial = MagicMock()
        mock_trial.suggest_int.return_value = 100
        mock_trial.suggest_float.return_value = 0.5
        
        integration = OptunaIntegration({'config': 'value'})
        param_space = {
            'param1': {
                'type': 'int',
                'low': 1,
                'high': 1000,
                'log': True
            },
            'param2': {
                'type': 'float',
                'low': 0.1,
                'high': 1.0
                # No log flag - should default to False
            },
            'param3': {
                'type': 'int',
                'low': 1,
                'high': 10,
                'log': 0  # Falsy value
            }
        }
        
        result = integration.suggest_hyperparameters(mock_trial, param_space)
        
        # Verify log flags are properly converted to bool
        mock_trial.suggest_int.assert_any_call('param1', 1, 1000, log=True)
        mock_trial.suggest_float.assert_called_once_with('param2', 0.1, 1.0, log=False)
        mock_trial.suggest_int.assert_any_call('param3', 1, 10, log=False)


class TestOptunaEdgeCases:
    """Test error handling and edge case scenarios."""

    def test_suggest_parameters_empty_space(self):
        """Test suggestion with empty parameter space."""
        mock_trial = MagicMock()
        integration = OptunaIntegration({'config': 'value'})
        
        result = integration.suggest_hyperparameters(mock_trial, {})
        
        assert result == {}

    @patch('src.utils.integrations.optuna_integration._require_optuna')
    def test_suggest_parameters_missing_choices(self, mock_require_optuna):
        """Test categorical parameter with missing choices."""
        mock_trial = MagicMock()
        mock_trial.suggest_categorical.return_value = None
        
        integration = OptunaIntegration({'config': 'value'})
        param_space = {
            'criterion': {
                'type': 'categorical'
                # Missing 'choices' key
            }
        }
        
        result = integration.suggest_hyperparameters(mock_trial, param_space)
        
        # Should call with empty list when choices missing
        mock_trial.suggest_categorical.assert_called_once_with('criterion', [])

    @patch('src.utils.integrations.optuna_integration._require_optuna')
    def test_suggest_parameters_type_conversion(self, mock_require_optuna):
        """Test parameter type conversion handling."""
        mock_trial = MagicMock()
        mock_trial.suggest_int.return_value = 50
        mock_trial.suggest_float.return_value = 0.5
        
        integration = OptunaIntegration({'config': 'value'})
        param_space = {
            'int_param': {
                'type': 'int',
                'low': '10',  # String values
                'high': '100'
            },
            'float_param': {
                'type': 'float',
                'low': '0.1',
                'high': '1.0'
            }
        }
        
        result = integration.suggest_hyperparameters(mock_trial, param_space)
        
        # Verify type conversion to int/float
        mock_trial.suggest_int.assert_called_once_with('int_param', 10, 100, log=False)
        mock_trial.suggest_float.assert_called_once_with('float_param', 0.1, 1.0, log=False)

    def test_complex_nested_parameter_space(self):
        """Test handling of complex nested parameter structures."""
        mock_trial = MagicMock()
        integration = OptunaIntegration({'config': 'value'})
        
        param_space = {
            'model': {
                'nested_config': {
                    'param1': 'value1',
                    'param2': 42
                }
            },
            'preprocessing': ['step1', 'step2']
        }
        
        result = integration.suggest_hyperparameters(mock_trial, param_space)
        
        # Should return complex structures as-is
        assert result == param_space

    @patch('src.utils.integrations.optuna_integration._require_optuna')
    def test_parameter_suggestion_with_exception(self, mock_require_optuna):
        """Test parameter suggestion when trial methods raise exceptions."""
        mock_trial = MagicMock()
        mock_trial.suggest_int.side_effect = ValueError("Invalid range")
        
        integration = OptunaIntegration({'config': 'value'})
        param_space = {
            'bad_param': {
                'type': 'int',
                'low': 100,
                'high': 10  # Invalid range
            }
        }
        
        # Should propagate the exception
        with pytest.raises(ValueError):
            integration.suggest_hyperparameters(mock_trial, param_space)

    def test_integration_with_complex_tuning_config(self):
        """Test integration with complex tuning configuration."""
        complex_config = {
            'optimization': {
                'direction': 'maximize',
                'n_trials': 100,
                'timeout': 3600
            },
            'pruning': {
                'type': 'median',
                'n_startup_trials': 5
            },
            'sampler': {
                'type': 'tpe',
                'n_startup_trials': 10
            }
        }
        
        integration = OptunaIntegration(
            tuning_config=complex_config,
            seed=42,
            timeout=3600,
            n_jobs=-1,
            pruning={'type': 'median'}
        )
        
        assert integration.tuning_config == complex_config
        assert integration.seed == 42
        assert integration.timeout == 3600
        assert integration.n_jobs == -1
        assert integration.pruning == {'type': 'median'}
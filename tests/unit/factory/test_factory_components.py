"""
Unit tests for Factory component creation methods.
Tests all create_* methods with registry integration, caching, and error handling.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from src.factory.factory import Factory
from tests.helpers.builders import RecipeBuilder


class TestFactoryDataAdapter:
    """Test Factory create_data_adapter method."""
    
    @patch('src.factory.factory.AdapterRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_data_adapter_with_explicit_type(self, mock_ensure_components, mock_adapter_registry):
        """Test create_data_adapter with explicitly specified adapter type."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        mock_adapter_instance = MagicMock()
        mock_adapter_registry.create.return_value = mock_adapter_instance
        
        factory = Factory(mock_settings)
        
        # Act
        result = factory.create_data_adapter(adapter_type='sql')
        
        # Assert
        assert result is mock_adapter_instance
        mock_adapter_registry.create.assert_called_once_with('sql', mock_settings)
        # Should be cached
        assert factory._component_cache['adapter_sql'] is mock_adapter_instance
    
    @patch('src.factory.factory.AdapterRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_data_adapter_auto_detection(self, mock_ensure_components, mock_adapter_registry):
        """Test create_data_adapter with automatic type detection from URI."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_settings.recipe.data.loader.source_uri = "data/file.csv"
        
        mock_adapter_instance = MagicMock()
        mock_adapter_registry.create.return_value = mock_adapter_instance
        
        factory = Factory(mock_settings)
        
        # Act
        result = factory.create_data_adapter()
        
        # Assert
        assert result is mock_adapter_instance
        mock_adapter_registry.create.assert_called_once_with('storage', mock_settings)
        # Should be cached with auto key
        assert factory._component_cache['adapter_auto'] is mock_adapter_instance
    
    @patch('src.factory.factory.AdapterRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_data_adapter_returns_cached_instance(self, mock_ensure_components, mock_adapter_registry):
        """Test create_data_adapter returns cached instance on second call."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        mock_adapter_instance = MagicMock()
        mock_adapter_registry.create.return_value = mock_adapter_instance
        
        factory = Factory(mock_settings)
        
        # Act - Call twice
        result1 = factory.create_data_adapter(adapter_type='storage')
        result2 = factory.create_data_adapter(adapter_type='storage')
        
        # Assert
        assert result1 is result2
        assert result1 is mock_adapter_instance
        # Registry.create should only be called once due to caching
        mock_adapter_registry.create.assert_called_once_with('storage', mock_settings)
    
    @patch('src.factory.factory.AdapterRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_data_adapter_handles_registry_error(self, mock_ensure_components, mock_adapter_registry):
        """Test create_data_adapter handles registry creation errors."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        mock_adapter_registry.create.side_effect = Exception("Adapter creation failed")
        mock_adapter_registry.list_adapters.return_value = {'storage': 'StorageAdapter', 'sql': 'SqlAdapter'}
        
        factory = Factory(mock_settings)
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            factory.create_data_adapter(adapter_type='invalid')
        
        assert "Failed to create adapter 'invalid'" in str(exc_info.value)
        assert "Available: ['storage', 'sql']" in str(exc_info.value)


class TestFactoryFetcher:
    """Test Factory create_fetcher method."""
    
    @patch('src.factory.factory.FetcherRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_fetcher_batch_mode_passthrough(self, mock_ensure_components, mock_fetcher_registry):
        """Test create_fetcher in batch mode with passthrough configuration."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_settings.config.environment.env_name = "local"
        mock_settings.feature_store = None
        
        mock_fetcher_instance = MagicMock()
        mock_fetcher_registry.create.return_value = mock_fetcher_instance
        
        factory = Factory(mock_settings)
        
        # Act
        result = factory.create_fetcher(run_mode='batch')
        
        # Assert
        assert result is mock_fetcher_instance
        mock_fetcher_registry.create.assert_called_once_with('pass_through')
        # Should be cached
        assert factory._component_cache['fetcher_batch'] is mock_fetcher_instance
    
    @patch('src.factory.factory.FetcherRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_fetcher_feature_store_mode(self, mock_ensure_components, mock_fetcher_registry):
        """Test create_fetcher with feature store configuration."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_settings.config.environment.env_name = "dev"
        mock_settings.feature_store.provider = "feast"
        mock_settings.recipe.data.fetcher.type = "feature_store"
        
        mock_fetcher_instance = MagicMock()
        mock_fetcher_registry.create.return_value = mock_fetcher_instance
        
        factory = Factory(mock_settings)
        
        # Act
        result = factory.create_fetcher(run_mode='batch')
        
        # Assert
        assert result is mock_fetcher_instance
        mock_fetcher_registry.create.assert_called_once_with(
            'feature_store', 
            settings=mock_settings, 
            factory=factory
        )
    
    @patch('src.factory.factory.FetcherRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_fetcher_serving_mode_validation(self, mock_ensure_components, mock_fetcher_registry):
        """Test create_fetcher validates serving mode requirements."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_settings.config.environment.env_name = "local"
        mock_settings.feature_store = None
        mock_settings.recipe.data.fetcher = None
        
        factory = Factory(mock_settings)
        
        # Act & Assert
        with pytest.raises(TypeError) as exc_info:
            factory.create_fetcher(run_mode='serving')
        
        assert "Serving 모드에서는 Feature Store 연결이 필요합니다" in str(exc_info.value)
    
    @patch('src.factory.factory.FetcherRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_fetcher_caching_by_mode(self, mock_ensure_components, mock_fetcher_registry):
        """Test create_fetcher caches instances separately by run mode."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_settings.config.environment.env_name = "local"
        mock_settings.feature_store = None
        
        mock_fetcher_batch = MagicMock()
        mock_fetcher_serving = MagicMock()
        mock_fetcher_registry.create.side_effect = [mock_fetcher_batch, mock_fetcher_serving]
        
        factory = Factory(mock_settings)
        
        # Act
        result_batch = factory.create_fetcher(run_mode='batch')
        result_serving = factory.create_fetcher(run_mode='batch')  # Same mode, should be cached
        
        # Assert
        assert result_batch is result_serving  # Same instance due to caching
        assert result_batch is mock_fetcher_batch
        # Should only call create once due to caching
        mock_fetcher_registry.create.assert_called_once()


class TestFactoryPreprocessor:
    """Test Factory create_preprocessor method."""
    
    @patch('src.factory.factory.Preprocessor')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_preprocessor_with_config(self, mock_ensure_components, mock_preprocessor_class):
        """Test create_preprocessor when preprocessor is configured."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_settings.recipe.preprocessor = MagicMock()  # Preprocessor configured
        
        mock_preprocessor_instance = MagicMock()
        mock_preprocessor_class.return_value = mock_preprocessor_instance
        
        factory = Factory(mock_settings)
        
        # Act
        result = factory.create_preprocessor()
        
        # Assert
        assert result is mock_preprocessor_instance
        mock_preprocessor_class.assert_called_once_with(settings=mock_settings)
        # Should be cached
        assert factory._component_cache['preprocessor'] is mock_preprocessor_instance
    
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_preprocessor_no_config_returns_none(self, mock_ensure_components):
        """Test create_preprocessor returns None when not configured."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        # Remove preprocessor config
        del mock_settings.recipe.preprocessor
        
        factory = Factory(mock_settings)
        
        # Act
        result = factory.create_preprocessor()
        
        # Assert
        assert result is None
    
    @patch('src.factory.factory.Preprocessor')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_preprocessor_handles_creation_error(self, mock_ensure_components, mock_preprocessor_class):
        """Test create_preprocessor handles preprocessor creation errors."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_settings.recipe.preprocessor = MagicMock()
        
        mock_preprocessor_class.side_effect = Exception("Preprocessor creation failed")
        
        factory = Factory(mock_settings)
        
        # Act & Assert
        with pytest.raises(Exception, match="Preprocessor creation failed"):
            factory.create_preprocessor()


class TestFactoryModel:
    """Test Factory create_model method."""
    
    @patch.object(Factory, '_create_from_class_path')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_model_with_fixed_hyperparameters(self, mock_ensure_components, mock_create_from_class_path):
        """Test create_model with fixed hyperparameters."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_settings.recipe.model.class_path = "sklearn.ensemble.RandomForestClassifier"
        
        # Mock hyperparameters with get_fixed_params method
        mock_hyperparams = MagicMock()
        mock_hyperparams.get_fixed_params.return_value = {'n_estimators': 100, 'random_state': 42}
        mock_settings.recipe.model.hyperparameters = mock_hyperparams
        
        mock_model_instance = MagicMock()
        mock_create_from_class_path.return_value = mock_model_instance
        
        factory = Factory(mock_settings)
        
        # Act
        result = factory.create_model()
        
        # Assert
        assert result is mock_model_instance
        mock_create_from_class_path.assert_called_once_with(
            "sklearn.ensemble.RandomForestClassifier",
            {'n_estimators': 100, 'random_state': 42}
        )
        # Should be cached
        assert factory._component_cache['model'] is mock_model_instance
    
    @patch.object(Factory, '_create_from_class_path')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_model_with_dict_hyperparameters(self, mock_ensure_components, mock_create_from_class_path):
        """Test create_model with dict-style hyperparameters."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_settings.recipe.model.class_path = "sklearn.svm.SVC"
        
        # Mock hyperparameters without get_fixed_params (fallback to dict)
        mock_hyperparams = MagicMock()
        mock_hyperparams.__dict__ = {'C': 1.0, 'kernel': 'rbf'}
        # Remove get_fixed_params to trigger fallback
        del mock_hyperparams.get_fixed_params
        mock_settings.recipe.model.hyperparameters = mock_hyperparams
        
        mock_model_instance = MagicMock()
        mock_create_from_class_path.return_value = mock_model_instance
        
        factory = Factory(mock_settings)
        
        # Act
        result = factory.create_model()
        
        # Assert
        assert result is mock_model_instance
        mock_create_from_class_path.assert_called_once_with(
            "sklearn.svm.SVC",
            {'C': 1.0, 'kernel': 'rbf'}
        )
    
    @patch.object(Factory, '_create_from_class_path')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_model_returns_cached_instance(self, mock_ensure_components, mock_create_from_class_path):
        """Test create_model returns cached instance on second call."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_settings.recipe.model.class_path = "sklearn.tree.DecisionTreeClassifier"
        mock_settings.recipe.model.hyperparameters = MagicMock()
        mock_settings.recipe.model.hyperparameters.get_fixed_params.return_value = {}
        
        mock_model_instance = MagicMock()
        mock_create_from_class_path.return_value = mock_model_instance
        
        factory = Factory(mock_settings)
        
        # Act - Call twice
        result1 = factory.create_model()
        result2 = factory.create_model()
        
        # Assert
        assert result1 is result2
        assert result1 is mock_model_instance
        # Should only call _create_from_class_path once due to caching
        mock_create_from_class_path.assert_called_once()


class TestFactoryEvaluator:
    """Test Factory create_evaluator method."""
    
    @patch('src.factory.factory.EvaluatorRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_evaluator_classification_task(self, mock_ensure_components, mock_evaluator_registry):
        """Test create_evaluator for classification task."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_data_interface = MagicMock()
        mock_data_interface.task_choice="classification"
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        mock_evaluator_instance = MagicMock()
        mock_evaluator_registry.create.return_value = mock_evaluator_instance
        
        factory = Factory(mock_settings)
        
        # Act
        result = factory.create_evaluator()
        
        # Assert
        assert result is mock_evaluator_instance
        mock_evaluator_registry.create.assert_called_once_with("classification", mock_data_interface)
        # Should be cached
        assert factory._component_cache['evaluator'] is mock_evaluator_instance
    
    @patch('src.factory.factory.EvaluatorRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_evaluator_handles_unknown_task_type(self, mock_ensure_components, mock_evaluator_registry):
        """Test create_evaluator handles unknown task type."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_data_interface = MagicMock()
        mock_data_interface.task_choice="unknown_task"
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        mock_evaluator_registry.create.side_effect = Exception("Unknown evaluator type")
        mock_evaluator_registry.list_evaluators.return_value = {
            'classification': 'ClassificationEvaluator',
            'regression': 'RegressionEvaluator'
        }
        
        factory = Factory(mock_settings)
        
        # Act & Assert
        with pytest.raises(Exception):
            factory.create_evaluator()


class TestFactoryTrainer:
    """Test Factory create_trainer method."""
    
    @patch('src.components.trainer.TrainerRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_trainer_default_type(self, mock_ensure_components, mock_trainer_registry):
        """Test create_trainer with default trainer type."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        mock_trainer_instance = MagicMock()
        mock_trainer_registry.create.return_value = mock_trainer_instance
        
        factory = Factory(mock_settings)
        
        # Act
        result = factory.create_trainer()
        
        # Assert
        assert result is mock_trainer_instance
        # Should be called with default type and factory provider
        mock_trainer_registry.create.assert_called_once()
        call_args = mock_trainer_registry.create.call_args
        assert call_args[0][0] == 'default'  # trainer_type
        assert 'settings' in call_args[1]
        assert 'factory_provider' in call_args[1]
        # Should be cached
        assert factory._component_cache['trainer_default'] is mock_trainer_instance
    
    @patch('src.components.trainer.TrainerRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_trainer_custom_type(self, mock_ensure_components, mock_trainer_registry):
        """Test create_trainer with custom trainer type."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        mock_trainer_instance = MagicMock()
        mock_trainer_registry.create.return_value = mock_trainer_instance
        
        factory = Factory(mock_settings)
        
        # Act
        result = factory.create_trainer(trainer_type='optuna')
        
        # Assert
        assert result is mock_trainer_instance
        call_args = mock_trainer_registry.create.call_args
        assert call_args[0][0] == 'optuna'  # trainer_type
        # Should be cached with custom key
        assert factory._component_cache['trainer_optuna'] is mock_trainer_instance
    
    @patch('src.components.trainer.TrainerRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_trainer_factory_provider_callable(self, mock_ensure_components, mock_trainer_registry):
        """Test create_trainer passes factory provider as callable."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        mock_trainer_instance = MagicMock()
        mock_trainer_registry.create.return_value = mock_trainer_instance
        
        factory = Factory(mock_settings)
        
        # Act
        result = factory.create_trainer()
        
        # Assert
        call_args = mock_trainer_registry.create.call_args
        factory_provider = call_args[1]['factory_provider']
        
        # Verify factory_provider is callable and returns the factory instance
        assert callable(factory_provider)
        assert factory_provider() is factory


class TestFactoryDataHandler:
    """Test Factory create_datahandler method."""
    
    @patch('src.components.datahandler.DataHandlerRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_datahandler_classification_task(self, mock_ensure_components, mock_datahandler_registry):
        """Test create_datahandler for classification task."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_data_interface = MagicMock()
        mock_data_interface.task_choice="classification"
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        mock_datahandler_instance = MagicMock()
        mock_datahandler_registry.get_handler_for_task.return_value = mock_datahandler_instance
        
        factory = Factory(mock_settings)
        
        # Act
        result = factory.create_datahandler()
        
        # Assert
        assert result is mock_datahandler_instance
        mock_datahandler_registry.get_handler_for_task.assert_called_once_with("classification", mock_settings)
        # Should be cached
        assert factory._component_cache['datahandler'] is mock_datahandler_instance
    
    @patch('src.components.datahandler.DataHandlerRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_datahandler_timeseries_task(self, mock_ensure_components, mock_datahandler_registry):
        """Test create_datahandler for timeseries task."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_data_interface = MagicMock()
        mock_data_interface.task_choice="timeseries"
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        mock_datahandler_instance = MagicMock()
        mock_datahandler_registry.get_handler_for_task.return_value = mock_datahandler_instance
        
        factory = Factory(mock_settings)
        
        # Act
        result = factory.create_datahandler()
        
        # Assert
        assert result is mock_datahandler_instance
        mock_datahandler_registry.get_handler_for_task.assert_called_once_with("timeseries", mock_settings)
    
    @patch('src.components.datahandler.DataHandlerRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_datahandler_handles_error(self, mock_ensure_components, mock_datahandler_registry):
        """Test create_datahandler handles creation errors."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_data_interface = MagicMock()
        mock_data_interface.task_choice="unknown_task"
        mock_settings.recipe.data.data_interface = mock_data_interface
        
        mock_datahandler_registry.get_handler_for_task.side_effect = Exception("Handler creation failed")
        mock_datahandler_registry.get_available_handlers.return_value = {
            'tabular': 'TabularDataHandler',
            'timeseries': 'TimeSeriesDataHandler'
        }
        
        factory = Factory(mock_settings)
        
        # Act & Assert
        with pytest.raises(Exception):
            factory.create_datahandler()


class TestFactoryFeatureStoreAdapter:
    """Test Factory create_feature_store_adapter method."""
    
    @patch('src.factory.factory.AdapterRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_feature_store_adapter_success(self, mock_ensure_components, mock_adapter_registry):
        """Test create_feature_store_adapter with valid feature store settings."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_settings.feature_store = MagicMock()  # Feature store configured
        
        mock_adapter_instance = MagicMock()
        mock_adapter_registry.create.return_value = mock_adapter_instance
        
        factory = Factory(mock_settings)
        
        # Act
        result = factory.create_feature_store_adapter()
        
        # Assert
        assert result is mock_adapter_instance
        mock_adapter_registry.create.assert_called_once_with('feature_store', mock_settings)
        # Should be cached
        assert factory._component_cache['feature_store_adapter'] is mock_adapter_instance
    
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_feature_store_adapter_no_config_raises_error(self, mock_ensure_components):
        """Test create_feature_store_adapter raises error when feature store not configured."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_settings.feature_store = None  # Not configured
        
        factory = Factory(mock_settings)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Feature Store settings are not configured"):
            factory.create_feature_store_adapter()


class TestFactoryOptuna:
    """Test Factory create_optuna_integration method."""
    
    @patch('src.utils.integrations.optuna_integration.OptunaIntegration')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_optuna_integration_success(self, mock_ensure_components, mock_optuna_class):
        """Test create_optuna_integration with valid hyperparameter tuning config."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_tuning_config = MagicMock()
        mock_settings.recipe.model.hyperparameters = mock_tuning_config
        
        mock_optuna_instance = MagicMock()
        mock_optuna_class.return_value = mock_optuna_instance
        
        factory = Factory(mock_settings)
        
        # Act
        result = factory.create_optuna_integration()
        
        # Assert
        assert result is mock_optuna_instance
        mock_optuna_class.assert_called_once_with(mock_tuning_config)
        # Should be cached
        assert factory._component_cache['optuna_integration'] is mock_optuna_instance
    
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_optuna_integration_no_config_raises_error(self, mock_ensure_components):
        """Test create_optuna_integration raises error when tuning not configured."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_settings.recipe.model.hyperparameters = None  # Not configured
        
        factory = Factory(mock_settings)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Hyperparameter tuning settings are not configured"):
            factory.create_optuna_integration()
    
    @patch('src.utils.integrations.optuna_integration.OptunaIntegration')
    @patch.object(Factory, '_ensure_components_registered')
    def test_create_optuna_integration_import_error(self, mock_ensure_components, mock_optuna_class):
        """Test create_optuna_integration handles optuna import error."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_settings.recipe.model.hyperparameters = MagicMock()
        
        mock_optuna_class.side_effect = ImportError("No module named 'optuna'")
        
        factory = Factory(mock_settings)
        
        # Act & Assert
        with pytest.raises(ImportError):
            factory.create_optuna_integration()


class TestFactoryCachingBehavior:
    """Test Factory caching behavior across different components."""
    
    @patch('src.factory.factory.AdapterRegistry')
    @patch('src.factory.factory.FetcherRegistry')
    @patch.object(Factory, '_ensure_components_registered')
    def test_different_components_cached_separately(self, mock_ensure_components, mock_fetcher_registry, mock_adapter_registry):
        """Test that different components are cached with separate keys."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        mock_settings.config.environment.env_name = "local"
        mock_settings.feature_store = None
        
        mock_adapter = MagicMock()
        mock_fetcher = MagicMock()
        mock_adapter_registry.create.return_value = mock_adapter
        mock_fetcher_registry.create.return_value = mock_fetcher
        
        factory = Factory(mock_settings)
        
        # Act
        adapter_result = factory.create_data_adapter(adapter_type='storage')
        fetcher_result = factory.create_fetcher(run_mode='batch')
        
        # Assert
        assert len(factory._component_cache) == 2
        assert factory._component_cache['adapter_storage'] is mock_adapter
        assert factory._component_cache['fetcher_batch'] is mock_fetcher
        assert adapter_result is mock_adapter
        assert fetcher_result is mock_fetcher
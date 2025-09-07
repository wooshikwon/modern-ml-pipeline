"""
Unit tests for Factory class.
Tests central component factory with caching, registry integration, and comprehensive component creation.
"""

import pytest
import importlib
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from src.factory.factory import Factory
from src.settings import Settings
from tests.helpers.builders import RecipeBuilder


class TestFactoryInitialization:
    """Test Factory initialization and validation."""
    
    @patch.object(Factory, '_ensure_components_registered')
    def test_factory_initialization_success(self, mock_ensure_components):
        """Test successful Factory initialization with valid settings."""
        # Arrange
        mock_settings = MagicMock()
        mock_recipe = MagicMock()
        mock_recipe.name = "test_recipe"
        mock_settings.recipe = mock_recipe
        mock_settings.config = MagicMock()
        
        # Act
        factory = Factory(mock_settings)
        
        # Assert
        assert factory.settings == mock_settings
        assert factory._recipe == mock_recipe
        assert factory._config == mock_settings.config
        assert factory._data == mock_recipe.data
        assert factory._model == mock_recipe.model
        assert isinstance(factory._component_cache, dict)
        assert len(factory._component_cache) == 0
        mock_ensure_components.assert_called_once()
    
    @patch.object(Factory, '_ensure_components_registered')
    def test_factory_initialization_missing_recipe_raises_error(self, mock_ensure_components):
        """Test Factory initialization fails when recipe is missing."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="현대화된 Recipe 구조가 필요합니다"):
            Factory(mock_settings)
        
        mock_ensure_components.assert_called_once()
    
    @patch.object(Factory, '_ensure_components_registered')
    def test_factory_caches_frequently_accessed_paths(self, mock_ensure_components):
        """Test that Factory caches frequently accessed configuration paths."""
        # Arrange
        mock_settings = MagicMock()
        mock_recipe = MagicMock()
        mock_config = MagicMock()
        mock_data = MagicMock()
        mock_model = MagicMock()
        
        mock_settings.recipe = mock_recipe
        mock_settings.config = mock_config
        mock_recipe.data = mock_data
        mock_recipe.model = mock_model
        mock_recipe.name = "cached_recipe"
        
        # Act
        factory = Factory(mock_settings)
        
        # Assert - Cached paths should be accessible without going through settings
        assert factory._recipe is mock_recipe
        assert factory._config is mock_config
        assert factory._data is mock_data
        assert factory._model is mock_model


class TestFactoryComponentRegistration:
    """Test Factory component registration mechanism."""
    
    @patch('src.factory.factory.importlib')
    @patch('src.factory.factory.logger')
    def test_ensure_components_registered_imports_all_components(self, mock_logger, mock_importlib):
        """Test that _ensure_components_registered imports all component modules."""
        # Arrange - Reset class variable to test registration
        original_state = Factory._components_registered
        Factory._components_registered = False
        
        try:
            # Act
            Factory._ensure_components_registered()
            
            # Assert
            assert Factory._components_registered is True
            
            # Check that all component modules are imported
            expected_imports = [
                'src.components.adapter',
                'src.components.evaluator', 
                'src.components.fetcher',
                'src.components.trainer',
                'src.components.preprocessor',
                'src.components.datahandler'
            ]
            
            import_calls = [call[0][0] for call in mock_importlib.import_module.call_args_list]
            for expected_import in expected_imports:
                assert expected_import in import_calls
                
            mock_logger.debug.assert_any_call("Initializing component registries...")
            mock_logger.debug.assert_any_call("Component registries initialized successfully")
            
        finally:
            # Reset state
            Factory._components_registered = original_state
    
    @patch('src.factory.factory.importlib')
    @patch('src.factory.factory.logger')
    def test_ensure_components_registered_handles_import_error(self, mock_logger, mock_importlib):
        """Test that _ensure_components_registered handles import errors gracefully."""
        # Arrange
        original_state = Factory._components_registered
        Factory._components_registered = False
        mock_importlib.import_module.side_effect = ImportError("Module not found")
        
        try:
            # Act
            Factory._ensure_components_registered()
            
            # Assert
            assert Factory._components_registered is True
            mock_logger.warning.assert_called_with("Some components could not be imported: Module not found")
            
        finally:
            # Reset state and mock
            Factory._components_registered = original_state
            mock_importlib.import_module.side_effect = None
    
    @patch('src.factory.factory.importlib')
    def test_ensure_components_registered_only_runs_once(self, mock_importlib):
        """Test that _ensure_components_registered only runs once even with multiple calls."""
        # Arrange - Ensure components are already registered
        Factory._components_registered = True
        
        # Act
        Factory._ensure_components_registered()
        Factory._ensure_components_registered()  # Call again
        
        # Assert - No imports should happen since already registered
        mock_importlib.import_module.assert_not_called()


class TestFactoryHelperMethods:
    """Test Factory helper methods."""
    
    def test_process_hyperparameters_converts_callable_strings(self):
        """Test _process_hyperparameters converts string references to callables."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        with patch.object(Factory, '_ensure_components_registered'):
            factory = Factory(mock_settings)
        
        params = {
            'n_estimators': 100,
            'criterion': 'gini',
            'loss_fn': 'sklearn.metrics.log_loss',  # Should be converted
            'custom_class': 'numpy.random.RandomState',  # Should be converted
            'regular_param': 'just_a_string'  # Should remain string
        }
        
        # Act
        with patch('src.factory.factory.importlib.import_module') as mock_import:
            with patch('src.factory.factory.getattr') as mock_getattr:
                # Mock the import and getattr calls
                mock_sklearn_metrics = MagicMock()
                mock_numpy_random = MagicMock()
                mock_log_loss = MagicMock()
                mock_random_state = MagicMock()
                
                def import_side_effect(module_path):
                    if module_path == 'sklearn.metrics':
                        return mock_sklearn_metrics
                    elif module_path == 'numpy.random':
                        return mock_numpy_random
                    else:
                        raise ImportError(f"No module named '{module_path}'")
                
                def getattr_side_effect(module, attr_name):
                    if module == mock_sklearn_metrics and attr_name == 'log_loss':
                        return mock_log_loss
                    elif module == mock_numpy_random and attr_name == 'RandomState':
                        return mock_random_state
                    else:
                        raise AttributeError(f"'{module}' has no attribute '{attr_name}'")
                
                mock_import.side_effect = import_side_effect
                mock_getattr.side_effect = getattr_side_effect
                
                result = factory._process_hyperparameters(params)
        
        # Assert
        assert result['n_estimators'] == 100
        assert result['criterion'] == 'gini'
        assert result['loss_fn'] == mock_log_loss
        assert result['custom_class'] == mock_random_state
        assert result['regular_param'] == 'just_a_string'
    
    def test_process_hyperparameters_handles_import_error(self):
        """Test _process_hyperparameters handles import errors gracefully."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        with patch.object(Factory, '_ensure_components_registered'):
            factory = Factory(mock_settings)
        
        params = {
            'valid_param': 100,
            'invalid_fn': 'nonexistent.module.function'
        }
        
        # Act
        with patch('src.factory.factory.importlib.import_module', side_effect=ImportError("Module not found")):
            result = factory._process_hyperparameters(params)
        
        # Assert - Should keep original string when import fails
        assert result['valid_param'] == 100
        assert result['invalid_fn'] == 'nonexistent.module.function'
    
    def test_detect_adapter_type_from_uri_sql_patterns(self):
        """Test _detect_adapter_type_from_uri correctly identifies SQL patterns."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        with patch.object(Factory, '_ensure_components_registered'):
            factory = Factory(mock_settings)
        
        # Test cases for SQL patterns
        sql_uris = [
            'query.sql',
            'SELECT * FROM table',
            'select id, name from users',
            'data/query.SQL',
            'SELECT COUNT(*) FROM products WHERE category = "electronics"'
        ]
        
        # Act & Assert
        for uri in sql_uris:
            result = factory._detect_adapter_type_from_uri(uri)
            assert result == 'sql', f"Failed for URI: {uri}"
    
    def test_detect_adapter_type_from_uri_storage_patterns(self):
        """Test _detect_adapter_type_from_uri correctly identifies storage patterns."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        with patch.object(Factory, '_ensure_components_registered'):
            factory = Factory(mock_settings)
        
        # Test cases for storage patterns
        storage_uris = [
            'data/file.csv',
            'data.parquet',
            'output.json',
            's3://bucket/data.csv',
            'gs://bucket/file.parquet',
            'az://container/data.tsv',
            'DATA.CSV',  # Test case sensitivity
        ]
        
        # Act & Assert
        for uri in storage_uris:
            result = factory._detect_adapter_type_from_uri(uri)
            assert result == 'storage', f"Failed for URI: {uri}"
    
    def test_detect_adapter_type_from_uri_bigquery_pattern(self):
        """Test _detect_adapter_type_from_uri correctly identifies BigQuery patterns."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        with patch.object(Factory, '_ensure_components_registered'):
            factory = Factory(mock_settings)
        
        # Test cases for BigQuery patterns
        bigquery_uris = [
            'bigquery://project.dataset.table',
            'BIGQUERY://project.dataset.view'
        ]
        
        # Act & Assert
        for uri in bigquery_uris:
            result = factory._detect_adapter_type_from_uri(uri)
            assert result == 'bigquery', f"Failed for URI: {uri}"
    
    def test_detect_adapter_type_from_uri_defaults_to_storage(self):
        """Test _detect_adapter_type_from_uri defaults to storage for unknown patterns."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        with patch.object(Factory, '_ensure_components_registered'):
            factory = Factory(mock_settings)
        
        unknown_uris = [
            'unknown://protocol',
            'file.unknown',
            'just_a_string',
            'ftp://server/file.txt'
        ]
        
        # Act & Assert
        with patch('src.factory.factory.logger') as mock_logger:
            for uri in unknown_uris:
                result = factory._detect_adapter_type_from_uri(uri)
                assert result == 'storage', f"Failed for URI: {uri}"
            
            # Should log warnings for unknown patterns
            assert mock_logger.warning.call_count == len(unknown_uris)
    
    @patch('src.factory.factory.importlib.import_module')
    def test_create_from_class_path_success(self, mock_import_module):
        """Test _create_from_class_path successfully creates instances."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        with patch.object(Factory, '_ensure_components_registered'):
            factory = Factory(mock_settings)
        
        # Mock the module and class
        mock_module = MagicMock()
        mock_class = MagicMock()
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        
        mock_import_module.return_value = mock_module
        mock_module.TestClass = mock_class
        
        hyperparameters = {'param1': 'value1', 'param2': 42}
        
        # Act
        with patch.object(factory, '_process_hyperparameters', return_value=hyperparameters) as mock_process:
            result = factory._create_from_class_path('test.module.TestClass', hyperparameters)
        
        # Assert
        assert result == mock_instance
        mock_import_module.assert_called_once_with('test.module')
        mock_class.assert_called_once_with(**hyperparameters)
        mock_process.assert_called_once_with(hyperparameters)
    
    @patch('src.factory.factory.importlib.import_module')
    def test_create_from_class_path_import_error(self, mock_import_module):
        """Test _create_from_class_path handles import errors."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        with patch.object(Factory, '_ensure_components_registered'):
            factory = Factory(mock_settings)
        
        mock_import_module.side_effect = ImportError("Module not found")
        
        # Act & Assert
        with pytest.raises(ValueError, match="Could not load class: invalid.module.Class"):
            factory._create_from_class_path('invalid.module.Class', {})
    
    @patch('src.factory.factory.importlib.import_module')
    def test_create_from_class_path_attribute_error(self, mock_import_module):
        """Test _create_from_class_path handles attribute errors."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        with patch.object(Factory, '_ensure_components_registered'):
            factory = Factory(mock_settings)
        
        mock_module = MagicMock()
        mock_import_module.return_value = mock_module
        # Simulate missing class attribute
        del mock_module.NonExistentClass
        
        # Act & Assert
        with pytest.raises(ValueError, match="Could not load class: test.module.NonExistentClass"):
            factory._create_from_class_path('test.module.NonExistentClass', {})


class TestFactoryCaching:
    """Test Factory component caching functionality."""
    
    def test_component_cache_initialized_empty(self):
        """Test that component cache is initialized as empty dict."""
        # Arrange & Act
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        with patch.object(Factory, '_ensure_components_registered'):
            factory = Factory(mock_settings)
        
        # Assert
        assert isinstance(factory._component_cache, dict)
        assert len(factory._component_cache) == 0
    
    def test_cache_stores_and_retrieves_components(self):
        """Test that cache correctly stores and retrieves components."""
        # Arrange
        mock_settings = MagicMock()
        mock_settings.recipe = MagicMock()
        mock_settings.recipe.name = "test"
        
        with patch.object(Factory, '_ensure_components_registered'):
            factory = Factory(mock_settings)
        
        # Act - Manually add items to cache
        mock_component = MagicMock()
        factory._component_cache['test_component'] = mock_component
        
        # Assert
        assert 'test_component' in factory._component_cache
        assert factory._component_cache['test_component'] is mock_component
        assert len(factory._component_cache) == 1


class TestFactoryErrorHandling:
    """Test Factory error handling scenarios."""
    
    @patch.object(Factory, '_ensure_components_registered')
    def test_factory_validates_recipe_structure(self, mock_ensure_components):
        """Test Factory validates recipe structure during initialization."""
        # Arrange - Settings without recipe
        mock_settings = MagicMock()
        mock_settings.recipe = None
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            Factory(mock_settings)
        
        assert "현대화된 Recipe 구조가 필요합니다" in str(exc_info.value)
        mock_ensure_components.assert_called_once()
    
    @patch.object(Factory, '_ensure_components_registered')
    def test_factory_handles_missing_nested_attributes_gracefully(self, mock_ensure_components):
        """Test Factory handles missing nested attributes in settings."""
        # Arrange - Recipe without expected nested structure
        mock_settings = MagicMock()
        mock_recipe = MagicMock()
        
        # Remove some expected attributes to test robustness
        mock_recipe.data = None
        mock_recipe.model = None
        mock_recipe.name = "incomplete_recipe"
        
        mock_settings.recipe = mock_recipe
        mock_settings.config = MagicMock()
        
        # Act - Should not raise error during initialization
        factory = Factory(mock_settings)
        
        # Assert
        assert factory._recipe is mock_recipe
        assert factory._data is None
        assert factory._model is None
        mock_ensure_components.assert_called_once()
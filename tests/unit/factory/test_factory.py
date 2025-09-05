"""
Unit tests for the Factory class.
Tests component creation and caching mechanisms.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from typing import Dict, Any

from src.factory import Factory
from src.settings import Settings
from tests.helpers.builders import ConfigBuilder, RecipeBuilder


class TestFactoryInitialization:
    """Test Factory initialization and setup."""
    
    def test_factory_initialization_with_valid_settings(self):
        """Test Factory initialization with valid settings."""
        settings = Mock(spec=Settings)
        settings.recipe = RecipeBuilder.build()
        settings.config = ConfigBuilder.build()
        
        factory = Factory(settings)
        
        assert factory.settings == settings
        assert factory._recipe == settings.recipe
        assert factory._config == settings.config
        assert factory._component_cache == {}
    
    def test_factory_initialization_without_recipe(self):
        """Test Factory initialization fails without recipe."""
        settings = Mock(spec=Settings)
        settings.recipe = None
        
        with pytest.raises(ValueError) as exc_info:
            Factory(settings)
        
        assert "recipe" in str(exc_info.value).lower()
    
    @patch('src.factory.factory.importlib.import_module')
    def test_factory_component_registration(self, mock_import):
        """Test that components are registered on first initialization."""
        # Reset class variable for testing
        Factory._components_registered = False
        
        settings = Mock(spec=Settings)
        settings.recipe = RecipeBuilder.build()
        settings.config = ConfigBuilder.build()
        
        factory = Factory(settings)
        
        # Should have triggered imports
        assert Factory._components_registered is True
        
        # Create another factory - should not re-register
        Factory._components_registered = True
        mock_import.reset_mock()
        
        factory2 = Factory(settings)
        # Should not call import_module again
        assert mock_import.call_count == 0


class TestFactoryComponentCreation:
    """Test Factory component creation methods."""
    
    @pytest.fixture
    def factory(self):
        """Create a Factory instance with mock settings."""
        settings = Mock(spec=Settings)
        settings.recipe = RecipeBuilder.build()
        settings.config = ConfigBuilder.build()
        
        factory = Factory(settings)
        return factory
    
    @patch('src.factory.factory.AdapterRegistry')
    def test_create_data_adapter_with_type(self, mock_registry, factory):
        """Test creating data adapter with explicit type."""
        mock_adapter = Mock()
        mock_registry.create.return_value = mock_adapter
        
        adapter = factory.create_data_adapter(adapter_type="sql")
        
        assert adapter == mock_adapter
        mock_registry.create.assert_called_once_with("sql", factory.settings)
        
        # Test caching
        adapter2 = factory.create_data_adapter(adapter_type="sql")
        assert adapter2 == mock_adapter
        assert mock_registry.create.call_count == 1  # Should not create again
    
    @patch('src.factory.factory.AdapterRegistry')
    def test_create_data_adapter_auto_detect(self, mock_registry, factory):
        """Test creating data adapter with auto-detection."""
        mock_adapter = Mock()
        mock_registry.create.return_value = mock_adapter
        
        # Set source_uri for auto-detection
        factory._data.loader.source_uri = "data.csv"
        
        adapter = factory.create_data_adapter()
        
        assert adapter == mock_adapter
        mock_registry.create.assert_called_once_with("storage", factory.settings)
    
    def test_detect_adapter_type_from_uri(self, factory):
        """Test adapter type detection from URI patterns."""
        # SQL patterns
        assert factory._detect_adapter_type_from_uri("query.sql") == "sql"
        assert factory._detect_adapter_type_from_uri("SELECT * FROM table") == "sql"
        
        # BigQuery pattern
        assert factory._detect_adapter_type_from_uri("bigquery://project.dataset.table") == "bigquery"
        
        # Cloud storage patterns
        assert factory._detect_adapter_type_from_uri("s3://bucket/file.csv") == "storage"
        assert factory._detect_adapter_type_from_uri("gs://bucket/file.parquet") == "storage"
        
        # File patterns
        assert factory._detect_adapter_type_from_uri("data.csv") == "storage"
        assert factory._detect_adapter_type_from_uri("data.parquet") == "storage"
        assert factory._detect_adapter_type_from_uri("data.json") == "storage"
        
        # Unknown pattern (defaults to storage)
        assert factory._detect_adapter_type_from_uri("unknown.xyz") == "storage"
    
    @patch('src.factory.factory.FetcherRegistry')
    def test_create_fetcher_batch_mode(self, mock_registry, factory):
        """Test creating fetcher in batch mode."""
        mock_fetcher = Mock()
        mock_registry.create.return_value = mock_fetcher
        
        # Setup for pass_through fetcher
        factory._config.environment = Mock()
        factory._config.environment.env_name = "local"
        factory.settings.feature_store = Mock(provider="none")
        
        fetcher = factory.create_fetcher(run_mode="batch")
        
        assert fetcher == mock_fetcher
        mock_registry.create.assert_called_once_with("pass_through")
        
        # Test caching
        fetcher2 = factory.create_fetcher(run_mode="batch")
        assert fetcher2 == mock_fetcher
        assert mock_registry.create.call_count == 1
    
    @patch('src.factory.factory.FetcherRegistry')
    def test_create_fetcher_serving_mode_error(self, mock_registry, factory):
        """Test creating fetcher in serving mode without feature store fails."""
        factory._config.environment = Mock()
        factory._config.environment.env_name = "local"
        factory.settings.feature_store = Mock(provider="none")
        factory._recipe.data.fetcher = Mock(type="pass_through")
        
        with pytest.raises(TypeError) as exc_info:
            factory.create_fetcher(run_mode="serving")
        
        assert "serving" in str(exc_info.value).lower()
        assert "feature store" in str(exc_info.value).lower()
    
    @patch('src.factory.factory.FetcherRegistry')
    def test_create_fetcher_feature_store(self, mock_registry, factory):
        """Test creating feature store fetcher."""
        mock_fetcher = Mock()
        mock_registry.create.return_value = mock_fetcher
        
        # Setup for feature store fetcher
        factory._config.environment = Mock()
        factory._config.environment.env_name = "production"
        factory.settings.feature_store = Mock(provider="feast")
        factory._recipe.data.fetcher = Mock(type="feature_store")
        
        fetcher = factory.create_fetcher()
        
        assert fetcher == mock_fetcher
        mock_registry.create.assert_called_once_with(
            "feature_store",
            settings=factory.settings,
            factory=factory
        )
    
    @patch('src.factory.factory.Preprocessor')
    def test_create_preprocessor_with_config(self, mock_preprocessor_class, factory):
        """Test creating preprocessor when configured."""
        mock_preprocessor = Mock()
        mock_preprocessor_class.return_value = mock_preprocessor
        
        factory._recipe.preprocessor = Mock()
        
        preprocessor = factory.create_preprocessor()
        
        assert preprocessor == mock_preprocessor
        mock_preprocessor_class.assert_called_once_with(settings=factory.settings)
        
        # Test caching
        preprocessor2 = factory.create_preprocessor()
        assert preprocessor2 == mock_preprocessor
        assert mock_preprocessor_class.call_count == 1
    
    def test_create_preprocessor_without_config(self, factory):
        """Test creating preprocessor when not configured."""
        factory._recipe.preprocessor = None
        
        preprocessor = factory.create_preprocessor()
        
        assert preprocessor is None
    
    @patch('src.factory.factory.importlib.import_module')
    def test_create_model(self, mock_import, factory):
        """Test creating model from class path."""
        # Setup mock model class
        mock_module = Mock()
        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        
        mock_import.return_value = mock_module
        mock_module.RandomForestClassifier = mock_model_class
        
        factory._model.class_path = "sklearn.ensemble.RandomForestClassifier"
        factory._model.hyperparameters = Mock()
        factory._model.hyperparameters.get_fixed_params = Mock(
            return_value={"n_estimators": 100, "max_depth": 10}
        )
        
        model = factory.create_model()
        
        assert model == mock_model_instance
        mock_import.assert_called_once_with("sklearn.ensemble")
        mock_model_class.assert_called_once_with(n_estimators=100, max_depth=10)
        
        # Test caching
        model2 = factory.create_model()
        assert model2 == mock_model_instance
        assert mock_model_class.call_count == 1
    
    @patch('src.factory.factory.EvaluatorRegistry')
    def test_create_evaluator(self, mock_registry, factory):
        """Test creating evaluator."""
        mock_evaluator = Mock()
        mock_registry.create.return_value = mock_evaluator
        
        factory._recipe.data.data_interface = Mock(task_type="classification")
        
        evaluator = factory.create_evaluator()
        
        assert evaluator == mock_evaluator
        mock_registry.create.assert_called_once_with(
            "classification",
            factory._recipe.data.data_interface
        )
        
        # Test caching
        evaluator2 = factory.create_evaluator()
        assert evaluator2 == mock_evaluator
        assert mock_registry.create.call_count == 1


class TestFactoryHelperMethods:
    """Test Factory helper methods."""
    
    @pytest.fixture
    def factory(self):
        """Create a Factory instance with mock settings."""
        settings = Mock(spec=Settings)
        settings.recipe = RecipeBuilder.build()
        settings.config = ConfigBuilder.build()
        
        factory = Factory(settings)
        return factory
    
    @patch('src.factory.factory.importlib.import_module')
    def test_create_from_class_path_success(self, mock_import, factory):
        """Test successful object creation from class path."""
        # Setup mock
        mock_module = Mock()
        mock_class = Mock()
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        
        mock_import.return_value = mock_module
        mock_module.TestClass = mock_class
        
        result = factory._create_from_class_path(
            "test.module.TestClass",
            {"param1": "value1"}
        )
        
        assert result == mock_instance
        mock_import.assert_called_once_with("test.module")
        mock_class.assert_called_once_with(param1="value1")
    
    @patch('src.factory.factory.importlib.import_module')
    def test_create_from_class_path_import_error(self, mock_import, factory):
        """Test error handling when import fails."""
        mock_import.side_effect = ImportError("Module not found")
        
        with pytest.raises(ValueError) as exc_info:
            factory._create_from_class_path(
                "invalid.module.Class",
                {}
            )
        
        assert "could not load class" in str(exc_info.value).lower()
    
    @patch('src.factory.factory.importlib.import_module')
    def test_process_hyperparameters_callable_conversion(self, mock_import, factory):
        """Test hyperparameter processing for callable parameters."""
        # Setup mock callable
        mock_module = Mock()
        mock_function = Mock()
        mock_module.accuracy_score = mock_function
        mock_import.return_value = mock_module
        
        params = {
            "scoring_fn": "sklearn.metrics.accuracy_score",
            "regular_param": "value",
            "number_param": 42
        }
        
        processed = factory._process_hyperparameters(params)
        
        assert processed["scoring_fn"] == mock_function
        assert processed["regular_param"] == "value"
        assert processed["number_param"] == 42
        
        mock_import.assert_called_once_with("sklearn.metrics")
    
    def test_process_hyperparameters_no_conversion(self, factory):
        """Test hyperparameters that don't need conversion."""
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "criterion": "gini"
        }
        
        processed = factory._process_hyperparameters(params)
        
        assert processed == params  # Should remain unchanged


class TestFactoryCaching:
    """Test Factory caching mechanism."""
    
    @pytest.fixture
    def factory(self):
        """Create a Factory instance with mock settings."""
        settings = Mock(spec=Settings)
        settings.recipe = RecipeBuilder.build()
        settings.config = ConfigBuilder.build()
        
        factory = Factory(settings)
        return factory
    
    def test_component_cache_initialization(self, factory):
        """Test that component cache is initialized empty."""
        assert factory._component_cache == {}
    
    @patch('src.factory.factory.AdapterRegistry')
    def test_adapter_caching(self, mock_registry, factory):
        """Test that adapters are properly cached."""
        mock_adapter = Mock()
        mock_registry.create.return_value = mock_adapter
        
        # First creation
        adapter1 = factory.create_data_adapter(adapter_type="sql")
        assert "adapter_sql" in factory._component_cache
        assert factory._component_cache["adapter_sql"] == mock_adapter
        
        # Second call should use cache
        adapter2 = factory.create_data_adapter(adapter_type="sql")
        assert adapter1 == adapter2
        assert mock_registry.create.call_count == 1
        
        # Different type should create new adapter
        mock_adapter2 = Mock()
        mock_registry.create.return_value = mock_adapter2
        
        adapter3 = factory.create_data_adapter(adapter_type="storage")
        assert "adapter_storage" in factory._component_cache
        assert factory._component_cache["adapter_storage"] == mock_adapter2
        assert mock_registry.create.call_count == 2
    
    @patch('src.factory.factory.FetcherRegistry')
    def test_fetcher_caching_by_mode(self, mock_registry, factory):
        """Test that fetchers are cached by mode."""
        mock_fetcher_batch = Mock()
        mock_fetcher_serving = Mock()
        
        factory._config.environment = Mock()
        factory._config.environment.env_name = "local"
        factory.settings.feature_store = Mock(provider="feast")
        factory._recipe.data.fetcher = Mock(type="feature_store")
        
        # Batch mode
        mock_registry.create.return_value = mock_fetcher_batch
        fetcher_batch = factory.create_fetcher(run_mode="batch")
        assert "fetcher_batch" in factory._component_cache
        
        # Serving mode (different cache key)
        mock_registry.create.return_value = mock_fetcher_serving
        fetcher_serving = factory.create_fetcher(run_mode="serving")
        assert "fetcher_serving" in factory._component_cache
        
        assert fetcher_batch != fetcher_serving
        assert len(factory._component_cache) == 2


class TestFactoryErrorHandling:
    """Test Factory error handling."""
    
    @pytest.fixture
    def factory(self):
        """Create a Factory instance with mock settings."""
        settings = Mock(spec=Settings)
        settings.recipe = RecipeBuilder.build()
        settings.config = ConfigBuilder.build()
        
        factory = Factory(settings)
        return factory
    
    @patch('src.factory.factory.AdapterRegistry')
    def test_adapter_creation_error(self, mock_registry, factory):
        """Test error handling when adapter creation fails."""
        mock_registry.create.side_effect = Exception("Adapter creation failed")
        mock_registry.list_adapters.return_value = {"sql": Mock(), "storage": Mock()}
        
        with pytest.raises(ValueError) as exc_info:
            factory.create_data_adapter(adapter_type="invalid")
        
        assert "failed to create adapter" in str(exc_info.value).lower()
        assert "available" in str(exc_info.value).lower()
    
    @patch('src.factory.factory.EvaluatorRegistry')
    def test_evaluator_creation_error(self, mock_registry, factory):
        """Test error handling when evaluator creation fails."""
        mock_registry.create.side_effect = Exception("Evaluator creation failed")
        mock_registry.list_evaluators.return_value = {
            "classification": Mock(),
            "regression": Mock()
        }
        
        factory._recipe.data.data_interface = Mock(task_type="invalid_task")
        
        with pytest.raises(Exception):
            factory.create_evaluator()
    
    def test_invalid_class_path_format(self, factory):
        """Test error handling for invalid class path format."""
        with pytest.raises(ValueError) as exc_info:
            factory._create_from_class_path(
                "invalid_class_path",  # No dots
                {}
            )
        
        assert "could not load class" in str(exc_info.value).lower()
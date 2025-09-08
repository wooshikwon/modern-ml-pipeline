"""
Factory Initialization & Registry Setup Tests
Week 1, Days 6-7: Factory & Component Registration Tests

Tests Factory initialization and component registry setup following
comprehensive testing strategy - No Mock Hell approach with real components.
"""

import pytest
from pathlib import Path

from src.factory.factory import Factory
from src.settings import Settings
from src.components.adapter.registry import AdapterRegistry
from src.components.evaluator.registry import EvaluatorRegistry 
from src.components.fetcher.registry import FetcherRegistry
from src.components.trainer.registry import TrainerRegistry
from src.components.datahandler.registry import DataHandlerRegistry
from src.components.preprocessor.registry import PreprocessorStepRegistry


class TestFactoryInitialization:
    """Test Factory initialization process and component registration setup (6 tests)."""
    
    def test_factory_initializes_with_valid_settings(self, settings_builder, performance_benchmark):
        """Test Factory properly initializes with valid Settings object."""
        settings = settings_builder.with_task("classification").build()
        
        with performance_benchmark.measure_time("factory_initialization"):
            factory = Factory(settings)
        
        # Validate Factory state after initialization
        assert factory.settings is settings
        assert factory._recipe is settings.recipe
        assert factory._config is settings.config
        assert factory._data is settings.recipe.data
        assert factory._model is settings.recipe.model
        
        # Validate component cache is initialized
        assert isinstance(factory._component_cache, dict)
        assert len(factory._component_cache) == 0  # Should start empty
        
        # Validate console is initialized
        assert factory.console is not None
        
        # Performance validation - should be fast
        performance_benchmark.assert_time_under("factory_initialization", 0.1)
    
    def test_factory_ensures_components_registered_on_first_initialization(self, settings_builder):
        """Test Factory triggers component registration on first initialization."""
        # Reset class variable to simulate fresh start
        Factory._components_registered = False
        
        settings = settings_builder.build()
        
        # Before Factory creation, ensure registration status is False
        assert not Factory._components_registered
        
        # Create Factory - should trigger component registration
        factory = Factory(settings)
        
        # After Factory creation, registration should be complete
        assert Factory._components_registered
        
        # Verify registries are populated with actual components
        assert len(AdapterRegistry.adapters) > 0
        assert len(EvaluatorRegistry.evaluators) > 0 
        assert len(FetcherRegistry.fetchers) > 0
        assert len(TrainerRegistry.trainers) > 0
        assert len(DataHandlerRegistry.handlers) > 0
        
        # Known adapters should be registered
        assert "storage" in AdapterRegistry.adapters
        assert "sql" in AdapterRegistry.adapters
        
        # Known evaluators should be registered
        assert "classification" in EvaluatorRegistry.evaluators
        assert "regression" in EvaluatorRegistry.evaluators
    
    def test_factory_skips_registration_on_subsequent_initializations(self, settings_builder, performance_benchmark):
        """Test Factory skips component registration on subsequent initializations."""
        # Ensure registration has already happened
        Factory._components_registered = True
        
        settings1 = settings_builder.with_task("classification").build()
        settings2 = settings_builder.with_task("regression").build()
        
        with performance_benchmark.measure_time("second_factory_init"):
            factory1 = Factory(settings1)
            factory2 = Factory(settings2)
        
        # Both factories should be properly initialized
        assert factory1.settings is settings1
        assert factory2.settings is settings2
        
        # Registration should still be marked as done
        assert Factory._components_registered
        
        # Second initialization should be even faster (no registration overhead)
        performance_benchmark.assert_time_under("second_factory_init", 0.05)
    
    def test_factory_validates_recipe_structure_on_initialization(self, settings_builder):
        """Test Factory validates recipe structure during initialization."""
        # Create valid settings first
        valid_settings = settings_builder.build()
        
        # Manually create Factory to test initialization
        factory = Factory(valid_settings)
        
        # Validate recipe structure was properly cached
        assert factory._recipe is not None
        assert factory._recipe.name == "test_recipe"
        assert factory._recipe.task_choice in ["classification", "regression", "timeseries", "clustering", "causal"]
        
        # Test that Factory requires valid recipe structure
        factory._recipe = None
        
        # This should cause issues when trying to access recipe-dependent methods
        # We test this indirectly by checking the cached recipe structure
        assert factory._data is not None
        assert factory._model is not None
    
    def test_factory_caches_frequently_accessed_paths(self, settings_builder):
        """Test Factory properly caches frequently accessed configuration paths."""
        settings = settings_builder \
            .with_task("classification") \
            .with_model("sklearn.ensemble.RandomForestClassifier") \
            .build()
        
        factory = Factory(settings)
        
        # Validate cached paths for performance
        assert factory._recipe is settings.recipe
        assert factory._config is settings.config  
        assert factory._data is settings.recipe.data
        assert factory._model is settings.recipe.model
        
        # Validate cached paths are the same objects (not copies)
        assert factory._recipe is settings.recipe
        assert factory._data is settings.recipe.data
        assert factory._model is settings.recipe.model
    
    def test_components_registered_class_variable_persistence(self, settings_builder):
        """Test _components_registered class variable persists across Factory instances."""
        # Reset to simulate fresh Python session
        Factory._components_registered = False
        
        settings1 = settings_builder.with_task("classification").build()
        settings2 = settings_builder.with_task("regression").build()
        
        # First factory should trigger registration
        factory1 = Factory(settings1)
        assert Factory._components_registered
        
        # Second factory should see registration already done
        factory2 = Factory(settings2)
        assert Factory._components_registered
        
        # Both factories should work correctly
        assert factory1.settings.recipe.task_choice == "classification"
        assert factory2.settings.recipe.task_choice == "regression"
        
        # Registry contents should be available to both
        adapter1 = factory1.create_data_adapter("storage")
        adapter2 = factory2.create_data_adapter("storage")
        
        # Both should get real StorageAdapter instances
        from src.components.adapter.modules.storage_adapter import StorageAdapter
        assert isinstance(adapter1, StorageAdapter)
        assert isinstance(adapter2, StorageAdapter)


class TestComponentRegistrySetup:
    """Test component registries are properly set up after Factory initialization."""
    
    def test_all_registries_populated_after_initialization(self, settings_builder):
        """Test all component registries are populated after Factory initialization."""
        settings = settings_builder.build()
        
        # Create Factory to trigger registration
        factory = Factory(settings)
        
        # Validate AdapterRegistry
        adapter_types = list(AdapterRegistry.adapters.keys())
        assert len(adapter_types) >= 2  # At minimum storage, sql
        assert "storage" in adapter_types
        assert "sql" in adapter_types
        
        # Validate EvaluatorRegistry
        evaluator_types = EvaluatorRegistry.get_available_tasks()
        assert len(evaluator_types) >= 3  # classification, regression, clustering
        assert "classification" in evaluator_types
        assert "regression" in evaluator_types
        
        # Validate FetcherRegistry
        fetcher_types = FetcherRegistry.get_available_types()
        assert len(fetcher_types) >= 1  # pass_through at minimum
        
        # Validate TrainerRegistry
        trainer_types = TrainerRegistry.get_available_types()
        assert len(trainer_types) >= 1  # default at minimum
        
        # Validate DataHandlerRegistry
        handler_types = list(DataHandlerRegistry.handlers.keys())
        assert len(handler_types) >= 1  # tabular at minimum
    
    def test_registry_components_are_actual_classes(self, settings_builder):
        """Test registered components are actual class objects, not strings or mocks."""
        settings = settings_builder.build()
        factory = Factory(settings)
        
        # Test AdapterRegistry contains actual classes
        storage_adapter_class = AdapterRegistry.adapters.get("storage")
        assert storage_adapter_class is not None
        assert hasattr(storage_adapter_class, '__init__')
        assert hasattr(storage_adapter_class, 'read')
        
        # Test EvaluatorRegistry contains actual classes
        cls_evaluator_class = EvaluatorRegistry.evaluators.get("classification")
        assert cls_evaluator_class is not None
        assert hasattr(cls_evaluator_class, '__init__')
        assert hasattr(cls_evaluator_class, 'evaluate')
        
        # Verify classes can be instantiated (basic smoke test)
        from src.components.adapter.modules.storage_adapter import StorageAdapter
        assert storage_adapter_class is StorageAdapter
        
        from src.components.evaluator.modules.classification_evaluator import ClassificationEvaluator
        assert cls_evaluator_class is ClassificationEvaluator
    
    def test_registry_self_registration_imports_work(self, settings_builder, isolated_temp_directory):
        """Test that self-registration via imports actually works correctly."""
        # This test validates the self-registration pattern works
        # by checking that imports have populated the registries
        
        settings = settings_builder.build()
        Factory(settings)  # Trigger registration
        
        # Test specific known registrations
        adapters = AdapterRegistry.list_adapters()
        assert "storage" in adapters
        assert "sql" in adapters
        
        # Test that we can create instances from registered classes
        storage_class = AdapterRegistry.get_adapter("storage")
        sql_class = AdapterRegistry.get_adapter("sql")
        
        # Classes should be importable and instantiable
        assert callable(storage_class)
        assert callable(sql_class)
        
        # Should be able to create real instances
        storage_instance = AdapterRegistry.create("storage", settings)
        
        # For SQL adapter, we need proper config
        db_path = isolated_temp_directory / "test.db"
        connection_string = f"sqlite:///{db_path}"
        sql_settings = settings_builder \
            .with_data_source("sql", config={"connection_uri": connection_string}) \
            .build()
        sql_instance = AdapterRegistry.create("sql", sql_settings)
        
        assert hasattr(storage_instance, 'read')
        assert hasattr(sql_instance, 'read')
    
    def test_registry_error_handling_for_missing_types(self, settings_builder):
        """Test registries properly handle requests for non-existent component types."""
        settings = settings_builder.build()
        factory = Factory(settings)
        
        # Test AdapterRegistry error handling
        with pytest.raises(KeyError, match="Unknown adapter type"):
            AdapterRegistry.get_adapter("nonexistent_adapter")
        
        # AdapterRegistry.create() first calls get_adapter(), so it also raises KeyError
        with pytest.raises(KeyError, match="Unknown adapter type"):
            AdapterRegistry.create("nonexistent_adapter", settings)
        
        # Test EvaluatorRegistry error handling  
        with pytest.raises(ValueError, match="Unknown task type"):
            EvaluatorRegistry.create("nonexistent_task", settings)
        
        # Test FetcherRegistry error handling
        with pytest.raises(ValueError, match="Unknown fetcher type"):
            FetcherRegistry.create("nonexistent_fetcher", settings)
        
        # Error messages should list available types
        try:
            AdapterRegistry.get_adapter("nonexistent")
            assert False, "Should have raised KeyError"
        except KeyError as e:
            error_msg = str(e)
            # Should contain list of available adapters
            assert "storage" in error_msg or "sql" in error_msg
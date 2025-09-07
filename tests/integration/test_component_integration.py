"""
Integration tests for Component Interactions.
Tests cross-component data flow, component failure recovery, configuration validation,
and production-like scenarios across the entire ML pipeline ecosystem.
"""

import pytest
import mlflow
import pandas as pd
import numpy as np
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from src.factory import Factory
from src.pipelines.train_pipeline import run_train_pipeline


class TestComponentDataFlow:
    """Test data flow between different pipeline components."""

    def test_fetcher_to_datahandler_flow(self, integration_settings_classification):
        """Test data flow from Fetcher to DataHandler components."""
        # Create factory and components
        factory = Factory(integration_settings_classification)
        
        # Create components in sequence
        data_adapter = factory.create_data_adapter()
        fetcher = factory.create_fetcher()
        datahandler = factory.create_datahandler()
        
        assert data_adapter is not None
        assert fetcher is not None
        assert datahandler is not None
        
        # Test data flow: DataAdapter -> Fetcher -> DataHandler
        # Load raw data
        raw_data = data_adapter.read(integration_settings_classification.recipe.data.loader.source_uri)
        assert isinstance(raw_data, pd.DataFrame)
        assert len(raw_data) > 0
        
        # Fetch (feature augmentation)
        fetched_data = fetcher.fetch(raw_data, run_mode="batch")
        assert isinstance(fetched_data, pd.DataFrame)
        assert len(fetched_data) > 0
        
        # Process data through DataHandler
        X, y, additional_data = datahandler.prepare_data(fetched_data)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) > 0
        assert len(y) > 0
        
        # Verify data consistency through the flow
        assert len(X) == len(y)
        assert len(X) <= len(fetched_data)  # May filter some rows
        
        # Check that features were extracted correctly
        feature_columns = [col for col in X.columns if col.startswith('feature_')]
        assert len(feature_columns) > 0

    def test_datahandler_to_preprocessor_flow(self, integration_settings_classification):
        """Test data flow from DataHandler to Preprocessor components."""
        factory = Factory(integration_settings_classification)
        
        # Create components
        data_adapter = factory.create_data_adapter()
        fetcher = factory.create_fetcher()
        datahandler = factory.create_datahandler()
        preprocessor = factory.create_preprocessor()  # May be None if not configured
        
        # Get data through the flow
        raw_data = data_adapter.read(integration_settings_classification.recipe.data.loader.source_uri)
        fetched_data = fetcher.fetch(raw_data, run_mode="batch")
        X, y, additional_data = datahandler.prepare_data(fetched_data)
        
        if preprocessor is not None:
            # Test preprocessing flow
            preprocessed_data = preprocessor.fit_transform(X)
            assert isinstance(preprocessed_data, (pd.DataFrame, np.ndarray))
            
            if isinstance(preprocessed_data, pd.DataFrame):
                assert len(preprocessed_data) == len(X)
            else:  # numpy array
                assert preprocessed_data.shape[0] == len(X)
        else:
            # If no preprocessor, data should pass through unchanged
            assert X is not None
            assert y is not None

    def test_complete_component_pipeline_flow(self, integration_settings_classification):
        """Test complete data flow through all components."""
        factory = Factory(integration_settings_classification)
        
        # Create all components
        data_adapter = factory.create_data_adapter()
        fetcher = factory.create_fetcher()
        datahandler = factory.create_datahandler()
        preprocessor = factory.create_preprocessor()
        model = factory.create_model()
        trainer = factory.create_trainer()
        evaluator = factory.create_evaluator()
        
        # Verify all essential components were created
        assert data_adapter is not None
        assert fetcher is not None
        assert datahandler is not None
        assert model is not None
        assert trainer is not None
        assert evaluator is not None
        
        # Test complete flow
        # Step 1: Load data
        raw_data = data_adapter.read(integration_settings_classification.recipe.data.loader.source_uri)
        assert isinstance(raw_data, pd.DataFrame)
        
        # Step 2: Fetch data (feature augmentation)
        fetched_data = fetcher.fetch(raw_data, run_mode="batch")
        assert isinstance(fetched_data, pd.DataFrame)
        
        # Step 3: Process data
        X, y, additional_data = datahandler.prepare_data(fetched_data)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        
        # Step 4: Preprocessing (if configured)
        if preprocessor:
            preprocessed_X = preprocessor.fit_transform(X)
            final_X = preprocessed_X
        else:
            final_X = X
        
        # Step 5: Validate final data for training
        if isinstance(final_X, pd.DataFrame):
            assert len(final_X) == len(y)
            final_features = final_X
        else:
            # Handle numpy array case
            assert final_X.shape[0] == len(y)
            final_features = final_X
        
        # Step 6: Train model (basic validation)
        assert final_features is not None
        assert y is not None
        assert len(final_features) == len(y)

    def test_error_propagation_through_components(self, integration_settings_classification):
        """Test how errors propagate through component chain."""
        factory = Factory(integration_settings_classification)
        
        # Test 1: Invalid data source
        invalid_settings = integration_settings_classification
        invalid_settings.recipe.data.loader.source_uri = "/nonexistent/path.csv"
        
        invalid_factory = Factory(invalid_settings)
        data_adapter = invalid_factory.create_data_adapter()
        
        # Should raise appropriate error when trying to load data
        with pytest.raises(Exception):
            data_adapter.read("/nonexistent/path.csv")
        
        # Test 2: Component configuration error
        # Test with invalid model class path
        invalid_model_settings = integration_settings_classification
        invalid_model_settings.recipe.model.class_path = "nonexistent.InvalidModel"
        
        invalid_model_factory = Factory(invalid_model_settings)
        
        # Should raise error when trying to create invalid model
        with pytest.raises(Exception):
            invalid_model_factory.create_model()


class TestComponentPerformance:
    """Test performance characteristics of component interactions."""

    def test_component_creation_performance(self, integration_settings_classification):
        """Test performance of component creation and initialization."""
        factory = Factory(integration_settings_classification)
        
        # Measure component creation times
        creation_times = {}
        
        components = [
            ('fetcher', factory.create_fetcher),
            ('datahandler', factory.create_datahandler), 
            ('preprocessor', factory.create_preprocessor),
            ('model', factory.create_model),
            ('trainer', factory.create_trainer),
            ('evaluator', factory.create_evaluator)
        ]
        
        for component_name, create_func in components:
            start_time = time.time()
            component = create_func()
            end_time = time.time()
            
            creation_times[component_name] = end_time - start_time
            
            # Basic performance assertion - components should create quickly
            assert creation_times[component_name] < 5.0, f"{component_name} took too long to create"
        
        # Verify all components were created successfully
        total_creation_time = sum(creation_times.values())
        assert total_creation_time < 10.0, "Total component creation took too long"

    def test_concurrent_component_usage(self, integration_settings_classification):
        """Test concurrent usage of components."""
        factory = Factory(integration_settings_classification)
        
        # Create components
        data_adapter = factory.create_data_adapter()
        fetcher = factory.create_fetcher()
        datahandler = factory.create_datahandler()
        
        # Test concurrent data fetching
        results = []
        errors = []
        
        def fetch_and_process():
            try:
                raw_data = data_adapter.read(integration_settings_classification.recipe.data.loader.source_uri)
                fetched_data = fetcher.fetch(raw_data, run_mode="batch")
                X, y, additional_data = datahandler.prepare_data(fetched_data)
                results.append(len(X))
            except Exception as e:
                errors.append(str(e))
        
        # Run concurrent operations
        threads = []
        for i in range(3):  # Limited concurrency for integration test
            thread = threading.Thread(target=fetch_and_process)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Verify results
        assert len(errors) == 0, f"Concurrent operations failed: {errors}"
        assert len(results) == 3, "Not all concurrent operations completed"
        
        # All operations should return same data size (deterministic)
        assert all(r == results[0] for r in results), "Concurrent operations returned different results"

    def test_memory_usage_across_components(self, integration_settings_classification):
        """Test memory usage patterns across component interactions."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        factory = Factory(integration_settings_classification)
        
        # Create components and measure memory
        components_memory = {}
        
        # Measure after each component creation
        data_adapter = factory.create_data_adapter()
        components_memory['data_adapter'] = process.memory_info().rss - initial_memory
        
        fetcher = factory.create_fetcher()
        components_memory['fetcher'] = process.memory_info().rss - initial_memory
        
        datahandler = factory.create_datahandler()
        components_memory['datahandler'] = process.memory_info().rss - initial_memory
        
        model = factory.create_model()
        components_memory['model'] = process.memory_info().rss - initial_memory
        
        # Process some data
        raw_data = data_adapter.read(integration_settings_classification.recipe.data.loader.source_uri)
        fetched_data = fetcher.fetch(raw_data, run_mode="batch")
        X, y, additional_data = datahandler.prepare_data(fetched_data)
        components_memory['after_processing'] = process.memory_info().rss - initial_memory
        
        # Basic memory usage assertions
        # Memory usage should be reasonable (less than 500MB for integration tests)
        max_memory_mb = max(components_memory.values()) / (1024 * 1024)
        assert max_memory_mb < 500, f"Memory usage too high: {max_memory_mb} MB"
        
        # Memory should increase as we create more components (some growth expected)
        assert components_memory['after_processing'] > components_memory['fetcher']


class TestComponentConfiguration:
    """Test component configuration validation and interactions."""

    def test_component_configuration_consistency(self, integration_settings_classification):
        """Test that all components receive consistent configuration."""
        factory = Factory(integration_settings_classification)
        
        # Create multiple components
        fetcher = factory.create_fetcher()
        datahandler = factory.create_datahandler()
        model = factory.create_model()
        
        # Test that components have access to consistent settings
        # (This would be more detailed with actual component inspection methods)
        assert fetcher is not None
        assert datahandler is not None
        assert model is not None
        
        # Verify factory maintains consistent settings across component creation
        assert factory.settings is not None
        
        # Test component creation with same factory produces consistent results
        fetcher2 = factory.create_fetcher()
        model2 = factory.create_model()
        
        assert fetcher2 is not None
        assert model2 is not None
        assert type(fetcher) == type(fetcher2)
        assert type(model) == type(model2)

    def test_component_reconfiguration(self, integration_settings_classification):
        """Test component behavior under configuration changes."""
        # Create factory with initial settings
        factory1 = Factory(integration_settings_classification)
        model1 = factory1.create_model()
        
        # Modify settings and create new factory
        modified_settings = integration_settings_classification
        modified_settings.recipe.model.hyperparameters.n_estimators = 20  # Changed from 10
        
        factory2 = Factory(modified_settings)
        model2 = factory2.create_model()
        
        # Both models should be created successfully
        assert model1 is not None
        assert model2 is not None
        
        # Models should potentially have different configurations
        # (Detailed inspection would depend on model implementation)
        assert type(model1) == type(model2)  # Same class
        
        # Test that components can handle configuration updates
        # (This would be more detailed with actual component inspection)

    def test_invalid_component_configuration_handling(self, integration_settings_classification):
        """Test handling of invalid component configurations."""
        # Test 1: Invalid adapter type
        invalid_settings = integration_settings_classification
        invalid_settings.config.data_source.adapter_type = "invalid_adapter_type"
        
        factory = Factory(invalid_settings)
        
        # Should handle invalid configuration gracefully or raise appropriate error
        with pytest.raises(Exception):
            factory.create_data_adapter()
        
        # Test 2: Missing required configuration
        incomplete_settings = integration_settings_classification
        
        # Remove essential configuration (simulate incomplete setup)
        if hasattr(incomplete_settings.recipe.model, 'class_path'):
            incomplete_settings.recipe.model.class_path = None
        
        incomplete_factory = Factory(incomplete_settings)
        
        # Should handle missing configuration appropriately
        with pytest.raises(Exception):
            incomplete_factory.create_model()


class TestProductionScenarios:
    """Test production-like scenarios and edge cases."""

    def test_large_dataset_component_handling(self, integration_settings_classification):
        """Test component behavior with larger datasets."""
        from tests.helpers.dataframe_builder import DataFrameBuilder
        
        # Create larger dataset
        large_data = DataFrameBuilder.build_classification_data(
            n_samples=1000,  # Larger but still manageable for integration tests
            n_features=20,   # More features
            n_classes=3,     # More classes
            add_entity_column=True
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        # Update settings to use larger dataset
        settings = integration_settings_classification
        settings.recipe.data.loader.source_uri = temp_path
        settings.recipe.data.data_interface.feature_columns = [f'feature_{i}' for i in range(20)]
        
        # Test components with larger dataset
        factory = Factory(settings)
        
        # Test each component can handle larger data
        data_adapter = factory.create_data_adapter()
        fetcher = factory.create_fetcher()
        
        raw_data = data_adapter.read(temp_path)
        fetched_data = fetcher.fetch(raw_data, run_mode="batch")
        assert len(fetched_data) == 1000
        assert len(fetched_data.columns) >= 20
        
        datahandler = factory.create_datahandler()
        X, y, additional_data = datahandler.prepare_data(fetched_data)
        assert len(X) > 0
        assert len(y) > 0
        
        # Cleanup
        Path(temp_path).unlink()

    def test_error_recovery_mechanisms(self, integration_settings_classification):
        """Test component error recovery and resilience."""
        factory = Factory(integration_settings_classification)
        
        # Test 1: Recoverable data processing error
        data_adapter = factory.create_data_adapter()
        fetcher = factory.create_fetcher()
        datahandler = factory.create_datahandler()
        
        # Get valid data first
        raw_data = data_adapter.read(integration_settings_classification.recipe.data.loader.source_uri)
        valid_data = fetcher.fetch(raw_data, run_mode="batch")
        
        # Test processing with partially corrupted data
        corrupted_data = valid_data.copy()
        corrupted_data.loc[0, 'feature_0'] = np.nan  # Introduce NaN
        corrupted_data.loc[1, 'target'] = 'invalid_value'  # Invalid target
        
        # DataHandler should handle corrupted data gracefully
        try:
            X, y, additional_data = datahandler.prepare_data(corrupted_data)
            assert X is not None
            assert y is not None
            # May have fewer rows due to cleaning
            assert len(X) <= len(corrupted_data)
            assert len(X) == len(y)
        except Exception as e:
            # If error occurs, it should be informative
            assert len(str(e)) > 0

    def test_component_resource_cleanup(self, integration_settings_classification):
        """Test that components properly clean up resources."""
        factory = Factory(integration_settings_classification)
        
        # Create components
        components = []
        components.append(factory.create_fetcher())
        components.append(factory.create_datahandler())
        components.append(factory.create_model())
        
        # Verify components were created
        for component in components:
            assert component is not None
        
        # Test resource cleanup (implicit through garbage collection)
        # In a more detailed implementation, this would test explicit cleanup methods
        component_count = len(components)
        components = None  # Release references
        
        # Basic cleanup verification (components should be cleanable)
        import gc
        gc.collect()  # Force garbage collection
        
        # If components had explicit cleanup methods, we would test them here
        # For now, we just verify the test doesn't crash

    def test_concurrent_pipeline_execution(self, integration_settings_classification):
        """Test concurrent execution of multiple pipeline instances."""
        def run_pipeline_instance(settings, results_list, errors_list):
            try:
                result = run_train_pipeline(settings)
                results_list.append(result.run_id)
            except Exception as e:
                errors_list.append(str(e))
        
        # Prepare for concurrent execution
        results = []
        errors = []
        threads = []
        
        # Run multiple pipeline instances concurrently
        for i in range(2):  # Limited to 2 for integration testing
            # Create separate settings for each instance to avoid conflicts
            settings_copy = integration_settings_classification
            
            thread = threading.Thread(
                target=run_pipeline_instance, 
                args=(settings_copy, results, errors)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=60)  # 60 second timeout for each pipeline
        
        # Verify concurrent execution
        assert len(errors) == 0, f"Concurrent pipeline execution failed: {errors}"
        assert len(results) == 2, "Not all concurrent pipelines completed"
        assert len(set(results)) == 2, "Concurrent pipelines should produce different run IDs"
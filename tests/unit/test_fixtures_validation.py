"""
Test to validate Phase 1 fixtures are working correctly.
This is a critical validation test for the test infrastructure itself.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

from src.settings import Settings, Config, Recipe
from src.factory import Factory


class TestFixtureValidation:
    """Validate that all Phase 1 fixtures work correctly."""
    
    def test_silence_logger_is_active(self, caplog):
        """Verify logger is silenced by default."""
        import logging
        logger = logging.getLogger(__name__)
        logger.critical("This should not appear")
        # If silence_logger works, nothing should be captured
        assert len(caplog.records) == 0
    
    def test_isolated_env(self, isolated_env):
        """Test isolated environment fixture."""
        assert isolated_env.exists()
        assert (isolated_env / "configs").exists()
        assert (isolated_env / "data").exists()
        assert (isolated_env / "mlruns").exists()
        
        import os
        assert os.getenv("ENV_NAME") == "test"
        assert os.getenv("MLFLOW_TRACKING_URI") == str(isolated_env / "mlruns")
    
    def test_temp_workspace(self, temp_workspace):
        """Test temporary workspace fixture."""
        assert temp_workspace.exists()
        assert (temp_workspace / "configs").exists()
        assert (temp_workspace / "recipes").exists()
        assert (temp_workspace / "data").exists()
    
    def test_clean_registries(self):
        """Test registry cleaning fixture."""
        from src.components.adapter import AdapterRegistry
        from src.components.fetcher import FetcherRegistry
        from src.components.evaluator import EvaluatorRegistry
        
        # Registries should have their default components
        assert "storage" in AdapterRegistry.adapters
        assert "pass_through" in FetcherRegistry.fetchers
        assert "classification" in EvaluatorRegistry.evaluators
    
    def test_test_settings(self, test_settings):
        """Test settings fixture."""
        assert isinstance(test_settings, Settings)
        assert test_settings.config is not None
        assert test_settings.recipe is not None
        assert test_settings.config.environment.name == "test"
    
    def test_test_config(self, test_config):
        """Test config fixture."""
        assert isinstance(test_config, Config)
        assert test_config.environment.name == "test"
        assert test_config.mlflow.tracking_uri == "./mlruns"
    
    def test_test_recipe(self, test_recipe):
        """Test recipe fixture."""
        assert isinstance(test_recipe, Recipe)
        assert test_recipe.name == "test_recipe"
        assert test_recipe.model.class_path == "sklearn.ensemble.RandomForestClassifier"
    
    def test_factory_with_clean_cache(self, factory_with_clean_cache, test_settings):
        """Test factory with clean cache fixture."""
        assert isinstance(factory_with_clean_cache, Factory)
        assert len(factory_with_clean_cache._component_cache) == 0
        
        # Create a component to test caching
        adapter = factory_with_clean_cache.create_data_adapter("storage")
        assert adapter is not None
        assert len(factory_with_clean_cache._component_cache) == 1
    
    def test_mock_factory(self, mock_factory):
        """Test mock factory fixture."""
        assert isinstance(mock_factory, MagicMock)
        
        # Test mock methods work
        model = mock_factory.create_model()
        assert model is not None
        
        adapter = mock_factory.create_data_adapter()
        assert adapter is not None
        
        fetcher = mock_factory.create_fetcher()
        assert fetcher is not None
        
        evaluator = mock_factory.create_evaluator()
        assert evaluator is not None
        assert evaluator.evaluate.return_value is not None
    
    def test_mock_mlflow(self, mock_mlflow):
        """Test MLflow mocking fixture."""
        assert 'start_run' in mock_mlflow
        assert 'log_metric' in mock_mlflow
        assert mock_mlflow['run'].info.run_id == 'test_run_id'
    
    def test_sample_dataframe(self, sample_dataframe):
        """Test sample dataframe fixture."""
        assert isinstance(sample_dataframe, pd.DataFrame)
        assert len(sample_dataframe) == 100
        assert 'user_id' in sample_dataframe.columns
        assert 'target' in sample_dataframe.columns
    
    def test_classification_data(self, classification_data):
        """Test classification data fixture."""
        X, y = classification_data
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert X.shape[1] == 5
    
    def test_regression_data(self, regression_data):
        """Test regression data fixture."""
        X, y = regression_data
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
    
    def test_sample_csv_data(self, sample_csv_data):
        """Test sample CSV data fixture."""
        assert sample_csv_data.exists()
        df = pd.read_csv(sample_csv_data)
        assert len(df) == 5
        assert 'user_id' in df.columns
    
    async def test_async_client(self, async_client):
        """Test async client fixture."""
        # This will fail if FastAPI app is not properly initialized
        # but we're testing the fixture exists and is callable
        assert async_client is not None
    
    def test_event_loop(self, event_loop):
        """Test event loop fixture."""
        import asyncio
        assert isinstance(event_loop, asyncio.AbstractEventLoop)
        assert not event_loop.is_closed()


class TestHelperModules:
    """Test helper modules are importable and functional."""
    
    def test_assertions_module(self):
        """Test assertions.py module."""
        from tests.helpers import assertions
        
        # Test a simple assertion helper
        config_dict = {
            'environment': {'name': 'test'},
            'data_source': {
                'name': 'test_ds',
                'adapter_type': 'storage',
                'config': {}
            }
        }
        from src.settings import Config
        config = Config(**config_dict)
        assertions.assert_config_valid(config)
    
    def test_builders_module(self):
        """Test builders.py module."""
        from tests.helpers.builders import (
            ConfigBuilder, RecipeBuilder, SettingsBuilder,
            DataFrameBuilder, MockBuilder
        )
        
        # Test builders work
        config = ConfigBuilder.build()
        assert config is not None
        
        recipe = RecipeBuilder.build()
        assert recipe is not None
        
        df = DataFrameBuilder.build()
        assert isinstance(df, pd.DataFrame)
        
        mock_model = MockBuilder.build_mock_model()
        assert mock_model is not None
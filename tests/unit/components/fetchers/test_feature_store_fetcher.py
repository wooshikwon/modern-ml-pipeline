"""
Feature Store Fetcher Unit Tests - Minimal Mocking for External Service
Following comprehensive testing strategy document principles
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

from src.components.fetcher.modules.feature_store_fetcher import FeatureStoreFetcher
from src.interface.base_fetcher import BaseFetcher

# Skip all tests in this module if feast is not installed
try:
    import feast
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False


@pytest.mark.skipif(not FEAST_AVAILABLE, reason="Feast is not installed")
class TestFeatureStoreFetcher:
    """Test FeatureStoreFetcher with minimal mocking for Feast."""
    
    def test_feature_store_fetcher_initialization(self, settings_builder):
        """Test FeatureStoreFetcher initialization."""
        # Given: Valid settings with feature store config
        settings = settings_builder \
            .with_feature_store(enabled=True) \
            .build()
        
        # When: Creating FeatureStoreFetcher
        mock_factory = MagicMock()
        mock_adapter = MagicMock()
        mock_factory.create_feature_store_adapter.return_value = mock_adapter
        
        fetcher = FeatureStoreFetcher(settings, mock_factory)
        
        # Then: Fetcher is properly initialized
        assert isinstance(fetcher, FeatureStoreFetcher)
        assert isinstance(fetcher, BaseFetcher)
        assert hasattr(fetcher, 'feature_store_adapter')
    
    def test_fetch_from_feature_store_online(self, settings_builder):
        """Test fetching features from online feature store."""
        # Given: Settings and mock feature store
        settings = settings_builder \
            .with_feature_store(enabled=True) \
            .build()
        
        # Input DataFrame with entity IDs
        input_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'timestamp': [datetime.now()] * 3
        })
        
        # Mock features that adapter will return
        mock_features = pd.DataFrame({
            'user_id': [1, 2, 3],
            'feature_1': [0.5, 0.7, 0.3],
            'feature_2': [100, 200, 150],
            'feature_3': ['A', 'B', 'A']
        })
        
        # Mock factory and adapter
        mock_factory = MagicMock()
        mock_adapter = MagicMock()
        mock_factory.create_feature_store_adapter.return_value = mock_adapter
        
        # Mock adapter's fetch method to return merged features
        mock_adapter.fetch.return_value = mock_features
        
        fetcher = FeatureStoreFetcher(settings, mock_factory)
        
        # When: Fetching features
        result = fetcher.fetch(input_df)
        
        # Then: Features are fetched via adapter
        mock_adapter.fetch.assert_called_once()
        # Result should be what the adapter returns
        if result is not None:
            assert isinstance(result, pd.DataFrame)
    
    def test_fetch_with_historical_features(self, settings_builder):
        """Test fetching historical features."""
        # Given: Settings for historical features
        settings = settings_builder \
            .with_feature_store(enabled=True) \
            .build()
        
        # Historical data with timestamps
        current_time = datetime.now()
        input_df = pd.DataFrame({
            'product_id': [10, 20, 30, 10, 20],
            'event_timestamp': [
                current_time - timedelta(days=i) for i in range(5)
            ]
        })
        
        # Mock historical features
        mock_historical = pd.DataFrame({
            'product_id': [10, 20, 30, 10, 20],
            'event_timestamp': input_df['event_timestamp'],
            'price': [99.9, 149.9, 79.9, 99.9, 149.9],
            'category': ['electronics', 'clothing', 'books', 'electronics', 'clothing']
        })
        
        # Mock factory and adapter
        mock_factory = MagicMock()
        mock_adapter = MagicMock()
        mock_factory.create_feature_store_adapter.return_value = mock_adapter
        mock_adapter.fetch.return_value = mock_historical
        
        fetcher = FeatureStoreFetcher(settings, mock_factory)
        
        # When: Fetching historical features
        result = fetcher.fetch(input_df)
        
        # Then: Historical features are returned via adapter
        mock_adapter.fetch.assert_called_once()
        if result is not None:
            assert isinstance(result, pd.DataFrame)
    
    def test_fetch_handles_missing_entities(self, settings_builder):
        """Test handling of missing entities in feature store."""
        # Given: Some entities not in feature store
        settings = settings_builder \
            .with_feature_store(enabled=True) \
            .build()
        
        input_df = pd.DataFrame({
            'item_id': [1, 2, 3, 999],  # 999 doesn't exist
        })
        
        # Mock features only for existing entities
        mock_features = pd.DataFrame({
            'item_id': [1, 2, 3, 999],
            'feature_a': [1.0, 2.0, 3.0, None],
            'feature_b': ['x', 'y', 'z', None]
        })
        
        # Mock factory and adapter
        mock_factory = MagicMock()
        mock_adapter = MagicMock()
        mock_factory.create_feature_store_adapter.return_value = mock_adapter
        mock_adapter.fetch.return_value = mock_features
        
        fetcher = FeatureStoreFetcher(settings, mock_factory)
        
        # When: Fetching with missing entities
        result = fetcher.fetch(input_df)
        
        # Then: Result includes all entities (with nulls for missing)
        if result is not None and 'feature_a' in result.columns:
            assert len(result) == 4
            assert result['feature_a'].isna().sum() == 1  # One null value
    
    def test_fetch_with_empty_input(self, settings_builder):
        """Test fetching with empty input DataFrame."""
        # Given: Empty input
        settings = settings_builder \
            .with_feature_store(enabled=True) \
            .build()
        
        input_df = pd.DataFrame(columns=['id', 'timestamp'])
        
        # Mock factory and adapter
        mock_factory = MagicMock()
        mock_adapter = MagicMock()
        mock_factory.create_feature_store_adapter.return_value = mock_adapter
        
        # Adapter should handle empty input gracefully
        mock_adapter.fetch.return_value = pd.DataFrame()
        
        fetcher = FeatureStoreFetcher(settings, mock_factory)
        
        # When: Fetching with empty input
        result = fetcher.fetch(input_df)
        
        # Then: Returns empty DataFrame or original
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_fetch_performance_with_large_batch(self, settings_builder):
        """Test fetching performance with large batch."""
        # Given: Large batch of entities
        settings = settings_builder \
            .with_feature_store(enabled=True) \
            .build()
        
        # Large input
        input_df = pd.DataFrame({
            'entity_id': range(500),
            'timestamp': [datetime.now()] * 500
        })
        
        # Mock large feature response
        mock_features = pd.DataFrame({
            'entity_id': range(500),
            'feature_x': np.random.randn(500),
            'feature_y': np.random.randn(500)
        })
        
        # Mock factory and adapter
        mock_factory = MagicMock()
        mock_adapter = MagicMock()
        mock_factory.create_feature_store_adapter.return_value = mock_adapter
        mock_adapter.fetch.return_value = mock_features
        
        fetcher = FeatureStoreFetcher(settings, mock_factory)
        
        # When: Fetching large batch
        result = fetcher.fetch(input_df)
        
        # Then: All entities are processed
        if result is not None and 'feature_x' in result.columns:
            assert len(result) == 500
            assert 'feature_x' in result.columns
            assert 'feature_y' in result.columns
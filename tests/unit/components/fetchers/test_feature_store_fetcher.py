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


class TestFeatureStoreFetcher:
    """Test FeatureStoreFetcher with minimal mocking for Feast."""
    
    def test_feature_store_fetcher_initialization(self, settings_builder):
        """Test FeatureStoreFetcher initialization."""
        # Given: Valid settings with feature store config
        settings = settings_builder \
            .with_feature_store(config={
                "feature_service": "customer_features",
                "entity_column": "customer_id"
            }) \
            .build()
        
        # When: Creating FeatureStoreFetcher
        with patch('src.components.fetcher.modules.feature_store_fetcher.FeatureStore'):
            fetcher = FeatureStoreFetcher(settings)
        
        # Then: Fetcher is properly initialized
        assert isinstance(fetcher, FeatureStoreFetcher)
        assert isinstance(fetcher, BaseFetcher)
        assert hasattr(fetcher, 'feature_service')
    
    def test_fetch_from_feature_store_online(self, settings_builder):
        """Test fetching features from online feature store."""
        # Given: Settings and mock feature store
        settings = settings_builder \
            .with_feature_store(config={
                "feature_service": "user_features",
                "entity_column": "user_id"
            }) \
            .build()
        
        # Input DataFrame with entity IDs
        input_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'timestamp': [datetime.now()] * 3
        })
        
        # Mock feature store response
        mock_features = pd.DataFrame({
            'user_id': [1, 2, 3],
            'feature_1': [0.5, 0.7, 0.3],
            'feature_2': [100, 200, 150],
            'feature_3': ['A', 'B', 'A']
        })
        
        with patch('src.components.fetcher.modules.feature_store_fetcher.FeatureStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            
            mock_response = MagicMock()
            mock_response.to_df.return_value = mock_features
            mock_store.get_online_features.return_value = mock_response
            
            fetcher = FeatureStoreFetcher(settings)
            
            # When: Fetching features
            result = fetcher.fetch(input_df)
            
            # Then: Features are fetched and merged
            assert isinstance(result, pd.DataFrame)
            assert 'feature_1' in result.columns
            assert 'feature_2' in result.columns
            assert len(result) == 3
    
    def test_fetch_with_historical_features(self, settings_builder):
        """Test fetching historical features."""
        # Given: Settings for historical features
        settings = settings_builder \
            .with_feature_store(config={
                "feature_service": "product_features",
                "entity_column": "product_id",
                "use_historical": True
            }) \
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
        
        with patch('src.components.fetcher.modules.feature_store_fetcher.FeatureStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            
            mock_job = MagicMock()
            mock_job.to_df.return_value = mock_historical
            mock_store.get_historical_features.return_value = mock_job
            
            fetcher = FeatureStoreFetcher(settings)
            
            # When: Fetching historical features
            result = fetcher.fetch(input_df)
            
            # Then: Historical features are returned
            assert isinstance(result, pd.DataFrame)
            assert 'price' in result.columns
            assert 'category' in result.columns
            assert len(result) == 5
    
    def test_fetch_handles_missing_entities(self, settings_builder):
        """Test handling of missing entities in feature store."""
        # Given: Some entities not in feature store
        settings = settings_builder \
            .with_feature_store(config={
                "feature_service": "item_features",
                "entity_column": "item_id"
            }) \
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
        
        with patch('src.components.fetcher.modules.feature_store_fetcher.FeatureStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            
            mock_response = MagicMock()
            mock_response.to_df.return_value = mock_features
            mock_store.get_online_features.return_value = mock_response
            
            fetcher = FeatureStoreFetcher(settings)
            
            # When: Fetching with missing entities
            result = fetcher.fetch(input_df)
            
            # Then: Result includes all entities (with nulls for missing)
            assert len(result) == 4
            assert result['feature_a'].isna().sum() == 1  # One null value
    
    def test_fetch_with_empty_input(self, settings_builder):
        """Test fetching with empty input DataFrame."""
        # Given: Empty input
        settings = settings_builder \
            .with_feature_store(config={
                "feature_service": "test_features",
                "entity_column": "id"
            }) \
            .build()
        
        input_df = pd.DataFrame(columns=['id', 'timestamp'])
        
        with patch('src.components.fetcher.modules.feature_store_fetcher.FeatureStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            
            fetcher = FeatureStoreFetcher(settings)
            
            # When: Fetching with empty input
            result = fetcher.fetch(input_df)
            
            # Then: Returns empty DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
    
    def test_fetch_performance_with_large_batch(self, settings_builder):
        """Test fetching performance with large batch."""
        # Given: Large batch of entities
        settings = settings_builder \
            .with_feature_store(config={
                "feature_service": "batch_features",
                "entity_column": "entity_id",
                "batch_size": 100
            }) \
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
        
        with patch('src.components.fetcher.modules.feature_store_fetcher.FeatureStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            
            mock_response = MagicMock()
            mock_response.to_df.return_value = mock_features
            mock_store.get_online_features.return_value = mock_response
            
            fetcher = FeatureStoreFetcher(settings)
            
            # When: Fetching large batch
            result = fetcher.fetch(input_df)
            
            # Then: All entities are processed
            assert len(result) == 500
            assert 'feature_x' in result.columns
            assert 'feature_y' in result.columns
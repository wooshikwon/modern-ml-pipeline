"""
Feast Adapter Unit Tests - Minimal Mocking for Feature Store
Following comprehensive testing strategy document principles
External services exception: Using minimal mock for Feast store only
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

from src.components.adapter.modules.feast_adapter import FeastAdapter
from src.interface.base_adapter import BaseAdapter


class TestFeastAdapterWithMinimalMocking:
    """Test FeastAdapter with minimal mocking for external Feast feature store."""
    
    def test_feast_adapter_initialization(self, settings_builder):
        """Test FeastAdapter initialization with mock feature store."""
        # Given: Valid Feast settings
        settings = settings_builder \
            .with_data_source("feast", config={
                "feature_store_yaml_path": "/path/to/feature_store.yaml",
                "entity_df_event_timestamp_col": "event_timestamp"
            }) \
            .build()
        
        # When: Creating FeastAdapter with mocked feature store
        with patch('feast.FeatureStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            adapter = FeastAdapter(settings)
        
        # Then: Adapter is properly initialized
        assert isinstance(adapter, FeastAdapter)
        assert isinstance(adapter, BaseAdapter)
        assert adapter.event_timestamp_col == "event_timestamp"
    
    def test_read_from_feast_online_store(self, settings_builder):
        """Test reading features from Feast online store."""
        # Given: Settings and mock Feast store
        settings = settings_builder \
            .with_data_source("feast", config={
                "feature_store_yaml_path": "/path/to/feature_store.yaml"
            }) \
            .build()
        
        # Create mock feature data
        mock_features = pd.DataFrame({
            "entity_id": [1, 2, 3],
            "feature_1": [0.5, 0.7, 0.3],
            "feature_2": [1.2, 2.3, 3.4],
            "feature_3": [10, 20, 30]
        })
        
        with patch('feast.FeatureStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            
            # Mock get_online_features response
            mock_response = MagicMock()
            mock_response.to_df.return_value = mock_features
            mock_store.get_online_features.return_value = mock_response
            
            adapter = FeastAdapter(settings)
            
            # When: Reading from online store
            entity_df = pd.DataFrame({"entity_id": [1, 2, 3]})
            features = ["feature_1", "feature_2", "feature_3"]
            df = adapter.read(entity_df)
            
            # Then: Features are returned correctly
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
            assert set(df.columns) == {"entity_id", "feature_1", "feature_2", "feature_3"}
            mock_store.get_online_features.assert_called_once()
    
    def test_read_from_feast_historical_features(self, settings_builder):
        """Test reading historical features from Feast."""
        # Given: Settings and mock Feast store
        settings = settings_builder \
            .with_data_source("feast", config={
                "feature_store_yaml_path": "/path/to/feature_store.yaml",
                "entity_df_event_timestamp_col": "event_timestamp"
            }) \
            .build()
        
        # Create mock historical feature data
        current_time = datetime.now()
        mock_historical = pd.DataFrame({
            "entity_id": [1, 1, 2, 2, 3, 3],
            "event_timestamp": [
                current_time - timedelta(days=i) 
                for i in range(6)
            ],
            "feature_1": np.random.randn(6),
            "feature_2": np.random.randn(6),
            "target": [0, 1, 1, 0, 1, 0]
        })
        
        with patch('feast.FeatureStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store
            
            # Mock get_historical_features response
            mock_retrieval_job = MagicMock()
            mock_retrieval_job.to_df.return_value = mock_historical
            mock_store.get_historical_features.return_value = mock_retrieval_job
            
            adapter = FeastAdapter(settings)
            
            # When: Reading historical features
            entity_df = pd.DataFrame({
                "entity_id": [1, 2, 3],
                "event_timestamp": [current_time] * 3
            })
            features = ["feature_1", "feature_2"]
            df = adapter.read(
                source="historical",
                params={
                    "entity_df": entity_df,
                    "features": features
                }
            )
            
            # Then: Historical features are returned
            assert isinstance(df, pd.DataFrame)
            assert "entity_id" in df.columns
            assert "event_timestamp" in df.columns
            assert "feature_1" in df.columns
            mock_store.get_historical_features.assert_called_once()
"""
Unit tests for FeatureStoreFetcher.
Tests feature store fetcher functionality with Feast integration.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch

from src.components.fetcher.modules.feature_store_fetcher import FeatureStoreFetcher
from src.interface.base_fetcher import BaseFetcher
from src.settings import Settings


class TestFeatureStoreFetcherInitialization:
    """Test FeatureStoreFetcher initialization."""
    
    def test_feature_store_fetcher_inherits_base_fetcher(self):
        """Test that FeatureStoreFetcher properly inherits from BaseFetcher."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_factory = Mock()
        mock_factory.create_feature_store_adapter.return_value = Mock()
        
        # Act
        fetcher = FeatureStoreFetcher(mock_settings, mock_factory)
        
        # Assert
        assert isinstance(fetcher, BaseFetcher)
        assert isinstance(fetcher, FeatureStoreFetcher)
    
    def test_init_creates_feature_store_adapter(self):
        """Test that initialization creates feature store adapter."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_factory = Mock()
        mock_adapter = Mock()
        mock_factory.create_feature_store_adapter.return_value = mock_adapter
        
        # Act
        fetcher = FeatureStoreFetcher(mock_settings, mock_factory)
        
        # Assert
        assert fetcher.settings == mock_settings
        assert fetcher.factory == mock_factory
        assert fetcher.feature_store_adapter == mock_adapter
        mock_factory.create_feature_store_adapter.assert_called_once()
    
    def test_init_stores_settings_and_factory(self):
        """Test that initialization properly stores settings and factory."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_factory = Mock()
        mock_factory.create_feature_store_adapter.return_value = Mock()
        
        # Act
        fetcher = FeatureStoreFetcher(mock_settings, mock_factory)
        
        # Assert
        assert hasattr(fetcher, 'settings')
        assert hasattr(fetcher, 'factory')
        assert hasattr(fetcher, 'feature_store_adapter')


class TestFeatureStoreFetcherTrainMode:
    """Test FeatureStoreFetcher train mode operations."""
    
    def test_fetch_train_mode_success(self):
        """Test successful fetch operation in train mode."""
        # Arrange
        mock_settings = self._create_mock_settings()
        mock_factory, mock_adapter = self._create_mock_factory_and_adapter()
        
        input_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03']
        })
        
        enriched_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'feature1': [0.1, 0.2, 0.3],
            'feature2': [10, 20, 30]
        })
        
        mock_adapter.get_historical_features_with_validation.return_value = enriched_df
        fetcher = FeatureStoreFetcher(mock_settings, mock_factory)
        
        # Act
        with patch('src.components.fetcher.modules.feature_store_fetcher.logger') as mock_logger:
            result = fetcher.fetch(input_df, run_mode="train")
        
        # Assert
        pd.testing.assert_frame_equal(result, enriched_df)
        mock_adapter.get_historical_features_with_validation.assert_called_once()
        mock_logger.info.assert_any_call("Feature Store를 통해 피처 증강을 시작합니다.")
        mock_logger.info.assert_any_call("피처 증강 완료(offline).")
    
    def test_fetch_batch_mode_success(self):
        """Test successful fetch operation in batch mode."""
        # Arrange
        mock_settings = self._create_mock_settings()
        mock_factory, mock_adapter = self._create_mock_factory_and_adapter()
        
        input_df = pd.DataFrame({
            'entity_id': [100, 200],
            'event_timestamp': ['2023-01-01 10:00:00', '2023-01-01 11:00:00']
        })
        
        result_df = pd.DataFrame({
            'entity_id': [100, 200],
            'event_timestamp': ['2023-01-01 10:00:00', '2023-01-01 11:00:00'],
            'historical_feature': [1.5, 2.5]
        })
        
        mock_adapter.get_historical_features_with_validation.return_value = result_df
        fetcher = FeatureStoreFetcher(mock_settings, mock_factory)
        
        # Act
        result = fetcher.fetch(input_df, run_mode="batch")
        
        # Assert
        pd.testing.assert_frame_equal(result, result_df)
        mock_adapter.get_historical_features_with_validation.assert_called_once()
    
    def test_fetch_train_mode_with_features_configuration(self):
        """Test train mode with specific features configuration."""
        # Arrange
        mock_settings = self._create_mock_settings_with_features()
        mock_factory, mock_adapter = self._create_mock_factory_and_adapter()
        
        input_df = pd.DataFrame({'user_id': [1]})
        expected_features = ['user_stats:login_count', 'user_stats:avg_session_time']
        
        fetcher = FeatureStoreFetcher(mock_settings, mock_factory)
        
        # Act
        fetcher.fetch(input_df, run_mode="train")
        
        # Assert
        call_args = mock_adapter.get_historical_features_with_validation.call_args
        assert call_args[1]['features'] == expected_features
    
    def _create_mock_settings(self):
        """Create mock settings with basic configuration."""
        mock_settings = Mock(spec=Settings)
        mock_settings.recipe = Mock()
        mock_settings.recipe.model = Mock()
        
        # Entity schema
        mock_settings.recipe.model.loader = Mock()
        mock_settings.recipe.model.loader.entity_schema = Mock()
        mock_settings.recipe.model.loader.entity_schema.entity_columns = ['user_id']
        mock_settings.recipe.model.loader.entity_schema.timestamp_column = 'timestamp'
        
        # Fetcher config (no features)
        mock_settings.recipe.model.fetcher = None
        
        # Data interface
        mock_settings.recipe.model.data_interface = Mock()
        mock_settings.recipe.model.data_interface.task_type = 'classification'
        mock_settings.recipe.model.data_interface.target_column = 'target'
        
        return mock_settings
    
    def _create_mock_settings_with_features(self):
        """Create mock settings with features configuration."""
        mock_settings = self._create_mock_settings()
        
        # Add features configuration
        mock_namespace = Mock()
        mock_namespace.feature_namespace = 'user_stats'
        mock_namespace.features = ['login_count', 'avg_session_time']
        
        mock_settings.recipe.model.fetcher = Mock()
        mock_settings.recipe.model.fetcher.features = [mock_namespace]
        
        return mock_settings
    
    def _create_mock_factory_and_adapter(self):
        """Create mock factory and adapter."""
        mock_factory = Mock()
        mock_adapter = Mock()
        mock_adapter.get_historical_features_with_validation.return_value = pd.DataFrame()
        mock_adapter.get_online_features.return_value = pd.DataFrame()
        mock_factory.create_feature_store_adapter.return_value = mock_adapter
        return mock_factory, mock_adapter


class TestFeatureStoreFetcherServingMode:
    """Test FeatureStoreFetcher serving mode operations."""
    
    def test_fetch_serving_mode_success(self):
        """Test successful fetch operation in serving mode."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.recipe = Mock()
        mock_settings.recipe.model = Mock()
        
        # Entity schema for serving
        mock_settings.recipe.model.loader = Mock()
        mock_settings.recipe.model.loader.entity_schema = Mock()
        mock_settings.recipe.model.loader.entity_schema.entity_columns = ['user_id']
        
        mock_settings.recipe.model.fetcher = Mock()
        mock_settings.recipe.model.fetcher.features = []
        
        mock_factory = Mock()
        mock_adapter = Mock()
        
        # Serving mode returns online features
        online_result = pd.DataFrame({
            'user_id': [42],
            'feature1': [0.5],
            'feature2': [100]
        })
        mock_adapter.get_online_features.return_value = online_result
        mock_factory.create_feature_store_adapter.return_value = mock_adapter
        
        input_df = pd.DataFrame({'user_id': [42]})
        fetcher = FeatureStoreFetcher(mock_settings, mock_factory)
        
        # Act
        with patch('src.components.fetcher.modules.feature_store_fetcher.logger') as mock_logger:
            result = fetcher.fetch(input_df, run_mode="serving")
        
        # Assert
        pd.testing.assert_frame_equal(result, online_result)
        mock_adapter.get_online_features.assert_called_once()
        mock_logger.info.assert_any_call("피처 증강 완료(online).")
    
    def test_fetch_serving_mode_entity_rows_conversion(self):
        """Test that serving mode properly converts DataFrame to entity_rows."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.recipe = Mock()
        mock_settings.recipe.model = Mock()
        
        mock_settings.recipe.model.loader = Mock()
        mock_settings.recipe.model.loader.entity_schema = Mock()
        mock_settings.recipe.model.loader.entity_schema.entity_columns = ['user_id', 'session_id']
        
        mock_settings.recipe.model.fetcher = Mock()
        mock_settings.recipe.model.fetcher.features = []
        
        mock_factory = Mock()
        mock_adapter = Mock()
        mock_adapter.get_online_features.return_value = pd.DataFrame()
        mock_factory.create_feature_store_adapter.return_value = mock_adapter
        
        input_df = pd.DataFrame({
            'user_id': [1, 2],
            'session_id': ['sess1', 'sess2'],
            'extra_col': ['a', 'b']  # Should not be included in entity_rows
        })
        
        fetcher = FeatureStoreFetcher(mock_settings, mock_factory)
        
        # Act
        fetcher.fetch(input_df, run_mode="serving")
        
        # Assert
        call_args = mock_adapter.get_online_features.call_args
        entity_rows = call_args[1]['entity_rows']
        expected_entity_rows = [
            {'user_id': 1, 'session_id': 'sess1'},
            {'user_id': 2, 'session_id': 'sess2'}
        ]
        assert entity_rows == expected_entity_rows


class TestFeatureStoreFetcherValidation:
    """Test FeatureStoreFetcher validation and error handling."""
    
    def test_fetch_invalid_run_mode_raises_error(self):
        """Test that invalid run_mode raises ValueError."""
        # Arrange
        mock_settings = self._create_comprehensive_mock_settings()
        mock_factory = Mock()
        mock_factory.create_feature_store_adapter.return_value = Mock()
        
        input_df = pd.DataFrame({'data': [1, 2, 3]})
        fetcher = FeatureStoreFetcher(mock_settings, mock_factory)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported run_mode: invalid"):
            fetcher.fetch(input_df, run_mode="invalid")
    
    def test_fetch_supported_run_modes(self):
        """Test that all supported run modes work."""
        # Arrange
        mock_settings = self._create_comprehensive_mock_settings()
        mock_factory = Mock()
        mock_adapter = Mock()
        mock_adapter.get_historical_features_with_validation.return_value = pd.DataFrame()
        mock_adapter.get_online_features.return_value = pd.DataFrame()
        mock_factory.create_feature_store_adapter.return_value = mock_adapter
        
        input_df = pd.DataFrame({'user_id': [1]})
        fetcher = FeatureStoreFetcher(mock_settings, mock_factory)
        
        supported_modes = ["train", "batch", "serving"]
        
        # Act & Assert
        for mode in supported_modes:
            result = fetcher.fetch(input_df, run_mode=mode)
            assert isinstance(result, pd.DataFrame)
    
    def _create_comprehensive_mock_settings(self):
        """Create comprehensive mock settings for testing."""
        mock_settings = Mock(spec=Settings)
        mock_settings.recipe = Mock()
        mock_settings.recipe.model = Mock()
        
        # Loader config
        mock_settings.recipe.model.loader = Mock()
        mock_settings.recipe.model.loader.entity_schema = Mock()
        mock_settings.recipe.model.loader.entity_schema.entity_columns = ['user_id']
        mock_settings.recipe.model.loader.entity_schema.timestamp_column = 'timestamp'
        
        # Fetcher config
        mock_settings.recipe.model.fetcher = Mock()
        mock_settings.recipe.model.fetcher.features = []
        
        # Data interface
        mock_settings.recipe.model.data_interface = Mock()
        mock_settings.recipe.model.data_interface.task_type = 'classification'
        mock_settings.recipe.model.data_interface.target_column = 'target'
        
        return mock_settings


class TestFeatureStoreFetcherDataInterfaceConfig:
    """Test FeatureStoreFetcher data interface configuration."""
    
    def test_fetch_data_interface_config_construction(self):
        """Test that data_interface_config is properly constructed."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.recipe = Mock()
        mock_settings.recipe.model = Mock()
        
        # Entity schema
        mock_settings.recipe.model.loader = Mock()
        mock_settings.recipe.model.loader.entity_schema = Mock()
        mock_settings.recipe.model.loader.entity_schema.entity_columns = ['user_id', 'item_id']
        mock_settings.recipe.model.loader.entity_schema.timestamp_column = 'event_timestamp'
        
        # Data interface with treatment column (causal inference)
        mock_settings.recipe.model.data_interface = Mock()
        mock_settings.recipe.model.data_interface.task_type = 'causal_inference'
        mock_settings.recipe.model.data_interface.target_column = 'outcome'
        mock_settings.recipe.model.data_interface.treatment_column = 'treatment'
        
        mock_settings.recipe.model.fetcher = None
        
        mock_factory = Mock()
        mock_adapter = Mock()
        mock_factory.create_feature_store_adapter.return_value = mock_adapter
        
        input_df = pd.DataFrame({'user_id': [1], 'item_id': [100]})
        fetcher = FeatureStoreFetcher(mock_settings, mock_factory)
        
        # Act
        fetcher.fetch(input_df, run_mode="train")
        
        # Assert
        call_args = mock_adapter.get_historical_features_with_validation.call_args
        data_interface_config = call_args[1]['data_interface_config']
        
        expected_config = {
            'entity_columns': ['user_id', 'item_id'],
            'timestamp_column': 'event_timestamp',
            'task_type': 'causal_inference',
            'target_column': 'outcome',
            'treatment_column': 'treatment'
        }
        
        assert data_interface_config == expected_config
    
    def test_fetch_data_interface_config_no_treatment_column(self):
        """Test data_interface_config when treatment_column is not present."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.recipe = Mock()
        mock_settings.recipe.model = Mock()
        
        mock_settings.recipe.model.loader = Mock()
        mock_settings.recipe.model.loader.entity_schema = Mock()
        mock_settings.recipe.model.loader.entity_schema.entity_columns = ['id']
        mock_settings.recipe.model.loader.entity_schema.timestamp_column = 'ts'
        
        # Data interface without treatment column
        mock_settings.recipe.model.data_interface = Mock()
        mock_settings.recipe.model.data_interface.task_type = 'regression'
        mock_settings.recipe.model.data_interface.target_column = 'target'
        # Remove treatment_column attribute if it exists to simulate missing attribute
        if hasattr(mock_settings.recipe.model.data_interface, 'treatment_column'):
            delattr(mock_settings.recipe.model.data_interface, 'treatment_column')
        
        mock_settings.recipe.model.fetcher = None
        
        mock_factory = Mock()
        mock_adapter = Mock()
        mock_factory.create_feature_store_adapter.return_value = mock_adapter
        
        input_df = pd.DataFrame({'id': [1]})
        fetcher = FeatureStoreFetcher(mock_settings, mock_factory)
        
        # Act
        fetcher.fetch(input_df, run_mode="batch")
        
        # Assert
        call_args = mock_adapter.get_historical_features_with_validation.call_args
        data_interface_config = call_args[1]['data_interface_config']
        
        # treatment_column should be None when not present
        assert data_interface_config['treatment_column'] is None


class TestFeatureStoreFetcherLogging:
    """Test FeatureStoreFetcher logging behavior."""
    
    def test_fetch_logs_start_and_completion_messages(self):
        """Test that fetch operation logs appropriate start and completion messages."""
        # Arrange
        mock_settings = Mock(spec=Settings)
        mock_settings.recipe = Mock()
        mock_settings.recipe.model = Mock()
        mock_settings.recipe.model.loader = Mock()
        mock_settings.recipe.model.loader.entity_schema = Mock()
        mock_settings.recipe.model.loader.entity_schema.entity_columns = ['id']
        mock_settings.recipe.model.loader.entity_schema.timestamp_column = 'ts'
        mock_settings.recipe.model.fetcher = None
        mock_settings.recipe.model.data_interface = Mock()
        mock_settings.recipe.model.data_interface.task_type = 'classification'
        mock_settings.recipe.model.data_interface.target_column = 'target'
        
        mock_factory = Mock()
        mock_adapter = Mock()
        mock_adapter.get_historical_features_with_validation.return_value = pd.DataFrame()
        mock_factory.create_feature_store_adapter.return_value = mock_adapter
        
        input_df = pd.DataFrame({'id': [1]})
        fetcher = FeatureStoreFetcher(mock_settings, mock_factory)
        
        # Act
        with patch('src.components.fetcher.modules.feature_store_fetcher.logger') as mock_logger:
            fetcher.fetch(input_df, run_mode="train")
        
        # Assert
        mock_logger.info.assert_any_call("Feature Store를 통해 피처 증강을 시작합니다.")
        mock_logger.info.assert_any_call("피처 증강 완료(offline).")
    
    def test_fetch_logs_different_messages_for_modes(self):
        """Test that different run modes log different completion messages."""
        # Arrange
        mock_settings = self._create_mock_settings_for_logging()
        mock_factory = Mock()
        mock_adapter = Mock()
        mock_adapter.get_historical_features_with_validation.return_value = pd.DataFrame()
        mock_adapter.get_online_features.return_value = pd.DataFrame()
        mock_factory.create_feature_store_adapter.return_value = mock_adapter
        
        input_df = pd.DataFrame({'user_id': [1]})
        fetcher = FeatureStoreFetcher(mock_settings, mock_factory)
        
        # Test train mode
        with patch('src.components.fetcher.modules.feature_store_fetcher.logger') as mock_logger:
            fetcher.fetch(input_df, run_mode="train")
            mock_logger.info.assert_any_call("피처 증강 완료(offline).")
        
        # Test serving mode
        with patch('src.components.fetcher.modules.feature_store_fetcher.logger') as mock_logger:
            fetcher.fetch(input_df, run_mode="serving")
            mock_logger.info.assert_any_call("피처 증강 완료(online).")
    
    def _create_mock_settings_for_logging(self):
        """Create mock settings for logging tests."""
        mock_settings = Mock(spec=Settings)
        mock_settings.recipe = Mock()
        mock_settings.recipe.model = Mock()
        
        mock_settings.recipe.model.loader = Mock()
        mock_settings.recipe.model.loader.entity_schema = Mock()
        mock_settings.recipe.model.loader.entity_schema.entity_columns = ['user_id']
        mock_settings.recipe.model.loader.entity_schema.timestamp_column = 'timestamp'
        
        mock_settings.recipe.model.fetcher = None
        
        mock_settings.recipe.model.data_interface = Mock()
        mock_settings.recipe.model.data_interface.task_type = 'classification'
        mock_settings.recipe.model.data_interface.target_column = 'target'
        
        return mock_settings


class TestFeatureStoreFetcherSelfRegistration:
    """Test FeatureStoreFetcher self-registration mechanism."""
    
    def test_feature_store_fetcher_self_registration(self):
        """Test that FeatureStoreFetcher registers itself in FetcherRegistry."""
        # Act - Import triggers self-registration
        from src.components.fetcher.modules import feature_store_fetcher
        from src.components.fetcher.registry import FetcherRegistry
        
        # Assert
        assert "feature_store" in FetcherRegistry.fetchers
        assert FetcherRegistry.fetchers["feature_store"] == FeatureStoreFetcher
        
        # Verify can create instance through registry with proper arguments
        mock_settings = Mock(spec=Settings)
        mock_factory = Mock()
        mock_factory.create_feature_store_adapter.return_value = Mock()
        
        instance = FetcherRegistry.create("feature_store", mock_settings, mock_factory)
        assert isinstance(instance, FeatureStoreFetcher)


class TestFeatureStoreFetcherIntegration:
    """Test FeatureStoreFetcher integration scenarios."""
    
    def test_feature_store_fetcher_end_to_end_workflow(self):
        """Test complete end-to-end workflow with FeatureStoreFetcher."""
        # Arrange - Complex realistic scenario
        mock_settings = Mock(spec=Settings)
        mock_settings.recipe = Mock()
        mock_settings.recipe.model = Mock()
        
        # Multi-entity scenario
        mock_settings.recipe.model.loader = Mock()
        mock_settings.recipe.model.loader.entity_schema = Mock()
        mock_settings.recipe.model.loader.entity_schema.entity_columns = ['user_id', 'product_id']
        mock_settings.recipe.model.loader.entity_schema.timestamp_column = 'event_timestamp'
        
        # Multiple feature namespaces
        mock_user_features = Mock()
        mock_user_features.feature_namespace = 'user_features'
        mock_user_features.features = ['age', 'income', 'city']
        
        mock_product_features = Mock()
        mock_product_features.feature_namespace = 'product_features'  
        mock_product_features.features = ['price', 'category', 'rating']
        
        mock_settings.recipe.model.fetcher = Mock()
        mock_settings.recipe.model.fetcher.features = [mock_user_features, mock_product_features]
        
        mock_settings.recipe.model.data_interface = Mock()
        mock_settings.recipe.model.data_interface.task_type = 'recommendation'
        mock_settings.recipe.model.data_interface.target_column = 'rating'
        
        mock_factory = Mock()
        mock_adapter = Mock()
        
        # Simulated enriched result
        enriched_result = pd.DataFrame({
            'user_id': [1, 2],
            'product_id': [100, 200],
            'event_timestamp': ['2023-01-01 10:00:00', '2023-01-01 11:00:00'],
            'user_features__age': [25, 30],
            'user_features__income': [50000, 75000],
            'product_features__price': [19.99, 29.99],
            'product_features__category': ['electronics', 'books']
        })
        
        mock_adapter.get_historical_features_with_validation.return_value = enriched_result
        mock_factory.create_feature_store_adapter.return_value = mock_adapter
        
        input_df = pd.DataFrame({
            'user_id': [1, 2],
            'product_id': [100, 200],
            'event_timestamp': ['2023-01-01 10:00:00', '2023-01-01 11:00:00']
        })
        
        fetcher = FeatureStoreFetcher(mock_settings, mock_factory)
        
        # Act
        result = fetcher.fetch(input_df, run_mode="train")
        
        # Assert
        pd.testing.assert_frame_equal(result, enriched_result)
        
        # Verify feature list construction
        call_args = mock_adapter.get_historical_features_with_validation.call_args
        features_called = call_args[1]['features']
        expected_features = [
            'user_features:age', 'user_features:income', 'user_features:city',
            'product_features:price', 'product_features:category', 'product_features:rating'
        ]
        assert sorted(features_called) == sorted(expected_features)
        
        # Verify data interface config
        data_interface_config = call_args[1]['data_interface_config']
        assert data_interface_config['entity_columns'] == ['user_id', 'product_id']
        assert data_interface_config['timestamp_column'] == 'event_timestamp'
        assert data_interface_config['task_type'] == 'recommendation'
        assert data_interface_config['target_column'] == 'rating'
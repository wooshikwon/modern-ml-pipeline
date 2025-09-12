"""
Unit tests for API lifespan and context setup with DataInterface v2.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pydantic import BaseModel

from src.serving._lifespan import setup_api_context, lifespan
from src.serving._context import app_context


class TestAPIContextSetup:
    """Test API context initialization with v2 schema"""
    
    @patch('src.serving._lifespan.mlflow')
    @patch('src.serving._lifespan.bootstrap')
    @patch('src.serving._lifespan.create_datainterface_based_prediction_request_v2')
    @patch('src.serving._lifespan.create_batch_prediction_request')
    def test_setup_with_datainterface_v2(
        self,
        mock_create_batch,
        mock_create_v2,
        mock_bootstrap,
        mock_mlflow
    ):
        """Test context setup uses v2 schema generation"""
        # Mock PyfuncWrapper with DataInterface schema
        mock_wrapper = Mock()
        mock_wrapper.data_interface_schema = {
            'target_column': 'price',
            'entity_columns': ['product_id', 'store_id'],
            'feature_columns': ['brand', 'category', 'stock'],
            'task_type': 'regression'
        }
        
        # Mock model
        mock_model = Mock()
        mock_model.unwrap_python_model.return_value = mock_wrapper
        mock_mlflow.pyfunc.load_model.return_value = mock_model
        
        # Mock v2 schema creation
        mock_request_model = Mock(spec=BaseModel)
        mock_request_model.__fields__ = {
            'product_id': Mock(),
            'store_id': Mock(),
            'brand': Mock(),
            'category': Mock(),
            'stock': Mock()
            # Note: 'price' excluded (target)
        }
        mock_create_v2.return_value = mock_request_model
        
        # Mock batch creation
        mock_batch_model = Mock(spec=BaseModel)
        mock_create_batch.return_value = mock_batch_model
        
        # Mock settings
        mock_settings = Mock()
        
        # Setup context
        setup_api_context(run_id='test_run_123', settings=mock_settings)
        
        # Verify bootstrap was called
        mock_bootstrap.assert_called_once_with(mock_settings)
        
        # Verify model was loaded
        mock_mlflow.pyfunc.load_model.assert_called_once_with('runs:/test_run_123/model')
        
        # Verify v2 schema creation was called with correct params
        mock_create_v2.assert_called_once_with(
            model_name="DataInterfacePredictionRequest",
            data_interface_schema=mock_wrapper.data_interface_schema,
            exclude_target=True  # Target should be excluded
        )
        
        # Verify batch request creation
        mock_create_batch.assert_called_once_with(mock_request_model)
        
        # Verify context was set
        assert app_context.model == mock_model
        assert app_context.model_uri == 'runs:/test_run_123/model'
        assert app_context.settings == mock_settings
        assert app_context.PredictionRequest == mock_request_model
        assert app_context.BatchPredictionRequest == mock_batch_model
        
    @patch('src.serving._lifespan.mlflow')
    @patch('src.serving._lifespan.bootstrap')
    @patch('src.serving._lifespan.create_dynamic_prediction_request')
    @patch('src.serving._lifespan.parse_select_columns')
    def test_fallback_to_legacy_without_datainterface(
        self,
        mock_parse_sql,
        mock_create_legacy,
        mock_bootstrap,
        mock_mlflow
    ):
        """Test fallback to legacy schema when no DataInterface"""
        # Mock wrapper without DataInterface
        mock_wrapper = Mock()
        mock_wrapper.data_interface_schema = None  # No DataInterface
        mock_wrapper.data_schema = None  # No data_schema either
        mock_wrapper.loader_sql_snapshot = 'SELECT user_id, item_id FROM table'
        
        # Mock model
        mock_model = Mock()
        mock_model.unwrap_python_model.return_value = mock_wrapper
        mock_mlflow.pyfunc.load_model.return_value = mock_model
        
        # Mock SQL parsing
        mock_parse_sql.return_value = ['user_id', 'item_id']
        
        # Mock legacy schema creation
        mock_legacy_model = Mock(spec=BaseModel)
        mock_create_legacy.return_value = mock_legacy_model
        
        mock_settings = Mock()
        
        # Setup context
        setup_api_context(run_id='legacy_run', settings=mock_settings)
        
        # Verify fallback to SQL parsing
        mock_parse_sql.assert_called_once_with('SELECT user_id, item_id FROM table')
        
        # Verify legacy schema creation
        mock_create_legacy.assert_called_once_with(
            model_name="DynamicPredictionRequest",
            pk_fields=['user_id', 'item_id']
        )
        
        # Context should still be set
        assert app_context.PredictionRequest == mock_legacy_model
        
    @patch('src.serving._lifespan.mlflow')
    @patch('src.serving._lifespan.bootstrap')
    @patch('src.serving._lifespan.create_dynamic_prediction_request')
    def test_fallback_with_data_schema_entity_columns(
        self,
        mock_create_legacy,
        mock_bootstrap,
        mock_mlflow
    ):
        """Test fallback using data_schema entity_columns"""
        # Mock wrapper with data_schema but no DataInterface
        mock_wrapper = Mock()
        mock_wrapper.data_interface_schema = None
        mock_wrapper.data_schema = {
            'entity_columns': ['customer_id', 'session_id']
        }
        
        mock_model = Mock()
        mock_model.unwrap_python_model.return_value = mock_wrapper
        mock_mlflow.pyfunc.load_model.return_value = mock_model
        
        mock_legacy_model = Mock(spec=BaseModel)
        mock_create_legacy.return_value = mock_legacy_model
        
        mock_settings = Mock()
        
        # Setup context
        setup_api_context(run_id='schema_run', settings=mock_settings)
        
        # Verify legacy creation with entity columns
        mock_create_legacy.assert_called_once_with(
            model_name="DynamicPredictionRequest",
            pk_fields=['customer_id', 'session_id']
        )
        
    @patch('src.serving._lifespan.mlflow')
    @patch('src.serving._lifespan.bootstrap')
    @patch('src.serving._lifespan.logger')
    def test_error_handling_in_setup(
        self,
        mock_logger,
        mock_bootstrap,
        mock_mlflow
    ):
        """Test error handling during context setup"""
        # Mock model load failure
        mock_mlflow.pyfunc.load_model.side_effect = Exception("Model not found")
        
        mock_settings = Mock()
        
        # Should raise and log error
        with pytest.raises(Exception) as exc_info:
            setup_api_context(run_id='bad_run', settings=mock_settings)
        
        assert "Model not found" in str(exc_info.value)
        mock_logger.error.assert_called()


class TestLifespanEvents:
    """Test FastAPI lifespan events"""
    
    @pytest.mark.asyncio
    async def test_lifespan_startup_shutdown(self):
        """Test lifespan async context manager"""
        from fastapi import FastAPI
        
        app = FastAPI()
        
        # Test lifespan context
        async with lifespan(app):
            # Startup completed
            pass
        # Shutdown completed
        
        # Should complete without errors
        assert True
        
    @pytest.mark.asyncio
    @patch('src.serving._lifespan.logger')
    async def test_lifespan_logging(self, mock_logger):
        """Test that lifespan logs startup/shutdown"""
        from fastapi import FastAPI
        
        app = FastAPI()
        
        async with lifespan(app):
            # Check startup log
            mock_logger.info.assert_called_with("🚀 Modern ML Pipeline API 서버 시작...")
            
        # Check shutdown log
        mock_logger.info.assert_called_with("✅ Modern ML Pipeline API 서버 종료.")


class TestSchemaGenerationPriority:
    """Test schema generation priority order"""
    
    @patch('src.serving._lifespan.mlflow')
    @patch('src.serving._lifespan.bootstrap')
    @patch('src.serving._lifespan.create_datainterface_based_prediction_request_v2')
    @patch('src.serving._lifespan.logger')
    def test_datainterface_priority_over_legacy(
        self,
        mock_logger,
        mock_create_v2,
        mock_bootstrap,
        mock_mlflow
    ):
        """Test DataInterface schema takes priority over legacy methods"""
        # Mock wrapper with BOTH DataInterface and legacy schemas
        mock_wrapper = Mock()
        mock_wrapper.data_interface_schema = {
            'target_column': 'target',
            'entity_columns': ['id'],
            'feature_columns': ['f1', 'f2']
        }
        mock_wrapper.data_schema = {
            'entity_columns': ['old_id']  # Different from DataInterface
        }
        mock_wrapper.loader_sql_snapshot = 'SELECT legacy_id FROM table'
        
        mock_model = Mock()
        mock_model.unwrap_python_model.return_value = mock_wrapper
        mock_mlflow.pyfunc.load_model.return_value = mock_model
        
        mock_v2_model = Mock(spec=BaseModel)
        mock_create_v2.return_value = mock_v2_model
        
        mock_settings = Mock()
        
        # Setup context
        setup_api_context(run_id='priority_test', settings=mock_settings)
        
        # Should use DataInterface (v2), not legacy
        mock_create_v2.assert_called_once()
        
        # Should log that DataInterface was used
        mock_logger.info.assert_called_with(
            "✅ DataInterface 스키마 기반 API 스키마 생성 (target_column 자동 제외)"
        )
        
    @patch('src.serving._lifespan.mlflow')
    @patch('src.serving._lifespan.bootstrap')
    @patch('src.serving._lifespan.create_dynamic_prediction_request')
    @patch('src.serving._lifespan.logger')
    def test_warning_when_no_datainterface(
        self,
        mock_logger,
        mock_create_legacy,
        mock_bootstrap,
        mock_mlflow
    ):
        """Test warning is logged when falling back to legacy"""
        mock_wrapper = Mock()
        mock_wrapper.data_interface_schema = None  # No DataInterface
        mock_wrapper.data_schema = None
        mock_wrapper.loader_sql_snapshot = 'SELECT id FROM table'
        
        mock_model = Mock()
        mock_model.unwrap_python_model.return_value = mock_wrapper
        mock_mlflow.pyfunc.load_model.return_value = mock_model
        
        mock_settings = Mock()
        
        with patch('src.serving._lifespan.parse_select_columns', return_value=['id']):
            setup_api_context(run_id='no_di_run', settings=mock_settings)
        
        # Should log warning about fallback
        mock_logger.warning.assert_called_with(
            "⚠️ DataInterface 스키마 없음 - 기존 방식으로 폴백"
        )
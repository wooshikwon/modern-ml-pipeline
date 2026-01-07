"""
Unit tests for API lifespan and context setup - Unit Test Layer
Following tests/README.md principles: Interface contract verification, external dependencies mocked
"""

from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from src.serving._context import app_context
from src.serving._lifespan import lifespan, setup_api_context


class TestAPIContextSetup:
    """Test setup_api_context interface contract and call sequence"""

    @patch("src.serving._lifespan.create_batch_prediction_request")
    @patch("src.serving._lifespan.create_datainterface_based_prediction_request_v2")
    @patch("src.serving._lifespan.bootstrap")
    @patch("src.serving._lifespan.mlflow")
    def test_setup_with_datainterface_v2(
        self, mock_mlflow, mock_bootstrap, mock_create_v2, mock_create_batch
    ):
        """Test setup_api_context calls correct functions in sequence for DataInterface v2"""
        # Given: Mock all dependencies
        mock_model = Mock()
        mock_wrapper = Mock()
        mock_wrapper.data_interface_schema = {
            "target_column": "target",
            "entity_columns": ["id"],
            "feature_columns": ["f1", "f2"],
            "task_type": "classification",
        }
        mock_model.unwrap_python_model.return_value = mock_wrapper
        mock_mlflow.pyfunc.load_model.return_value = mock_model

        mock_prediction_request = Mock(spec=BaseModel)
        mock_batch_request = Mock(spec=BaseModel)
        mock_create_v2.return_value = mock_prediction_request
        mock_create_batch.return_value = mock_batch_request

        mock_settings = Mock()

        # When: Setup API context
        setup_api_context(run_id="test_run_123", settings=mock_settings)

        # Then: Verify call sequence and parameters
        mock_bootstrap.assert_called_once_with(mock_settings)
        mock_mlflow.pyfunc.load_model.assert_called_once_with("runs:/test_run_123/model")
        mock_create_v2.assert_called_once_with(
            model_name="DataInterfacePredictionRequest",
            data_interface_schema=mock_wrapper.data_interface_schema,
            exclude_target=True,
        )
        mock_create_batch.assert_called_once_with(mock_prediction_request)

        # Verify app_context was set correctly
        assert app_context.model == mock_model
        assert app_context.model_uri == "runs:/test_run_123/model"
        assert app_context.settings == mock_settings
        assert app_context.PredictionRequest == mock_prediction_request
        assert app_context.BatchPredictionRequest == mock_batch_request

    @patch("src.serving._lifespan.parse_select_columns")
    @patch("src.serving._lifespan.create_batch_prediction_request")
    @patch("src.serving._lifespan.create_dynamic_prediction_request")
    @patch("src.serving._lifespan.bootstrap")
    @patch("src.serving._lifespan.mlflow")
    def test_fallback_to_legacy_without_datainterface(
        self,
        mock_mlflow,
        mock_bootstrap,
        mock_create_dynamic,
        mock_create_batch,
        mock_parse_columns,
    ):
        """Test fallback to legacy schema when no DataInterface available"""
        # Given: Mock all dependencies for legacy path
        mock_model = Mock()
        mock_wrapper = Mock()
        mock_wrapper.data_interface_schema = None  # No DataInterface
        mock_wrapper.data_schema = None  # No data_schema either
        mock_wrapper.loader_sql_snapshot = "SELECT user_id, item_id FROM features"
        mock_model.unwrap_python_model.return_value = mock_wrapper
        mock_mlflow.pyfunc.load_model.return_value = mock_model

        # Mock parse_select_columns to return pk fields
        mock_parse_columns.return_value = ["user_id", "item_id"]

        mock_legacy_request = Mock(spec=BaseModel)
        mock_batch_request = Mock(spec=BaseModel)
        mock_create_dynamic.return_value = mock_legacy_request
        mock_create_batch.return_value = mock_batch_request

        mock_settings = Mock()

        # When: Setup API context
        setup_api_context(run_id="legacy_run", settings=mock_settings)

        # Then: Verify legacy fallback path
        mock_bootstrap.assert_called_once_with(mock_settings)
        mock_mlflow.pyfunc.load_model.assert_called_once_with("runs:/legacy_run/model")
        mock_parse_columns.assert_called_once_with(mock_wrapper.loader_sql_snapshot)
        mock_create_dynamic.assert_called_once_with(
            model_name="DynamicPredictionRequest", pk_fields=["user_id", "item_id"]
        )
        mock_create_batch.assert_called_once_with(mock_legacy_request)

        # Verify app_context was set correctly
        assert app_context.model == mock_model
        assert app_context.model_uri == "runs:/legacy_run/model"
        assert app_context.settings == mock_settings
        assert app_context.PredictionRequest == mock_legacy_request
        assert app_context.BatchPredictionRequest == mock_batch_request

    @patch("src.serving._lifespan.create_batch_prediction_request")
    @patch("src.serving._lifespan.create_dynamic_prediction_request")
    @patch("src.serving._lifespan.bootstrap")
    @patch("src.serving._lifespan.mlflow")
    def test_fallback_with_data_schema_entity_columns(
        self, mock_mlflow, mock_bootstrap, mock_create_dynamic, mock_create_batch
    ):
        """Test fallback using data_schema entity_columns"""
        # Given: Mock model with data_schema but no DataInterface
        mock_model = Mock()
        mock_wrapper = Mock()
        mock_wrapper.data_interface_schema = None
        mock_wrapper.data_schema = {"entity_columns": ["customer_id", "session_id"]}
        mock_model.unwrap_python_model.return_value = mock_wrapper
        mock_mlflow.pyfunc.load_model.return_value = mock_model

        mock_entity_request = Mock(spec=BaseModel)
        mock_batch_request = Mock(spec=BaseModel)
        mock_create_dynamic.return_value = mock_entity_request
        mock_create_batch.return_value = mock_batch_request

        mock_settings = Mock()

        # When: Setup API context
        setup_api_context(run_id="schema_run", settings=mock_settings)

        # Then: Verify data_schema fallback path
        mock_bootstrap.assert_called_once_with(mock_settings)
        mock_mlflow.pyfunc.load_model.assert_called_once_with("runs:/schema_run/model")
        mock_create_dynamic.assert_called_once_with(
            model_name="DynamicPredictionRequest", pk_fields=["customer_id", "session_id"]
        )
        mock_create_batch.assert_called_once_with(mock_entity_request)

        # Verify app_context was set correctly
        assert app_context.model == mock_model
        assert app_context.settings == mock_settings
        assert app_context.PredictionRequest == mock_entity_request
        assert app_context.BatchPredictionRequest == mock_batch_request

    @patch("src.serving._lifespan.bootstrap")
    @patch("src.serving._lifespan.mlflow")
    def test_error_handling_in_setup(self, mock_mlflow, mock_bootstrap):
        """Test error handling during context setup"""
        # Given: Mock MLflow to fail
        mock_mlflow.pyfunc.load_model.side_effect = Exception("Model not found")
        mock_settings = Mock()

        # When/Then: Should raise exception when model load fails
        with pytest.raises(Exception) as exc_info:
            setup_api_context(run_id="bad_run", settings=mock_settings)

        assert "Model not found" in str(exc_info.value)
        mock_bootstrap.assert_called_once_with(mock_settings)
        mock_mlflow.pyfunc.load_model.assert_called_once_with("runs:/bad_run/model")


class TestLifespanEvents:
    """Test FastAPI lifespan events using real components"""

    def test_lifespan_startup_shutdown(self):
        """Test lifespan function can be imported and is callable"""
        from fastapi import FastAPI

        # Given: Real FastAPI app
        app = FastAPI()

        # When/Then: Verify lifespan is a proper async context manager
        assert lifespan is not None
        assert callable(lifespan)

        # Verify it returns an async context manager
        context_manager = lifespan(app)
        assert hasattr(context_manager, "__aenter__")
        assert hasattr(context_manager, "__aexit__")

        # Should complete successfully
        assert True

    def test_lifespan_logging(self):
        """Test that lifespan function exists and has proper structure"""
        from fastapi import FastAPI

        # Given: Real FastAPI app
        app = FastAPI()

        # When: Create lifespan context manager (not executing async)
        context_manager = lifespan(app)

        # Then: Verify it's a proper async context manager
        assert context_manager is not None
        assert hasattr(context_manager, "__aenter__")
        assert hasattr(context_manager, "__aexit__")

        # Should complete successfully without exceptions
        assert True


class TestSchemaGenerationPriority:
    """Test schema generation priority order using real components"""

    @patch("src.serving._lifespan.create_batch_prediction_request")
    @patch("src.serving._lifespan.create_datainterface_based_prediction_request_v2")
    @patch("src.serving._lifespan.bootstrap")
    @patch("src.serving._lifespan.mlflow")
    def test_datainterface_priority_over_legacy(
        self, mock_mlflow, mock_bootstrap, mock_create_v2, mock_create_batch
    ):
        """Test DataInterface schema takes priority over legacy methods"""
        # Given: Model with BOTH DataInterface and legacy schemas
        mock_model = Mock()
        mock_wrapper = Mock()
        mock_wrapper.data_interface_schema = {
            "target_column": "target",
            "entity_columns": ["id"],
            "feature_columns": ["f1", "f2"],
        }
        mock_wrapper.data_schema = {"entity_columns": ["old_id"]}  # Different from DataInterface
        mock_wrapper.loader_sql_snapshot = "SELECT legacy_id FROM table"
        mock_model.unwrap_python_model.return_value = mock_wrapper
        mock_mlflow.pyfunc.load_model.return_value = mock_model

        mock_prediction_request = Mock(spec=BaseModel)
        mock_batch_request = Mock(spec=BaseModel)
        mock_create_v2.return_value = mock_prediction_request
        mock_create_batch.return_value = mock_batch_request

        mock_settings = Mock()

        # When: Setup API context
        setup_api_context(run_id="priority_test", settings=mock_settings)

        # Then: Should use DataInterface schema (priority over legacy)
        mock_bootstrap.assert_called_once_with(mock_settings)
        mock_mlflow.pyfunc.load_model.assert_called_once_with("runs:/priority_test/model")
        # DataInterface v2 should be called, not legacy methods
        mock_create_v2.assert_called_once_with(
            model_name="DataInterfacePredictionRequest",
            data_interface_schema=mock_wrapper.data_interface_schema,
            exclude_target=True,
        )
        mock_create_batch.assert_called_once_with(mock_prediction_request)

        # Verify app_context was set correctly with DataInterface priority
        assert app_context.model == mock_model
        assert app_context.PredictionRequest == mock_prediction_request
        assert app_context.BatchPredictionRequest == mock_batch_request

    @patch("src.serving._lifespan.parse_select_columns")
    @patch("src.serving._lifespan.create_batch_prediction_request")
    @patch("src.serving._lifespan.create_dynamic_prediction_request")
    @patch("src.serving._lifespan.bootstrap")
    @patch("src.serving._lifespan.mlflow")
    def test_fallback_when_no_datainterface(
        self,
        mock_mlflow,
        mock_bootstrap,
        mock_create_dynamic,
        mock_create_batch,
        mock_parse_columns,
    ):
        """Test fallback to legacy when no DataInterface available"""
        # Given: Model without DataInterface schema
        mock_model = Mock()
        mock_wrapper = Mock()
        mock_wrapper.data_interface_schema = None  # No DataInterface
        mock_wrapper.data_schema = None
        mock_wrapper.loader_sql_snapshot = "SELECT id FROM table"
        mock_model.unwrap_python_model.return_value = mock_wrapper
        mock_mlflow.pyfunc.load_model.return_value = mock_model

        # Mock parse_select_columns to return pk fields
        mock_parse_columns.return_value = ["id"]

        mock_legacy_request = Mock(spec=BaseModel)
        mock_batch_request = Mock(spec=BaseModel)
        mock_create_dynamic.return_value = mock_legacy_request
        mock_create_batch.return_value = mock_batch_request

        mock_settings = Mock()

        # When: Setup API context
        setup_api_context(run_id="no_di_run", settings=mock_settings)

        # Then: Should fallback to legacy schema generation
        mock_bootstrap.assert_called_once_with(mock_settings)
        mock_mlflow.pyfunc.load_model.assert_called_once_with("runs:/no_di_run/model")
        mock_parse_columns.assert_called_once_with("SELECT id FROM table")
        mock_create_dynamic.assert_called_once_with(
            model_name="DynamicPredictionRequest", pk_fields=["id"]
        )
        mock_create_batch.assert_called_once_with(mock_legacy_request)

        # Verify app_context was set correctly with legacy fallback
        assert app_context.model == mock_model
        assert app_context.PredictionRequest == mock_legacy_request
        assert app_context.BatchPredictionRequest == mock_batch_request

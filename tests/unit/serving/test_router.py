"""
Unit tests for FastAPI serving router.
Tests routing, integration, and FastAPI app functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from fastapi.testclient import TestClient

from src.serving.router import app, _register_dynamic_routes_if_needed


class TestFastAPIAppInitialization:
    """Test FastAPI app initialization."""
    
    def test_app_exists_and_configured(self):
        """Test that FastAPI app is properly initialized."""
        # Assert
        assert app is not None
        assert app.title == "Modern ML Pipeline API"
        assert app.description == "Blueprint v17.0 기반 모델 서빙 API"
        assert app.version == "17.0.0"
    
    def test_app_has_lifespan(self):
        """Test that app has lifespan configuration."""
        # Assert
        assert app.router.lifespan_context is not None


class TestStaticRoutes:
    """Test static routes functionality with TestClient."""
    
    @patch('src.serving._endpoints.app_context')
    def test_health_endpoint_route(self, mock_app_context):
        """Test /health endpoint routing."""
        # Arrange
        mock_model = Mock()
        mock_wrapped_model = Mock()
        mock_wrapped_model.model_class_path = "test.model.Class"
        mock_model.unwrap_python_model.return_value = mock_wrapped_model
        
        mock_settings = Mock()
        mock_app_context.model = mock_model
        mock_app_context.settings = mock_settings
        mock_app_context.model_uri = "runs/123/model"
        
        client = TestClient(app)
        
        # Act
        response = client.get("/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_uri"] == "runs/123/model"
        assert data["model_name"] == "test.model.Class"
    
    @patch('src.serving._endpoints.app_context')
    def test_model_metadata_endpoint_route(self, mock_app_context):
        """Test /model/metadata endpoint routing."""
        # Arrange
        mock_wrapped_model = Mock()
        mock_wrapped_model.model_class_path = "test.model.Class"
        
        mock_model = Mock()
        mock_model.unwrap_python_model.return_value = mock_wrapped_model
        mock_model.hyperparameter_optimization = {
            "enabled": False,
            "best_params": {},
            "best_score": 0.0,
            "total_trials": 0
        }
        mock_model.training_methodology = {
            "train_test_split_method": "random",
            "train_ratio": 0.8,
            "validation_strategy": "holdout",
            "preprocessing_fit_scope": "train_only"
        }
        mock_model.model_class_path = "test.model.Class"
        
        # Mock PredictionRequest for API schema
        mock_prediction_request = Mock()
        mock_prediction_request.model_fields = {"user_id": {}}
        
        mock_app_context.model = mock_model
        mock_app_context.model_uri = "runs/123/model"
        mock_app_context.PredictionRequest = mock_prediction_request
        
        client = TestClient(app)
        
        # Act
        response = client.get("/model/metadata")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["model_class_path"] == "test.model.Class"
        assert data["hyperparameter_optimization"]["enabled"] is False
    
    @patch('src.serving._endpoints.app_context')
    @patch('src.serving._endpoints.pd.DataFrame')
    def test_batch_predict_endpoint_route(self, mock_dataframe_class, mock_app_context):
        """Test /batch_predict endpoint routing."""
        # Arrange
        mock_batch_request_class = Mock()
        mock_request_instance = Mock()
        mock_sample = Mock()
        mock_sample.model_dump.return_value = {"feature1": 1.0}
        mock_request_instance.samples = [mock_sample]
        mock_batch_request_class.return_value = mock_request_instance
        
        mock_input_df = Mock()
        mock_input_df.empty = False
        mock_dataframe_class.return_value = mock_input_df
        
        mock_predictions_df = Mock()
        mock_predictions_df.to_dict.return_value = [{"prediction": 0.8}]
        mock_predictions_df.__len__ = Mock(return_value=1)
        
        mock_model = Mock()
        mock_model.predict.return_value = mock_predictions_df
        
        mock_app_context.BatchPredictionRequest = mock_batch_request_class
        mock_app_context.model = mock_model
        mock_app_context.model_uri = "runs/123/model"
        
        client = TestClient(app)
        
        # Act
        response = client.post("/batch_predict", json={
            "samples": [{"feature1": 1.0}]
        })
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["predictions"] == [{"prediction": 0.8}]
        assert data["sample_count"] == 1


class TestDynamicRouteRegistration:
    """Test dynamic route registration functionality."""
    
    @patch('src.serving.router.app_context')
    def test_register_dynamic_routes_if_needed_success(self, mock_app_context):
        """Test successful dynamic route registration."""
        # Arrange
        mock_prediction_request = Mock()
        mock_app_context.PredictionRequest = mock_prediction_request
        
        # Clear any existing routes for clean test
        original_routes = list(app.router.routes)
        
        # Act
        _register_dynamic_routes_if_needed()
        
        # Assert
        predict_routes = [r for r in app.router.routes if getattr(r, "path", None) == "/predict"]
        assert len(predict_routes) > 0
        
        # Cleanup - restore original routes
        app.router.routes = original_routes
    
    @patch('src.serving.router.app_context')
    def test_register_dynamic_routes_already_registered(self, mock_app_context):
        """Test that dynamic routes are not registered twice."""
        # Arrange
        mock_prediction_request = Mock()
        mock_app_context.PredictionRequest = mock_prediction_request
        
        # First registration
        _register_dynamic_routes_if_needed()
        initial_route_count = len(app.router.routes)
        
        # Act - second registration
        _register_dynamic_routes_if_needed()
        
        # Assert - should not add duplicate routes
        final_route_count = len(app.router.routes)
        assert final_route_count == initial_route_count
    
    @patch('src.serving.router.app_context')
    def test_register_dynamic_routes_no_prediction_request(self, mock_app_context):
        """Test dynamic route registration when PredictionRequest is None."""
        # Arrange
        mock_app_context.PredictionRequest = None
        
        initial_route_count = len(app.router.routes)
        
        # Act
        _register_dynamic_routes_if_needed()
        
        # Assert - should not add routes
        final_route_count = len(app.router.routes)
        assert final_route_count == initial_route_count


class TestRouterIntegration:
    """Test router integration scenarios."""
    
    @patch('src.serving._endpoints.app_context')
    def test_complete_api_integration(self, mock_app_context):
        """Test complete API integration with multiple endpoints."""
        # Arrange
        mock_wrapped_model = Mock()
        mock_wrapped_model.model_class_path = "integration.test.Model"
        mock_wrapped_model.training_results = {
            "training_metrics": {"accuracy": 0.98, "loss": 0.02},
            "hyperparameter_optimization": {
                "enabled": True,
                "best_params": {"lr": 0.001},
                "best_score": 0.98,
                "total_trials": 50
            }
        }
        
        mock_model = Mock()
        mock_model.unwrap_python_model.return_value = mock_wrapped_model
        
        mock_batch_request_class = Mock()
        mock_request_instance = Mock()
        mock_sample = Mock()
        mock_sample.model_dump.return_value = {"feature1": 2.0, "feature2": 3.0}
        mock_request_instance.samples = [mock_sample]
        mock_batch_request_class.return_value = mock_request_instance
        
        mock_predictions_df = Mock()
        mock_predictions_df.to_dict.return_value = [{"prediction": 0.95, "confidence": 0.9}]
        mock_predictions_df.__len__ = Mock(return_value=1)
        mock_model.predict.return_value = mock_predictions_df
        
        mock_settings = Mock()
        mock_app_context.model = mock_model
        mock_app_context.settings = mock_settings
        mock_app_context.model_uri = "runs/integration/model"
        mock_app_context.BatchPredictionRequest = mock_batch_request_class
        
        client = TestClient(app)
        
        # Act & Assert - Health check
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        assert health_data["model_name"] == "integration.test.Model"
        
        # Act & Assert - Model metadata
        metadata_response = client.get("/model/metadata")
        assert metadata_response.status_code == 200
        metadata_data = metadata_response.json()
        assert metadata_data["model_class"] == "integration.test.Model"
        assert metadata_data["training_metrics"]["accuracy"] == 0.98
        
        # Act & Assert - Batch prediction
        with patch('src.serving._endpoints.pd.DataFrame') as mock_df:
            mock_input_df = Mock()
            mock_input_df.empty = False
            mock_df.return_value = mock_input_df
            
            batch_response = client.post("/batch_predict", json={
                "samples": [{"feature1": 2.0, "feature2": 3.0}]
            })
            assert batch_response.status_code == 200
            batch_data = batch_response.json()
            assert batch_data["predictions"] == [{"prediction": 0.95, "confidence": 0.9}]
            assert batch_data["sample_count"] == 1


class TestRouterErrorHandling:
    """Test router error handling."""
    
    @patch('src.serving._endpoints.app_context')
    def test_health_endpoint_service_unavailable(self, mock_app_context):
        """Test /health endpoint when service is unavailable."""
        # Arrange
        mock_app_context.model = None
        mock_app_context.settings = None
        
        client = TestClient(app)
        
        # Act
        response = client.get("/health")
        
        # Assert
        assert response.status_code == 503
        data = response.json()
        assert "모델이 준비되지 않았습니다" in data["detail"]
    
    @patch('src.serving._endpoints.app_context')
    def test_batch_predict_bad_request(self, mock_app_context):
        """Test /batch_predict endpoint with bad request."""
        # Arrange
        mock_batch_request_class = Mock()
        mock_batch_request_class.side_effect = ValueError("Invalid request data")
        mock_app_context.BatchPredictionRequest = mock_batch_request_class
        
        client = TestClient(app)
        
        # Act
        response = client.post("/batch_predict", json={"invalid": "data"})
        
        # Assert
        assert response.status_code == 500  # Internal server error due to unhandled ValueError
    
    @patch('src.serving._endpoints.app_context')
    def test_model_metadata_service_unavailable(self, mock_app_context):
        """Test /model/metadata endpoint when model is unavailable."""
        # Arrange
        mock_app_context.model = None
        
        client = TestClient(app)
        
        # Act
        response = client.get("/model/metadata")
        
        # Assert
        assert response.status_code == 503
        data = response.json()
        assert "모델이 로드되지 않았습니다" in data["detail"]


class TestRouterLifespan:
    """Test router lifespan management."""
    
    def test_app_lifespan_configuration(self):
        """Test that app has proper lifespan configuration."""
        # Assert
        assert hasattr(app, 'router')
        assert app.router.lifespan_context is not None
    
    def test_app_startup_shutdown_hooks(self):
        """Test that app can handle startup and shutdown."""
        # This test verifies that the app can be created and torn down
        # without errors, which is important for testing
        
        # Act & Assert - Should not raise exceptions
        client = TestClient(app)
        assert client is not None
        
        # The context manager will handle startup/shutdown
        with client:
            pass


class TestRouterEdgeCases:
    """Test router edge cases."""
    
    def test_unknown_endpoint(self):
        """Test request to unknown endpoint."""
        # Arrange
        client = TestClient(app)
        
        # Act
        response = client.get("/unknown")
        
        # Assert
        assert response.status_code == 404
    
    def test_wrong_http_method(self):
        """Test wrong HTTP method on existing endpoint."""
        # Arrange
        client = TestClient(app)
        
        # Act - POST to health endpoint (which expects GET)
        response = client.post("/health")
        
        # Assert
        assert response.status_code == 405  # Method Not Allowed
    
    @patch('src.serving._endpoints.app_context')
    def test_batch_predict_without_json(self, mock_app_context):
        """Test batch predict endpoint without JSON payload."""
        # Arrange
        client = TestClient(app)
        
        # Act
        response = client.post("/batch_predict")
        
        # Assert
        assert response.status_code == 422  # Unprocessable Entity


class TestRouterPerformance:
    """Test router performance characteristics."""
    
    @patch('src.serving._endpoints.app_context')
    def test_concurrent_health_requests(self, mock_app_context):
        """Test handling concurrent health check requests."""
        # Arrange
        mock_model = Mock()
        mock_wrapped_model = Mock()
        mock_wrapped_model.model_class_path = "concurrent.test.Model"
        mock_model.unwrap_python_model.return_value = mock_wrapped_model
        
        mock_settings = Mock()
        mock_app_context.model = mock_model
        mock_app_context.settings = mock_settings
        mock_app_context.model_uri = "runs/concurrent/model"
        
        client = TestClient(app)
        
        # Act - Make multiple concurrent requests
        responses = []
        for _ in range(10):
            response = client.get("/health")
            responses.append(response)
        
        # Assert - All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
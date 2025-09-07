"""
Unit tests for FastAPI serving endpoints.
Tests core API logic for health, prediction, and metadata endpoints.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from fastapi import HTTPException

from src.serving import _endpoints as endpoints
from src.serving.schemas import (
    HealthCheckResponse,
    BatchPredictionResponse,
    ModelMetadataResponse,
)


class TestHealthEndpoint:
    """Test health endpoint functionality."""
    
    @patch('src.serving._endpoints.app_context')
    def test_health_endpoint_success(self, mock_app_context):
        """Test successful health check."""
        # Arrange
        mock_model = Mock()
        mock_wrapped_model = Mock()
        mock_wrapped_model.model_class_path = "test.model.Class"
        mock_model.unwrap_python_model.return_value = mock_wrapped_model
        
        mock_settings = Mock()
        mock_app_context.model = mock_model
        mock_app_context.settings = mock_settings
        mock_app_context.model_uri = "runs/123/model"
        
        # Act
        result = endpoints.health()
        
        # Assert
        assert isinstance(result, HealthCheckResponse)
        assert result.status == "healthy"
        assert result.model_uri == "runs/123/model"
        assert result.model_name == "test.model.Class"
    
    @patch('src.serving._endpoints.app_context')
    def test_health_endpoint_model_not_ready(self, mock_app_context):
        """Test health check when model is not ready."""
        # Arrange
        mock_app_context.model = None
        mock_app_context.settings = Mock()
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            endpoints.health()
        
        assert exc_info.value.status_code == 503
        assert "모델이 준비되지 않았습니다" in str(exc_info.value.detail)
    
    @patch('src.serving._endpoints.app_context')
    def test_health_endpoint_settings_not_ready(self, mock_app_context):
        """Test health check when settings are not ready."""
        # Arrange
        mock_app_context.model = Mock()
        mock_app_context.settings = None
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            endpoints.health()
        
        assert exc_info.value.status_code == 503
        assert "모델이 준비되지 않았습니다" in str(exc_info.value.detail)
    
    @patch('src.serving._endpoints.app_context')
    def test_health_endpoint_model_info_exception(self, mock_app_context):
        """Test health check when model info extraction fails."""
        # Arrange
        mock_model = Mock()
        mock_model.unwrap_python_model.side_effect = Exception("Failed to unwrap")
        
        mock_settings = Mock()
        mock_app_context.model = mock_model
        mock_app_context.settings = mock_settings
        mock_app_context.model_uri = "runs/123/model"
        
        # Act
        result = endpoints.health()
        
        # Assert - should handle exception gracefully
        assert isinstance(result, HealthCheckResponse)
        assert result.status == "healthy"
        assert result.model_name == "unknown"


class TestPredictBatchEndpoint:
    """Test batch prediction endpoint functionality."""
    
    @patch('src.serving._endpoints.app_context')
    @patch('src.serving._endpoints.pd.DataFrame')
    def test_predict_batch_success(self, mock_dataframe_class, mock_app_context):
        """Test successful batch prediction."""
        # Arrange
        mock_batch_request_class = Mock()
        mock_request_instance = Mock()
        mock_sample = Mock()
        mock_sample.model_dump.return_value = {"feature1": 1.0, "feature2": 2.0}
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
        
        request_data = {"samples": [{"feature1": 1.0, "feature2": 2.0}]}
        
        # Act
        result = endpoints.predict_batch(request_data)
        
        # Assert
        assert isinstance(result, BatchPredictionResponse)
        assert result.predictions == [{"prediction": 0.8}]
        assert result.model_uri == "runs/123/model"
        assert result.sample_count == 1
        
        # Verify model.predict was called with correct parameters
        mock_model.predict.assert_called_once_with(
            mock_input_df, 
            params={"run_mode": "serving", "return_intermediate": False}
        )
    
    @patch('src.serving._endpoints.app_context')
    def test_predict_batch_empty_samples(self, mock_app_context):
        """Test batch prediction with empty samples."""
        # Arrange
        mock_batch_request_class = Mock()
        mock_request_instance = Mock()
        mock_request_instance.samples = []
        mock_batch_request_class.return_value = mock_request_instance
        
        mock_app_context.BatchPredictionRequest = mock_batch_request_class
        
        request_data = {"samples": []}
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            endpoints.predict_batch(request_data)
        
        assert exc_info.value.status_code == 400
        assert "입력 샘플이 비어있습니다" in str(exc_info.value.detail)
    
    @patch('src.serving._endpoints.app_context')
    @patch('src.serving._endpoints.pd.DataFrame')
    def test_predict_batch_model_prediction_error(self, mock_dataframe_class, mock_app_context):
        """Test batch prediction when model prediction fails."""
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
        
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model prediction failed")
        
        mock_app_context.BatchPredictionRequest = mock_batch_request_class
        mock_app_context.model = mock_model
        
        request_data = {"samples": [{"feature1": 1.0}]}
        
        # Act & Assert
        with pytest.raises(Exception, match="Model prediction failed"):
            endpoints.predict_batch(request_data)


class TestGetModelMetadataEndpoint:
    """Test model metadata endpoint functionality."""
    
    @patch('src.serving._endpoints.app_context')
    def test_get_model_metadata_success(self, mock_app_context):
        """Test successful model metadata retrieval."""
        # Arrange
        mock_wrapped_model = Mock()
        mock_wrapped_model.model_class_path = "src.models.TestModel"
        
        # Mock model with the attributes that the endpoint actually looks for
        mock_model = Mock()
        mock_model.unwrap_python_model.return_value = mock_wrapped_model
        mock_model.hyperparameter_optimization = {
            "enabled": True,
            "best_params": {"lr": 0.001, "batch_size": 32},
            "best_score": 0.95,
            "total_trials": 100,
            "engine": "optuna",
            "pruned_trials": 10,
            "optimization_time": "00:30:00"
        }
        mock_model.training_methodology = {
            "train_test_split_method": "temporal",
            "train_ratio": 0.8,
            "validation_strategy": "cross_validation",
            "preprocessing_fit_scope": "train_only",
            "random_state": 42
        }
        mock_model.model_class_path = "src.models.TestModel"
        
        # Mock PredictionRequest for API schema
        mock_prediction_request = Mock()
        mock_prediction_request.model_fields = {"user_id": {}, "campaign_id": {}}
        
        mock_app_context.model = mock_model
        mock_app_context.model_uri = "runs/123/model"
        mock_app_context.PredictionRequest = mock_prediction_request
        
        # Act
        result = endpoints.get_model_metadata()
        
        # Assert
        assert isinstance(result, ModelMetadataResponse)
        assert result.model_uri == "runs/123/model"
        assert result.model_class_path == "src.models.TestModel"
        assert result.hyperparameter_optimization.enabled is True
        assert result.hyperparameter_optimization.best_params == {"lr": 0.001, "batch_size": 32}
        assert result.training_methodology.train_test_split_method == "temporal"
    
    @patch('src.serving._endpoints.app_context')
    def test_get_model_metadata_no_model(self, mock_app_context):
        """Test model metadata when no model is loaded."""
        # Arrange
        mock_app_context.model = None
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            endpoints.get_model_metadata()
        
        assert exc_info.value.status_code == 503
        assert "모델이 로드되지 않았습니다" in str(exc_info.value.detail)
    
    @patch('src.serving._endpoints.app_context')
    def test_get_model_metadata_unwrap_error(self, mock_app_context):
        """Test model metadata when model unwrapping fails."""
        # Arrange
        mock_model = Mock()
        mock_model.unwrap_python_model.side_effect = Exception("Failed to unwrap model")
        mock_model.hyperparameter_optimization = {}
        mock_model.training_methodology = {}
        mock_model.model_class_path = "test.model"
        
        # Mock PredictionRequest for API schema
        mock_prediction_request = Mock()
        mock_prediction_request.model_fields = {"user_id": {}}
        
        mock_app_context.model = mock_model
        mock_app_context.model_uri = "runs/123/model"
        mock_app_context.PredictionRequest = mock_prediction_request
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            endpoints.get_model_metadata()
        
        assert "Failed to unwrap model" in str(exc_info.value)


class TestEndpointIntegration:
    """Test endpoint integration scenarios."""
    
    @patch('src.serving._endpoints.app_context')
    def test_full_api_workflow(self, mock_app_context):
        """Test complete API workflow from health to prediction."""
        # Arrange
        mock_wrapped_model = Mock()
        mock_wrapped_model.model_class_path = "test.model.Class"
        
        mock_model = Mock()
        mock_model.unwrap_python_model.return_value = mock_wrapped_model
        mock_model.hyperparameter_optimization = {"enabled": False}
        mock_model.training_methodology = {
            "train_test_split_method": "random",
            "train_ratio": 0.8,
            "validation_strategy": "holdout",
            "preprocessing_fit_scope": "train_only"
        }
        mock_model.model_class_path = "test.model.Class"
        
        mock_predictions_df = Mock()
        mock_predictions_df.to_dict.return_value = [{"prediction": 0.7}]
        mock_predictions_df.__len__ = Mock(return_value=1)
        mock_model.predict.return_value = mock_predictions_df
        
        mock_batch_request_class = Mock()
        mock_request_instance = Mock()
        mock_sample = Mock()
        mock_sample.model_dump.return_value = {"feature1": 1.0}
        mock_request_instance.samples = [mock_sample]
        mock_batch_request_class.return_value = mock_request_instance
        
        # Mock PredictionRequest for API schema
        mock_prediction_request = Mock()
        mock_prediction_request.model_fields = {"feature1": {}}
        
        mock_settings = Mock()
        
        mock_app_context.model = mock_model
        mock_app_context.settings = mock_settings
        mock_app_context.model_uri = "runs/123/model"
        mock_app_context.BatchPredictionRequest = mock_batch_request_class
        mock_app_context.PredictionRequest = mock_prediction_request
        
        # Act & Assert - Health check
        health_result = endpoints.health()
        assert health_result.status == "healthy"
        assert health_result.model_name == "test.model.Class"
        
        # Act & Assert - Batch prediction
        with patch('src.serving._endpoints.pd.DataFrame') as mock_df:
            mock_input_df = Mock()
            mock_input_df.empty = False
            mock_df.return_value = mock_input_df
            
            prediction_result = endpoints.predict_batch({"samples": [{"feature1": 1.0}]})
            assert prediction_result.predictions == [{"prediction": 0.7}]
            assert prediction_result.sample_count == 1
        
        # Act & Assert - Model metadata
        metadata_result = endpoints.get_model_metadata()
        assert metadata_result.model_class_path == "test.model.Class"
        assert metadata_result.hyperparameter_optimization.enabled is False


class TestEndpointErrorHandling:
    """Test endpoint error handling scenarios."""
    
    @patch('src.serving._endpoints.app_context')
    def test_predict_batch_validation_error(self, mock_app_context):
        """Test batch prediction with validation errors."""
        # Arrange
        mock_batch_request_class = Mock()
        mock_batch_request_class.side_effect = ValueError("Invalid request format")
        
        mock_app_context.BatchPredictionRequest = mock_batch_request_class
        
        request_data = {"invalid": "data"}
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid request format"):
            endpoints.predict_batch(request_data)
    
    @patch('src.serving._endpoints.app_context')
    def test_health_check_partial_failure(self, mock_app_context):
        """Test health check with partial system failures."""
        # Arrange - model exists but has issues
        mock_model = Mock()
        mock_model.unwrap_python_model.side_effect = AttributeError("Missing attribute")
        
        mock_settings = Mock()
        mock_app_context.model = mock_model
        mock_app_context.settings = mock_settings
        mock_app_context.model_uri = "runs/456/model"
        
        # Act
        result = endpoints.health()
        
        # Assert - should still return healthy but with unknown model name
        assert result.status == "healthy"
        assert result.model_name == "unknown"
        assert result.model_uri == "runs/456/model"


class TestEndpointEdgeCases:
    """Test endpoint edge cases."""
    
    @patch('src.serving._endpoints.app_context')
    @patch('src.serving._endpoints.pd.DataFrame')
    def test_predict_batch_large_payload(self, mock_dataframe_class, mock_app_context):
        """Test batch prediction with large payload."""
        # Arrange
        mock_batch_request_class = Mock()
        mock_request_instance = Mock()
        
        # Create large number of samples
        large_samples = []
        for i in range(1000):
            mock_sample = Mock()
            mock_sample.model_dump.return_value = {"feature1": float(i)}
            large_samples.append(mock_sample)
        
        mock_request_instance.samples = large_samples
        mock_batch_request_class.return_value = mock_request_instance
        
        mock_input_df = Mock()
        mock_input_df.empty = False
        mock_dataframe_class.return_value = mock_input_df
        
        mock_predictions_df = Mock()
        mock_predictions_df.to_dict.return_value = [{"prediction": i/1000.0} for i in range(1000)]
        mock_predictions_df.__len__ = Mock(return_value=1000)
        
        mock_model = Mock()
        mock_model.predict.return_value = mock_predictions_df
        
        mock_app_context.BatchPredictionRequest = mock_batch_request_class
        mock_app_context.model = mock_model
        mock_app_context.model_uri = "runs/123/model"
        
        request_data = {"samples": [{"feature1": float(i)} for i in range(1000)]}
        
        # Act
        result = endpoints.predict_batch(request_data)
        
        # Assert
        assert result.sample_count == 1000
        assert len(result.predictions) == 1000
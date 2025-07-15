"""
Serving API 테스트

Blueprint v17.0의 새로운 API 엔드포인트들과 기존 엔드포인트 확장 기능 테스트
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from src.settings import Settings


class TestServingAPI:
    """기존 API 엔드포인트 테스트"""
    
    @pytest.fixture
    def mock_app_context(self):
        """Mock AppContext 설정"""
        from serving.api import AppContext
        context = AppContext()
        
        # Mock 모델 설정
        mock_model = Mock()
        mock_model.hyperparameter_optimization = {
            "enabled": True,
            "best_params": {"learning_rate": 0.1, "n_estimators": 100},
            "best_score": 0.92,
            "total_trials": 50
        }
        mock_model.training_methodology = {
            "train_test_split_method": "stratified",
            "preprocessing_fit_scope": "train_only",
            "random_state": 42
        }
        mock_model.model_class_path = "causalml.inference.meta.XGBTRegressor"
        mock_model.training_metadata = {"timestamp": "2024-01-01"}
        mock_model.loader_sql_snapshot = "SELECT user_id, product_id FROM users"
        
        context.model = mock_model
        context.model_uri = "runs:/test123/model"
        context.feature_columns = ["feature1", "feature2"] 
        context.join_key = "user_id"
        
        # Mock settings
        mock_settings = Mock()
        mock_settings.serving.realtime_feature_store = {"type": "redis"}
        context.settings = mock_settings
        context.feature_store_config = {"type": "redis"}
        
        # Mock Pydantic 모델
        from pydantic import BaseModel, Field
        class MockPredictionRequest(BaseModel):
            user_id: str = Field(..., description="User ID")
            product_id: str = Field(..., description="Product ID")
        
        context.PredictionRequest = MockPredictionRequest
        context.BatchPredictionRequest = Mock()
        
        return context

    @patch('serving.api.app_context')
    def test_predict_endpoint_with_optimization_metadata(self, mock_context, mock_app_context):
        """predict 엔드포인트가 최적화 메타데이터를 포함하는지 테스트"""
        mock_context.return_value = mock_app_context
        
        from serving.api import create_app
        app = create_app("test_run_id")
        client = TestClient(app)
        
        # Mock 예측 결과
        mock_predictions = pd.DataFrame({"uplift_score": [0.85]})
        mock_app_context.model.predict.return_value = mock_predictions
        
        # API 호출
        response = client.post("/predict", json={"user_id": "user123", "product_id": "prod456"})
        
        # 응답 확인
        assert response.status_code == 200
        data = response.json()
        
        # 기존 필드들 확인
        assert "uplift_score" in data
        assert "model_uri" in data
        
        # 🆕 Blueprint v17.0: 새로운 최적화 메타데이터 필드들 확인
        assert "optimization_enabled" in data
        assert "best_score" in data
        assert data["optimization_enabled"] is True
        assert data["best_score"] == 0.92

    @patch('serving.api.app_context')
    def test_predict_batch_endpoint_with_optimization_metadata(self, mock_context, mock_app_context):
        """predict_batch 엔드포인트가 최적화 메타데이터를 포함하는지 테스트"""
        mock_context.return_value = mock_app_context
        
        from serving.api import create_app
        app = create_app("test_run_id")
        client = TestClient(app)
        
        # Mock 배치 예측 결과
        mock_predictions = pd.DataFrame({
            "user_id": ["user1", "user2"],
            "product_id": ["prod1", "prod2"],
            "uplift_score": [0.85, 0.75]
        })
        mock_app_context.model.predict.return_value = mock_predictions
        
        # API 호출
        request_data = {
            "samples": [
                {"user_id": "user1", "product_id": "prod1"},
                {"user_id": "user2", "product_id": "prod2"}
            ]
        }
        response = client.post("/predict_batch", json=request_data)
        
        # 응답 확인
        assert response.status_code == 200
        data = response.json()
        
        # 기존 필드들 확인
        assert "predictions" in data
        assert "model_uri" in data
        assert "sample_count" in data
        
        # 🆕 Blueprint v17.0: 새로운 최적화 메타데이터 필드들 확인
        assert "optimization_enabled" in data
        assert "best_score" in data
        assert data["optimization_enabled"] is True
        assert data["best_score"] == 0.92


# 🆕 Blueprint v17.0: 새로운 메타데이터 엔드포인트 테스트 클래스
class TestServingAPIMetadataEndpoints:
    """새로운 메타데이터 API 엔드포인트들 테스트"""
    
    @pytest.fixture
    def mock_app_context_full(self):
        """전체 메타데이터를 포함한 Mock AppContext 설정"""
        from serving.api import AppContext
        context = AppContext()
        
        # Mock 모델 설정 (완전한 메타데이터 포함)
        mock_model = Mock()
        mock_model.hyperparameter_optimization = {
            "enabled": True,
            "engine": "optuna",
            "best_params": {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 5},
            "best_score": 0.924,
            "total_trials": 50,
            "pruned_trials": 12,
            "optimization_time": "0:15:32",
            "optimization_history": [
                {"trial": 1, "score": 0.85, "params": {"learning_rate": 0.05}},
                {"trial": 2, "score": 0.90, "params": {"learning_rate": 0.1}}
            ],
            "search_space": {
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
                "n_estimators": {"type": "int", "low": 50, "high": 200}
            },
            "timeout_occurred": False
        }
        mock_model.training_methodology = {
            "train_test_split_method": "stratified",
            "train_ratio": 0.8,
            "validation_strategy": "train_validation_split",
            "preprocessing_fit_scope": "train_only",
            "random_state": 42
        }
        mock_model.model_class_path = "causalml.inference.meta.XGBTRegressor"
        mock_model.training_metadata = {
            "timestamp": "2024-01-01T10:30:00",
            "mlflow_run_id": "abc123def456"
        }
        mock_model.loader_sql_snapshot = "SELECT user_id, product_id, session_id FROM users WHERE active = 1"
        
        context.model = mock_model
        context.model_uri = "runs:/abc123def456/model"
        context.feature_columns = ["age", "country", "ltv", "click_count"]
        context.join_key = "user_id"
        
        # Mock settings
        mock_settings = Mock()
        mock_settings.serving.realtime_feature_store = {"type": "redis", "host": "localhost"}
        context.settings = mock_settings
        context.feature_store_config = {"type": "redis", "host": "localhost"}
        
        # Mock Pydantic 모델
        from pydantic import BaseModel, Field
        class MockPredictionRequest(BaseModel):
            user_id: str = Field(..., description="User ID")
            product_id: str = Field(..., description="Product ID")
            session_id: str = Field(..., description="Session ID")
        
        context.PredictionRequest = MockPredictionRequest
        context.BatchPredictionRequest = Mock()
        
        return context

    @patch('serving.api.app_context')
    def test_model_metadata_endpoint(self, mock_context, mock_app_context_full):
        """GET /model/metadata 엔드포인트 테스트"""
        mock_context.return_value = mock_app_context_full
        
        from serving.api import create_app
        app = create_app("test_run_id")
        client = TestClient(app)
        
        # API 호출
        response = client.get("/model/metadata")
        
        # 응답 확인
        assert response.status_code == 200
        data = response.json()
        
        # 기본 메타데이터 확인
        assert data["model_uri"] == "runs:/abc123def456/model"
        assert data["model_class_path"] == "causalml.inference.meta.XGBTRegressor"
        
        # 하이퍼파라미터 최적화 정보 확인
        hpo = data["hyperparameter_optimization"]
        assert hpo["enabled"] is True
        assert hpo["engine"] == "optuna"
        assert hpo["best_score"] == 0.924
        assert hpo["total_trials"] == 50
        assert hpo["pruned_trials"] == 12
        assert "learning_rate" in hpo["best_params"]
        
        # 학습 방법론 정보 확인
        tm = data["training_methodology"]
        assert tm["train_test_split_method"] == "stratified"
        assert tm["preprocessing_fit_scope"] == "train_only"
        assert tm["random_state"] == 42
        
        # API 스키마 정보 확인
        api_schema = data["api_schema"]
        assert "input_fields" in api_schema
        assert "user_id" in api_schema["input_fields"]
        assert "feature_columns" in api_schema
        assert "join_key" in api_schema

    @patch('serving.api.app_context')
    def test_optimization_history_endpoint_enabled(self, mock_context, mock_app_context_full):
        """GET /model/optimization 엔드포인트 테스트 (최적화 활성화)"""
        mock_context.return_value = mock_app_context_full
        
        from serving.api import create_app
        app = create_app("test_run_id")
        client = TestClient(app)
        
        # API 호출
        response = client.get("/model/optimization")
        
        # 응답 확인
        assert response.status_code == 200
        data = response.json()
        
        # 최적화 활성화 확인
        assert data["enabled"] is True
        
        # 최적화 히스토리 확인
        assert "optimization_history" in data
        assert len(data["optimization_history"]) == 2
        
        # 탐색 공간 확인
        assert "search_space" in data
        search_space = data["search_space"]
        assert "learning_rate" in search_space
        assert search_space["learning_rate"]["type"] == "float"
        
        # 수렴 정보 확인
        convergence = data["convergence_info"]
        assert convergence["best_score"] == 0.924
        assert convergence["total_trials"] == 50
        assert data["timeout_occurred"] is False

    @patch('serving.api.app_context')
    def test_optimization_history_endpoint_disabled(self, mock_context, mock_app_context_full):
        """GET /model/optimization 엔드포인트 테스트 (최적화 비활성화)"""
        # 최적화 비활성화로 설정
        mock_app_context_full.model.hyperparameter_optimization["enabled"] = False
        mock_context.return_value = mock_app_context_full
        
        from serving.api import create_app
        app = create_app("test_run_id")
        client = TestClient(app)
        
        # API 호출
        response = client.get("/model/optimization")
        
        # 응답 확인
        assert response.status_code == 200
        data = response.json()
        
        # 최적화 비활성화 확인
        assert data["enabled"] is False
        assert data["optimization_history"] == []
        assert data["search_space"] == {}
        assert "하이퍼파라미터 최적화가 비활성화되었습니다" in data["convergence_info"]["message"]

    @patch('serving.api.app_context')
    def test_api_schema_endpoint(self, mock_context, mock_app_context_full):
        """GET /model/schema 엔드포인트 테스트"""
        mock_context.return_value = mock_app_context_full
        
        from serving.api import create_app
        app = create_app("test_run_id")
        client = TestClient(app)
        
        # API 호출
        response = client.get("/model/schema")
        
        # 응답 확인
        assert response.status_code == 200
        data = response.json()
        
        # 동적 스키마 정보 확인
        assert "prediction_request_schema" in data
        assert "batch_prediction_request_schema" in data
        assert "loader_sql_snapshot" in data
        assert "extracted_fields" in data
        
        # SQL 스냅샷 확인
        sql_snapshot = data["loader_sql_snapshot"]
        assert "SELECT user_id, product_id, session_id" in sql_snapshot
        
        # 추출된 필드 확인
        extracted_fields = data["extracted_fields"]
        assert "user_id" in extracted_fields
        assert "product_id" in extracted_fields
        assert "session_id" in extracted_fields
        
        # Feature Store 정보 확인
        fs_info = data["feature_store_info"]
        assert fs_info["feature_columns"] == ["age", "country", "ltv", "click_count"]
        assert fs_info["join_key"] == "user_id"
        assert fs_info["feature_store_config"]["type"] == "redis"

    @patch('serving.api.app_context')
    def test_api_endpoints_without_model(self, mock_context):
        """모델이 로드되지 않은 상태에서 메타데이터 엔드포인트 테스트"""
        from serving.api import AppContext
        context = AppContext()
        context.model = None  # 모델 없음
        mock_context.return_value = context
        
        from serving.api import create_app
        app = create_app("test_run_id")
        client = TestClient(app)
        
        # 모든 메타데이터 엔드포인트에서 503 오류 확인
        endpoints = ["/model/metadata", "/model/optimization", "/model/schema"]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 503
            assert "모델이 로드되지 않았습니다" in response.json()["detail"]


# 🆕 Blueprint v17.0: 호환성 보장 테스트 클래스
class TestServingAPICompatibility:
    """API 호환성 보장 테스트"""
    
    @patch('serving.api.app_context')
    def test_backward_compatibility_predict(self, mock_context):
        """기존 predict 엔드포인트 하위 호환성 테스트"""
        from serving.api import AppContext
        context = AppContext()
        
        # 최적화가 비활성화된 모델 Mock
        mock_model = Mock()
        mock_model.hyperparameter_optimization = {"enabled": False}
        mock_model.predict.return_value = pd.DataFrame({"uplift_score": [0.75]})
        
        context.model = mock_model
        context.model_uri = "runs:/legacy123/model"
        context.settings = Mock()
        context.settings.serving.realtime_feature_store = {}
        context.feature_store_config = {}
        context.feature_columns = []
        
        # Mock Pydantic 모델
        from pydantic import BaseModel, Field
        class MockPredictionRequest(BaseModel):
            user_id: str = Field(..., description="User ID")
        
        context.PredictionRequest = MockPredictionRequest
        mock_context.return_value = context
        
        from serving.api import create_app
        app = create_app("legacy_run_id")
        client = TestClient(app)
        
        # API 호출
        response = client.post("/predict", json={"user_id": "user123"})
        
        # 응답 확인
        assert response.status_code == 200
        data = response.json()
        
        # 기존 필드들 확인
        assert "uplift_score" in data
        assert "model_uri" in data
        
        # 새로운 필드들이 기본값으로 설정되었는지 확인
        assert "optimization_enabled" in data
        assert "best_score" in data
        assert data["optimization_enabled"] is False
        assert data["best_score"] == 0.0  # 기본값
    
    def test_response_schema_extensions(self):
        """응답 스키마 확장이 올바르게 적용되었는지 테스트"""
        from serving.schemas import PredictionResponse, BatchPredictionResponse
        
        # PredictionResponse 스키마 확인
        pred_schema = PredictionResponse.schema()
        properties = pred_schema["properties"]
        
        # 기존 필드들 확인
        assert "uplift_score" in properties
        assert "model_uri" in properties
        
        # 새로운 필드들 확인
        assert "optimization_enabled" in properties
        assert "best_score" in properties
        
        # 새로운 필드들이 Optional인지 확인 (기본값 있음)
        assert properties["optimization_enabled"]["default"] is False
        assert properties["best_score"]["default"] == 0.0
        
        # BatchPredictionResponse도 동일하게 확인
        batch_schema = BatchPredictionResponse.schema()
        batch_properties = batch_schema["properties"]
        
        assert "optimization_enabled" in batch_properties
        assert "best_score" in batch_properties
        assert batch_properties["optimization_enabled"]["default"] is False
        assert batch_properties["best_score"]["default"] == 0.0 
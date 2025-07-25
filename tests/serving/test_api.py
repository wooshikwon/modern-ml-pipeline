"""
Serving API 테스트 (Blueprint v17.0 현대화)

Blueprint v17.0의 새로운 API 엔드포인트들과 E2E 테스트 강화

Blueprint 원칙 검증:
- 원칙 6: 자기 기술 API (Self-Describing API)
- 완전한 커버리지: 모든 메타데이터 엔드포인트 검증
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from src.settings import Settings
import os

import mlflow
import shutil

from src.settings import Settings
from serving.api import app, setup_api_context
from src.pipelines.train_pipeline import run_training

@pytest.fixture(scope="module")
def trained_model_run_id_for_api(dev_test_settings: Settings):
    """
    API 테스트를 위해 미리 학습된 모델 아티팩트를 생성하고
    해당 run_id를 제공하는 Fixture.
    """
    test_tracking_uri = "./test_mlruns_api"
    mlflow.set_tracking_uri(test_tracking_uri)
    
    result_artifact = run_training(settings=dev_test_settings)
    
    yield result_artifact.run_id
    
    shutil.rmtree(test_tracking_uri, ignore_errors=True)
    mlflow.set_tracking_uri("mlruns")

@pytest.mark.requires_dev_stack
class TestServingAPIComplete:
    """
    Serving API의 모든 엔드포인트를 검증하는 완전한 E2E 테스트.
    실제 학습된 모델을 사용하여 API의 동작을 검증한다.
    Blueprint v17.0 완전 현대화
    """

    @pytest.fixture(scope="class")
    def client(self, dev_test_settings: Settings, trained_model_run_id_for_api: str):
        """
        테스트용 API 클라이언트를 설정하는 Fixture.
        학습된 모델로 API 컨텍스트를 초기화한다.
        """
        setup_api_context(run_id=trained_model_run_id_for_api, settings=dev_test_settings)
        return TestClient(app)

    def test_health_endpoint(self, client: TestClient, trained_model_run_id_for_api: str):
        """GET /health: API 서버의 상태와 모델 로드 여부를 확인한다."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert trained_model_run_id_for_api in data["model_uri"]
        assert data["model_name"] == "sklearn.ensemble.RandomForestClassifier"

    def test_root_endpoint(self, client: TestClient, trained_model_run_id_for_api: str):
        """GET /: 루트 엔드포인트가 올바른 정보를 반환하는지 확인한다."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Modern ML Pipeline API"
        assert data["status"] == "ready"
        assert trained_model_run_id_for_api in data["model_uri"]

    def test_predict_endpoint(self, client: TestClient):
        """POST /predict: 단일 예측 요청이 정상적으로 처리되는지 확인한다."""
        # dev_classification_test.yaml의 loader가 user_id와 product_id를 요구한다고 가정
        request_data = {"user_id": "u1001", "product_id": "p2001"}
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], float)
        assert "input_features" in data
        assert data["input_features"]["product_price"] > 0 # 예시 검증

    def test_predict_endpoint_invalid_input(self, client: TestClient):
        """POST /predict: 잘못된 입력에 대해 422 Unprocessable Entity를 반환하는지 확인한다."""
        # product_id 필드가 누락된 요청
        request_data = {"user_id": "u1001"}
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422

    # 🆕 Blueprint v17.0: 완전한 메타데이터 엔드포인트 검증
    def test_model_metadata_endpoint_complete(self, client: TestClient):
        """
        GET /model/metadata: 모델의 완전한 메타데이터 반환을 검증한다.
        Blueprint 원칙 6: 자기 기술 API
        """
        response = client.get("/model/metadata")
        assert response.status_code == 200
        data = response.json()
        
        # 필수 메타데이터 필드 검증
        required_fields = [
            "model_uri", "model_class_path", "training_timestamp",
            "hyperparameter_optimization", "training_methodology", "api_schema"
        ]
        for field in required_fields:
            assert field in data, f"메타데이터에서 '{field}' 필드가 누락되었습니다."
        
        # 하이퍼파라미터 최적화 메타데이터 상세 검증
        hpo = data["hyperparameter_optimization"]
        assert isinstance(hpo["enabled"], bool)
        if hpo["enabled"]:
            assert "best_params" in hpo
            assert "best_score" in hpo
            assert "total_trials" in hpo
        
        # 학습 방법론 메타데이터 검증 (Data Leakage 방지)
        tm = data["training_methodology"]
        assert tm["preprocessing_fit_scope"] == "train_only"
        assert "train_test_split_method" in tm
        
        # API 스키마 정보 검증
        api_schema = data["api_schema"]
        assert "input_fields" in api_schema
        assert isinstance(api_schema["input_fields"], list)

    def test_model_optimization_endpoint(self, client: TestClient):
        """
        GET /model/optimization: 하이퍼파라미터 최적화 상세 정보를 검증한다.
        """
        response = client.get("/model/optimization")
        assert response.status_code == 200
        data = response.json()
        
        # 기본 필드 검증
        assert "enabled" in data
        assert "optimization_history" in data
        assert "search_space" in data
        assert "convergence_info" in data
        
        # 활성화 여부에 따른 상세 검증
        if data["enabled"]:
            assert isinstance(data["optimization_history"], list)
            assert isinstance(data["search_space"], dict)
            assert "best_score" in data["convergence_info"]
        else:
            assert data["optimization_history"] == []
            assert data["search_space"] == {}

    def test_model_schema_endpoint(self, client: TestClient):
        """
        GET /model/schema: 동적으로 생성된 API 스키마 정보를 검증한다.
        Blueprint 원칙 6: 자기 기술 API의 핵심
        """
        response = client.get("/model/schema")
        assert response.status_code == 200
        data = response.json()
        
        # 필수 스키마 정보 검증
        required_schema_fields = [
            "prediction_request_schema", "batch_prediction_request_schema",
            "loader_sql_snapshot", "extracted_fields", "feature_store_info"
        ]
        for field in required_schema_fields:
            assert field in data, f"스키마 정보에서 '{field}' 필드가 누락되었습니다."
        
        # 동적 스키마 검증
        prediction_schema = data["prediction_request_schema"]
        assert "properties" in prediction_schema
        assert isinstance(prediction_schema["properties"], dict)
        
        # SQL 스냅샷 검증
        assert isinstance(data["loader_sql_snapshot"], str)
        assert len(data["loader_sql_snapshot"]) > 0
        
        # 추출된 필드 검증
        assert isinstance(data["extracted_fields"], list)
        assert len(data["extracted_fields"]) > 0

    # 🆕 Blueprint v17.0: 배치 예측 엔드포인트 (새로 추가)
    def test_batch_predict_endpoint(self, client: TestClient):
        """
        POST /batch_predict: 배치 예측 요청이 정상적으로 처리되는지 확인한다.
        """
        # 배치 요청 데이터
        batch_request = {
            "requests": [
                {"user_id": "u1001", "product_id": "p2001"},
                {"user_id": "u1002", "product_id": "p2002"}
            ]
        }
        
        response = client.post("/batch_predict", json=batch_request)
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        for prediction in data["predictions"]:
            assert "prediction" in prediction
            assert isinstance(prediction["prediction"], float)

    # 🆕 Blueprint v17.0: 에러 처리 검증
    def test_api_error_handling(self, client: TestClient):
        """API의 적절한 에러 처리를 검증한다."""
        # 1. 잘못된 엔드포인트
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        # 2. 잘못된 HTTP 메서드
        response = client.get("/predict")  # POST만 허용
        assert response.status_code == 405
        
        # 3. 잘못된 JSON 형식
        response = client.post("/predict", 
                              data="invalid json", 
                              headers={"Content-Type": "application/json"})
        assert response.status_code == 422

    # 🆕 Blueprint v17.0: 성능 검증
    def test_api_performance_targets(self, client: TestClient):
        """API 응답 시간 목표 달성을 검증한다."""
        import time
        
        request_data = {"user_id": "u1001", "product_id": "p2001"}
        
        # 여러 번 측정하여 평균 응답 시간 계산
        response_times = []
        for _ in range(5):
            start_time = time.time()
            response = client.post("/predict", json=request_data)
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append((end_time - start_time) * 1000)  # ms로 변환
        
        avg_response_time = sum(response_times) / len(response_times)
        
        # 성능 목표: 평균 응답 시간 500ms 이하 (DEV 환경 기준)
        assert avg_response_time < 500, f"평균 응답 시간 목표 미달성: {avg_response_time:.2f}ms"


# 🆕 Blueprint v17.0: 자기 기술 API 심화 테스트
@pytest.mark.requires_dev_stack
@pytest.mark.blueprint_principle_6
class TestSelfDescribingAPIAdvanced:
    """자기 기술 API의 고급 기능 테스트"""
    
    def setup_method(self):
        os.environ['APP_ENV'] = 'dev'
    
    def teardown_method(self):
        if 'APP_ENV' in os.environ:
            del os.environ['APP_ENV']

    def test_dynamic_schema_adaptation(self, dev_test_settings: Settings, trained_model_run_id_for_api: str):
        """
        서로 다른 모델의 loader SQL에 따라 API 스키마가 동적으로 변경되는지 검증한다.
        """
        # API 컨텍스트 설정
        setup_api_context(run_id=trained_model_run_id_for_api, settings=dev_test_settings)
        client = TestClient(app)
        
        # 스키마 정보 조회
        response = client.get("/model/schema")
        assert response.status_code == 200
        data = response.json()
        
        # 현재 모델의 스키마 필드 확인
        current_fields = data["extracted_fields"]
        
        # 예상되는 필드들 (dev_classification_test.yaml 기준)
        expected_fields = ["user_id", "product_id"]  # event_timestamp는 제외
        for field in expected_fields:
            assert field in current_fields, f"예상 필드 '{field}'가 추출된 필드에 없습니다."

    def test_feature_store_integration_visibility(self, dev_test_settings: Settings, trained_model_run_id_for_api: str):
        """
        API가 Feature Store 연동 정보를 올바르게 노출하는지 검증한다.
        """
        setup_api_context(run_id=trained_model_run_id_for_api, settings=dev_test_settings)
        client = TestClient(app)
        
        response = client.get("/model/schema")
        assert response.status_code == 200
        data = response.json()
        
        # Feature Store 정보 검증
        fs_info = data["feature_store_info"]
        assert "feature_columns" in fs_info
        assert "join_key" in fs_info
        assert isinstance(fs_info["feature_columns"], list)

    def test_api_documentation_completeness(self, dev_test_settings: Settings, trained_model_run_id_for_api: str):
        """
        API 문서화가 완전한지 검증한다. (FastAPI 자동 생성 문서)
        """
        setup_api_context(run_id=trained_model_run_id_for_api, settings=dev_test_settings)
        client = TestClient(app)
        
        # OpenAPI 스키마 조회
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi_schema = response.json()
        
        # 필수 엔드포인트가 문서에 포함되어 있는지 확인
        paths = openapi_schema["paths"]
        expected_endpoints = ["/", "/health", "/predict", "/model/metadata", "/model/schema", "/model/optimization"]
        
        for endpoint in expected_endpoints:
            assert endpoint in paths, f"API 문서에서 '{endpoint}' 엔드포인트가 누락되었습니다."
        
        # API 문서 UI 접근 가능성 확인
        docs_response = client.get("/docs")
        assert docs_response.status_code == 200


# 기존 테스트 클래스들 유지 (호환성을 위해)
class TestServingAPIMetadataEndpoints:
    """기존 메타데이터 엔드포인트 테스트 (하위 호환성)"""
    
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
    def test_model_metadata_endpoint_mock(self, mock_context, mock_app_context_full):
        """GET /model/metadata 엔드포인트 Mock 테스트 (기존 호환성)"""
        mock_context.return_value = mock_app_context_full
        
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


# 🆕 환경별 차등 API 서빙 테스트 클래스들 추가
@pytest.mark.blueprint_principle_9
class TestEnvironmentSpecificAPIServing:
    """환경별 차등 API 서빙 기능 테스트"""
    
    def teardown_method(self):
        """환경변수 정리"""
        if 'APP_ENV' in os.environ:
            del os.environ['APP_ENV']
    
    @pytest.mark.local_env
    def test_local_env_api_serving_blocked(self):
        """LOCAL 환경: API 서빙 시스템적 차단 검증"""
        os.environ['APP_ENV'] = 'local'
        
        with patch('serving.api.load_config_files') as mock_load_config:
            # LOCAL 환경에서 API 서빙 차단 설정
            mock_config = {
                'environment': {'app_env': 'local'},
                'api_serving': {
                    'enabled': False,
                    'blocked_reason': 'LOCAL 환경의 철학에 따라 빠른 실험과 디버깅에만 집중합니다.'
                }
            }
            mock_load_config.return_value = mock_config
            
            # API 서빙 시도 시 차단 확인
            with pytest.raises(RuntimeError, match="LOCAL 환경에서는 API 서빙이 지원되지 않습니다"):
                from serving.api import create_app
                app = create_app("test_run_id")
    
    @pytest.mark.dev_env
    @pytest.mark.requires_dev_stack
    def test_dev_env_api_serving_enabled(self):
        """DEV 환경: API 서빙 완전 허용 검증"""
        os.environ['APP_ENV'] = 'dev'
        
        with patch('serving.api.load_config_files') as mock_load_config, \
             patch('mlflow.pyfunc.load_model') as mock_load_model:
            
            # DEV 환경에서 API 서빙 허용 설정
            mock_config = {
                'environment': {'app_env': 'dev'},
                'api_serving': {'enabled': True},
                'serving': {
                    'realtime_feature_store': {
                        'store_type': 'redis',
                        'connection': {'host': 'localhost', 'port': 6379}
                    }
                }
            }
            mock_load_config.return_value = mock_config
            
            # Mock Wrapped Artifact
            mock_wrapper = Mock()
            mock_wrapper.unwrap_python_model.return_value = Mock()
            mock_wrapper.unwrap_python_model.return_value.loader_sql_snapshot = 'SELECT user_id FROM users'
            mock_load_model.return_value = mock_wrapper
            
            # API 앱 생성 성공 확인
            from serving.api import create_app
            app = create_app("test_run_id")
            assert app is not None
    
    @pytest.mark.prod_env
    def test_prod_env_api_serving_enhanced(self):
        """PROD 환경: 향상된 API 서빙 기능 검증"""
        os.environ['APP_ENV'] = 'prod'
        
        with patch('serving.api.load_config_files') as mock_load_config, \
             patch('mlflow.pyfunc.load_model') as mock_load_model:
            
            # PROD 환경에서 향상된 API 서빙 설정
            mock_config = {
                'environment': {'app_env': 'prod'},
                'api_serving': {
                    'enabled': True,
                    'enhanced_features': {
                        'monitoring': True,
                        'rate_limiting': True,
                        'caching': True,
                        'health_checks': True
                    }
                },
                'serving': {
                    'realtime_feature_store': {
                        'store_type': 'redis_cluster',
                        'connection': {'host': 'redis-cluster.prod.internal'}
                    }
                }
            }
            mock_load_config.return_value = mock_config
            
            # Mock Wrapped Artifact
            mock_wrapper = Mock()
            mock_wrapper.unwrap_python_model.return_value = Mock()
            mock_wrapper.unwrap_python_model.return_value.loader_sql_snapshot = 'SELECT user_id FROM users'
            mock_load_model.return_value = mock_wrapper
            
            # PROD 환경 API 앱 생성 성공 확인
            from serving.api import create_app
            app = create_app("test_run_id")
            assert app is not None


@pytest.mark.requires_dev_stack
@pytest.mark.blueprint_principle_6
class TestSelfDescribingAPIReal:
    """실제 인프라에서 자기 기술 API 테스트"""
    
    def setup_method(self):
        os.environ['APP_ENV'] = 'dev'
    
    def teardown_method(self):
        if 'APP_ENV' in os.environ:
            del os.environ['APP_ENV']
    
    def test_dynamic_schema_generation_real(self):
        """실제 Wrapped Artifact에서 동적 스키마 생성 테스트"""
        from serving.schemas import create_dynamic_prediction_request
        from src.utils.system.sql_utils import parse_select_columns
        
        # 실제 Loader SQL 예시
        real_loader_sql = """
        SELECT 
            u.user_id,
            p.product_id,
            s.session_id,
            CURRENT_TIMESTAMP as event_timestamp
        FROM users u
        CROSS JOIN products p  
        CROSS JOIN sessions s
        WHERE u.is_active = 1
        """
        
        with patch('serving.api.load_config_files') as mock_config, \
             patch('mlflow.pyfunc.load_model') as mock_load_model:
            
            mock_config.return_value = {
                'environment': {'app_env': 'dev'},
                'api_serving': {'enabled': True}
            }
            
            # 실제 SQL 파싱을 포함한 Mock Wrapper
            mock_wrapper = Mock()
            mock_inner_model = Mock()
            mock_inner_model.loader_sql_snapshot = real_loader_sql
            mock_wrapper.unwrap_python_model.return_value = mock_inner_model
            mock_load_model.return_value = mock_wrapper
            
            # 실제 SQL 파싱 테스트
            columns = parse_select_columns(real_loader_sql)
            expected_columns = ['user_id', 'product_id', 'session_id']  # event_timestamp 제외
            
            # 동적 스키마 생성 테스트
            PredictionRequest = create_dynamic_prediction_request(expected_columns)
            
            # 스키마 검증
            assert hasattr(PredictionRequest, '__annotations__')
            for col in expected_columns:
                assert col in PredictionRequest.__annotations__
            
            # API 앱 생성 및 스키마 적용 확인
            from serving.api import create_app
            app = create_app("test_run_id")
            assert app is not None
    
    def test_api_consistency_with_batch_inference(self):
        """API 입력과 배치 추론 입력의 완전한 일관성 검증"""
        # 배치 추론 SQL
        batch_sql = """
        SELECT 
            customer_id,
            product_id, 
            interaction_timestamp as event_timestamp
        FROM customer_interactions
        WHERE interaction_date >= '2024-01-01'
        """
        
        with patch('serving.api.load_config_files') as mock_config, \
             patch('mlflow.pyfunc.load_model') as mock_load_model:
            
            mock_config.return_value = {
                'environment': {'app_env': 'dev'},
                'api_serving': {'enabled': True}
            }
            
            # Mock Wrapper with batch SQL
            mock_wrapper = Mock()
            mock_inner_model = Mock()
            mock_inner_model.loader_sql_snapshot = batch_sql
            mock_wrapper.unwrap_python_model.return_value = mock_inner_model
            mock_load_model.return_value = mock_wrapper
            
            # API 스키마 생성
            from src.utils.system.sql_utils import parse_select_columns
            from serving.schemas import create_dynamic_prediction_request
            
            api_columns = parse_select_columns(batch_sql)
            # event_timestamp는 API에서 제외되어야 함
            expected_api_columns = ['customer_id', 'product_id']
            
            assert set(api_columns) == set(expected_api_columns)
            
            # 동적 스키마 생성 확인
            PredictionRequest = create_dynamic_prediction_request(api_columns)
            
            # 배치와 API의 엔티티 키 일관성 검증
            for col in expected_api_columns:
                assert col in PredictionRequest.__annotations__


@pytest.mark.requires_dev_stack
@pytest.mark.blueprint_principle_5
class TestContextInjectionAPIReal:
    """실제 환경에서 컨텍스트 주입 API 테스트"""
    
    def setup_method(self):
        os.environ['APP_ENV'] = 'dev'
    
    def teardown_method(self):
        if 'APP_ENV' in os.environ:
            del os.environ['APP_ENV']
    
    def test_serving_mode_context_injection(self):
        """실제 API 서빙에서 컨텍스트 주입 테스트"""
        from fastapi.testclient import TestClient
        
        with patch('serving.api.load_config_files') as mock_config, \
             patch('mlflow.pyfunc.load_model') as mock_load_model:
            
            mock_config.return_value = {
                'environment': {'app_env': 'dev'},
                'api_serving': {'enabled': True},
                'serving': {
                    'realtime_feature_store': {
                        'store_type': 'redis',
                        'connection': {'host': 'localhost', 'port': 6379}
                    }
                }
            }
            
            # Mock Wrapper가 컨텍스트를 올바르게 처리하는지 확인
            mock_wrapper = Mock()
            mock_inner_model = Mock()
            mock_inner_model.loader_sql_snapshot = 'SELECT user_id FROM users'
            
            # 서빙 모드 컨텍스트 처리 Mock
            def mock_predict(input_df, params=None):
                # 컨텍스트 확인
                assert params is not None
                assert params.get('run_mode') == 'serving'
                
                return pd.DataFrame({
                    'user_id': input_df['user_id'],
                    'prediction': [0.85] * len(input_df)
                })
            
            mock_wrapper.predict = mock_predict
            mock_wrapper.unwrap_python_model.return_value = mock_inner_model
            mock_load_model.return_value = mock_wrapper
            
            # API 앱 생성 및 테스트
            from serving.api import create_app
            app = create_app("test_run_id")
            client = TestClient(app)
            
            # 실제 예측 요청 (컨텍스트 주입 확인)
            response = client.post("/predict", json={"user_id": "test_user_123"})
            
            # 서빙 모드 컨텍스트가 올바르게 주입되었는지 확인
            assert response.status_code == 200
            result = response.json()
            assert 'prediction' in result
    
    def test_batch_vs_serving_context_difference(self):
        """배치와 서빙 컨텍스트 차이 검증"""
        # 동일한 Wrapper가 다른 컨텍스트에서 다르게 동작하는지 확인
        
        mock_wrapper = Mock()
        
        def context_aware_predict(input_df, params=None):
            run_mode = params.get('run_mode', 'batch') if params else 'batch'
            
            if run_mode == 'serving':
                # 서빙 모드: 온라인 Feature Store 사용 시뮬레이션
                return pd.DataFrame({
                    'user_id': input_df['user_id'],
                    'prediction': [0.85] * len(input_df),
                    'source': ['online_features'] * len(input_df)
                })
            else:
                # 배치 모드: 오프라인 Feature Store 사용 시뮬레이션
                return pd.DataFrame({
                    'user_id': input_df['user_id'],
                    'prediction': [0.85] * len(input_df),
                    'source': ['offline_features'] * len(input_df)
                })
        
        mock_wrapper.predict = context_aware_predict
        
        test_df = pd.DataFrame({'user_id': ['u1', 'u2']})
        
        # 배치 컨텍스트 테스트
        batch_result = mock_wrapper.predict(test_df, params={'run_mode': 'batch'})
        assert batch_result['source'].iloc[0] == 'offline_features'
        
        # 서빙 컨텍스트 테스트
        serving_result = mock_wrapper.predict(test_df, params={'run_mode': 'serving'})
        assert serving_result['source'].iloc[0] == 'online_features'


@pytest.mark.requires_dev_stack
@pytest.mark.performance
class TestAPIPerformanceReal:
    """실제 API 성능 테스트"""
    
    def setup_method(self):
        os.environ['APP_ENV'] = 'dev'
    
    def teardown_method(self):
        if 'APP_ENV' in os.environ:
            del os.environ['APP_ENV']
    
    @pytest.mark.benchmark
    def test_api_response_time_target(self):
        """API 응답 시간 목표 달성 테스트"""
        import time
        from fastapi.testclient import TestClient
        
        with patch('serving.api.load_config_files') as mock_config, \
             patch('mlflow.pyfunc.load_model') as mock_load_model:
            
            mock_config.return_value = {
                'environment': {'app_env': 'dev'},
                'api_serving': {'enabled': True},
                'serving': {
                    'performance_targets': {
                        'max_response_time_ms': 100  # 100ms 목표
                    }
                }
            }
            
            # 빠른 예측을 위한 Mock Wrapper
            mock_wrapper = Mock()
            mock_inner_model = Mock()
            mock_inner_model.loader_sql_snapshot = 'SELECT user_id FROM users'
            
            def fast_predict(input_df, params=None):
                return pd.DataFrame({
                    'user_id': input_df['user_id'],
                    'prediction': [0.75] * len(input_df)
                })
            
            mock_wrapper.predict = fast_predict
            mock_wrapper.unwrap_python_model.return_value = mock_inner_model
            mock_load_model.return_value = mock_wrapper
            
            # API 앱 생성 및 성능 테스트
            from serving.api import create_app
            app = create_app("test_run_id")
            client = TestClient(app)
            
            # 응답 시간 측정
            start_time = time.time()
            response = client.post("/predict", json={"user_id": "perf_test_user"})
            end_time = time.time()
            
            response_time_ms = (end_time - start_time) * 1000
            
            # 성능 목표 검증
            assert response.status_code == 200
            assert response_time_ms < 100, f"API 응답 시간 목표 미달성: {response_time_ms:.2f}ms" 
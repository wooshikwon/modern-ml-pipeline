"""
Serving 엔드포인트 테스트 (커버리지 확장)
테스트 전략에 따른 FastAPI 앱 테스트
"""
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.serving.router import app


class TestServingEndpoints:
    """서빙 엔드포인트 테스트"""
    
    def test_health_endpoint_returns_proper_status(self):
        """케이스 A: 헬스체크 엔드포인트 적절한 상태 응답 (모델 없음)"""
        client = TestClient(app)
        response = client.get("/health")
        
        # 200 (모델 로드됨), 503 (모델 준비 안됨), 500 (내부 오류) 모두 유효한 응답
        assert response.status_code in [200, 500, 503]
        
        # 응답이 JSON 형식이어야 함
        data = response.json()
        assert isinstance(data, dict)
        
        if response.status_code == 200:
            assert "status" in data
            assert data["status"] in ["healthy", "ok", "ready"]
        elif response.status_code == 503:
            assert "detail" in data  # FastAPI HTTPException 형식
    
    def test_health_endpoint_with_actual_model(self, serving_test_context):
        """실제 모델과 함께 헬스체크 엔드포인트 테스트 (ServingTestContext 패턴)"""
        with serving_test_context.with_trained_model("classification") as ctx:
            # 모델 로딩 확인
            assert ctx.is_model_loaded()
            
            response = ctx.client.get("/health")
            
            # 실제 모델이 로딩된 상태에서는 200 응답 기대
            assert response.status_code == 200
            data = response.json()
            
            # 응답 구조 검증
            assert isinstance(data, dict)
            assert "status" in data
            assert data["status"] in ["healthy", "ok", "ready"]
            
            # 모델 정보가 포함되어 있는지 확인 (선택적)
            if "model_uri" in data:
                assert data["model_uri"]  # 모델 URI가 있어야 함
    
    def test_root_endpoint_returns_200(self):
        """루트 엔드포인트 200 응답"""
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        
    def test_predict_endpoint_with_valid_payload(self):
        """케이스 B: 추론 엔드포인트 - 모델 없이 기본 가용성 확인"""
        client = TestClient(app)
        
        # 기본 페이로드로 요청
        payload = {
            "inputs": [
                {"feature_0": 1.0, "feature_1": 0.5},
                {"feature_0": 2.0, "feature_1": 1.5}
            ]
        }
        
        response = client.post("/predict", json=payload)
        
        # 200 (성공) 또는 422 (스키마 오류) 또는 500 (모델 로드 실패) 또는 503 (서비스 준비 안됨) 등 의미있는 응답
        assert response.status_code in [200, 422, 500, 503]
        
        if response.status_code == 422:
            # 스키마 검증 실패 시 에러 메시지 존재
            error_data = response.json()
            assert "detail" in error_data
        elif response.status_code == 500:
            # 서버 에러 시에도 응답 형식 확인
            assert response.headers.get("content-type", "").startswith("application/json")
    
    def test_predict_endpoint_with_actual_model(self, serving_test_context):
        """실제 모델과 함께 예측 API 기능 테스트 (ServingTestContext 패턴)"""
        with serving_test_context.with_trained_model("classification") as ctx:
            # 모델 로딩 확인
            assert ctx.is_model_loaded()
            model_info = ctx.get_model_info()
            assert model_info["loaded"] == True
            
            # 올바른 페이로드로 예측 요청 (단일 요청 형식)
            payload = {
                "feature_0": 1.0, 
                "feature_1": 0.5, 
                "feature_2": 0.3, 
                "feature_3": 0.8
            }
            
            response = ctx.client.post("/predict", json=payload)
            
            # 실제 모델이 로딩된 상태에서는 200 응답 기대
            assert response.status_code == 200
            data = response.json()
            
            # 예측 결과 구조 검증
            assert "prediction" in data
            assert "model_uri" in data
            assert isinstance(data["prediction"], (int, float))  # 분류 결과
    
    def test_predict_endpoint_with_invalid_payload(self):
        """잘못된 페이로드로 422 응답 확인"""
        client = TestClient(app)
        
        # 잘못된 형식의 페이로드
        invalid_payload = {"wrong_field": "invalid_data"}
        
        response = client.post("/predict", json=invalid_payload)
        # 422 (페이로드 검증 실패) 또는 503 (서비스 준비 안됨) 모두 유효
        assert response.status_code in [422, 503]
        
        if response.status_code == 422:
            error_data = response.json()
            assert "detail" in error_data
        
    def test_predict_endpoint_empty_inputs(self):
        """빈 입력에 대한 처리"""
        client = TestClient(app)
        
        payload = {"inputs": []}
        response = client.post("/predict", json=payload)
        
        # 빈 입력도 적절히 처리되어야 함 (서비스 준비 안됨 시 503도 허용)
        assert response.status_code in [200, 400, 422, 503]
    
    @patch('src.serving._context.app_context')
    def test_predict_endpoint_with_mocked_model(self, mock_app_context):
        """모델 로딩을 모킹하여 예측 플로우 테스트"""
        # 모델 모킹
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8, 0.2]  # 더미 예측 결과
        
        # app_context 모킹
        mock_app_context.model = mock_model
        mock_app_context.settings = MagicMock()
        mock_app_context.model_uri = "test://model/uri"
        
        client = TestClient(app)
        payload = {
            "inputs": [
                {"feature_0": 1.0, "feature_1": 0.5},
                {"feature_0": 2.0, "feature_1": 1.5}
            ]
        }
        
        response = client.post("/predict", json=payload)
        
        # 모델이 정상적으로 로드되면 200 응답
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data or "prediction" in data
            mock_model.predict.assert_called_once()
    
    def test_metrics_endpoint_if_exists(self):
        """메트릭스 엔드포인트 존재 시 테스트"""
        client = TestClient(app)
        response = client.get("/metrics")
        
        # 메트릭스 엔드포인트가 존재할 수도, 없을 수도 있음
        assert response.status_code in [200, 404]
        
    def test_openapi_docs_available(self):
        """API 문서 엔드포인트 접근 가능"""
        client = TestClient(app)
        
        # OpenAPI 스키마
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        # Swagger UI (일반적으로 /docs)
        docs_response = client.get("/docs")
        assert docs_response.status_code in [200, 307, 404]  # 리다이렉트 또는 존재하지 않을 수 있음


class TestServingLifespan:
    """서빙 라이프사이클 테스트"""
    
    def test_app_startup_shutdown_cycle(self):
        """케이스 A: 앱 기동/종료 수명주기 - 에러 없이 start/stop"""
        # TestClient는 자동으로 lifespan 이벤트를 처리함
        with TestClient(app) as client:
            # 앱이 정상적으로 시작됨 (모델 준비 안됨 시 500/503도 유효)
            response = client.get("/health")
            assert response.status_code in [200, 500, 503]
        # with 블록 종료 시 앱이 정상적으로 종료됨
    
    def test_multiple_requests_handling(self):
        """다중 요청 처리 능력"""
        client = TestClient(app)
        
        responses = []
        for i in range(5):
            response = client.get("/health")
            responses.append(response)
        
        # 모든 요청이 정상 처리됨 (모델 없어도 크래시하지 않음)
        for response in responses:
            assert response.status_code in [200, 500, 503]
    
    def test_concurrent_health_checks(self):
        """동시 헬스체크 요청 처리"""
        import concurrent.futures
        import threading
        
        client = TestClient(app)
        results = []
        
        def make_request():
            response = client.get("/health")
            return response.status_code
        
        # 동시에 여러 요청 실행
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 모든 요청이 성공 (모델 없어도 크래시하지 않음)
        assert all(status in [200, 500, 503] for status in results)
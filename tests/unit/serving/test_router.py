"""
FastAPI router configuration and endpoint testing
Follows tests/README.md philosophy with Context classes
Tests for src/serving/router.py

Author: Phase 2B Development
Date: 2025-09-13
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, Any

from src.serving.router import app, _register_dynamic_routes_if_needed, run_api_server
from src.serving._context import app_context
from src.serving.schemas import (
    HealthCheckResponse,
    ModelMetadataResponse,
    OptimizationHistoryResponse,
    MinimalPredictionResponse,
    HyperparameterOptimizationInfo,
    TrainingMethodologyInfo,
)


class TestFastAPIAppConfiguration:
    """FastAPI 앱 설정 테스트 - Context 클래스 기반"""

    def test_fastapi_app_basic_configuration(self, component_test_context):
        """기본 FastAPI 앱 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Test FastAPI app configuration (Real Object Testing)
            assert app.title == "Modern ML Pipeline API"
            assert app.description == "MMP 모델 서빙 API"
            assert app.version == "17.0.0"

            # Verify app is properly configured
            assert app is not None
            assert hasattr(app, 'router')
            # Lifespan is configured during app initialization, not as an attribute

    def test_fastapi_app_routes_existence(self, component_test_context):
        """FastAPI 앱 라우트 존재 확인 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Get all registered routes
            routes = [route.path for route in app.routes if hasattr(route, 'path')]

            # Expected static routes
            expected_routes = ["/", "/health", "/predict", "/model/metadata", "/model/optimization", "/model/schema"]

            # Verify routes exist (some may be duplicated due to dynamic registration)
            for expected_route in expected_routes:
                route_exists = any(expected_route == route for route in routes)
                assert route_exists, f"Route {expected_route} not found in {routes}"

    def test_fastapi_app_route_methods(self, component_test_context):
        """FastAPI 앱 라우트 메소드 확인 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Check route methods
            route_methods = {}
            for route in app.routes:
                if hasattr(route, 'path') and hasattr(route, 'methods'):
                    route_methods[route.path] = route.methods

            # Verify specific route methods
            assert "GET" in route_methods.get("/", set())
            assert "GET" in route_methods.get("/health", set())
            assert "POST" in route_methods.get("/predict", set())
            assert "GET" in route_methods.get("/model/metadata", set())


class TestStaticEndpoints:
    """정적 엔드포인트 테스트"""

    def test_root_endpoint_without_model(self, component_test_context):
        """루트 엔드포인트 (모델 없음) 테스트"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                # Clear app_context for this test
                original_model = app_context.model
                original_uri = app_context.model_uri
                app_context.model = None
                app_context.model_uri = ""

                try:
                    response = client.get("/")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["message"] == "Modern ML Pipeline API"
                    assert data["status"] == "error"  # No model loaded
                    assert data["model_uri"] == ""

                finally:
                    # Restore original state
                    app_context.model = original_model
                    app_context.model_uri = original_uri

    def test_root_endpoint_with_model(self, component_test_context):
        """루트 엔드포인트 (모델 있음) 테스트"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                # Set up mock model in app_context
                original_model = app_context.model
                original_uri = app_context.model_uri

                mock_model = Mock()
                test_uri = "runs:/test-run/model"
                app_context.model = mock_model
                app_context.model_uri = test_uri

                try:
                    response = client.get("/")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["message"] == "Modern ML Pipeline API"
                    assert data["status"] == "ready"  # Model loaded
                    assert data["model_uri"] == test_uri

                finally:
                    # Restore original state
                    app_context.model = original_model
                    app_context.model_uri = original_uri

    def test_health_endpoint_success(self, component_test_context):
        """Health 체크 엔드포인트 성공 테스트"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                # Mock the health handler to return success
                with patch('src.serving.router.handlers.health') as mock_health:
                    mock_health.return_value = HealthCheckResponse(
                        status="healthy",
                        model_uri="runs:/test-run/model",
                        model_name="test_model"
                    )

                    response = client.get("/health")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "healthy"
                    assert data["model_uri"] == "runs:/test-run/model"
                    assert data["model_name"] == "test_model"

    def test_health_endpoint_error_handling(self, component_test_context):
        """Health 체크 엔드포인트 에러 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                # Mock the health handler to raise exception
                with patch('src.serving.router.handlers.health') as mock_health:
                    mock_health.side_effect = Exception("Health check failed")

                    response = client.get("/health")

                    assert response.status_code == 500
                    data = response.json()
                    assert "Health check failed" in data["detail"]

    def test_model_metadata_endpoint_success(self, component_test_context):
        """모델 메타데이터 엔드포인트 성공 테스트"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                # Mock the metadata handler
                with patch('src.serving.router.handlers.get_model_metadata') as mock_metadata:
                    mock_metadata.return_value = ModelMetadataResponse(
                        model_uri="runs:/test-run/model",
                        model_class_path="sklearn.ensemble.RandomForestClassifier",
                        hyperparameter_optimization=HyperparameterOptimizationInfo(
                            enabled=True,
                            engine="optuna",
                            best_params={"n_estimators": 100},
                            best_score=0.95
                        ),
                        training_methodology=TrainingMethodologyInfo(
                            train_test_split_method="stratified",
                            train_ratio=0.8
                        )
                    )

                    response = client.get("/model/metadata")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["model_uri"] == "runs:/test-run/model"
                    assert data["hyperparameter_optimization"]["enabled"] is True

    def test_model_optimization_endpoint_success(self, component_test_context):
        """모델 최적화 히스토리 엔드포인트 테스트"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                # Mock the optimization handler
                with patch('src.serving.router.handlers.get_optimization_history') as mock_opt:
                    mock_opt.return_value = OptimizationHistoryResponse(
                        enabled=True,
                        optimization_history=[{"trial": 1, "value": 0.90}],
                        search_space={"n_estimators": [50, 200]}
                    )

                    response = client.get("/model/optimization")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["enabled"] is True
                    assert len(data["optimization_history"]) > 0

    def test_model_schema_endpoint_success(self, component_test_context):
        """모델 스키마 엔드포인트 테스트"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                # Mock the schema handler
                with patch('src.serving.router.handlers.get_api_schema') as mock_schema:
                    test_schema = {
                        "input_schema": {"feature1": "float", "feature2": "string"},
                        "output_schema": {"prediction": "float", "probability": "float"}
                    }
                    mock_schema.return_value = test_schema

                    response = client.get("/model/schema")

                    assert response.status_code == 200
                    data = response.json()
                    assert "input_schema" in data
                    assert "output_schema" in data


class TestPredictEndpoint:
    """Predict 엔드포인트 테스트"""

    def test_predict_endpoint_model_not_ready(self, component_test_context):
        """Predict 엔드포인트 - 모델 준비되지 않음 테스트"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                # Clear app_context
                original_model = app_context.model
                original_settings = app_context.settings
                app_context.model = None
                app_context.settings = None

                try:
                    test_request = {"feature1": 1.0, "feature2": "test"}
                    response = client.post("/predict", json=test_request)

                    assert response.status_code == 503
                    data = response.json()
                    assert "모델이 준비되지 않았습니다" in data["detail"]

                finally:
                    # Restore original state
                    app_context.model = original_model
                    app_context.settings = original_settings

    def test_predict_endpoint_success(self, component_test_context):
        """Predict 엔드포인트 성공 테스트"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                # Set up app_context with model and settings
                original_model = app_context.model
                original_settings = app_context.settings

                app_context.model = Mock()
                app_context.settings = ctx.settings

                try:
                    # Mock the predict handler
                    with patch('src.serving.router.handlers.predict') as mock_predict:
                        mock_predict.return_value = {
                            "prediction": 0.85,
                            "model_uri": "runs:/test-run/model"
                        }

                        test_request = {"feature1": 1.0, "feature2": "test"}
                        response = client.post("/predict", json=test_request)

                        assert response.status_code == 200
                        data = response.json()
                        assert "prediction" in data
                        assert data["prediction"] == 0.85

                finally:
                    # Restore original state
                    app_context.model = original_model
                    app_context.settings = original_settings

    def test_predict_endpoint_handler_exception(self, component_test_context):
        """Predict 엔드포인트 핸들러 예외 테스트"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                # Set up app_context
                original_model = app_context.model
                original_settings = app_context.settings

                app_context.model = Mock()
                app_context.settings = ctx.settings

                try:
                    # Mock the predict handler to raise exception
                    with patch('src.serving.router.handlers.predict') as mock_predict:
                        mock_predict.side_effect = Exception("Prediction failed")

                        test_request = {"feature1": 1.0}
                        response = client.post("/predict", json=test_request)

                        assert response.status_code == 500
                        data = response.json()
                        assert "Prediction failed" in data["detail"]

                finally:
                    # Restore original state
                    app_context.model = original_model
                    app_context.settings = original_settings


class TestDynamicRouteRegistration:
    """동적 라우트 등록 테스트"""

    def test_register_dynamic_routes_if_needed_no_model(self, component_test_context):
        """동적 라우트 등록 - 모델 없음 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Clear PredictionRequest schema
            original_request = app_context.PredictionRequest
            app_context.PredictionRequest = None

            try:
                # Should not register routes when PredictionRequest is None
                initial_route_count = len(app.routes)
                _register_dynamic_routes_if_needed()
                final_route_count = len(app.routes)

                # Route count should not change
                assert final_route_count == initial_route_count

            finally:
                # Restore original state
                app_context.PredictionRequest = original_request

    def test_register_dynamic_routes_if_needed_with_model(self, component_test_context):
        """동적 라우트 등록 - 모델 있음 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Set up PredictionRequest schema
            from pydantic import create_model
            test_schema = create_model(
                "TestPredictionRequest",
                feature1=(float, ...),
                feature2=(str, ...)
            )

            original_request = app_context.PredictionRequest
            app_context.PredictionRequest = test_schema

            try:
                # Get initial state
                initial_routes = [route.path for route in app.routes if hasattr(route, 'path')]

                # Register dynamic routes
                _register_dynamic_routes_if_needed()

                # Should have prediction route
                final_routes = [route.path for route in app.routes if hasattr(route, 'path')]
                predict_routes = [r for r in final_routes if r == "/predict"]

                # Verify predict route exists (may be multiple due to static + dynamic)
                assert len(predict_routes) >= 1

            finally:
                # Restore original state
                app_context.PredictionRequest = original_request

    def test_register_dynamic_routes_idempotent(self, component_test_context):
        """동적 라우트 등록 멱등성 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Set up schema
            from pydantic import create_model
            test_schema = create_model("IdempotentTestRequest", field=(int, ...))

            original_request = app_context.PredictionRequest
            app_context.PredictionRequest = test_schema

            try:
                # Register multiple times
                initial_route_count = len(app.routes)
                _register_dynamic_routes_if_needed()
                first_call_count = len(app.routes)
                _register_dynamic_routes_if_needed()
                second_call_count = len(app.routes)

                # Should not keep adding routes
                assert first_call_count == second_call_count

            finally:
                # Restore original state
                app_context.PredictionRequest = original_request


class TestRunAPIServer:
    """run_api_server 함수 테스트"""

    def test_run_api_server_disabled_serving(self, component_test_context):
        """서빙 비활성화된 환경에서 API 서버 실행 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings with serving disabled
            mock_settings = Mock()
            mock_settings.serving = Mock()
            mock_settings.serving.enabled = False
            mock_settings.environment = Mock()
            mock_settings.environment.env_name = "production"

            # Mock setup_api_context and uvicorn
            with patch('src.serving.router.setup_api_context') as mock_setup, \
                 patch('src.serving.router.uvicorn.run') as mock_run:

                # Should return early without starting server
                run_api_server(mock_settings, "test-run-id")

                # When serving is disabled, should return early without calling anything
                mock_setup.assert_not_called()  # setup_api_context should NOT be called
                # uvicorn.run should not be called
                mock_run.assert_not_called()

    def test_run_api_server_enabled_serving(self, component_test_context):
        """서빙 활성화된 환경에서 API 서버 실행 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings with serving enabled
            mock_settings = Mock()
            mock_settings.serving = Mock()
            mock_settings.serving.enabled = True

            # Mock setup_api_context and uvicorn
            with patch('src.serving.router.setup_api_context') as mock_setup, \
                 patch('src.serving.router.uvicorn.run') as mock_run:

                run_api_server(mock_settings, "test-run-id", host="127.0.0.1", port=9000)

                # Both functions should be called
                mock_setup.assert_called_once_with(run_id="test-run-id", settings=mock_settings)
                mock_run.assert_called_once_with(app, host="127.0.0.1", port=9000)

    def test_run_api_server_no_serving_config(self, component_test_context):
        """서빙 설정 없는 환경에서 API 서버 실행 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings without serving config
            mock_settings = Mock()
            mock_settings.serving = None

            # Mock setup_api_context and uvicorn
            with patch('src.serving.router.setup_api_context') as mock_setup, \
                 patch('src.serving.router.uvicorn.run') as mock_run:

                run_api_server(mock_settings, "test-run-id")

                # Should proceed normally (serving enabled by default)
                mock_setup.assert_called_once()
                mock_run.assert_called_once()

    def test_run_api_server_custom_host_port(self, component_test_context):
        """커스텀 호스트/포트로 API 서버 실행 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_settings = Mock()
            mock_settings.serving = None  # Default enabled

            with patch('src.serving.router.setup_api_context'), \
                 patch('src.serving.router.uvicorn.run') as mock_run:

                run_api_server(mock_settings, "test-run", host="192.168.1.100", port=8080)

                # Verify custom host and port passed to uvicorn
                mock_run.assert_called_once_with(app, host="192.168.1.100", port=8080)


class TestRouterIntegration:
    """라우터 통합 테스트"""

    def test_complete_router_workflow(self, component_test_context, mlflow_test_context):
        """완전한 라우터 워크플로우 테스트"""
        with component_test_context.classification_stack() as comp_ctx:
            with mlflow_test_context.for_classification(experiment="router_integration") as mlflow_ctx:
                with TestClient(app) as client:
                    # Step 1: Set up app context
                    original_model = app_context.model
                    original_settings = app_context.settings
                    original_uri = app_context.model_uri

                    mock_model = Mock()
                    app_context.model = mock_model
                    app_context.settings = mlflow_ctx.settings
                    app_context.model_uri = "runs:/integration-test/model"

                    try:
                        # Step 2: Test root endpoint
                        response = client.get("/")
                        assert response.status_code == 200
                        root_data = response.json()
                        assert root_data["status"] == "ready"

                        # Step 3: Test health endpoint with mocked handler
                        with patch('src.serving.router.handlers.health') as mock_health:
                            mock_health.return_value = HealthCheckResponse(
                                status="healthy",
                                model_uri="runs:/integration-test/model",
                                model_name="integration_model"
                            )

                            health_response = client.get("/health")
                            assert health_response.status_code == 200
                            health_data = health_response.json()
                            assert health_data["status"] == "healthy"

                        # Step 4: Test prediction endpoint with mocked handler
                        with patch('src.serving.router.handlers.predict') as mock_predict:
                            mock_predict.return_value = {
                                "prediction": 0.92,
                                "model_uri": "runs:/integration-test/model"
                            }

                            pred_response = client.post("/predict", json={"feature1": 2.5})
                            assert pred_response.status_code == 200
                            pred_data = pred_response.json()
                            assert pred_data["prediction"] == 0.92

                    finally:
                        # Restore original state
                        app_context.model = original_model
                        app_context.settings = original_settings
                        app_context.model_uri = original_uri

    def test_router_error_resilience(self, component_test_context):
        """라우터 오류 복원력 테스트"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                # Test multiple error scenarios

                # 1. All handlers fail
                with patch('src.serving.router.handlers.health', side_effect=Exception("Health failed")), \
                     patch('src.serving.router.handlers.get_model_metadata', side_effect=Exception("Metadata failed")), \
                     patch('src.serving.router.handlers.get_optimization_history', side_effect=Exception("Opt failed")):

                    # Health endpoint should return 500
                    health_response = client.get("/health")
                    assert health_response.status_code == 500

                    # Metadata endpoint should return 500
                    metadata_response = client.get("/model/metadata")
                    assert metadata_response.status_code == 500

                    # Optimization endpoint should return 500
                    opt_response = client.get("/model/optimization")
                    assert opt_response.status_code == 500

                    # Root endpoint should still work (no handler dependency)
                    root_response = client.get("/")
                    assert root_response.status_code == 200
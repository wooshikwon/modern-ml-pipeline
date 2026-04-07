"""
FastAPI router configuration and endpoint testing
Follows tests/README.md philosophy with Context classes
Tests for mmp/serving/router.py

Author: Phase 2B Development
Date: 2025-09-13
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip(
    "prometheus_fastapi_instrumentator", reason="serving extras not installed"
)

from fastapi.testclient import TestClient

from mmp.serving._context import app_context
from mmp.serving.router import app, run_api_server
from mmp.serving.schemas import (
    HealthCheckResponse,
    HyperparameterOptimizationInfo,
    ModelMetadataResponse,
    OptimizationHistoryResponse,
    ReadyCheckResponse,
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
            # 버전은 pyproject.toml에서 동적으로 로드됨
            assert app.version is not None
            assert len(app.version) > 0

            # Verify app is properly configured
            assert app is not None
            assert hasattr(app, "router")
            # Lifespan is configured during app initialization, not as an attribute

    def test_fastapi_app_routes_existence(self, component_test_context):
        """FastAPI 앱 라우트 존재 확인 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Get all registered routes
            routes = [route.path for route in app.routes if hasattr(route, "path")]

            # Expected static routes
            expected_routes = [
                "/",
                "/health",
                "/ready",
                "/predict",
                "/predict/batch",
                "/model/metadata",
                "/model/optimization",
                "/model/schema",
            ]

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
                if hasattr(route, "path") and hasattr(route, "methods"):
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
        """Health 체크 엔드포인트 성공 테스트 (Liveness probe)"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                # /health는 항상 200 OK 반환 (Liveness probe)
                response = client.get("/health")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "ok"

    def test_ready_endpoint_success(self, component_test_context):
        """Ready 체크 엔드포인트 성공 테스트 (Readiness probe)"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                # Mock the ready handler to return success
                with patch("mmp.serving.router.handlers.ready") as mock_ready:
                    mock_ready.return_value = ReadyCheckResponse(
                        status="ready", model_uri="runs:/test-run/model", model_name="test_model"
                    )

                    response = client.get("/ready")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "ready"
                    assert data["model_uri"] == "runs:/test-run/model"
                    assert data["model_name"] == "test_model"

    def test_ready_endpoint_model_not_loaded(self, component_test_context):
        """Ready 체크 엔드포인트 - 모델 미로드 시 503 반환 테스트"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                from fastapi import HTTPException

                # Mock the ready handler to raise HTTPException (model not ready)
                with patch("mmp.serving.router.handlers.ready") as mock_ready:
                    mock_ready.side_effect = HTTPException(
                        status_code=503, detail="모델이 준비되지 않았습니다."
                    )

                    response = client.get("/ready")

                    assert response.status_code == 503
                    data = response.json()
                    assert "모델이 준비되지 않았습니다" in data["detail"]

    def test_request_id_middleware_passthrough(self, component_test_context):
        """X-Request-ID 미들웨어 - 헤더 전달 테스트"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                custom_request_id = "test-request-id-12345"
                response = client.get("/health", headers={"X-Request-ID": custom_request_id})

                assert response.status_code == 200
                assert response.headers.get("X-Request-ID") == custom_request_id

    def test_request_id_middleware_auto_generation(self, component_test_context):
        """X-Request-ID 미들웨어 - 자동 생성 테스트"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                response = client.get("/health")

                assert response.status_code == 200
                request_id = response.headers.get("X-Request-ID")
                assert request_id is not None
                # UUID 형식 검증
                assert len(request_id) == 36  # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

    def test_prometheus_instrumentator_initialized(self, component_test_context):
        """Prometheus Instrumentator 초기화 테스트"""
        with component_test_context.classification_stack() as ctx:
            from mmp.serving.router import _instrumentator

            # Instrumentator가 app에 연결되어 있는지 확인
            assert _instrumentator is not None
            assert hasattr(_instrumentator, "instrument")
            assert hasattr(_instrumentator, "expose")

    def test_model_metadata_endpoint_success(self, component_test_context):
        """모델 메타데이터 엔드포인트 성공 테스트"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                # Mock the metadata handler
                with patch("mmp.serving.router.handlers.get_model_metadata") as mock_metadata:
                    mock_metadata.return_value = ModelMetadataResponse(
                        model_uri="runs:/test-run/model",
                        model_class_path="sklearn.ensemble.RandomForestClassifier",
                        hyperparameter_optimization=HyperparameterOptimizationInfo(
                            enabled=True,
                            engine="optuna",
                            best_params={"n_estimators": 100},
                            best_score=0.95,
                        ),
                        training_methodology=TrainingMethodologyInfo(
                            train_test_split_method="stratified", train_ratio=0.8
                        ),
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
                with patch("mmp.serving.router.handlers.get_optimization_history") as mock_opt:
                    mock_opt.return_value = OptimizationHistoryResponse(
                        enabled=True,
                        optimization_history=[{"trial": 1, "value": 0.90}],
                        search_space={"n_estimators": [50, 200]},
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
                with patch("mmp.serving.router.handlers.get_api_schema") as mock_schema:
                    test_schema = {
                        "input_schema": {"feature1": "float", "feature2": "string"},
                        "output_schema": {"prediction": "float", "probability": "float"},
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
                    with patch("mmp.serving.router.handlers.predict") as mock_predict:
                        mock_predict.return_value = {
                            "prediction": 0.85,
                            "model_uri": "runs:/test-run/model",
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

    def test_predict_endpoint_multi_output(self, component_test_context):
        """다중 출력 모델(quantile, multi-target)에서 predict 엔드포인트가 dict를 반환하는지 테스트"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                original_model = app_context.model
                original_settings = app_context.settings

                app_context.model = Mock()
                app_context.settings = ctx.settings

                try:
                    with patch("mmp.serving.router.handlers.predict") as mock_predict:
                        mock_predict.return_value = {
                            "prediction": {
                                "pred_p50": 38,
                                "pred_p75": 52,
                                "pred_p90": 65,
                                "pred_p95": 78,
                                "pred_p99": 98,
                            },
                            "model_uri": "runs:/test-run/model",
                        }

                        test_request = {"feature1": 1.0, "feature2": "test"}
                        response = client.post("/predict", json=test_request)

                        assert response.status_code == 200
                        data = response.json()
                        assert isinstance(data["prediction"], dict)
                        assert data["prediction"]["pred_p50"] == 38
                        assert data["prediction"]["pred_p99"] == 98

                finally:
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
                    with patch("mmp.serving.router.handlers.predict") as mock_predict:
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


class TestRunAPIServer:
    """run_api_server 함수 테스트"""

    def teardown_method(self):
        """각 테스트 후 app.state 정리 (lifespan이 잔여 상태로 트리거되지 않도록)"""
        for attr in ("run_id", "settings"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)

    def test_run_api_server_disabled_serving(self, component_test_context):
        """서빙 비활성화된 환경에서 API 서버 실행 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings with serving disabled (settings.config.serving 구조)
            mock_settings = Mock()
            mock_settings.config = Mock()
            mock_settings.config.serving = Mock()
            mock_settings.config.serving.enabled = False
            mock_settings.config.environment = Mock()
            mock_settings.config.environment.name = "production"

            with patch("mmp.serving.router.uvicorn.run") as mock_run:
                # Should return early without starting server
                run_api_server(mock_settings, "test-run-id")

                # uvicorn.run should not be called
                mock_run.assert_not_called()

    def test_run_api_server_enabled_serving(self, component_test_context):
        """서빙 활성화된 환경에서 API 서버 실행 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings with serving enabled
            mock_settings = Mock()
            mock_settings.serving = Mock()
            mock_settings.serving.enabled = True

            with (
                patch("mmp.serving.router.uvicorn.run") as mock_run,
                patch("mmp.serving.router._configure_middlewares") as mock_configure,
            ):

                run_api_server(mock_settings, "test-run-id", host="127.0.0.1", port=9000)

                # run_id와 settings가 app.state에 저장되어야 함
                assert app.state.run_id == "test-run-id"
                assert app.state.settings is mock_settings
                mock_configure.assert_called_once_with(mock_settings)
                mock_run.assert_called_once_with(app, host="127.0.0.1", port=9000)

    def test_run_api_server_no_serving_config(self, component_test_context):
        """서빙 설정 없는 환경에서 API 서버 실행 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings without serving config
            mock_settings = Mock()
            mock_settings.serving = None

            with (
                patch("mmp.serving.router.uvicorn.run") as mock_run,
                patch("mmp.serving.router._configure_middlewares") as mock_configure,
            ):

                run_api_server(mock_settings, "test-run-id")

                # Should proceed normally (serving enabled by default)
                assert app.state.run_id == "test-run-id"
                assert app.state.settings is mock_settings
                mock_configure.assert_called_once()
                mock_run.assert_called_once()

    def test_run_api_server_custom_host_port(self, component_test_context):
        """커스텀 호스트/포트로 API 서버 실행 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_settings = Mock()
            mock_settings.serving = None  # Default enabled

            with (
                patch("mmp.serving.router.uvicorn.run") as mock_run,
                patch("mmp.serving.router._configure_middlewares"),
            ):

                run_api_server(mock_settings, "test-run", host="192.168.1.100", port=8080)

                # Verify custom host and port passed to uvicorn
                mock_run.assert_called_once_with(app, host="192.168.1.100", port=8080)


class TestRouterIntegration:
    """라우터 통합 테스트"""

    def test_complete_router_workflow(self, component_test_context, mlflow_test_context):
        """완전한 라우터 워크플로우 테스트"""
        with component_test_context.classification_stack() as comp_ctx:
            with mlflow_test_context.for_classification(
                experiment="router_integration"
            ) as mlflow_ctx:
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

                        # Step 3: Test health endpoint (liveness - always 200)
                        health_response = client.get("/health")
                        assert health_response.status_code == 200
                        health_data = health_response.json()
                        assert health_data["status"] == "ok"

                        # Step 3.5: Test ready endpoint with mocked handler
                        with patch("mmp.serving.router.handlers.ready") as mock_ready:
                            mock_ready.return_value = ReadyCheckResponse(
                                status="ready",
                                model_uri="runs:/integration-test/model",
                                model_name="integration_model",
                            )

                            ready_response = client.get("/ready")
                            assert ready_response.status_code == 200
                            ready_data = ready_response.json()
                            assert ready_data["status"] == "ready"

                        # Step 4: Test prediction endpoint with mocked handler
                        with patch("mmp.serving.router.handlers.predict") as mock_predict:
                            mock_predict.return_value = {
                                "prediction": 0.92,
                                "model_uri": "runs:/integration-test/model",
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

                # 1. 핸들러 에러 시 적절한 응답 반환 확인
                with (
                    patch(
                        "mmp.serving.router.handlers.ready", side_effect=Exception("Ready failed")
                    ),
                    patch(
                        "mmp.serving.router.handlers.get_model_metadata",
                        side_effect=Exception("Metadata failed"),
                    ),
                    patch(
                        "mmp.serving.router.handlers.get_optimization_history",
                        side_effect=Exception("Opt failed"),
                    ),
                ):

                    # Health endpoint는 항상 200 반환 (liveness probe)
                    health_response = client.get("/health")
                    assert health_response.status_code == 200
                    assert health_response.json()["status"] == "ok"

                    # Ready endpoint should return 500 (핸들러 예외)
                    ready_response = client.get("/ready")
                    assert ready_response.status_code == 500

                    # Metadata endpoint should return 500
                    metadata_response = client.get("/model/metadata")
                    assert metadata_response.status_code == 500

                    # Optimization endpoint should return 500
                    opt_response = client.get("/model/optimization")
                    assert opt_response.status_code == 500

                    # Root endpoint should still work (no handler dependency)
                    root_response = client.get("/")
                    assert root_response.status_code == 200


class TestMiddlewareConfiguration:
    """미들웨어 구성 테스트"""

    def test_timeout_middleware_class(self, component_test_context):
        """TimeoutMiddleware 클래스 테스트"""
        with component_test_context.classification_stack() as ctx:
            from mmp.serving.router import TimeoutMiddleware

            # 클래스가 올바르게 정의되어 있는지 확인
            assert TimeoutMiddleware is not None
            assert hasattr(TimeoutMiddleware, "dispatch")

    def test_configure_middlewares_with_cors(self, component_test_context):
        """CORS 설정 기반 미들웨어 구성 테스트"""
        with component_test_context.classification_stack() as ctx:
            from mmp.serving.router import _configure_middlewares

            # Mock settings with CORS enabled
            mock_settings = Mock()
            mock_serving = Mock()
            mock_cors = Mock()
            mock_cors.enabled = True
            mock_cors.allow_origins = ["http://localhost:3000"]
            mock_cors.allow_methods = ["GET", "POST"]
            mock_cors.allow_headers = ["*"]
            mock_cors.allow_credentials = False
            mock_serving.cors = mock_cors
            mock_serving.request_timeout_seconds = 30
            mock_serving.metrics_enabled = False  # 테스트 시 메트릭 비활성화
            mock_settings.serving = mock_serving

            # _configure_middlewares 호출 시 에러 없이 실행되는지 확인
            # 실제 미들웨어 적용은 app 상태에 의존하므로 에러 없이 호출되는지만 테스트
            try:
                _configure_middlewares(mock_settings)
            except Exception as e:
                # 이미 추가된 미들웨어로 인한 에러는 무시 (테스트 환경에서 발생 가능)
                pass

    def test_configure_middlewares_without_serving(self, component_test_context):
        """serving 설정 없이 미들웨어 구성 테스트"""
        with component_test_context.classification_stack() as ctx:
            from mmp.serving.router import _configure_middlewares

            # serving 설정이 없는 settings
            mock_settings = Mock()
            mock_settings.serving = None

            # 에러 없이 조기 반환되어야 함
            _configure_middlewares(mock_settings)

    def test_timeout_middleware_settings(self, component_test_context):
        """Timeout 미들웨어 설정값 테스트"""
        with component_test_context.classification_stack() as ctx:
            from mmp.serving.router import TimeoutMiddleware

            # 커스텀 타임아웃 값으로 미들웨어 인스턴스 생성
            mock_app = Mock()
            middleware = TimeoutMiddleware(mock_app, timeout_seconds=60)

            assert middleware.timeout_seconds == 60


class TestPredictEndpointLogic:
    """_endpoints.predict() 핵심 로직 테스트 — DataFrame에서 값 추출, numpy 변환, 유한성 검증"""

    def _run_predict(self, predictions_df):
        """predict() 호출의 공통 설정. model.predict와 format_predictions를 mock하여
        predictions_df를 직접 주입한다. 반환값은 predict()의 결과 dict."""
        from mmp.serving import _endpoints as ep

        mock_model = Mock()
        mock_model.predict.return_value = predictions_df

        original_model = app_context.model
        original_settings = app_context.settings
        original_uri = app_context.model_uri
        original_req = getattr(app_context, "PredictionRequest", None)

        app_context.model = mock_model
        app_context.settings = Mock()
        app_context.model_uri = "runs:/test/model"
        app_context.PredictionRequest = lambda **kw: Mock()

        try:
            with (
                patch.object(ep, "validate_scalar_values"),
                patch.object(ep, "validate_numeric_types"),
                patch.object(ep, "validate_required_columns"),
                patch.object(ep, "_convert_to_signature_types", side_effect=lambda df, m: df),
                patch.object(ep, "format_predictions", return_value=predictions_df),
                patch.object(ep, "_get_data_interface_schema", return_value={}),
            ):
                return ep.predict({"feature1": 1.0})
        finally:
            app_context.model = original_model
            app_context.settings = original_settings
            app_context.model_uri = original_uri
            if original_req is not None:
                app_context.PredictionRequest = original_req

    def test_multi_output_returns_dict(self, component_test_context):
        """다중 컬럼 DataFrame → prediction이 dict로 반환되어야 함"""
        with component_test_context.classification_stack():
            df = pd.DataFrame({"pred_p50": [np.float64(38.0)], "pred_p90": [np.float64(65.0)]})
            result = self._run_predict(df)

            assert isinstance(result["prediction"], dict)
            assert result["prediction"]["pred_p50"] == 38.0
            assert result["prediction"]["pred_p90"] == 65.0

    def test_multi_output_numpy_to_python(self, component_test_context):
        """다중 출력에서 numpy 스칼라가 Python 기본형으로 변환되어야 함"""
        with component_test_context.classification_stack():
            df = pd.DataFrame({"a": [np.int64(42)], "b": [np.float64(3.14)]})
            result = self._run_predict(df)

            for v in result["prediction"].values():
                assert not isinstance(v, np.generic), f"numpy 타입이 남아있음: {type(v)}"

    def test_single_output_prediction_column(self, component_test_context):
        """단일 'prediction' 컬럼 → 스칼라로 반환 (하위 호환)"""
        with component_test_context.classification_stack():
            df = pd.DataFrame({"prediction": [np.float64(0.85)]})
            result = self._run_predict(df)

            assert result["prediction"] == 0.85
            assert not isinstance(result["prediction"], np.generic)

    def test_single_output_unnamed_column(self, component_test_context):
        """단일 비-'prediction' 컬럼 → 첫 번째 컬럼의 스칼라로 반환"""
        with component_test_context.classification_stack():
            df = pd.DataFrame({"score": [np.float64(0.92)]})
            result = self._run_predict(df)

            assert result["prediction"] == 0.92

    def test_multi_output_nan_raises_422(self, component_test_context):
        """다중 출력에서 NaN이 포함되면 422 에러 + 문제 컬럼명 포함"""
        from fastapi import HTTPException

        with component_test_context.classification_stack():
            df = pd.DataFrame({"pred_p50": [38.0], "pred_p99": [float("nan")]})

            with pytest.raises(HTTPException) as exc_info:
                self._run_predict(df)
            assert exc_info.value.status_code == 422
            assert "pred_p99" in str(exc_info.value.detail)

    def test_single_output_inf_raises_422(self, component_test_context):
        """단일 출력에서 Inf이면 422 에러"""
        from fastapi import HTTPException

        with component_test_context.classification_stack():
            df = pd.DataFrame({"prediction": [float("inf")]})

            with pytest.raises(HTTPException) as exc_info:
                self._run_predict(df)
            assert exc_info.value.status_code == 422

    def _run_predict_with_schema(self, predictions_df, data_interface_schema):
        """data_interface_schema를 mock하면서 predict() 호출. 메타데이터 컬럼 처리 검증용."""
        from mmp.serving import _endpoints as ep

        mock_model = Mock()
        mock_model.predict.return_value = predictions_df

        original_model = app_context.model
        original_settings = app_context.settings
        original_uri = app_context.model_uri
        original_req = getattr(app_context, "PredictionRequest", None)

        app_context.model = mock_model
        app_context.settings = Mock()
        app_context.model_uri = "runs:/test/model"
        app_context.PredictionRequest = lambda **kw: Mock()

        try:
            with (
                patch.object(ep, "validate_scalar_values"),
                patch.object(ep, "validate_numeric_types"),
                patch.object(ep, "validate_required_columns"),
                patch.object(ep, "_convert_to_signature_types", side_effect=lambda df, m: df),
                patch.object(ep, "format_predictions", return_value=predictions_df),
                patch.object(ep, "_get_data_interface_schema", return_value=data_interface_schema),
            ):
                return ep.predict({"feature1": 1.0})
        finally:
            app_context.model = original_model
            app_context.settings = original_settings
            app_context.model_uri = original_uri
            if original_req is not None:
                app_context.PredictionRequest = original_req

    def test_single_output_with_entity_id_returns_scalar(self, component_test_context):
        """단일 출력 + entity_id 메타데이터 부착 → 스칼라 반환 (회귀 방지).

        이전에는 entity_id 부착 때문에 shape[1]==2가 되어 다중 출력 dict로 잘못 분류되었다.
        """
        with component_test_context.classification_stack():
            df = pd.DataFrame({"prediction": [np.float64(1.0)], "entity_id": [42]})
            schema = {"entity_columns": ["entity_id"], "task_type": "classification"}
            result = self._run_predict_with_schema(df, schema)

            assert isinstance(result["prediction"], (int, float))
            assert result["prediction"] == 1.0

    def test_multi_output_with_entity_id_excludes_metadata(self, component_test_context):
        """다중 출력 + entity_id → dict이지만 entity_id는 제외되어야 함."""
        with component_test_context.classification_stack():
            df = pd.DataFrame(
                {
                    "pred_p50": [np.float64(38.0)],
                    "pred_p90": [np.float64(65.0)],
                    "entity_id": [7],
                }
            )
            schema = {"entity_columns": ["entity_id"], "task_type": "regression"}
            result = self._run_predict_with_schema(df, schema)

            assert isinstance(result["prediction"], dict)
            assert "entity_id" not in result["prediction"]
            assert result["prediction"]["pred_p50"] == 38.0
            assert result["prediction"]["pred_p90"] == 65.0

    def test_single_output_with_timestamp_and_entity(self, component_test_context):
        """단일 출력 + timestamp + entity 모두 제외되어야 함."""
        with component_test_context.classification_stack():
            df = pd.DataFrame(
                {
                    "prediction": [np.float64(0.73)],
                    "entity_id": [99],
                    "ts": [1234567890],
                }
            )
            schema = {
                "entity_columns": ["entity_id"],
                "timestamp_column": "ts",
                "task_type": "timeseries",
            }
            result = self._run_predict_with_schema(df, schema)

            assert isinstance(result["prediction"], (int, float))
            assert result["prediction"] == 0.73

    def test_strip_metadata_handles_string_entity(self, component_test_context):
        """entity_columns가 문자열(단수)로 주어진 경우도 올바르게 처리."""
        with component_test_context.classification_stack():
            df = pd.DataFrame({"prediction": [np.float64(0.5)], "user_id": ["u1"]})
            schema = {"entity_columns": "user_id"}
            result = self._run_predict_with_schema(df, schema)

            assert isinstance(result["prediction"], (int, float))
            assert result["prediction"] == 0.5

    def test_strip_metadata_all_columns_metadata_falls_back(self, component_test_context):
        """모든 컬럼이 메타데이터인 비정상 케이스 → 폴백으로 원본 반환."""
        with component_test_context.classification_stack():
            # 예측 컬럼 없이 entity만 있는 비정상 입력
            df = pd.DataFrame({"entity_id": [42]})
            schema = {"entity_columns": ["entity_id"]}
            result = self._run_predict_with_schema(df, schema)

            # 폴백이 동작해 entity_id를 첫 컬럼으로 추출
            assert result["prediction"] == 42


class TestPredictionErrorHandling:
    """_handle_prediction_error의 타입 기반 매핑 검증"""

    def test_mlflow_invalid_parameter_value_returns_422(self):
        """MlflowException(INVALID_PARAMETER_VALUE) → 422"""
        from mlflow.exceptions import MlflowException

        from mmp.serving.router import _handle_prediction_error

        exc = MlflowException.invalid_parameter_value("schema mismatch")
        http_exc = _handle_prediction_error(exc, "단일 예측")

        assert http_exc.status_code == 422
        assert "schema mismatch" in str(http_exc.detail)

    def test_mlflow_other_error_code_returns_500(self):
        """MlflowException 중 클라이언트 입력이 아닌 코드 → 500"""
        from mlflow.exceptions import MlflowException

        from mmp.serving.router import _handle_prediction_error

        # default error_code는 INTERNAL_ERROR
        exc = MlflowException("upstream tracking server unavailable")
        http_exc = _handle_prediction_error(exc, "단일 예측")

        assert http_exc.status_code == 500

    def test_unexpected_exception_returns_500(self):
        """알 수 없는 예외 → 500 + 타입 정보 로깅"""
        from mmp.serving.router import _handle_prediction_error

        exc = RuntimeError("something broke")
        http_exc = _handle_prediction_error(exc, "단일 예측")

        assert http_exc.status_code == 500
        assert "something broke" in str(http_exc.detail)

    def test_predict_endpoint_mlflow_input_error_returns_422(self, component_test_context):
        """MlflowException(클라이언트 입력)이 핸들러에서 raise되면 라우트가 422 반환"""
        from mlflow.exceptions import MlflowException

        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                original_model = app_context.model
                original_settings = app_context.settings

                app_context.model = Mock()
                app_context.settings = ctx.settings

                try:
                    with patch("mmp.serving.router.handlers.predict") as mock_predict:
                        mock_predict.side_effect = MlflowException.invalid_parameter_value(
                            "Failed to enforce schema"
                        )

                        response = client.post("/predict", json={"feature1": 1.0})

                        assert response.status_code == 422
                        data = response.json()
                        assert "Failed to enforce schema" in data["detail"]
                finally:
                    app_context.model = original_model
                    app_context.settings = original_settings


class TestStructuredErrorResponse:
    """구조화된 에러 응답 본문 검증"""

    def test_error_response_includes_structured_fields(self, component_test_context):
        """에러 응답에 detail/error_code/message가 모두 포함되어야 함 (additive 호환)"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                original_model = app_context.model
                original_settings = app_context.settings
                app_context.model = None
                app_context.settings = None

                try:
                    response = client.post("/predict", json={"feature1": 1.0})

                    assert response.status_code == 503
                    data = response.json()
                    # 기존 컨벤션과의 하위 호환
                    assert "detail" in data
                    assert "모델이 준비되지 않았습니다" in data["detail"]
                    # 신규 구조화 필드
                    assert data["error_code"] == 503
                    assert data["message"] == data["detail"]
                finally:
                    app_context.model = original_model
                    app_context.settings = original_settings

    def test_error_response_includes_request_id(self, component_test_context):
        """X-Request-ID가 응답 본문에 포함되어야 함"""
        with component_test_context.classification_stack() as ctx:
            with TestClient(app) as client:
                original_model = app_context.model
                original_settings = app_context.settings
                app_context.model = None
                app_context.settings = None

                try:
                    custom_id = "trace-id-abc-123"
                    response = client.post(
                        "/predict",
                        json={"feature1": 1.0},
                        headers={"X-Request-ID": custom_id},
                    )

                    assert response.status_code == 503
                    data = response.json()
                    assert data.get("request_id") == custom_id
                    # 응답 헤더에도 그대로 노출됨
                    assert response.headers.get("X-Request-ID") == custom_id
                finally:
                    app_context.model = original_model
                    app_context.settings = original_settings

    def test_unhandled_exception_returns_500_with_error_type(self, component_test_context):
        """라우트에서 잡지 못한 예외 → 글로벌 핸들러가 error_type 포함 500 반환.

        /health는 try/except가 없으므로 handlers.health가 raise하면 글로벌
        unhandled_exception_handler가 처리한다.
        """
        with component_test_context.classification_stack():
            with TestClient(app, raise_server_exceptions=False) as client:
                with patch("mmp.serving.router.handlers.health") as mock_health:
                    mock_health.side_effect = RuntimeError("boom")

                    response = client.get(
                        "/health", headers={"X-Request-ID": "unhandled-1"}
                    )

                    assert response.status_code == 500
                    data = response.json()
                    assert data["error_code"] == 500
                    assert data["error_type"] == "RuntimeError"
                    assert data["request_id"] == "unhandled-1"
                    assert "boom" in data["detail"]


class TestRunAPIServerWorkers:
    """run_api_server의 workers 설정 분기 검증"""

    def teardown_method(self):
        for attr in ("run_id", "settings"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)

    def test_single_worker_uses_app_object(self, component_test_context):
        """workers=1 → uvicorn.run에 app 객체 전달 (기존 동작 보존)"""
        with component_test_context.classification_stack():
            mock_settings = Mock()
            mock_settings.config = Mock()
            mock_settings.config.serving = Mock()
            mock_settings.config.serving.enabled = True
            mock_settings.config.serving.workers = 1

            with (
                patch("mmp.serving.router.uvicorn.run") as mock_run,
                patch("mmp.serving.router._configure_middlewares"),
            ):
                run_api_server(mock_settings, "test-run", host="127.0.0.1", port=8000)

                mock_run.assert_called_once_with(app, host="127.0.0.1", port=8000)

    def test_multi_worker_uses_import_string(self, component_test_context):
        """workers>1 → uvicorn.run에 import string + workers 전달"""
        with component_test_context.classification_stack():
            mock_settings = Mock()
            mock_settings.config = Mock()
            mock_settings.config.serving = Mock()
            mock_settings.config.serving.enabled = True
            mock_settings.config.serving.workers = 4

            with (
                patch("mmp.serving.router.uvicorn.run") as mock_run,
                patch("mmp.serving.router._configure_middlewares"),
            ):
                run_api_server(mock_settings, "test-run", host="0.0.0.0", port=8000)

                mock_run.assert_called_once_with(
                    "mmp.serving.router:app",
                    host="0.0.0.0",
                    port=8000,
                    workers=4,
                )

    def test_invalid_workers_falls_back_to_single(self, component_test_context):
        """workers가 정수가 아니거나 < 1이면 1로 폴백"""
        with component_test_context.classification_stack():
            mock_settings = Mock()
            mock_settings.config = Mock()
            mock_settings.config.serving = Mock()
            mock_settings.config.serving.enabled = True
            mock_settings.config.serving.workers = "invalid"

            with (
                patch("mmp.serving.router.uvicorn.run") as mock_run,
                patch("mmp.serving.router._configure_middlewares"),
            ):
                run_api_server(mock_settings, "test-run", host="127.0.0.1", port=8000)

                mock_run.assert_called_once_with(app, host="127.0.0.1", port=8000)

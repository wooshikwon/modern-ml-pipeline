import asyncio
import uuid
from importlib.metadata import version as get_pkg_version
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from mlflow.exceptions import MlflowException
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from mmp.serving import _endpoints as handlers
from mmp.serving._context import app_context
from mmp.serving._lifespan import lifespan
from mmp.serving.schemas import (
    BatchPredictionResponse,
    HealthCheckResponse,
    MinimalPredictionResponse,
    ModelMetadataResponse,
    OptimizationHistoryResponse,
    ReadyCheckResponse,
)
from mmp.settings import Settings
from mmp.utils.core.logger import log_api, log_error, log_warn


def _get_version() -> str:
    """패키지 메타데이터에서 버전 정보를 가져옵니다."""
    try:
        return get_pkg_version("modern-ml-pipeline")
    except Exception:
        return "0.0.0"


# 테스트와 실제 서빙 모두에서 사용될 수 있는 최상위 app 객체
app = FastAPI(
    title="Modern ML Pipeline API",
    description="MMP 모델 서빙 API",
    version=_get_version(),
    lifespan=lifespan,
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    X-Request-ID 헤더를 처리하는 미들웨어.
    - 요청에 X-Request-ID가 있으면 그대로 사용
    - 없으면 새로운 UUID 생성
    - 응답 헤더에 X-Request-ID 포함
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    요청 타임아웃을 처리하는 미들웨어.
    지정된 시간 내에 응답하지 않으면 504 Gateway Timeout 반환.
    """

    def __init__(self, app, timeout_seconds: int = 30):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(
                call_next(request), timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            log_warn(
                f"Request timeout: {request.method} {request.url.path} exceeded {self.timeout_seconds}s",
                "API",
            )
            return JSONResponse(
                status_code=504,
                content={"detail": f"Request timeout ({self.timeout_seconds}s)"},
            )


# 미들웨어 적용 순서: 바깥쪽부터 Timeout → CORS → RequestID
# add_middleware는 스택 형태로 동작하므로 역순으로 등록
app.add_middleware(RequestIDMiddleware)

# Prometheus 메트릭 (config.serving.metrics_enabled로 제어)
_instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/health", "/metrics"],
    inprogress_name="http_requests_inprogress",
    inprogress_labels=True,
)
_instrumentator.instrument(app)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """모든 HTTPException을 구조화된 형태로 변환한다.

    `detail`은 보존되고 `error_code`/`message`/`request_id`가 추가된다.
    """
    request_id = request.headers.get("X-Request-ID")
    body = _build_error_body(
        status_code=exc.status_code,
        detail=str(exc.detail) if exc.detail is not None else "",
        request_id=request_id,
    )
    headers = getattr(exc, "headers", None)
    return JSONResponse(status_code=exc.status_code, content=body, headers=headers)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """라우트에서 잡히지 않은 예외에 대한 마지막 안전망. 항상 500을 반환한다."""
    log_error(f"Unhandled exception: {type(exc).__name__}: {exc}", "API")
    request_id = request.headers.get("X-Request-ID")
    body = _build_error_body(
        status_code=500,
        detail=str(exc),
        error_type=type(exc).__name__,
        request_id=request_id,
    )
    return JSONResponse(status_code=500, content=body)


@app.get("/", tags=["General"])
def root() -> Dict[str, str]:
    return {
        "message": "Modern ML Pipeline API",
        "status": "ready" if app_context.model else "error",
        "model_uri": app_context.model_uri,
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["General"])
def health_check() -> HealthCheckResponse:
    """
    Liveness 체크 (K8s livenessProbe용).
    프로세스 생존 여부만 확인. 모델 로드 상태와 무관하게 항상 200 반환.
    """
    return handlers.health()


@app.get("/ready", response_model=ReadyCheckResponse, tags=["General"])
def ready_check() -> ReadyCheckResponse:
    """
    Readiness 체크 (K8s readinessProbe용).
    모델이 로드되어 트래픽을 받을 준비가 되었는지 확인.
    """
    try:
        return handlers.ready()
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Ready check 중 오류 발생: {e}", "API")
        raise HTTPException(status_code=500, detail=str(e))


# MlflowException error_code 중 클라이언트 입력 문제로 분류할 코드들.
# MLflow가 raise한 메시지 텍스트가 아닌 안정적인 error_code 식별자로 매칭한다.
_MLFLOW_CLIENT_ERROR_CODES = frozenset(
    {
        "INVALID_PARAMETER_VALUE",
        "BAD_REQUEST",
        "INVALID_STATE",
    }
)


def _handle_prediction_error(e: Exception, log_prefix: str) -> HTTPException:
    """prediction 엔드포인트 공통 에러 매핑.

    예외 타입(MlflowException, ValueError 등)으로 분기한다. 키워드 매칭은 사용하지 않는다.
    - MlflowException(클라이언트 입력 문제) → 422
    - 기타 MlflowException → 500 (로그 후)
    - 그 외 예상치 못한 예외 → 500 (로그 후)
    """
    if isinstance(e, MlflowException):
        error_code = getattr(e, "error_code", None)
        if error_code in _MLFLOW_CLIENT_ERROR_CODES:
            return HTTPException(status_code=422, detail=str(e))
        log_error(
            f"{log_prefix} 중 MLflow 오류 (error_code={error_code}): {e}", "API"
        )
        return HTTPException(status_code=500, detail=str(e))

    log_error(f"{log_prefix} 중 오류 발생: {type(e).__name__}: {e}", "API")
    return HTTPException(status_code=500, detail=str(e))


def _build_error_body(
    status_code: int,
    detail: str,
    error_type: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """에러 응답 본문을 통일된 형태로 빌드한다.

    `detail`은 FastAPI 기본 컨벤션과 호환되도록 유지하고, 추가 필드를 덧붙인다.
    클라이언트는 `error_code`(HTTP status)로 분류하고, `error_type`으로 원인을 식별할 수 있다.
    """
    body: Dict[str, Any] = {
        "detail": detail,
        "error_code": status_code,
        "message": detail,
    }
    if error_type:
        body["error_type"] = error_type
    if request_id:
        body["request_id"] = request_id
    return body


def _check_model_ready() -> None:
    """모델과 설정이 준비되지 않았으면 503을 발생시킨다."""
    if not app_context.model or not app_context.settings:
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")


@app.post("/predict", response_model=MinimalPredictionResponse, tags=["Prediction"])
def predict_generic(request: Dict[str, Any]) -> MinimalPredictionResponse:
    _check_model_ready()
    try:
        prediction_result = handlers.predict(request)
        return MinimalPredictionResponse(**prediction_result)
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise _handle_prediction_error(e, "단일 예측")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: Dict[str, Any]) -> BatchPredictionResponse:
    """배치 예측 엔드포인트 - 여러 샘플을 한 번에 예측합니다."""
    _check_model_ready()
    try:
        return handlers.predict_batch(request)
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise _handle_prediction_error(e, "배치 예측")


# 모델 메타데이터 자기 기술 엔드포인트들


@app.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Model Metadata"])
def get_model_metadata() -> ModelMetadataResponse:
    """
    모델의 완전한 메타데이터 반환 (하이퍼파라미터 최적화, Data Leakage 방지 정보 포함)
    """
    try:
        return handlers.get_model_metadata()
    except Exception as e:
        log_error(f"모델 메타데이터 조회 중 오류 발생: {e}", "API")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/optimization", response_model=OptimizationHistoryResponse, tags=["Model Metadata"])
def get_optimization_history() -> OptimizationHistoryResponse:
    """
    하이퍼파라미터 최적화 과정의 상세 히스토리 반환
    """
    try:
        return handlers.get_optimization_history()
    except Exception as e:
        log_error(f"최적화 히스토리 조회 중 오류 발생: {e}", "API")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/schema", tags=["Model Metadata"])
def get_api_schema() -> Dict[str, Any]:
    """
    동적으로 생성된 API 스키마 정보 반환
    """
    try:
        return handlers.get_api_schema()
    except Exception as e:
        log_error(f"API 스키마 조회 중 오류 발생: {e}", "API")
        raise HTTPException(status_code=500, detail=str(e))


# 미들웨어 구성 상태 추적
_middlewares_configured = False


def _configure_middlewares(settings: Settings) -> None:
    """설정 기반 미들웨어 구성. 서버 시작 전 한 번만 호출."""
    global _middlewares_configured
    if _middlewares_configured:
        return
    _middlewares_configured = True

    serving = getattr(settings.config, "serving", None)
    if not serving:
        return

    # CORS 미들웨어 (설정에서 활성화된 경우)
    cors_config = getattr(serving, "cors", None)
    if cors_config and getattr(cors_config, "enabled", False):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_config.allow_origins,
            allow_credentials=cors_config.allow_credentials,
            allow_methods=cors_config.allow_methods,
            allow_headers=cors_config.allow_headers,
        )
        log_api(f"CORS 미들웨어 활성화: origins={cors_config.allow_origins}")

    # Timeout 미들웨어
    timeout_seconds = getattr(serving, "request_timeout_seconds", 30)
    if timeout_seconds > 0:
        app.add_middleware(TimeoutMiddleware, timeout_seconds=timeout_seconds)
        log_api(f"Timeout 미들웨어 활성화: {timeout_seconds}초")

    # Prometheus 메트릭 엔드포인트 노출
    metrics_enabled = getattr(serving, "metrics_enabled", True)
    if metrics_enabled:
        _instrumentator.expose(app, endpoint="/metrics", tags=["Monitoring"])
        log_api("Prometheus 메트릭 엔드포인트 활성화: /metrics")


def run_api_server(settings: Settings, run_id: str, host: str = "0.0.0.0", port: int = 8000):
    """
    FastAPI 서버를 실행하고, Lifespan 이벤트를 통해 모델을 로드합니다.
    """
    # 서빙 비활성화 시 차단
    if (
        hasattr(settings.config, "serving")
        and settings.config.serving
        and not getattr(settings.config.serving, "enabled", True)
    ):
        env_name = (
            settings.config.environment.name
            if hasattr(settings.config, "environment")
            else "unknown"
        )
        log_error(
            f"'{env_name}' 환경에서는 API 서빙이 비활성화되어 있습니다 (serving.enabled=false)",
            "API",
        )
        return

    # lifespan에서 접근할 수 있도록 app.state에 저장
    app.state.run_id = run_id
    app.state.settings = settings

    # 설정 기반 미들웨어 구성
    _configure_middlewares(settings)

    # workers 설정 추출. config.serving.workers가 정수 ≥ 1일 때만 사용.
    # (Mock 등 비정상 값을 안전하게 무시)
    serving_cfg = getattr(getattr(settings, "config", None), "serving", None)
    workers = 1
    if serving_cfg is not None:
        cfg_workers = getattr(serving_cfg, "workers", 1)
        if isinstance(cfg_workers, int) and cfg_workers >= 1:
            workers = cfg_workers

    if workers > 1:
        # 다중 워커 모드는 import string이 필요하다 (uvicorn이 fork 후 재import)
        log_api(f"다중 워커 모드로 시작: workers={workers}")
        uvicorn.run(
            "mmp.serving.router:app",
            host=host,
            port=port,
            workers=workers,
        )
    else:
        uvicorn.run(app, host=host, port=port)

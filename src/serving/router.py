import asyncio
import uuid
from importlib.metadata import version as get_pkg_version
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.serving import _endpoints as handlers
from src.serving._context import app_context
from src.serving._lifespan import lifespan, setup_api_context
from src.serving.schemas import (
    BatchPredictionResponse,
    HealthCheckResponse,
    MinimalPredictionResponse,
    ModelMetadataResponse,
    OptimizationHistoryResponse,
    ReadyCheckResponse,
)
from src.settings import Settings
from src.utils.core.logger import logger


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
            logger.warning(
                f"Request timeout: {request.method} {request.url.path} "
                f"exceeded {self.timeout_seconds}s"
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
        logger.error(f"Ready check 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=MinimalPredictionResponse, tags=["Prediction"])
def predict_generic(request: Dict[str, Any]) -> MinimalPredictionResponse:
    if not app_context.model or not app_context.settings:
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")
    try:
        # Data Interface 기반 API 엔드포인트는 모든 fetcher 타입을 지원
        # target, entity, timestamp columns을 제외한 feature columns로 API 생성되므로
        # pass_through fetcher도 문제없이 작동
        prediction_result = handlers.predict(request)
        return MinimalPredictionResponse(**prediction_result)
    except HTTPException as he:
        # 정책 위반 등은 원래 상태코드로 전달
        raise he
    except ValueError as ve:
        # 입력 형식 오류 등은 422로 변환
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        # MLflow 입력 스키마 위반 메시지 검사 → 422로 매핑
        msg = str(e)
        lower = msg.lower()
        if any(
            k in lower
            for k in [
                "failed to enforce schema",
                "missing inputs",
                "failed to convert column",
                "invalid parameter value",
            ]
        ):
            raise HTTPException(status_code=422, detail=msg)
        logger.error(f"단일 예측 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: Dict[str, Any]) -> BatchPredictionResponse:
    """배치 예측 엔드포인트 - 여러 샘플을 한 번에 예측합니다."""
    if not app_context.model or not app_context.settings:
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")
    try:
        return handlers.predict_batch(request)
    except HTTPException as he:
        raise he
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        msg = str(e)
        lower = msg.lower()
        if any(
            k in lower
            for k in [
                "failed to enforce schema",
                "missing inputs",
                "failed to convert column",
                "invalid parameter value",
            ]
        ):
            raise HTTPException(status_code=422, detail=msg)
        logger.error(f"배치 예측 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# 모델 메타데이터 자기 기술 엔드포인트들


@app.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Model Metadata"])
def get_model_metadata() -> ModelMetadataResponse:
    """
    모델의 완전한 메타데이터 반환 (하이퍼파라미터 최적화, Data Leakage 방지 정보 포함)
    """
    try:
        return handlers.get_model_metadata()
    except Exception as e:
        logger.error(f"모델 메타데이터 조회 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/optimization", response_model=OptimizationHistoryResponse, tags=["Model Metadata"])
def get_optimization_history() -> OptimizationHistoryResponse:
    """
    하이퍼파라미터 최적화 과정의 상세 히스토리 반환
    """
    try:
        return handlers.get_optimization_history()
    except Exception as e:
        logger.error(f"최적화 히스토리 조회 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/schema", tags=["Model Metadata"])
def get_api_schema() -> Dict[str, Any]:
    """
    동적으로 생성된 API 스키마 정보 반환
    """
    try:
        return handlers.get_api_schema()
    except Exception as e:
        logger.error(f"API 스키마 조회 중 오류 발생: {e}", exc_info=True)
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
        logger.info(f"CORS 미들웨어 활성화: origins={cors_config.allow_origins}")

    # Timeout 미들웨어
    timeout_seconds = getattr(serving, "request_timeout_seconds", 30)
    if timeout_seconds > 0:
        app.add_middleware(TimeoutMiddleware, timeout_seconds=timeout_seconds)
        logger.info(f"Timeout 미들웨어 활성화: {timeout_seconds}초")

    # Prometheus 메트릭 엔드포인트 노출
    metrics_enabled = getattr(serving, "metrics_enabled", True)
    if metrics_enabled:
        _instrumentator.expose(app, endpoint="/metrics", tags=["Monitoring"])
        logger.info("Prometheus 메트릭 엔드포인트 활성화: /metrics")


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
        logger.error(
            f"'{env_name}' 환경에서는 API 서빙이 비활성화되어 있습니다. (serving.enabled=false)"
        )
        return

    # 서버 시작 시 컨텍스트 설정
    setup_api_context(run_id=run_id, settings=settings)

    # 설정 기반 미들웨어 구성
    _configure_middlewares(settings)

    uvicorn.run(app, host=host, port=port)

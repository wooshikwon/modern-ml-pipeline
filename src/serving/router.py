import uvicorn
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

from src.settings import Settings
from src.utils.system.logger import logger
from src.serving._context import app_context
from src.serving._lifespan import lifespan, setup_api_context
from src.serving import _endpoints as handlers
from src.components._fetcher import PassThroughAugmenter
from src.serving.schemas import (
    HealthCheckResponse,
    ModelMetadataResponse,
    OptimizationHistoryResponse,
    MinimalPredictionResponse,
)

# 테스트와 실제 서빙 모두에서 사용될 수 있는 최상위 app 객체
app = FastAPI(
    title="Modern ML Pipeline API",
    description="Blueprint v17.0 기반 모델 서빙 API",
    version="17.0.0",
    lifespan=lifespan,
)

def _register_dynamic_routes_if_needed():
    """Setup 후 동적 라우트를 보장한다(/predict)."""
    # 이미 등록되었는지 확인
    if any(getattr(r, "path", None) == "/predict" for r in app.router.routes):
        return
    PredictionRequest = app_context.PredictionRequest
    if PredictionRequest is None:
        return

    def predict(request: PredictionRequest) -> MinimalPredictionResponse:
        try:
            prediction_result = handlers.predict(request.model_dump())
            return MinimalPredictionResponse(**prediction_result)
        except Exception as e:
            logger.error(f"단일 예측 중 오류 발생: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    app.add_api_route(
        "/predict",
        predict,
        response_model=MinimalPredictionResponse,
        methods=["POST"],
        tags=["Prediction"],
    )

@app.get("/", tags=["General"])
def root() -> Dict[str, str]:
    return {
        "message": "Modern ML Pipeline API",
        "status": "ready" if app_context.model else "error",
        "model_uri": app_context.model_uri,
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["General"])
def health_check() -> HealthCheckResponse:
    try:
        return handlers.health()
    except Exception as e:
        logger.error(f"Health check 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=MinimalPredictionResponse, tags=["Prediction"])
def predict_generic(request: Dict[str, Any]) -> MinimalPredictionResponse:
    if not app_context.model or not app_context.settings:
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")
    try:
        # 서빙 정책: pass_through/폴백 차단
        try:
            wrapped_model = app_context.model.unwrap_python_model()
            if isinstance(wrapped_model.trained_augmenter, PassThroughAugmenter):
                raise HTTPException(status_code=503, detail="Serving with 'pass_through' augmenter is not allowed.")
        except HTTPException:
            raise
        except Exception:
            pass
        prediction_result = handlers.predict(request)
        return MinimalPredictionResponse(**prediction_result)
    except HTTPException as he:
        # 정책 위반 등은 원래 상태코드로 전달
        raise he
    except Exception as e:
        logger.error(f"단일 예측 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 🆕 Blueprint v17.0: 모델 메타데이터 자기 기술 엔드포인트들
    
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

def run_api_server(settings: Settings, run_id: str, host: str = "0.0.0.0", port: int = 8000):
    """
    FastAPI 서버를 실행하고, Lifespan 이벤트를 통해 모델을 로드합니다.
    """
    # Blueprint 원칙 9: 환경별 기능 분리 - API 서빙 시스템적 차단
    if hasattr(settings, 'serving') and settings.serving and not getattr(settings.serving, 'enabled', True):
        logger.error(f"'{settings.environment.env_name}' 환경에서는 API 서빙이 비활성화되어 있습니다.")
        return

    # 서버 시작 시 컨텍스트 설정
    setup_api_context(run_id=run_id, settings=settings)

    # [신규] Augmenter 타입 명시적 검증
    wrapped_model = app_context.model.unwrap_python_model()
    if isinstance(wrapped_model.trained_augmenter, PassThroughAugmenter):
        raise TypeError(
            "API serving is not supported when the augmenter is 'pass_through'. "
            "A feature store connection is required."
        )
    # 정적 predict 엔드포인트 사용

    uvicorn.run(app, host=host, port=port)

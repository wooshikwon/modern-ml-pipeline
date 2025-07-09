import os
import pandas as pd
import mlflow
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

# config.settings에서 통합 설정 객체를 import
from config.settings import settings
from src.utils.logger import logger
from serving.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthCheckResponse
)

# 전역 모델 객체
model: mlflow.pyfunc.PyFuncModel | None = None
model_uri_g: str = ""

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 시작 시 Model Registry에서 Production 모델을 로드합니다.
    """
    global model, model_uri_g
    logger.info("FastAPI 애플리케이션 시작...")
    
    model_stage = os.getenv("MODEL_STAGE", "Production")
    model_name = settings.model.name
    model_uri = f"models:/{model_name}/{model_stage}"
    model_uri_g = model_uri
    
    logger.info(f"'{model_uri}' 모델 로드를 시도합니다.")

    try:
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        logger.info(f"모델이 성공적으로 로드되었습니다: {model_uri}")
        
    except Exception as e:
        logger.error(f"모델 로딩 실패: {e}", exc_info=True)
        model = None

    yield
    
    logger.info("FastAPI 애플리케이션 종료.")


# FastAPI 앱 생성
app = FastAPI(
    title="Uplift Virtual Coupon API",
    description="가상 쿠폰 발송 효과 예측 API",
    version="1.0.1",
    lifespan=lifespan
)

@app.get("/", tags=["General"])
async def root() -> Dict[str, str]:
    """API 기본 정보 및 상태를 반환합니다."""
    return {
        "message": "Uplift Virtual Coupon Prediction API",
        "status": "ready" if model else "error",
        "model_uri": model_uri_g if model else "N/A"
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["General"])
async def health() -> HealthCheckResponse:
    """서비스의 헬스 체크를 수행합니다."""
    if not model:
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")
    
    return HealthCheckResponse(
        status="healthy",
        model_uri=model_uri_g,
        model_name=settings.model.name
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """단일 데이터에 대한 Uplift 스코어를 예측합니다."""
    if not model:
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")

    try:
        input_df = pd.DataFrame([request.model_dump()])
        result_df = model.predict(input_df)
        uplift_score = result_df['uplift_score'].iloc[0]
        
        return PredictionResponse(
            uplift_score=uplift_score,
            model_uri=model_uri_g
        )
    except Exception as e:
        logger.error(f"예측 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """배치 데이터에 대한 Uplift 스코어를 예측합니다."""
    if not model:
        raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")

    try:
        input_data = [sample.model_dump() for sample in request.samples]
        if not input_data:
            raise HTTPException(status_code=400, detail="입력 샘플이 비어있습니다.")
            
        input_df = pd.DataFrame(input_data)
        result_df = model.predict(input_df)
        uplift_scores = result_df['uplift_score'].tolist()
        
        return BatchPredictionResponse(
            predictions=uplift_scores,
            model_uri=model_uri_g,
            sample_count=len(uplift_scores)
        )
    except Exception as e:
        logger.error(f"배치 예측 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
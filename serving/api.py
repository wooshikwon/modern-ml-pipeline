import os
import pandas as pd
import mlflow
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from typing import Dict, Any, List, Type
from pydantic import BaseModel

from config.settings import Settings
from src.utils.logger import logger
from serving.schemas import (
    create_dynamic_prediction_request,
    create_batch_prediction_request,
    PredictionResponse,
    BatchPredictionResponse,
    HealthCheckResponse,
)

# --- 애플리케이션 컨텍스트 ---
# FastAPI 앱의 상태(로드된 모델, 설정 등)를 관리하는 중앙 저장소
class AppContext:
    def __init__(self):
        self.model: mlflow.pyfunc.PyFuncModel | None = None
        self.model_uri: str = ""
        self.settings: Settings | None = None
        self.PredictionRequest: Type[BaseModel] | None = None
        self.BatchPredictionRequest: Type[BaseModel] | None = None

# 앱 컨텍스트 인스턴스 생성
app_context = AppContext()


# --- FastAPI 앱 팩토리 ---
def create_app(settings: Settings) -> FastAPI:
    """
    설정 객체를 기반으로 FastAPI 애���리케이션을 생성하고 구성합니다.
    """
    # 동적 스키마 생성
    app_context.PredictionRequest = create_dynamic_prediction_request(settings)
    app_context.BatchPredictionRequest = create_batch_prediction_request(app_context.PredictionRequest)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        애플리케이션 시작 시 모델을 로드하고, 종료 시 정리합니다.
        """
        logger.info("FastAPI 애플리케이션 시작...")
        app_context.settings = settings
        
        model_stage = os.getenv("MODEL_STAGE", "Production")
        model_name = settings.model.name
        model_uri = f"models:/{model_name}/{model_stage}"
        app_context.model_uri = model_uri
        
        logger.info(f"'{model_uri}' 모델 로드를 시도합니다.")
        try:
            mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
            app_context.model = mlflow.pyfunc.load_model(model_uri=model_uri)
            logger.info(f"모델이 성공적으로 로드되었습니다: {model_uri}")
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}", exc_info=True)
            app_context.model = None
        
        yield
        
        logger.info("FastAPI 애플리케이션 종료.")

    # FastAPI 앱 생성
    app = FastAPI(
        title=f"Uplift Model API: {settings.model.name}",
        description="가상 쿠폰 발송 효과 예측 API",
        version="2.0.0",
        lifespan=lifespan
    )

    # --- API 엔드포인트 정의 ---
    @app.get("/", tags=["General"])
    async def root() -> Dict[str, str]:
        return {
            "message": "Uplift Model Prediction API",
            "status": "ready" if app_context.model else "error",
            "model_uri": app_context.model_uri if app_context.model else "N/A"
        }

    @app.get("/health", response_model=HealthCheckResponse, tags=["General"])
    async def health() -> HealthCheckResponse:
        if not app_context.model or not app_context.settings:
            raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")
        return HealthCheckResponse(
            status="healthy",
            model_uri=app_context.model_uri,
            model_name=app_context.settings.model.name
        )

    # 동적 스키마를 사용하기 위해 Depends를 사용
    @app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
    async def predict(request: app_context.PredictionRequest) -> PredictionResponse:
        if not app_context.model:
            raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")
        try:
            input_df = pd.DataFrame([request.dict()])
            result_df = app_context.model.predict(input_df)
            uplift_score = result_df[result_df.columns[0]].iloc[0]
            return PredictionResponse(uplift_score=uplift_score, model_uri=app_context.model_uri)
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
    async def predict_batch(request: app_context.BatchPredictionRequest) -> BatchPredictionResponse:
        if not app_context.model:
            raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")
        try:
            input_data = [sample.dict() for sample in request.samples]
            if not input_data:
                raise HTTPException(status_code=400, detail="입력 샘플이 비어있습니다.")
            
            input_df = pd.DataFrame(input_data)
            result_df = app_context.model.predict(input_df)
            uplift_scores = result_df[result_df.columns[0]].tolist()
            
            return BatchPredictionResponse(
                predictions=uplift_scores,
                model_uri=app_context.model_uri,
                sample_count=len(uplift_scores)
            )
        except Exception as e:
            logger.error(f"배치 예측 중 오류 발생: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return app

# --- 서버 실행 함수 ---
def run_api_server(settings: Settings, host: str, port: int):
    """
    Uvicorn을 사용하여 FastAPI 애플리케이션을 실행합니다.
    """
    app = create_app(settings)
    uvicorn.run(app, host=host, port=port)

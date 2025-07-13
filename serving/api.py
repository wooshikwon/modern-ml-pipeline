import uvicorn
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import Dict, Any, List, Type
from pydantic import BaseModel, create_model

from src.settings.settings import Settings
from src.utils.system.logger import logger
from src.utils.system import mlflow_utils
from serving.schemas import (
    get_pk_from_loader_sql,
    create_dynamic_prediction_request,
    create_batch_prediction_request,
    PredictionResponse,
    BatchPredictionResponse,
    HealthCheckResponse,
)

class AppContext:
    def __init__(self):
        self.model: mlflow.pyfunc.PyFuncModel | None = None
        self.model_uri: str = ""
        self.settings: Settings | None = None
        self.feature_store_config: Dict | None = None
        self.feature_columns: List[str] = []
        self.join_key: str = ""
        self.PredictionRequest: Type[BaseModel] = create_model("DefaultPredictionRequest")
        self.BatchPredictionRequest: Type[BaseModel] = create_model("DefaultBatchPredictionRequest")

app_context = AppContext()

def create_app(run_id: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("FastAPI 애플리케이션 시작...")
        
        try:
            # 1. 지정된 run_id로 완전한 Wrapped Artifact 로드
            model_uri = f"runs:/{run_id}/model"
            app_context.model = mlflow.pyfunc.load_model(model_uri)
            app_context.model_uri = model_uri
            logger.info(f"모델 로드 완료: {model_uri}")

            # 2. 현재 환경의 config 로드 (서빙 설정만 필요)
            from src.settings.settings import load_settings
            temp_settings = load_settings("xgboost_x_learner")  # 환경 설정만 사용
            app_context.settings = temp_settings
            app_context.feature_store_config = temp_settings.serving.realtime_feature_store

            # 3. Wrapper의 내장 SQL 스냅샷에서 PK 목록 추출 (동적 API 스키마 생성)
            from serving.schemas import get_pk_from_loader_sql
            pk_fields = get_pk_from_loader_sql(app_context.model.loader_sql_snapshot)
            logger.info(f"API 요청 PK 필드를 동적으로 생성합니다: {pk_fields}")

            # 4. Feature Store 조회 준비 (Blueprint 4.2.3의 4번)
            from src.utils.system.sql_utils import parse_feature_columns
            feature_columns, join_key = parse_feature_columns(app_context.model.augmenter_sql_snapshot)
            app_context.feature_columns = feature_columns
            app_context.join_key = join_key
            logger.info(f"Feature Store 조회 준비 완료: {len(feature_columns)}개 컬럼, JOIN 키: {join_key}")

            # 5. 동적 Pydantic 모델 생성
            app_context.PredictionRequest = create_dynamic_prediction_request(
                model_name="dynamic", pk_fields=pk_fields
            )
            app_context.BatchPredictionRequest = create_batch_prediction_request(
                app_context.PredictionRequest
            )
            logger.info("동적 API 스키마 생성이 완료되었습니다.")

        except Exception as e:
            logger.error(f"모델 로딩 또는 API 스키마 생성 실패: {e}", exc_info=True)
            app_context.model = None
        
        yield
        
        logger.info("FastAPI 애플리케이션 종료.")

    app = FastAPI(
        title=f"Uplift Model API (Run ID: {run_id})",
        description="가상 쿠폰 발송 효과 예측 API - Blueprint v13.0",
        version="13.0.0",
        lifespan=lifespan,
    )

    @app.get("/", tags=["General"])
    async def root() -> Dict[str, str]:
        return {
            "message": "Uplift Model Prediction API",
            "status": "ready" if app_context.model else "error",
            "model_uri": app_context.model_uri,
        }

    @app.get("/health", response_model=HealthCheckResponse, tags=["General"])
    async def health() -> HealthCheckResponse:
        if not app_context.model or not app_context.settings:
            raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")
        
        # 모델 정보를 Wrapper의 recipe_snapshot에서 가져오기
        model_info = "unknown"
        if app_context.model and hasattr(app_context.model, 'recipe_snapshot'):
            model_info = app_context.model.recipe_snapshot.get('class_path', 'unknown')
        
        return HealthCheckResponse(
            status="healthy",
            model_uri=app_context.model_uri,
            model_name=model_info,
        )

    @app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
    async def predict(request: app_context.PredictionRequest) -> PredictionResponse:
        if not app_context.model or not app_context.settings:
            raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")
        try:
            input_df = pd.DataFrame([request.dict()])
            predict_params = {
                "run_mode": "serving",
                "feature_store_config": app_context.feature_store_config,
                "feature_columns": app_context.feature_columns,
            }
            predictions = app_context.model.predict(input_df, params=predict_params)
            
            uplift_score = predictions["uplift_score"].iloc[0]
            
            return PredictionResponse(
                uplift_score=uplift_score, model_uri=app_context.model_uri
            )
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
    async def predict_batch(
        request: app_context.BatchPredictionRequest,
    ) -> BatchPredictionResponse:
        if not app_context.model or not app_context.settings:
            raise HTTPException(status_code=503, detail="모델이 준비되지 않았습니다.")
        try:
            input_df = pd.DataFrame([sample.dict() for sample in request.samples])
            if input_df.empty:
                raise HTTPException(status_code=400, detail="입력 샘플이 비어있습니다.")
            
            predict_params = {
                "run_mode": "serving",
                "feature_store_config": app_context.settings.serving.realtime_feature_store,
            }
            predictions_df = app_context.model.predict(input_df, params=predict_params)
            
            return BatchPredictionResponse(
                predictions=predictions_df.to_dict(orient="records"),
                model_uri=app_context.model_uri,
                sample_count=len(predictions_df),
            )
        except Exception as e:
            logger.error(f"배치 예측 중 오류 발생: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return app

def run_api_server(run_id: str, host: str = "0.0.0.0", port: int = 8000):
    """
    run_id 기반 API 서버 실행
    Blueprint v13.0 완전 구현: 정확한 모델 식별과 재현성 보장
    """
    app = create_app(run_id)
    uvicorn.run(app, host=host, port=port)

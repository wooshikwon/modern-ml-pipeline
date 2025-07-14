import uvicorn
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import Dict, Any, List, Type
from pydantic import BaseModel, create_model

from src.settings import Settings
from src.utils.system.logger import logger
from src.utils.system import mlflow_utils
from serving.schemas import (
    get_pk_from_loader_sql,
    create_dynamic_prediction_request,
    create_batch_prediction_request,
    PredictionResponse,
    BatchPredictionResponse,
    HealthCheckResponse,
    # 🆕 Blueprint v17.0: 새로운 메타데이터 스키마들
    ModelMetadataResponse,
    OptimizationHistoryResponse,
    HyperparameterOptimizationInfo,
    TrainingMethodologyInfo,
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
            from src.settings import load_settings
            temp_settings = load_settings("xgboost_x_learner")  # 환경 설정만 사용
            app_context.settings = temp_settings
            app_context.feature_store_config = temp_settings.serving.realtime_feature_store

            # 🆕 3. Blueprint v17.0: 정교한 SQL 파싱으로 API 스키마 생성
            from src.utils.system.sql_utils import parse_select_columns, parse_feature_columns
            
            # loader_sql_snapshot에서 API 입력 컬럼 추출
            pk_fields = parse_select_columns(app_context.model.loader_sql_snapshot)
            logger.info(f"API 요청 PK 필드를 동적으로 생성합니다: {pk_fields}")

            # 4. Feature Store 조회 준비 (Blueprint 4.2.3의 4번)
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
            
            # 🆕 Blueprint v17.0: 최적화 정보 포함
            hpo_info = app_context.model.hyperparameter_optimization
            optimization_enabled = hpo_info.get("enabled", False)
            best_score = hpo_info.get("best_score", 0.0) if optimization_enabled else 0.0
            
            return PredictionResponse(
                uplift_score=uplift_score, 
                model_uri=app_context.model_uri,
                optimization_enabled=optimization_enabled,
                best_score=best_score,
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
            
            # 🆕 Blueprint v17.0: 최적화 정보 포함
            hpo_info = app_context.model.hyperparameter_optimization
            optimization_enabled = hpo_info.get("enabled", False)
            best_score = hpo_info.get("best_score", 0.0) if optimization_enabled else 0.0
            
            return BatchPredictionResponse(
                predictions=predictions_df.to_dict(orient="records"),
                model_uri=app_context.model_uri,
                sample_count=len(predictions_df),
                optimization_enabled=optimization_enabled,
                best_score=best_score,
            )
        except Exception as e:
            logger.error(f"배치 예측 중 오류 발생: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # 🆕 Blueprint v17.0: 모델 메타데이터 자기 기술 엔드포인트들
    
    @app.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Model Metadata"])
    async def get_model_metadata() -> ModelMetadataResponse:
        """
        모델의 완전한 메타데이터 반환 (하이퍼파라미터 최적화, Data Leakage 방지 정보 포함)
        """
        try:
            if app_context.model is None:
                raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")
            
            # 하이퍼파라미터 최적화 정보 구성
            hpo_info = app_context.model.hyperparameter_optimization
            hyperparameter_optimization = HyperparameterOptimizationInfo(
                enabled=hpo_info.get("enabled", False),
                engine=hpo_info.get("engine", ""),
                best_params=hpo_info.get("best_params", {}),
                best_score=hpo_info.get("best_score", 0.0),
                total_trials=hpo_info.get("total_trials", 0),
                pruned_trials=hpo_info.get("pruned_trials", 0),
                optimization_time=str(hpo_info.get("optimization_time", "")),
            )
            
            # 학습 방법론 정보 구성
            tm_info = app_context.model.training_methodology
            training_methodology = TrainingMethodologyInfo(
                train_test_split_method=tm_info.get("train_test_split_method", ""),
                train_ratio=tm_info.get("train_ratio", 0.8),
                validation_strategy=tm_info.get("validation_strategy", ""),
                preprocessing_fit_scope=tm_info.get("preprocessing_fit_scope", ""),
                random_state=tm_info.get("random_state", 42),
            )
            
            # API 스키마 정보 구성
            api_schema = {
                "input_fields": [field for field in app_context.PredictionRequest.__fields__.keys()],
                "sql_source": "loader_sql_snapshot",
                "feature_columns": app_context.feature_columns,
                "join_key": app_context.join_key,
            }
            
            return ModelMetadataResponse(
                model_uri=app_context.model_uri,
                model_class_path=getattr(app_context.model, "model_class_path", ""),
                hyperparameter_optimization=hyperparameter_optimization,
                training_methodology=training_methodology,
                training_metadata=app_context.model.training_metadata,
                api_schema=api_schema,
            )
            
        except Exception as e:
            logger.error(f"모델 메타데이터 조회 중 오류 발생: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/model/optimization", response_model=OptimizationHistoryResponse, tags=["Model Metadata"])
    async def get_optimization_history() -> OptimizationHistoryResponse:
        """
        하이퍼파라미터 최적화 과정의 상세 히스토리 반환
        """
        try:
            if app_context.model is None:
                raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")
            
            hpo_info = app_context.model.hyperparameter_optimization
            
            if not hpo_info.get("enabled", False):
                return OptimizationHistoryResponse(
                    enabled=False,
                    optimization_history=[],
                    search_space={},
                    convergence_info={"message": "하이퍼파라미터 최적화가 비활성화되었습니다."},
                    timeout_occurred=False,
                )
            
            return OptimizationHistoryResponse(
                enabled=True,
                optimization_history=hpo_info.get("optimization_history", []),
                search_space=hpo_info.get("search_space", {}),
                convergence_info={
                    "best_score": hpo_info.get("best_score", 0.0),
                    "total_trials": hpo_info.get("total_trials", 0),
                    "pruned_trials": hpo_info.get("pruned_trials", 0),
                    "optimization_time": str(hpo_info.get("optimization_time", "")),
                },
                timeout_occurred=hpo_info.get("timeout_occurred", False),
            )
            
        except Exception as e:
            logger.error(f"최적화 히스토리 조회 중 오류 발생: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/model/schema", tags=["Model Metadata"])
    async def get_api_schema() -> Dict[str, Any]:
        """
        동적으로 생성된 API 스키마 정보 반환
        """
        try:
            if app_context.model is None:
                raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")
            
            return {
                "prediction_request_schema": app_context.PredictionRequest.schema(),
                "batch_prediction_request_schema": app_context.BatchPredictionRequest.schema(),
                "loader_sql_snapshot": app_context.model.loader_sql_snapshot,
                "extracted_fields": [field for field in app_context.PredictionRequest.__fields__.keys()],
                "feature_store_info": {
                    "feature_columns": app_context.feature_columns,
                    "join_key": app_context.join_key,
                    "feature_store_config": app_context.feature_store_config,
                },
            }
            
        except Exception as e:
            logger.error(f"API 스키마 조회 중 오류 발생: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return app

def run_api_server(run_id: str, host: str = "0.0.0.0", port: int = 8000):
    """
    run_id 기반 API 서버 실행
    Blueprint v13.0 완전 구현: 정확한 모델 식별과 재현성 보장
    """
    app = create_app(run_id)
    uvicorn.run(app, host=host, port=port)
